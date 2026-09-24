"""
Stage 3: Meta-Controller RL Fine-Tuning

Train the routing policy with PPO to optimize a multi-objective reward.

Each episode is one query (a contextual bandit): the engine samples routing
actions from the policy, executes them, and the trainer scores the response.
Updates use fresh on-policy batches of `batch_size` episodes.
"""

import json
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from mantis.inference.engine import MANTISInferenceEngine
from mantis.models.critic import CriticValueNetwork

DEMO_QA = [
    ("What is the capital of France?", "Paris"),
    ("Summarize the plot of the movie Inception.", "A thief who enters people's dreams to steal information."),
    ("Who wrote the book 'Pride and Prejudice'?", "Jane Austen"),
    ("Explain the theory of relativity in simple terms.", "Space and time are connected, and massive objects bend spacetime."),
    ("What are the main causes of climate change?", "Greenhouse gas emissions from human activities."),
    ("How does photosynthesis work?", "Plants convert sunlight into energy using chlorophyll."),
    ("What is the Pythagorean theorem?", "In a right triangle, the square of the hypotenuse equals the sum of squares of the other sides."),
    ("Describe the water cycle.", "Water evaporates, condenses into clouds, and falls as precipitation."),
    ("What is DNA?", "DNA is the molecule that carries genetic information."),
    ("How do computers store information?", "Computers use binary code to store data as 0s and 1s."),
]


class PPOTrainer:
    """
    PPO trainer for the meta-controller and its state encoder.

    Optimizes routing policy to balance:
    - Accuracy
    - Latency
    - Compute cost
    - Confidence calibration
    """

    def __init__(
        self,
        engine: MANTISInferenceEngine,
        value_network: nn.Module,
        alpha: float = 1.0,  # Accuracy weight
        beta: float = 0.3,   # Latency weight
        gamma: float = 0.2,  # Compute weight
        delta: float = 0.5,  # Calibration weight
        lr: float = 1e-5,
        ppo_epsilon: float = 0.2,
        batch_size: int = 256,
        n_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.engine = engine
        self.device = engine.device
        self.value_net = value_network.to(self.device).eval()

        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.epsilon = ppo_epsilon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size

        # Dropout stays off (modules in eval mode) so old and new log-probs agree
        policy_params = list(engine.meta.parameters()) + list(engine.state_encoder.parameters())
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_params = policy_params

        # Latency is scored relative to a running mean of observed latency
        self.latency_baseline = None

        self.stats = {'episodes': 0, 'total_reward': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

    def train(self, train_pairs: List[Tuple[str, str]], val_pairs: List[Tuple[str, str]], num_episodes: int = 50000):
        """
        Main RL training loop.

        Args:
            train_pairs: (query, ground_truth) pairs for rollouts
            val_pairs: held-out pairs scored with the deterministic policy
            num_episodes: Number of episodes to train
        """
        print(f"Starting PPO training for {num_episodes} episodes")
        batch = []

        for episode in range(1, num_episodes + 1):
            query, ground_truth = random.choice(train_pairs)
            transition = self.collect_episode(query, ground_truth)
            batch.append(transition)

            self.stats['episodes'] += 1
            self.stats['total_reward'] += transition['reward']

            if len(batch) == self.batch_size:
                self.stats['policy_loss'], self.stats['value_loss'] = self.update(batch)
                batch = []

            if episode % 100 == 0:
                avg_reward = self.stats['total_reward'] / self.stats['episodes']
                print(f"Episode {episode} | Avg Reward: {avg_reward:.4f} | "
                      f"Policy Loss: {self.stats['policy_loss']:.4f} | "
                      f"Value Loss: {self.stats['value_loss']:.4f}")

            if episode % 1000 == 0:
                print(f"Validation Reward: {self.validate(val_pairs):.4f}")

        if len(batch) > 1:
            self.update(batch)
        print("PPO training complete!")

    def collect_episode(self, query: str, ground_truth: str) -> Dict:
        """Run one exploratory episode and return its transition."""
        result = self.engine.generate(query, explore=True, return_details=True)
        transition = dict(result['rollout'])
        transition['reward'] = self.compute_reward(result, ground_truth)
        return transition

    def compute_reward(self, result: Dict, ground_truth: str, update_baseline: bool = True) -> float:
        """
        Multi-objective reward function.

        R = α·R_acc - β·R_lat - γ·R_comp + δ·R_calib
        """
        correct = self._check_correctness(result['response'], ground_truth)
        r_acc = 1.0 if correct else -1.0

        latency = result['latency']
        if self.latency_baseline is None:
            self.latency_baseline = latency
        r_lat = latency / max(self.latency_baseline, 1e-6)
        if update_baseline:
            self.latency_baseline = 0.95 * self.latency_baseline + 0.05 * latency

        # Memory gates only execute on the full path
        full = result['path'] == 'full'
        gates = result['routing_decisions']
        r_comp = (
            0.5 * self.engine.active_param_ratio +
            0.3 * float(full and gates['episodic']) +
            0.2 * float(full and gates['semantic'])
        )

        # Calibration: uncertainty should be low when correct, high when wrong
        u = result['uncertainty']
        r_calib = -((u - (0.0 if correct else 1.0)) ** 2)

        return self.alpha * r_acc - self.beta * r_lat - self.gamma * r_comp + self.delta * r_calib

    def update(self, batch: List[Dict]) -> Tuple[float, float]:
        """
        PPO update on one on-policy batch.

        Returns:
            (mean policy loss, mean value loss)
        """
        stack = lambda key: torch.stack([t[key] for t in batch]).to(self.device)
        query_emb = stack('query_emb')
        features = stack('features')
        gate_actions = stack('gate_actions')
        expert_bias = stack('expert_bias')
        gate_mask = stack('gate_mask')
        old_log_probs = stack('log_prob')
        rewards = torch.tensor([t['reward'] for t in batch], device=self.device)

        with torch.no_grad():
            summary = self.engine.encode_state(features)
            old_values = self.value_net(torch.cat([query_emb, summary], dim=-1)).squeeze(-1)
        advantages = rewards - old_values
        if len(batch) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses = [], []
        n = len(batch)
        for _ in range(self.n_epochs):
            for idx in torch.randperm(n, device=self.device).split(self.minibatch_size):
                summary = self.engine.encode_state(features[idx])
                decisions = self.engine.meta(query_emb[idx], summary)
                log_probs = self.engine.meta.log_prob(decisions, gate_actions[idx], expert_bias[idx], gate_mask[idx])
                assert log_probs.shape == old_log_probs[idx].shape

                ratio = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_params, 1.0)
                self.policy_optimizer.step()
                policy_losses.append(policy_loss.item())

                # Value function update with clipping
                values = self.value_net(torch.cat([query_emb[idx], summary.detach()], dim=-1)).squeeze(-1)
                values_clipped = old_values[idx] + torch.clamp(values - old_values[idx], -self.epsilon, self.epsilon)
                value_loss = torch.max(
                    (values - rewards[idx]) ** 2,
                    (values_clipped - rewards[idx]) ** 2,
                ).mean()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.value_optimizer.step()
                value_losses.append(value_loss.item())

        return sum(policy_losses) / len(policy_losses), sum(value_losses) / len(value_losses)

    def validate(self, pairs: List[Tuple[str, str]]) -> float:
        """Average reward of the deterministic policy on held-out pairs."""
        total_reward = 0.0
        for query, ground_truth in pairs:
            result = self.engine.generate(query, return_details=True)
            total_reward += self.compute_reward(result, ground_truth, update_baseline=False)
        return total_reward / len(pairs)

    def _check_correctness(self, response: str, ground_truth: str) -> bool:
        """Ground truth appears in the generated completion."""
        return ground_truth.lower() in response.lower()


def load_qa_pairs(path: str) -> List[Tuple[str, str]]:
    """JSONL with {"query": ..., "answer": ...} per line."""
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                pairs.append((record['query'], record['answer']))
    return pairs


def train_rl_stage(args):
    """
    Entry point for Stage 3: RL training of meta-controller.

    Called from train.py with --stage 3. Optional components:
    --memory-checkpoint (Stage 2) enables episodic memory,
    --semantic-store (Stage 2) enables semantic memory,
    --critic-checkpoint (Stage 4) enables verification.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    engine = MANTISInferenceEngine.from_checkpoints(
        base_checkpoint=args.resume,
        memory_checkpoint=args.memory_checkpoint,
        semantic_store=args.semantic_store,
        critic_checkpoint=args.critic_checkpoint,
        tokenizer_path=args.tokenizer_path,
        device=device,
    )
    config = engine.mantis_config
    available = [g for g, on in zip(('episodic', 'semantic', 'verification'), engine.gate_mask[0, 1:].tolist()) if on]
    print(f"✓ Engine ready. Optional gates available: {available or 'none'}")

    if args.train_file:
        pairs = load_qa_pairs(args.train_file)
        print(f"✓ Loaded {len(pairs)} query-answer pairs from {args.train_file}")
    else:
        pairs = list(DEMO_QA)
        print("⚠️  No data file given: using the 10-pair demo dataset.")
        print("   Pass a JSONL file of {\"query\": ..., \"answer\": ...} records for real training.")
    if len(pairs) < 2:
        raise ValueError("Stage 3 needs at least 2 query-answer pairs")

    random.Random(42).shuffle(pairs)
    n_val = max(1, len(pairs) // 10)
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    value_network = CriticValueNetwork(d_model=config.base_moe.d_model, state_dim=config.meta_controller.state_dim)
    ppo_trainer = PPOTrainer(
        engine=engine,
        value_network=value_network,
        alpha=config.training.alpha_accuracy,
        beta=config.training.beta_latency,
        gamma=config.training.gamma_compute,
        delta=config.training.delta_calibration,
        lr=args.learning_rate or config.training.rl_lr,
        ppo_epsilon=config.training.rl_ppo_epsilon,
        batch_size=args.rl_batch_size,
    )

    print(f"\nStarting PPO training: {args.rl_episodes} episodes, "
          f"{len(train_pairs)} train / {len(val_pairs)} val pairs, batch {args.rl_batch_size}")
    print("=" * 80 + "\n")
    ppo_trainer.train(train_pairs, val_pairs, num_episodes=args.rl_episodes)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "meta_controller_rl.pt")
    absolute = lambda p: os.path.abspath(p) if p else None
    torch.save({
        'meta_controller_state_dict': engine.meta.state_dict(),
        'state_encoder_state_dict': engine.state_encoder.state_dict(),
        'value_network_state_dict': value_network.state_dict(),
        'config': config,
        'tokenizer_fingerprint': engine.tokenizer.fingerprint(),
        'base_checkpoint': absolute(args.resume),
        'memory_checkpoint': absolute(args.memory_checkpoint),
        'semantic_store': absolute(args.semantic_store),
        'critic_checkpoint': absolute(args.critic_checkpoint),
    }, save_path)

    print(f"\n{'='*80}")
    print(f"✓ Policy saved to: {save_path}")
    print("  Rebuild the full engine with MANTISInferenceEngine.from_checkpoints(policy_checkpoint=...)")
    print("=" * 80 + "\n")
