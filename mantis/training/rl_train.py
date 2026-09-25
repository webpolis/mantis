"""
Stage 3: Meta-Controller RL Fine-Tuning

Train the routing policy with PPO to optimize a multi-objective reward.

Each episode is one query: the engine samples routing actions from the
policy, executes them, and the trainer scores the response. Training
episodes write to episodic memory, so the process is sequential and stateful
(later queries see earlier answers); rewards do not credit that future
usefulness. Validation and the supervised route-search warm start run with
memory frozen (no writes, no consolidation), so their comparisons are
between actions, not between memory states.

Updates use fresh on-policy batches of `batch_size` episodes.
"""

import itertools
import json
import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mantis.inference.engine import MANTISInferenceEngine
from mantis.models.critic import CriticValueNetwork
from mantis.models.meta_controller import GATES
from mantis.training.scoring import score_answer
from mantis.utils.checkpoints import model_fingerprint

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

    Reward: R = alpha*r_acc - beta*r_lat - gamma*r_comp + delta*r_calib
    - r_acc: +1 correct, `abstain_reward` for an abstention, -1 wrong
    - r_lat: latency over a declared budget (seconds)
    - r_comp: compute beyond a query-only answer, as a ratio (>= 0)
    - r_calib: -(confidence - correct)^2 on the final reported confidence
    """

    def __init__(
        self,
        engine: MANTISInferenceEngine,
        value_network: nn.Module,
        alpha: float = 1.0,  # Accuracy weight
        beta: float = 0.3,   # Latency weight
        gamma: float = 0.2,  # Compute weight
        delta: float = 0.5,  # Calibration weight
        latency_budget_s: float = 2.0,
        abstain_reward: float = -0.25,
        f1_threshold: float = 0.5,
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
        self.latency_budget_s = latency_budget_s
        self.abstain_reward = abstain_reward
        self.f1_threshold = f1_threshold
        self.epsilon = ppo_epsilon
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size

        # Dropout stays off (modules in eval mode) so old and new log-probs agree
        policy_params = list(engine.meta.parameters()) + list(engine.state_encoder.parameters())
        self.policy_optimizer = torch.optim.Adam(policy_params, lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_params = policy_params

        self.stats = {'episodes': 0, 'total_reward': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}

    # ---------------------------------------------------------------- reward

    def compute_reward(self, result: Dict, ground_truth: str) -> float:
        return self.reward_terms(result, ground_truth)['reward']

    def reward_terms(self, result: Dict, ground_truth: str) -> Dict:
        """Reward and its components for one engine result."""
        outcome = score_answer(result['response'], ground_truth, result['abstained'], self.f1_threshold)
        if outcome is None:
            r_acc = self.abstain_reward
        else:
            r_acc = 1.0 if outcome else -1.0

        r_lat = result['latency'] / self.latency_budget_s

        cost = result['cost']
        baseline = self.engine.active_params * max(1, cost['query_tokens'] + cost['generated_tokens'])
        r_comp = max(0.0, result['compute_units'] / baseline - 1.0)

        y = 1.0 if outcome else 0.0
        r_calib = -((result['confidence'] - y) ** 2)

        reward = self.alpha * r_acc - self.beta * r_lat - self.gamma * r_comp + self.delta * r_calib
        return {'reward': reward, 'outcome': outcome, 'r_acc': r_acc, 'r_lat': r_lat,
                'r_comp': r_comp, 'r_calib': r_calib}

    # --------------------------------------------------------------- episodes

    def collect_episode(self, query: str, ground_truth: str) -> Dict:
        """Run one exploratory episode (memory writes on) and return its transition."""
        result = self.engine.generate(query, explore=True, return_details=True)
        transition = dict(result['rollout'])
        transition['reward'] = self.compute_reward(result, ground_truth)
        return transition

    def gate_options(self) -> List[Dict[str, bool]]:
        """Gate combinations the engine can execute: bypass alone, then every combo of available gates."""
        available = [gate for gate, on in zip(GATES[1:], self.engine.gate_mask[0, 1:].tolist()) if on]
        options = [{'bypass': True}]
        for bits in itertools.product([False, True], repeat=len(available)):
            options.append({'bypass': False, **dict(zip(available, bits))})
        return options

    def route_search(self, query: str, ground_truth: str) -> Tuple[Dict[str, bool], float, Dict]:
        """
        Try every gate combination on a frozen memory and return the best
        (action, reward, rollout of the best run).
        """
        best = None
        with self.engine.frozen_memory():
            for gates in self.gate_options():
                result = self.engine.generate(query, force_gates=gates, return_details=True)
                reward = self.compute_reward(result, ground_truth)
                if best is None or reward > best[1]:
                    best = (gates, reward, result['rollout'])
        return best

    def supervised_warmup(self, pairs: List[Tuple[str, str]], episodes: int) -> float:
        """
        Supervised route selection: for `episodes` random pairs, find the
        best action by route search and train the gate logits toward it with
        BCE (masked by component availability). Returns the mean best reward.
        """
        print(f"Supervised route-search warm start: {episodes} episodes over {len(self.gate_options())} actions each")
        total = 0.0
        for episode in range(1, episodes + 1):
            query, ground_truth = random.choice(pairs)
            gates, reward, rollout = self.route_search(query, ground_truth)
            total += reward
            target = torch.tensor([[float(gates.get(g, False)) for g in GATES]], device=self.device)
            mask = self.engine.gate_mask

            query_emb = rollout['query_emb'].unsqueeze(0).to(self.device)
            features = rollout['features'].unsqueeze(0).to(self.device)
            decisions = self.engine.meta(query_emb, self.engine.encode_state(features))
            loss = (F.binary_cross_entropy_with_logits(decisions['gate_logits'], target, reduction='none') * mask).sum() / mask.sum()

            self.policy_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_params, 1.0)
            self.policy_optimizer.step()

            if episode % 10 == 0 or episode == episodes:
                print(f"  warmup {episode}/{episodes} | best action {gates} | reward {reward:.3f} | loss {loss.item():.4f}")
        return total / max(1, episodes)

    # ----------------------------------------------------------------- train

    def train(self, train_pairs: List[Tuple[str, str]], val_pairs: List[Tuple[str, str]], num_episodes: int = 50000):
        """
        Main RL training loop.

        Args:
            train_pairs: (query, ground_truth) pairs for rollouts
            val_pairs: held-out pairs scored with the deterministic policy on frozen memory
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
                print(f"Validation: {self.format_validation(self.validate(val_pairs))}")

        if len(batch) > 1:
            self.update(batch)
        print("PPO training complete!")

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
        expert_raw = stack('expert_raw')
        action_mask = stack('action_mask')
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
                log_probs = self.engine.meta.log_prob(decisions, gate_actions[idx], expert_raw[idx], action_mask[idx])
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

    # -------------------------------------------------------------- validate

    def validate(self, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Deterministic policy on held-out pairs with memory frozen.

        Returns mean reward, accuracy over all questions, coverage (fraction
        answered), error rate among answered questions, p95 latency and mean
        compute units.
        """
        rewards, outcomes, latencies, compute = [], [], [], []
        with self.engine.frozen_memory():
            for query, ground_truth in pairs:
                result = self.engine.generate(query, return_details=True)
                terms = self.reward_terms(result, ground_truth)
                rewards.append(terms['reward'])
                outcomes.append(terms['outcome'])
                latencies.append(result['latency'])
                compute.append(result['compute_units'])
        n = len(pairs)
        answered = [o for o in outcomes if o is not None]
        latencies.sort()
        return {
            'reward': sum(rewards) / n,
            'accuracy': sum(1 for o in outcomes if o) / n,
            'coverage': len(answered) / n,
            'answered_error_rate': (sum(1 for o in answered if not o) / len(answered)) if answered else 0.0,
            'latency_p95': latencies[min(n - 1, int(0.95 * n))],
            'mean_compute_units': sum(compute) / n,
        }

    @staticmethod
    def format_validation(metrics: Dict[str, float]) -> str:
        return (f"reward {metrics['reward']:.4f} | acc {metrics['accuracy']:.2%} | coverage {metrics['coverage']:.2%} | "
                f"answered err {metrics['answered_error_rate']:.2%} | p95 latency {metrics['latency_p95']:.3f}s | "
                f"compute {metrics['mean_compute_units']:.3e}")


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
    --rl-supervised-episodes N runs a route-search warm start before PPO.
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
    available = [g for g, on in zip(GATES[1:], engine.gate_mask[0, 1:].tolist()) if on]
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

    tc = config.training
    value_network = CriticValueNetwork(d_model=config.base_moe.d_model, state_dim=config.meta_controller.state_dim)
    ppo_trainer = PPOTrainer(
        engine=engine,
        value_network=value_network,
        alpha=tc.alpha_accuracy,
        beta=tc.beta_latency,
        gamma=tc.gamma_compute,
        delta=tc.delta_calibration,
        latency_budget_s=tc.latency_budget_s,
        abstain_reward=tc.abstain_reward,
        f1_threshold=tc.correctness_f1_threshold,
        lr=args.learning_rate or tc.rl_lr,
        ppo_epsilon=tc.rl_ppo_epsilon,
        batch_size=args.rl_batch_size,
    )

    warmup = getattr(args, 'rl_supervised_episodes', 0) or 0
    if warmup > 0:
        ppo_trainer.supervised_warmup(train_pairs, warmup)
        print(f"Validation after warm start: {ppo_trainer.format_validation(ppo_trainer.validate(val_pairs))}")

    print(f"\nStarting PPO training: {args.rl_episodes} episodes, "
          f"{len(train_pairs)} train / {len(val_pairs)} val pairs, batch {args.rl_batch_size}")
    print("=" * 80 + "\n")
    ppo_trainer.train(train_pairs, val_pairs, num_episodes=args.rl_episodes)
    print(f"Final validation: {ppo_trainer.format_validation(ppo_trainer.validate(val_pairs))}")
    engine.close()

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "meta_controller_rl.pt")
    absolute = lambda p: os.path.abspath(p) if p else None
    torch.save({
        'meta_controller_state_dict': engine.meta.state_dict(),
        'state_encoder_state_dict': engine.state_encoder.state_dict(),
        'value_network_state_dict': value_network.state_dict(),
        'config': config,
        'tokenizer_fingerprint': engine.tokenizer.fingerprint(),
        'embedding_fingerprint': getattr(engine.base, '_source_fingerprint', None)
                                 or model_fingerprint(engine.base, engine.tokenizer),
        'base_checkpoint': absolute(args.resume),
        'memory_checkpoint': absolute(args.memory_checkpoint),
        'semantic_store': absolute(args.semantic_store),
        'critic_checkpoint': absolute(args.critic_checkpoint),
    }, save_path)

    print(f"\n{'='*80}")
    print(f"✓ Policy saved to: {save_path}")
    print("  Rebuild the full engine with MANTISInferenceEngine.from_checkpoints(policy_checkpoint=...)")
    print("=" * 80 + "\n")
