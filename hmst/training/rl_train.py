"""
Stage 3: Meta-Controller RL Fine-Tuning

Train the routing policy with PPO to optimize multi-objective reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
from typing import Dict, List, Tuple
from collections import deque
import random


class PPOTrainer:
    """
    PPO trainer for meta-controller.

    Optimizes routing policy to balance:
    - Accuracy
    - Latency
    - Compute cost
    - Confidence calibration
    """

    def __init__(
        self,
        meta_controller: nn.Module,
        value_network: nn.Module,
        inference_engine,
        alpha: float = 1.0,  # Accuracy weight
        beta: float = 0.3,   # Latency weight
        gamma: float = 0.2,  # Compute weight
        delta: float = 0.5,  # Calibration weight
        lr: float = 1e-5,
        ppo_epsilon: float = 0.2,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        batch_size: int = 256,
        n_epochs: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.meta = meta_controller.to(device)
        self.value_net = value_network.to(device)
        self.engine = inference_engine
        self.device = device

        # Reward weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # PPO hyperparameters
        self.epsilon = ppo_epsilon
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(meta_controller.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(value_network.parameters(), lr=lr)

        # Experience buffer
        self.buffer = deque(maxlen=10000)

        # Training statistics
        self.stats = {
            'episodes': 0,
            'total_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }

    def train(
        self,
        queries: List[str],
        ground_truths: List[str],
        num_episodes: int = 50000
    ):
        """
        Main RL training loop.

        Args:
            queries: List of training queries
            ground_truths: Corresponding ground truth responses
            num_episodes: Number of episodes to train
        """
        print(f"Starting PPO training for {num_episodes} episodes")

        for episode in range(num_episodes):
            # Sample random query
            idx = random.randint(0, len(queries) - 1)
            query = queries[idx]
            ground_truth = ground_truths[idx]

            # Collect episode
            reward = self.collect_episode(query, ground_truth)

            # Update statistics
            self.stats['episodes'] += 1
            self.stats['total_reward'] += reward

            # PPO update
            if len(self.buffer) >= self.batch_size:
                policy_loss, value_loss = self.update()

                self.stats['policy_loss'] = policy_loss
                self.stats['value_loss'] = value_loss

            # Logging
            if episode % 100 == 0:
                avg_reward = self.stats['total_reward'] / max(self.stats['episodes'], 1)
                print(f"Episode {episode} | Avg Reward: {avg_reward:.4f} | "
                      f"Policy Loss: {self.stats['policy_loss']:.4f} | "
                      f"Value Loss: {self.stats['value_loss']:.4f}")

            # Validation
            if episode % 1000 == 0:
                val_reward = self.validate(queries[:100], ground_truths[:100])
                print(f"Validation Reward: {val_reward:.4f}")

        print("PPO training complete!")

    def collect_episode(self, query: str, ground_truth: str) -> float:
        """
        Collect one episode of experience.

        Returns:
            Episode reward
        """
        # Generate with current policy
        result = self.engine.generate(query, return_details=True)

        # Extract routing decisions
        routing = result['routing_decisions']

        # Compute reward
        metrics = {
            'latency': result['latency'],
            'baseline_latency': 0.15,  # Baseline comparison
            'active_params': 2e9,
            'total_params': 12e9,
            'episodic_accessed': routing['episodic'],
            'semantic_accessed': routing['semantic'],
            'uncertainty': result['uncertainty']
        }

        reward = self.compute_reward(
            query,
            result['response'],
            metrics,
            ground_truth
        )

        # Store transition
        state = self._get_state(query, result)
        action = self._encode_action(routing)
        log_prob = self._compute_log_prob(result['routing_continuous'], action)
        value = self.value_net(state).item()

        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value
        }

        self.buffer.append(transition)

        return reward

    def compute_reward(
        self,
        query: str,
        response: str,
        metrics: Dict,
        ground_truth: str
    ) -> float:
        """
        Multi-objective reward function.

        R = α·R_acc - β·R_lat - γ·R_comp + δ·R_calib
        """
        # Accuracy (simplified - could use BLEU, ROUGE, or critic)
        correct = self._check_correctness(response, ground_truth)
        r_acc = 1.0 if correct else -1.0

        # Latency (normalized)
        r_lat = metrics['latency'] / metrics['baseline_latency']

        # Compute cost
        r_comp = (
            0.5 * (metrics['active_params'] / metrics['total_params']) +
            0.3 * float(metrics['episodic_accessed']) +
            0.2 * float(metrics['semantic_accessed'])
        )

        # Calibration (uncertainty should match correctness)
        u = metrics['uncertainty']
        y_true = 1 if correct else 0
        r_calib = -((u - (1 - y_true)) ** 2)

        # Total reward
        reward = (
            self.alpha * r_acc -
            self.beta * r_lat -
            self.gamma * r_comp +
            self.delta * r_calib
        )

        return reward

    def update(self) -> Tuple[float, float]:
        """
        PPO update step.

        Returns:
            (policy_loss, value_loss)
        """
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)

        states = torch.stack([t['state'] for t in batch])
        actions = torch.stack([t['action'] for t in batch])
        old_log_probs = torch.stack([t['log_prob'] for t in batch])
        rewards = torch.tensor([t['reward'] for t in batch], device=self.device)
        old_values = torch.tensor([t['value'] for t in batch], device=self.device)

        # Compute advantages (GAE)
        values = self.value_net(states).squeeze(-1)
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO policy update (multiple epochs)
        total_policy_loss = 0.0

        for _ in range(self.n_epochs):
            # Forward pass
            routing = self.meta(
                states[:, :self.meta.d_model],
                states[:, self.meta.d_model:]
            )

            # Compute action log probabilities
            log_probs = self._compute_log_prob_batch(routing, actions)

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.meta.parameters(), 1.0)
            self.policy_optimizer.step()

            total_policy_loss += policy_loss.item()

        # Value function update
        values = self.value_net(states).squeeze(-1)
        value_loss = ((values - rewards) ** 2).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.value_optimizer.step()

        return total_policy_loss / self.n_epochs, value_loss.item()

    def validate(self, queries: List[str], ground_truths: List[str]) -> float:
        """
        Validate on held-out set.

        Returns:
            Average reward
        """
        total_reward = 0.0

        for query, gt in zip(queries, ground_truths):
            result = self.engine.generate(query, return_details=True)

            metrics = {
                'latency': result['latency'],
                'baseline_latency': 0.15,
                'active_params': 2e9,
                'total_params': 12e9,
                'episodic_accessed': result['routing_decisions']['episodic'],
                'semantic_accessed': result['routing_decisions']['semantic'],
                'uncertainty': result['uncertainty']
            }

            reward = self.compute_reward(query, result['response'], metrics, gt)
            total_reward += reward

        return total_reward / len(queries)

    def _get_state(self, query: str, result: Dict) -> torch.Tensor:
        """
        Encode state for value network.

        Returns:
            (d_model + state_dim,) state vector
        """
        # Get query embedding from inference engine
        query_tokens = self.engine._tokenize(query)
        query_emb = self.engine.base.encode(query_tokens).squeeze(0)  # (d_model,)

        # Create state summary
        state_summary = self.engine._create_state_summary(
            uncertainty=result['uncertainty'],
            context_length=query_tokens.size(1),
            confidence=result.get('confidence', 0.5)
        ).squeeze(0)  # (state_dim,)

        # Combine
        state = torch.cat([query_emb, state_summary], dim=0)
        return state

    def _encode_action(self, routing: Dict) -> torch.Tensor:
        """
        Encode routing decisions as action vector.

        Returns:
            (5,) action tensor [early_exit, episodic, semantic, verification, expert_id]
        """
        expert_id = routing['expert_weights'].argmax().item() if torch.is_tensor(routing['expert_weights']) else 0

        action = torch.tensor([
            float(routing['early_exit']),
            float(routing['episodic']),
            float(routing['semantic']),
            float(routing['verification']),
            float(expert_id)
        ], device=self.device)

        return action

    def _compute_log_prob(self, routing_continuous: Dict, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of action under current policy.
        """
        log_prob = torch.tensor(0.0, device=self.device)

        # Binary gates (Bernoulli)
        for i, gate in enumerate(['early_exit', 'episodic', 'semantic', 'verification']):
            prob = routing_continuous[gate].squeeze()
            if prob.dim() == 0:
                prob = prob.unsqueeze(0)

            # Create Bernoulli distribution
            dist = Bernoulli(probs=prob)
            action_val = action[i]

            # Add log probability
            log_prob = log_prob + dist.log_prob(action_val)

        # Expert selection (Categorical)
        expert_probs = routing_continuous['expert_weights']
        if expert_probs.dim() == 1:
            expert_probs = expert_probs.unsqueeze(0)

        dist = Categorical(probs=expert_probs)
        expert_id = action[4].long()
        log_prob = log_prob + dist.log_prob(expert_id)

        return log_prob

    def _compute_log_prob_batch(
        self,
        routing: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for batch of actions using torch.distributions.
        """
        batch_size = actions.size(0)
        log_probs = torch.zeros(batch_size, device=self.device)

        # Binary gates using Bernoulli distribution
        for i, gate in enumerate(['early_exit', 'episodic', 'semantic', 'verification']):
            probs = routing[gate].squeeze(-1)
            action_vals = actions[:, i]

            # Create Bernoulli distribution
            dist = Bernoulli(probs=probs)

            # Add log probabilities
            log_probs += dist.log_prob(action_vals)

        # Expert selection using Categorical distribution
        expert_probs = routing['expert_weights']
        expert_ids = actions[:, 4].long()

        # Create Categorical distribution
        dist = Categorical(probs=expert_probs)

        # Add log probabilities
        log_probs += dist.log_prob(expert_ids)

        return log_probs

    def _check_correctness(self, response: str, ground_truth: str) -> bool:
        """
        Simple correctness check (could be more sophisticated).
        """
        # Exact match or substring
        return ground_truth.lower() in response.lower()
