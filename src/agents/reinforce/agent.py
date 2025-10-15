"""
REINFORCE Agent (Monte Carlo Policy Gradient).

Algorithm:
    1. Collect full episode trajectory: (s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T)
    2. Compute returns: G_t = Σ(γ^k * r_{t+k}) for each timestep
    3. Update policy: ∇J(θ) = E[∇log π(a|s) * G_t]
    4. Optionally use baseline to reduce variance: G_t - b(s_t)

Key Features:
    - On-policy: Learns from its own experience
    - Monte Carlo: Updates after full episodes
    - Policy gradient: Directly optimizes policy parameters
    - Stochastic policy: Explores naturally through probability distribution

Reference:
    Williams, R. J. (1992). Simple statistical gradient-following algorithms 
    for connectionist reinforcement learning. Machine Learning, 8, 229-256.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import PolicyNetwork


@dataclass
class REINFORCEConfig:
    """Configuration for REINFORCE agent."""
    learning_rate: float = 0.001
    gamma: float = 0.99              # Discount factor
    hidden_dims: List[int] = None    # Network architecture
    use_baseline: bool = True        # Use baseline to reduce variance
    entropy_coef: float = 0.01       # Entropy regularization (encourage exploration)
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


class REINFORCEAgent:
    """
    REINFORCE (Monte Carlo Policy Gradient) agent.
    
    Learns a stochastic policy that directly maps states to action probabilities.
    Updates the policy after each episode using the policy gradient theorem.
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        config: Optional[REINFORCEConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config if config is not None else REINFORCEConfig()
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Policy network
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate
        )
        
        # Episode storage (cleared after each episode)
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_rewards: List[float] = []
        self.episode_states: List[np.ndarray] = []  # For baseline calculation
        
        # Statistics
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray, board=None) -> int:
        """
        Select action by sampling from policy distribution.
        
        Args:
            state: Current board state (flattened)
            board: GameBoard object (unused, for API compatibility)
        
        Returns:
            Action index (0-3)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy_net.sample_action(state_tensor)
        
        # Store for later update
        self.episode_log_probs.append(log_prob)
        
        return action
    
    def act_greedy(self, state: np.ndarray, board=None) -> int:
        """
        Select action greedily (for evaluation).
        
        Args:
            state: Current board state (flattened)
            board: GameBoard object (unused, for API compatibility)
        
        Returns:
            Action with highest probability
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.policy_net.get_action_probs(state_tensor)
            action = probs.argmax(dim=-1).item()
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store reward for current timestep.
        
        Note: REINFORCE only needs rewards (log_probs already stored in select_action)
        """
        self.episode_rewards.append(reward)
        self.episode_states.append(state)
    
    def train_step(self):
        """
        REINFORCE updates after full episodes, not after each step.
        This method does nothing - call finish_episode() instead.
        """
        pass
    
    def finish_episode(self):
        """
        Update policy after episode completes.
        
        Implements the REINFORCE algorithm:
            1. Compute discounted returns G_t
            2. Optionally subtract baseline (mean return)
            3. Compute policy gradient: ∇J = Σ ∇log π(a|s) * (G_t - baseline)
            4. Update policy parameters
        """
        if len(self.episode_rewards) == 0:
            return
        
        # ─────────────────────────────────────────────────────────────────
        # Step 1: Compute discounted returns G_t
        # ─────────────────────────────────────────────────────────────────
        returns = []
        G = 0
        
        # Compute returns backward (G_t = r_t + γ*r_{t+1} + γ^2*r_{t+2} + ...)
        for reward in reversed(self.episode_rewards):
            G = reward + self.config.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # ─────────────────────────────────────────────────────────────────
        # Step 2: Normalize returns (reduces variance)
        # ─────────────────────────────────────────────────────────────────
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # ─────────────────────────────────────────────────────────────────
        # Step 3: Compute policy loss
        # ─────────────────────────────────────────────────────────────────
        policy_loss = []
        entropy_loss = []
        
        for log_prob, G_t in zip(self.episode_log_probs, returns):
            # Policy gradient: -log π(a|s) * G_t
            policy_loss.append(-log_prob * G_t)
        
        # Add entropy regularization (encourages exploration)
        if self.config.entropy_coef > 0:
            # Recompute action probabilities for entropy
            states_tensor = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
            probs = self.policy_net.get_action_probs(states_tensor)
            
            # Entropy: -Σ p(a) * log p(a)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            total_loss = torch.stack(policy_loss).sum() - self.config.entropy_coef * entropy
        else:
            total_loss = torch.stack(policy_loss).sum()
        
        # ─────────────────────────────────────────────────────────────────
        # Step 4: Update policy network
        # ─────────────────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # ─────────────────────────────────────────────────────────────────
        # Step 5: Clear episode data
        # ─────────────────────────────────────────────────────────────────
        self.episode_log_probs.clear()
        self.episode_rewards.clear()
        self.episode_states.clear()
        self.episode_count += 1
    
    def save(self, path: Path, episode: int):
        """Save policy network to disk."""
        save_path = path / f"reinforce_ep{episode}.pth"
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, save_path)
        print(f"[SAVE] Model saved: {save_path}")
    
    def load(self, path: Path):
        """Load policy network from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[LOAD] Model loaded: {path}")
