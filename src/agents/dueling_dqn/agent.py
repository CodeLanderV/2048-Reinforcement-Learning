"""
Dueling DQN Agent Implementation

Uses dueling architecture with Double DQN training (best of both worlds).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .network import DuelingDQN, DuelingDQNModelConfig


@dataclass
class DuelingAgentConfig:
    """Hyperparameters for Dueling DQN agent."""
    gamma: float = 0.99
    batch_size: int = 512
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 300000
    target_update_interval: int = 1000
    replay_buffer_size: int = 500000
    gradient_clip: float = 5.0


class DuelingDQNAgent:
    """
    Dueling DQN Agent with Double DQN training.
    
    Features:
        - Dueling architecture (separate value/advantage streams)
        - Double DQN training (reduces overestimation)
        - Experience replay
        - Target network
        - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        model_config: DuelingDQNModelConfig,
        agent_config: DuelingAgentConfig,
        action_space: List[str],
    ):
        self.config = agent_config
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DuelingDQN(model_config).to(self.device)
        self.target_net = DuelingDQN(model_config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=agent_config.learning_rate
        )
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer: List[Tuple] = []
        
        # Training state
        self.epsilon = agent_config.epsilon_start
        self.steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Board state [16] - log2 normalized
            training: If False, uses greedy policy (no exploration)
            
        Returns:
            action: Integer 0-3 (up/down/left/right)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, len(self.action_space) - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        if len(self.replay_buffer) >= self.config.replay_buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def can_optimize(self) -> bool:
        """Check if enough samples for training."""
        return len(self.replay_buffer) >= self.config.batch_size
    
    def optimize_model(self):
        """
        Train policy network using Double DQN.
        
        Uses policy network for action SELECTION but target network
        for Q-value EVALUATION (reduces overestimation).
        """
        if not self.can_optimize():
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values: Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t)
        
        # Double DQN: Select action with policy net, evaluate with target net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1 - dones_t) * self.config.gamma * next_q
        
        # Compute loss and optimize
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            self.config.gradient_clip
        )
        self.optimizer.step()
        
        # Update epsilon and step counter
        self.steps += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - (self.steps / self.config.epsilon_decay)
        )
        
        # Update target network periodically
        if self.steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: Path, episode: int):
        """Save agent checkpoint."""
        torch.save({
            'episode': episode,
            'model_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'replay_buffer': self.replay_buffer[-10000:],  # Save recent experiences
            'model_config': {
                'output_dim': self.policy_net.config.output_dim,
                'hidden_dims': self.policy_net.config.hidden_dims,
                'stream_dim': self.policy_net.config.stream_dim,
            },
        }, path)
    
    def load(self, path: Path) -> int:
        """Load agent checkpoint. Returns episode number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        self.steps = checkpoint.get('steps', 0)
        
        if 'replay_buffer' in checkpoint:
            self.replay_buffer = checkpoint['replay_buffer']
        
        return checkpoint.get('episode', 0)
