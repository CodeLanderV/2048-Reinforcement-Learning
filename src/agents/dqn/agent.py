"""DQN Agent with Replay Buffer."""

from __future__ import annotations

from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import torch
from torch import nn, optim

from .network import DQN, DQNModelConfig


Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        """Store a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample random batch of transitions."""
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.memory)


@dataclass
class AgentConfig:
    """Configuration for DQN agent."""
    gamma: float = 0.99
    batch_size: int = 128
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 50000
    target_update_interval: int = 1000
    replay_buffer_size: int = 100_000
    gradient_clip: float = 5.0


@dataclass
class DQNAgent:
    """Deep Q-Network agent."""
    
    model_config: DQNModelConfig
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    action_space: List[str] = field(default_factory=lambda: ["up", "down", "left", "right"])
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self) -> None:
        # Build networks
        self.policy_net: DQN = self.model_config.build().to(self.device)
        self.target_net: DQN = self.model_config.build().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Setup optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.agent_config.learning_rate)
        self.replay_buffer = ReplayBuffer(self.agent_config.replay_buffer_size)

        # Training state
        self.steps_done = 0
        self.epsilon = self.agent_config.epsilon_start

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        self.steps_done += 1
        self._update_epsilon()
        
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, len(self.action_space)))
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def act_greedy(self, state: np.ndarray) -> int:
        """Select best action (no exploration)."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, *transition) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(*transition)

    def can_optimize(self) -> bool:
        """Check if enough samples for training."""
        return len(self.replay_buffer) >= self.agent_config.batch_size

    def optimize_model(self) -> Optional[float]:
        """Perform one optimization step."""
        if not self.can_optimize():
            return None
        
        # Sample batch
        transitions = self.replay_buffer.sample(self.agent_config.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors (use from_numpy for speed)
        state_batch = torch.from_numpy(np.stack(batch.state).astype(np.float32)).to(self.device)
        action_batch = torch.from_numpy(np.array(batch.action, dtype=np.int64)).to(self.device).unsqueeze(1)
        reward_batch = torch.from_numpy(np.array(batch.reward, dtype=np.float32)).to(self.device).unsqueeze(1)
        next_state_batch = torch.from_numpy(np.stack(batch.next_state).astype(np.float32)).to(self.device)
        done_batch = torch.from_numpy(np.array(batch.done, dtype=np.float32)).to(self.device).unsqueeze(1)

        # Compute Q-values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.agent_config.gamma * max_next_q_values

        # Optimize
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.agent_config.gradient_clip)
        self.optimizer.step()

        # Update target network periodically
        if self.steps_done % self.agent_config.target_update_interval == 0:
            self.update_target_network()
        
        return float(loss.item())

    def update_target_network(self) -> None:
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path, episode: int) -> None:
        """Save checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "episode": episode,
            "steps": self.steps_done,
            "model_state": self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "agent_config": self.agent_config.__dict__,
            "model_config": {
                "input_dim": self.model_config.input_dim,
                "hidden_dims": self.model_config.hidden_dims,
                "output_dim": self.model_config.output_dim,
            },
        }
        torch.save(checkpoint, path)

    def load(self, path: Path) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["model_state"])
        self.target_net.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps_done = checkpoint.get("steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.agent_config.epsilon_end)

    def _update_epsilon(self) -> None:
        """Decay epsilon for exploration."""
        decay = self.agent_config.epsilon_decay
        self.epsilon = self.agent_config.epsilon_end + (
            self.agent_config.epsilon_start - self.agent_config.epsilon_end
        ) * np.exp(-1.0 * self.steps_done / decay)
