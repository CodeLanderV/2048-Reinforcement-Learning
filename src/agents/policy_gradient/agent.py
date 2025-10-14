"""REINFORCE Policy Gradient Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical

from .network import PolicyNetwork, PolicyNetworkConfig


@dataclass
class PolicyGradientAgentConfig:
    """Configuration for Policy Gradient agent."""
    gamma: float = 0.99
    learning_rate: float = 1e-4
    gradient_clip: float = 5.0


@dataclass
class PolicyGradientAgent:
    """REINFORCE agent - learns policy directly."""
    
    model_config: PolicyNetworkConfig
    agent_config: PolicyGradientAgentConfig = field(default_factory=PolicyGradientAgentConfig)
    action_space: List[str] = field(default_factory=lambda: ["up", "down", "left", "right"])
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def __post_init__(self) -> None:
        self.policy_net: PolicyNetwork = self.model_config.build().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.agent_config.learning_rate)
        self.episode_buffer: List[tuple] = []
        self.steps_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from policy."""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        self.episode_buffer.append((state, action.item(), probs[0, action].item()))
        self.steps_done += 1
        return int(action.item())

    def act_greedy(self, state: np.ndarray) -> int:
        """Select best action (for evaluation)."""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.policy_net(state_tensor)
            return int(torch.argmax(probs, dim=1).item())

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Not used in policy gradient (batches at episode end)."""
        pass

    def finish_episode(self, rewards: List[float]) -> Optional[float]:
        """Update policy after episode completion."""
        if not self.episode_buffer:
            return None

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.agent_config.gamma * G
            returns.insert(0, G)

        # Normalize returns
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)

        # Compute policy loss
        policy_loss = []
        for (state, action, _), R in zip(self.episode_buffer, returns_tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs = self.policy_net(state_tensor)
            log_prob = torch.log(probs[0, action] + 1e-9)
            policy_loss.append(-log_prob * R)

        # Optimize
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.agent_config.gradient_clip)
        self.optimizer.step()

        self.episode_buffer.clear()
        return float(loss.item())

    def save(self, path: Path, episode: int) -> None:
        """Save checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "episode": episode,
            "steps": self.steps_done,
            "model_state": self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.steps_done = checkpoint.get("steps", 0)
