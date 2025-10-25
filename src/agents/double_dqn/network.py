"""Double Deep Q-Network architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..dqn.network import DQN


class DoubleDQN(DQN):
    """Double DQN - same architecture as DQN, different action selection logic."""
    pass


@dataclass
class DoubleDQNModelConfig:
    """Configuration for Double DQN model."""
    input_dim: int = 16
    hidden_dims: Tuple[int, ...] = (512, 512, 256)  # Deeper and wider network
    output_dim: int = 4

    def build(self) -> DoubleDQN:
        """Build Double DQN model from config."""
        return DoubleDQN(self.input_dim, self.hidden_dims, self.output_dim)
