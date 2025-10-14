"""Deep Q-Network architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


class DQN(nn.Module):
    """Fully-connected Q-network for 2048."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (256, 256),
        output_dim: int = 4
    ) -> None:
        super().__init__()
        
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

        # Initialize weights
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class DQNModelConfig:
    """Configuration for DQN model."""
    input_dim: int = 16
    hidden_dims: Tuple[int, ...] = (256, 256)
    output_dim: int = 4

    def build(self) -> DQN:
        """Build DQN model from config."""
        return DQN(self.input_dim, self.hidden_dims, self.output_dim)
