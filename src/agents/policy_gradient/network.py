"""Policy Network for REINFORCE algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Policy network outputting action probabilities."""

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
        """Forward pass returning action probabilities."""
        logits = self.network(x)
        return F.softmax(logits, dim=-1)


@dataclass
class PolicyNetworkConfig:
    """Configuration for Policy Network."""
    input_dim: int = 16
    hidden_dims: Tuple[int, ...] = (256, 256)
    output_dim: int = 4

    def build(self) -> PolicyNetwork:
        """Build Policy Network from config."""
        return PolicyNetwork(self.input_dim, self.hidden_dims, self.output_dim)
