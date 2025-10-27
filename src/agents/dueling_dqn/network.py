"""
Dueling DQN Neural Network Architecture

Separates Q(s,a) into Value V(s) and Advantage A(s,a) streams:
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,*)))

This decomposition helps the agent learn which states are valuable 
independently of the action taken.
"""

from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn


@dataclass
class DuelingDQNModelConfig:
    """Configuration for Dueling DQN network architecture."""
    output_dim: int = 4                     # Number of actions (up/down/left/right)
    hidden_dims: Tuple[int, ...] = (512, 512)  # Shared feature extractor dimensions
    stream_dim: int = 256                   # Dimension of value/advantage streams


class DuelingDQN(nn.Module):
    """
    Dueling DQN network with separate value and advantage streams.
    
    Architecture:
        Input (16D) → Shared Layers → Split into:
            1. Value stream V(s): Estimates state value
            2. Advantage stream A(s,a): Estimates action advantages
        
        Q(s,a) = V(s) + (A(s,a) - mean(A))
    """
    
    def __init__(self, config: DuelingDQNModelConfig):
        super().__init__()
        self.config = config
        
        # Shared feature extractor
        layers = []
        input_dim = 16  # 4x4 log2-normalized board
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        
        # Value stream: V(s) - scalar state value
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.stream_dim),
            nn.ReLU(),
            nn.Linear(config.stream_dim, 1)
        )
        
        # Advantage stream: A(s,a) - advantage per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_dims[-1], config.stream_dim),
            nn.ReLU(),
            nn.Linear(config.stream_dim, config.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.
        
        Args:
            x: Board state [batch, 16] - log2 normalized
            
        Returns:
            Q-values [batch, output_dim] - combined V(s) + A(s,a)
        """
        # Shared feature extraction
        features = self.shared(x)
        
        # Separate value and advantage
        value = self.value_stream(features)           # [batch, 1]
        advantage = self.advantage_stream(features)   # [batch, actions]
        
        # Combine using dueling formula: Q = V + (A - mean(A))
        # Subtracting mean makes advantage centered around 0
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
