"""Policy Network for REINFORCE algorithm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PolicyNetwork(nn.Module):
    """
    Policy Network that outputs action probabilities.
    
    Architecture:
        Input (16) → Hidden Layers → Softmax Output (4 action probabilities)
    
    The network learns to output π(a|s) - probability distribution over actions.
    """
    
    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 4,
        hidden_dims: List[int] = None
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            prev_dim = hidden_dim
        
        # Output layer (no activation - will use softmax)
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.
        
        Args:
            state: Flattened board state tensor [batch, 16]
        
        Returns:
            Action logits [batch, 4] (use softmax to get probabilities)
        """
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution.
        
        Args:
            state: Flattened board state tensor [batch, 16]
        
        Returns:
            Action probabilities [batch, 4] summing to 1.0
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor) -> tuple[int, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Flattened board state tensor [1, 16]
        
        Returns:
            Tuple of (action_index, log_probability)
        """
        probs = self.get_action_probs(state)
        
        # Sample action from categorical distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob
