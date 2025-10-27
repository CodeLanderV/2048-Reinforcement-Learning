"""
Dueling DQN Agent - Separates value and advantage estimation.

Exports:
    - DuelingDQNAgent: Main agent class
    - DuelingAgentConfig: Agent hyperparameters
    - DuelingDQN: Neural network architecture
    - DuelingDQNModelConfig: Model architecture config
"""

from .agent import DuelingDQNAgent, DuelingAgentConfig
from .network import DuelingDQN, DuelingDQNModelConfig

__all__ = [
    "DuelingDQNAgent",
    "DuelingAgentConfig", 
    "DuelingDQN",
    "DuelingDQNModelConfig",
]
