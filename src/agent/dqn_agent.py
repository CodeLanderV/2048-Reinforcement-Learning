"""
Deep Q-Network (DQN) Agent Implementation
Implements the DQN algorithm for learning to play 2048.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Optional


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores past experiences and samples random batches for training.
    This breaks correlation between consecutive samples and improves learning stability.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """
    Deep Q-Network Neural Network.
    
    Architecture:
    - Input: 4x4 game board (flattened to 16 values)
    - Hidden layers with ReLU activation
    - Output: Q-values for 4 actions (up, down, left, right)
    """
    
    def __init__(self, input_size: int = 16, hidden_sizes: List[int] = [256, 256, 128], 
                 output_size: int = 4):
        """
        Initialize the neural network.
        
        Args:
            input_size: Size of input (4x4 = 16 for 2048 board)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of actions (4 for up, down, left, right)
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Prevent overfitting
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch of states)
            
        Returns:
            Q-values for each action
        """
        # Normalize input by dividing by max possible tile value (2^17 = 131072)
        x = x / 131072.0
        return self.network(x)


class DQNAgent:
    """
    DQN Agent for playing 2048.
    
    Implements the Deep Q-Learning algorithm:
    1. Q-Network approximates Q-values for state-action pairs
    2. Target Network for stable learning
    3. Experience Replay for breaking correlations
    4. Epsilon-greedy exploration strategy
    """
    
    def __init__(
        self,
        state_size: int = 16,
        action_size: int = 4,
        hidden_sizes: List[int] = [256, 256, 128],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            hidden_sizes: Hidden layer sizes for neural network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: How often to update target network
            device: Device to use (cuda/cpu)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Q-Network and Target Network
        self.q_network = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is not trained directly
        
        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training metrics
        self.training_step = 0
        self.losses = []
    
    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess state for neural network input.
        
        Args:
            state: Game board state (4x4 array)
            
        Returns:
            Preprocessed state tensor
        """
        # Flatten the board
        flat_state = state.flatten().astype(np.float32)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(self.device)
        
        return state_tensor
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[List[int]] = None, 
                     training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current game state
            valid_actions: List of valid actions (optional)
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random action
            if valid_actions:
                return random.choice(valid_actions)
            return random.randint(0, self.action_size - 1)
        
        # Greedy action (exploitation)
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.q_network(state_tensor)
            
            if valid_actions:
                # Mask invalid actions
                mask = torch.full((self.action_size,), float('-inf'))
                mask[valid_actions] = 0
                q_values = q_values + mask.to(self.device)
            
            action = q_values.argmax().item()
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (sample batch and update Q-network).
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Need enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).view(self.batch_size, -1).to(self.device) / 131072.0
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).view(self.batch_size, -1).to(self.device) / 131072.0
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store loss for logging
        loss_value = loss.item()
        self.losses.append(loss_value)
        self.training_step += 1
        
        return loss_value
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")
    
    def get_stats(self) -> dict:
        """
        Get agent statistics.
        
        Returns:
            Dictionary of agent stats
        """
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0
        }
