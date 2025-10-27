import os
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from ..dqn.agent import DQNAgent, AgentConfig
from ..dqn.network import DQN as DQNNetwork
from .network import ManagerDQN, ManagerModelConfig


@dataclass
class HierarchicalConfig:
    # Manager config (high-level goal selection)
    manager_input_dim: int = 16
    manager_hidden: tuple = (256, 128)
    manager_output_dim: int = 4  # 4 goals: focus on corners
    manager_lr: float = 3e-4
    manager_gamma: float = 0.99
    manager_epsilon_decay: int = 100000  # Manager exploration steps
    
    # Goal execution horizon (controller executes K steps per goal)
    goal_horizon: int = 15
    
    # Intrinsic reward weight (balance between goal-seeking and score-maximizing)
    intrinsic_weight: float = 0.3

    # Low-level controller config (uses existing DQN agent config defaults)
    controller_config: AgentConfig = None  # Will be initialized in __post_init__

    # Training
    gamma: float = 0.99
    batch_size: int = 256
    device: str = None  # Will auto-detect CUDA
    save_path: str = "models/HierarchicalDQN"
    
    def __post_init__(self):
        if self.controller_config is None:
            self.controller_config = AgentConfig()
        # Auto-detect device if not specified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class HierarchicalDQNAgent:
    """Proper Hierarchical DQN Implementation.
    
    Manager: Selects high-level goals every K steps (which corner to focus on)
    Controller: Executes low-level actions to achieve the current goal
    
    The manager learns to select goals that maximize long-term rewards.
    The controller learns to execute actions to achieve the current goal.
    """

    def __init__(self, config: HierarchicalConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Enable CUDA optimizations if using GPU
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Manager network (selects goals)
        manager_cfg = ManagerModelConfig(
            input_dim=config.manager_input_dim,
            hidden_dims=config.manager_hidden,
            output_dim=config.manager_output_dim,
        )
        self.manager = ManagerDQN(manager_cfg).to(self.device)
        self.manager_target = ManagerDQN(manager_cfg).to(self.device)
        self.manager_target.load_state_dict(self.manager.state_dict())
        self.manager_opt = optim.Adam(self.manager.parameters(), lr=config.manager_lr)
        
        # Manager experience replay
        self.manager_transitions = []
        self.manager_buffer_size = 10000
        
        # Manager epsilon (separate from controller)
        self.manager_epsilon = 1.0
        self.manager_epsilon_end = 0.1
        self.manager_epsilon_decay = config.manager_epsilon_decay  # From config
        self.manager_steps = 0

        # Low-level controller: reuse DQNAgent implementation
        self.controller_config = config.controller_config
        self.controller: Optional[DQNAgent] = None
        
        # Goal tracking
        self.current_goal = None
        self.goal_step_count = 0

        os.makedirs(config.save_path, exist_ok=True)
        self.save_path = config.save_path

    def attach_controller(self, env):
        """Create a controller DQNAgent bound to the same device and environment."""
        from ..dqn.network import DQNModelConfig
        
        # Create model config (use config params)
        model_config = DQNModelConfig(
            output_dim=4,  # 4 actions: up/down/left/right
            hidden_dims=(512, 512, 256)
        )
        
        # Create DQN agent with proper dataclass initialization
        device_str = str(self.device).replace("cuda:0", "cuda")
        self.controller = DQNAgent(
            model_config=model_config,
            agent_config=self.controller_config,
            action_space=["up", "down", "left", "right"],
            device=device_str
        )

    def select_manager_action(self, state):
        """Select goal using epsilon-greedy (manager's decision)."""
        # Decay manager epsilon
        self.manager_epsilon = max(
            self.manager_epsilon_end,
            self.manager_epsilon * (1 - 1/self.manager_epsilon_decay)
        )
        
        if random.random() < self.manager_epsilon:
            return random.randint(0, self.config.manager_output_dim - 1)
        
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.manager(s)
            return int(q_values.argmax(dim=1).item())

    def get_intrinsic_reward(self, state, goal, action_taken, next_state):
        """Calculate intrinsic reward for controller based on goal achievement.
        
        Goals map to corner strategies:
        - Goal 0: Top-left corner strategy
        - Goal 1: Top-right corner strategy  
        - Goal 2: Bottom-left corner strategy
        - Goal 3: Bottom-right corner strategy
        """
        grid = np.array(state).reshape(4, 4)
        next_grid = np.array(next_state).reshape(4, 4)
        max_tile = next_grid.max()
        
        # Define corner positions
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        goal_corner = corners[goal]
        
        intrinsic = 0.0
        
        # Reward for having max tile in goal corner
        if next_grid[goal_corner] == max_tile and max_tile >= 32:
            intrinsic += np.log2(max_tile) * 2.0
        
        # Reward for keeping high tiles near goal corner
        r, c = goal_corner
        neighbors = []
        if r > 0: neighbors.append((r-1, c))
        if r < 3: neighbors.append((r+1, c))
        if c > 0: neighbors.append((r, c-1))
        if c < 3: neighbors.append((r, c+1))
        
        for nr, nc in neighbors:
            if next_grid[nr, nc] >= max_tile / 2 and max_tile >= 64:
                intrinsic += 1.0
        
        return intrinsic

    def train_manager(self):
        """Train manager network using stored transitions."""
        if len(self.manager_transitions) < 32:  # Min batch size
            return
        
        # Sample batch
        batch = random.sample(self.manager_transitions, min(32, len(self.manager_transitions)))
        
        # Convert to numpy arrays first (much faster than list of arrays)
        states = torch.from_numpy(np.array([t[0] for t in batch], dtype=np.float32)).to(self.device)
        goals = torch.from_numpy(np.array([t[1] for t in batch], dtype=np.int64)).to(self.device)
        rewards = torch.from_numpy(np.array([t[2] for t in batch], dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array([t[3] for t in batch], dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array([t[4] for t in batch], dtype=np.float32)).to(self.device)
        
        # Q-learning update
        with torch.no_grad():
            next_q_values = self.manager_target(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.manager_gamma * next_q_values
        
        current_q = self.manager(states).gather(1, goals.unsqueeze(1)).squeeze(1)
        
        loss = nn.functional.mse_loss(current_q, target_q)
        
        self.manager_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.manager.parameters(), 5.0)
        self.manager_opt.step()
        
        # Update target network periodically
        if self.manager_steps % 1000 == 0:
            self.manager_target.load_state_dict(self.manager.state_dict())

    def rollout_with_manager(self, env, max_steps=1000):
        """Hierarchical rollout with proper manager-controller interaction."""
        if self.controller is None:
            raise RuntimeError("Controller agent not attached. Call attach_controller(env) first.")

        obs = env.reset()
        total_reward = 0.0
        total_extrinsic_reward = 0.0
        steps = 0
        done = False
        
        # Manager selects initial goal
        self.current_goal = self.select_manager_action(obs)
        self.goal_step_count = 0
        manager_state = obs.copy()
        
        while not done and steps < max_steps:
            # Controller executes actions for goal_horizon steps
            goal_reward = 0.0
            
            for _ in range(self.config.goal_horizon):
                if done:
                    break
                
                # Controller selects action (with exploration)
                action = self.controller.select_action(obs)
                
                # Execute action
                result = env.step(action)
                
                # Calculate intrinsic reward for controller
                intrinsic_reward = self.get_intrinsic_reward(
                    obs, self.current_goal, action, result.state
                )
                
                # Controller learns from extrinsic + intrinsic reward
                combined_reward = result.reward + intrinsic_reward * self.config.intrinsic_weight
                
                # Store transition for controller
                self.controller.store_transition(
                    obs, action, combined_reward, result.state, result.done
                )
                
                # Train controller
                if self.controller.can_optimize():
                    self.controller.optimize_model()
                
                # Track rewards
                goal_reward += result.reward
                total_extrinsic_reward += result.reward
                total_reward += combined_reward
                
                obs = result.state
                done = result.done
                steps += 1
                self.goal_step_count += 1
            
            # Manager learns from goal completion
            self.manager_transitions.append((
                manager_state,
                self.current_goal,
                goal_reward,  # Manager gets extrinsic reward for this goal
                obs.copy(),
                done
            ))
            
            # Trim buffer
            if len(self.manager_transitions) > self.manager_buffer_size:
                self.manager_transitions.pop(0)
            
            # Train manager
            self.train_manager()
            self.manager_steps += 1
            
            # Select new goal
            if not done:
                manager_state = obs.copy()
                self.current_goal = self.select_manager_action(obs)
                self.goal_step_count = 0
        
        return total_reward, steps

    def save(self, tag: Optional[str] = None):
        """Save both manager and controller."""
        tag = tag or f"hdqn_{int(time.time())}"
        path = os.path.join(self.save_path, f"{tag}.pth")
        data = {
            "manager_state": self.manager.state_dict(),
            "manager_target_state": self.manager_target.state_dict(),
            "manager_opt": self.manager_opt.state_dict(),
            "manager_epsilon": self.manager_epsilon,
            "manager_steps": self.manager_steps,
        }
        if self.controller is not None:
            data["controller_policy"] = self.controller.policy_net.state_dict()
            data["controller_target"] = self.controller.target_net.state_dict()
            data["controller_opt"] = self.controller.optimizer.state_dict()
            data["controller_epsilon"] = self.controller.epsilon
            data["controller_steps"] = self.controller.steps_done
        torch.save(data, path)
        return path

    def load(self, path: str):
        """Load both manager and controller."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.manager.load_state_dict(data["manager_state"])
        self.manager_target.load_state_dict(data["manager_target_state"])
        self.manager_opt.load_state_dict(data.get("manager_opt", {}))
        self.manager_epsilon = data.get("manager_epsilon", 0.1)
        self.manager_steps = data.get("manager_steps", 0)
        
        if self.controller is not None and "controller_policy" in data:
            self.controller.policy_net.load_state_dict(data["controller_policy"])
            self.controller.target_net.load_state_dict(data["controller_target"])
            self.controller.optimizer.load_state_dict(data["controller_opt"])
            self.controller.epsilon = data.get("controller_epsilon", 0.01)
            self.controller.steps_done = data.get("controller_steps", 0)
