# 2048 Reinforcement Learning

A comprehensive reinforcement learning framework for training AI agents to play the 2048 game. This project implements multiple state-of-the-art RL algorithms with professional tooling for training, evaluation, and hyperparameter tuning.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [Algorithms](#algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Training Tips](#training-tips)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multiple RL Algorithms**: DQN, Double DQN, Policy Gradient (REINFORCE), and MCTS
- **Centralized Configuration**: Easy hyperparameter tuning via single config file
- **Real-time Visualization**: Pygame UI and live training plots
- **Training Utilities**: Automatic timing, logging, and checkpointing
- **Professional Structure**: Modular codebase with clear separation of concerns
- **Evaluation System**: Comprehensive logging and performance tracking

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `torch >= 2.0.0` - Neural network training
- `numpy >= 1.24.0` - Numerical computations
- `pygame >= 2.5.0` - Game visualization
- `matplotlib >= 3.7.0` - Training plots

## Quick Start

### Train an Agent

The easiest way to train is using the central control panel:

```bash
# Train with default settings (DQN, 2000 episodes)
python 2048RL.py train

# Train with custom episode count
python 2048RL.py train --episodes 5000

# View current configuration
python 2048RL.py config
```

### Watch a Trained Agent

```bash
# Play with the trained model
python 2048RL.py play

# Play specific model with custom settings
python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 10
```

### Customize Training

Edit the `CONFIG` dictionary in `2048RL.py`:

```python
CONFIG = {
    "algorithm": "dqn",        # Algorithm to use
    "episodes": 2000,          # Training episodes
    "enable_ui": True,         # Show game window
    "enable_plots": True,      # Show training graphs
    
    "dqn": {
        "learning_rate": 1e-4,
        "batch_size": 128,
        "epsilon_decay": 50000,
        "hidden_dims": (256, 256),
        # ... more settings
    }
}
```

## Project Architecture

### Directory Structure

```
2048-Reinforcement-Learning/
├── src/                           # Source code package
│   ├── game/                      # Game logic
│   │   ├── board.py              # Core 2048 mechanics
│   │   └── ui.py                 # Pygame visualization
│   ├── agents/                    # RL agent implementations
│   │   ├── dqn/                  # Deep Q-Network
│   │   │   ├── network.py        # DQN architecture
│   │   │   └── agent.py          # DQN agent + replay buffer
│   │   ├── double_dqn/           # Double DQN
│   │   │   ├── network.py
│   │   │   └── agent.py
│   │   ├── policy_gradient/      # Policy Gradient (REINFORCE)
│   │   │   ├── network.py
│   │   │   └── agent.py
│   │   └── mcts/                 # Monte Carlo Tree Search
│   │       └── agent.py
│   ├── environment.py             # Gym-style game wrapper
│   └── utils.py                   # Training timer & logger
├── scripts/                       # Training scripts
│   └── train_dqn.py              # DQN training with full pipeline
├── models/                        # Saved model checkpoints
├── evaluations/                   # Training logs and metrics
│   └── training_log.txt          # Consolidated training results
├── 2048RL.py                      # Central control panel
├── play.py                        # Model evaluation script
└── requirements.txt               # Python dependencies
```

### Module Overview

**`src/game/board.py`**
- Implements core 2048 game mechanics
- Methods: `step()`, `reset()`, `get_valid_actions()`
- Handles tile merging, move validation, score tracking

**`src/game/ui.py`**
- Pygame-based visualization
- Real-time rendering during training
- User input handling for manual play

**`src/environment.py`**
- Gym-style environment wrapper
- Provides standardized interface: `reset()`, `step()`, `render()`
- Handles reward shaping and state normalization

**`src/agents/*/agent.py`**
- Agent implementations with consistent interface
- Methods: `select_action()`, `act_greedy()`, `save()`, `load()`
- Algorithm-specific training logic

**`src/utils.py`**
- `TrainingTimer`: Tracks training duration with human-readable output
- `EvaluationLogger`: Logs training results to `evaluations/training_log.txt`

## Algorithms

### 1. DQN (Deep Q-Network)

**Type**: Value-based reinforcement learning

**Key Concepts**:
- Learns Q-function Q(s,a) estimating expected cumulative reward
- Uses neural network to approximate Q-values
- Experience replay buffer for stable training
- Separate target network updated periodically

**Architecture**:
```
Input (16 values) → FC(256) → ReLU → FC(256) → ReLU → FC(4 actions)
```

**Training Process**:
1. Agent selects action using epsilon-greedy policy
2. Environment returns next state and reward
3. Transition stored in replay buffer
4. Sample random batch from buffer
5. Compute TD error: `Q(s,a) - (r + γ * max_a' Q(s',a'))`
6. Update network via gradient descent
7. Periodically copy weights to target network

**Hyperparameters** (defaults):
- Learning rate: 1e-4
- Discount factor (γ): 0.99
- Batch size: 128
- Replay buffer: 100,000 transitions
- Epsilon decay: 1.0 → 0.05 over 50,000 steps
- Target network update: Every 1,000 steps

### 2. Double DQN

**Type**: Improved value-based RL

**Key Innovation**: Reduces Q-value overestimation bias

**Difference from DQN**:
- Uses policy network to **select** best action
- Uses target network to **evaluate** selected action
- Update formula: `Q(s,a) - (r + γ * Q_target(s', argmax_a' Q_policy(s',a')))`

**Benefits**:
- More accurate value estimates
- Better performance on complex tasks
- Same computational cost as DQN

### 3. Policy Gradient (REINFORCE)

**Type**: Policy-based reinforcement learning

**Key Concepts**:
- Directly learns policy π(a|s) outputting action probabilities
- No Q-function or value estimation
- Updates after full episode using Monte Carlo returns

**Architecture**:
```
Input (16 values) → FC(256) → ReLU → FC(256) → ReLU → FC(4 actions) → Softmax
```

**Training Process**:
1. Play full episode, store (state, action, reward) at each step
2. Compute returns: G_t = Σ(γ^k * r_{t+k})
3. Compute policy gradient: ∇J = Σ ∇log π(a_t|s_t) * G_t
4. Update policy parameters via gradient ascent

**Characteristics**:
- High variance (single episode samples)
- No replay buffer
- Better for stochastic environments
- Can learn stochastic policies

### 4. MCTS (Monte Carlo Tree Search)

**Type**: Planning algorithm (no learning)

**Key Concepts**:
- Builds search tree via simulations
- Uses UCB1 for exploration-exploitation balance
- Deterministic, no neural networks

**Algorithm**:
1. **Selection**: Traverse tree using UCB1 until leaf node
2. **Expansion**: Add child nodes for unexplored actions
3. **Simulation**: Play random game to terminal state
4. **Backpropagation**: Update visit counts and values

**UCB1 Formula**:
```
UCB(node) = Q(node) + c * sqrt(ln(N_parent) / N_node)
```

**Characteristics**:
- No training required
- Slower per move (runs simulations)
- Deterministic given same state
- Good baseline for comparison

## Evaluation Metrics

### Training Metrics

**Episode Reward**
- Total reward accumulated in one episode
- Primary optimization target
- Tracked in real-time during training

**Moving Average Reward**
- Average over last 50 episodes
- Smooths out variance for trend analysis
- Used for early stopping detection

**Max Tile Achieved**
- Highest tile reached (64, 128, 256, 512, 1024, 2048, etc.)
- Indicates game mastery level
- Key performance indicator

**Final Score**
- Game score at episode termination
- Sum of all merged tile values
- Correlates with max tile and strategy quality

**Training Loss**
- For value-based methods (DQN, Double DQN): TD error magnitude
- For policy gradient: Negative log-likelihood weighted by returns
- Indicates learning progress

**Epsilon Value** (DQN/Double DQN only)
- Current exploration rate
- Decays from 1.0 to 0.05
- Shows exploration-exploitation transition

### Evaluation Metrics

All metrics logged to `evaluations/training_log.txt`:

```
Algorithm:          DQN
Episodes:           2000
Training Time:      2h 15m 30s
Final Avg Reward:   150.25
Best Max Tile:      512
Best Score:         5234
Model Saved:        models/DQN/dqn_2048_final.pth
```

**Comparison Metrics**:
- Average score across evaluation episodes
- Consistency (standard deviation of scores)
- Success rate (% reaching 512+ tile)
- Training efficiency (time to convergence)

## Workflow

### Complete Training Pipeline

```
1. Configuration
   ├─ Edit CONFIG in 2048RL.py
   ├─ Set algorithm, episodes, hyperparameters
   └─ Choose UI/plot options

2. Initialization
   ├─ Load game environment (src/environment.py)
   ├─ Initialize agent (src/agents/*/agent.py)
   ├─ Create replay buffer (DQN/Double DQN)
   ├─ Setup training timer (src/utils.py)
   └─ Initialize evaluation logger

3. Training Loop (per episode)
   ├─ Reset environment → Initial state
   ├─ WHILE not done:
   │  ├─ Agent selects action (epsilon-greedy or policy)
   │  ├─ Environment executes action
   │  ├─ Receive reward, next_state, done
   │  ├─ Store transition in buffer (if applicable)
   │  ├─ Update agent (every N steps)
   │  └─ Render UI (if enabled)
   ├─ Log episode metrics
   ├─ Update plots
   └─ Save checkpoint (every 100 episodes)

4. Optimization (per update)
   ├─ Sample batch from replay buffer
   ├─ Compute loss (TD error or policy gradient)
   ├─ Backpropagate gradients
   ├─ Clip gradients (prevent instability)
   ├─ Update network parameters
   └─ Update target network (if applicable)

5. Completion
   ├─ Stop training timer
   ├─ Save final model
   ├─ Log results to evaluations/training_log.txt
   └─ Display summary statistics
```

### Data Flow

```
Game State (4x4 board)
    ↓
Environment.to_normalized_state()
    ↓
Flattened vector (16 values, normalized 0-1)
    ↓
Neural Network (agent)
    ↓
Q-values [4] or Action Probs [4]
    ↓
Action Selection (epsilon-greedy or sample)
    ↓
Environment.step(action)
    ↓
Next State, Reward, Done, Info
    ↓
Replay Buffer (DQN/Double DQN)
    ↓
Batch Sampling
    ↓
Loss Computation
    ↓
Gradient Descent
    ↓
Parameter Update
```

### File Interactions

```
2048RL.py (entry point)
    ↓
Imports: src.agents.dqn.agent, src.environment, src.utils
    ↓
Creates: GameEnvironment (wraps src.game.board + src.game.ui)
    ↓
Creates: DQNAgent (loads src.agents.dqn.network)
    ↓
Training Loop:
    - environment.reset() → board.reset() → ui.draw()
    - agent.select_action() → network.forward()
    - environment.step() → board.step() → reward calculation
    - agent.optimize_model() → loss computation → backprop
    ↓
Utilities:
    - TrainingTimer.elapsed_str() → human-readable duration
    - EvaluationLogger.log_training() → append to training_log.txt
    ↓
Outputs:
    - models/DQN/dqn_2048_final.pth (model checkpoint)
    - evaluations/training_log.txt (training metrics)
```

## Configuration

### Central Configuration File

All settings managed in `2048RL.py`:

```python
CONFIG = {
    # Algorithm Selection
    "algorithm": "dqn",  # "dqn", "double-dqn", "policy-gradient", "mcts"
    
    # Training Duration
    "episodes": 2000,
    
    # Visualization
    "enable_ui": True,       # Pygame window
    "enable_plots": True,    # Live matplotlib plots
    
    # Algorithm-Specific Settings
    "dqn": {
        "learning_rate": 1e-4,          # Optimizer learning rate
        "gamma": 0.99,                  # Discount factor
        "batch_size": 128,              # Samples per update
        "epsilon_start": 1.0,           # Initial exploration
        "epsilon_end": 0.05,            # Final exploration
        "epsilon_decay": 50000,         # Decay steps
        "replay_buffer_size": 100000,   # Memory capacity
        "target_update_interval": 1000, # Target net sync
        "gradient_clip": 5.0,           # Gradient clipping
        "hidden_dims": (256, 256),      # Network architecture
    },
    
    # Saving
    "save_dir": "models",
    "checkpoint_interval": 100,
    
    # Evaluation
    "eval_episodes": 5,
}
```

### Hyperparameter Tuning Guide

**Learning Rate**:
- Default: 1e-4 (good balance)
- Higher (1e-3): Faster learning, may be unstable
- Lower (1e-5): More stable, slower convergence

**Network Architecture** (`hidden_dims`):
- Smaller (128, 128): Faster, may underfit
- Default (256, 256): Balanced performance
- Larger (512, 512): More capacity, slower training

**Epsilon Decay**:
- Faster (25000): Quick transition to exploitation
- Default (50000): Balanced exploration
- Slower (100000): Extended exploration phase

**Batch Size**:
- Smaller (64): Less stable, faster updates
- Default (128): Good balance
- Larger (256): More stable, requires more memory

## Advanced Usage

### Training Specific Algorithms

```bash
# DQN with custom settings
python 2048RL.py train --episodes 5000

# Direct script usage (bypass central config)
python scripts/train_dqn.py --episodes 3000

# Train without UI (faster)
# Edit CONFIG: "enable_ui": False
python 2048RL.py train
```

### Evaluation and Testing

```bash
# Evaluate model performance
python 2048RL.py play --episodes 20

# Watch specific checkpoint
python play.py models/DQN/dqn_2048_episode_1000.pth

# View training history
cat evaluations/training_log.txt
```

### Comparing Algorithms

1. Train multiple algorithms:
```bash
# Edit CONFIG: "algorithm": "dqn"
python 2048RL.py train

# Edit CONFIG: "algorithm": "double-dqn"
python 2048RL.py train

# Edit CONFIG: "algorithm": "policy-gradient"
python 2048RL.py train
```

2. Check results:
```bash
cat evaluations/training_log.txt
```

3. Compare metrics:
- Training time efficiency
- Final average reward
- Max tile achieved
- Training stability (loss curves)

## Training Tips

### General Recommendations

- **Start Small**: Test with 100 episodes first to verify setup
- **Monitor Plots**: Watch for convergence or divergence
- **Check Logs**: Review `evaluations/training_log.txt` after training
- **Compare Baselines**: Train MCTS for comparison (no learning)
- **Save Checkpoints**: Don't rely only on final model

### Algorithm-Specific Tips

**DQN / Double DQN**:
- Requires 2000+ episodes for meaningful results
- Watch epsilon decay in console output
- If performance plateaus, try increasing network size
- Loss should generally decrease over time
- Target network updates critical for stability

**Policy Gradient**:
- More variance than DQN methods
- May need 3000+ episodes
- Consider adding baseline (not implemented)
- Check for policy collapse (always same action)
- Episode rewards will be noisier

**MCTS**:
- No training needed
- Increase simulations (100 → 500) for better play
- Slower per move but deterministic
- Good for understanding optimal play
- Use as benchmark for learned agents

### Troubleshooting

**Low scores (< 500)**:
- Increase training episodes
- Check learning rate (try 5e-4)
- Verify epsilon decay is working
- Ensure valid moves are prioritized

**Training instability**:
- Reduce learning rate (1e-5)
- Increase batch size (256)
- Check gradient clipping is enabled
- Monitor loss for NaN values

**Slow training**:
- Disable UI: `"enable_ui": False`
- Reduce checkpoint frequency
- Use smaller network (128, 128)
- Run on GPU if available

## Contributing

Contributions welcome! Areas of interest:

- Implement A3C or PPO algorithms
- Add prioritized experience replay
- Tensorboard logging integration
- Hyperparameter search automation
- Web-based UI
- Unit test coverage

## License

MIT License. See `LICENSE` for details.