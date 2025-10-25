# 2048 Reinforcement Learning with Deep Q-Network (DQN)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation of Deep Q-Network (DQN) for learning to play the 2048 game. Features episodic training, interactive Pygame UI, comprehensive logging, and detailed visualizations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Deep Q-Network Theory](#deep-q-network-theory)
- [Implementation Details](#implementation-details)
- [Training](#training)
- [Playing](#playing)
- [Results and Visualization](#results-and-visualization)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Code Explanation](#code-explanation)

## 🎯 Overview

This project implements a Deep Q-Network (DQN) agent that learns to play the 2048 game through reinforcement learning. The agent uses a neural network to approximate Q-values for state-action pairs and learns through experience replay and temporal difference learning.

### What is 2048?

2048 is a sliding tile puzzle game where the objective is to combine tiles with the same number to create a tile with the number 2048. The game is played on a 4×4 grid with tiles that slide in four directions (up, down, left, right). When two tiles with the same number touch, they merge into one tile with double the value.

## ✨ Features

### Core Features
- **Deep Q-Network (DQN)** implementation with experience replay
- **Target Network** for stable learning
- **Epsilon-greedy exploration** strategy
- **Episodic training** with customizable hyperparameters
- **Interactive Pygame UI** for real-time visualization
- **Comprehensive logging** of training and playing sessions
- **Advanced plotting** of training metrics
- **Game state saving** and loading

### Reward Function
The reward function is carefully designed to encourage good 2048 strategies:

```python
reward = log2(max_tile) * 2 + score_bonus + corner_bonus + snake_bonus + empty_cell_bonus
```

**Components:**
1. **Base Reward**: `log2(max_tile) × 2` - Encourages creating higher tiles
2. **Score Bonus**: Rewards points gained from merging tiles
3. **Corner Strategy**: Bonus for keeping the highest tile in a corner
4. **Snake Pattern**: Bonus for maintaining descending tile values in a snake pattern
5. **Empty Cells**: Encourages keeping the board open for more moves

### Training Features
- Automatic checkpointing every N episodes
- Real-time training visualization
- Episode vs Score plotting
- Episode vs Maximum Tile plotting
- Loss curve tracking
- Training statistics logging

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/CodeLanderV/2048-Reinforcement-Learning.git
cd 2048-Reinforcement-Learning
```

2. **Create virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎮 Quick Start

### Training a Model

**Basic training (1000 episodes):**
```bash
python train.py --episodes 1000
```

**Training with visualization:**
```bash
python train.py --episodes 1000 --visualize --viz-freq 10
```

**Training with custom hyperparameters:**
```bash
python train.py \
    --episodes 2000 \
    --lr 0.0001 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.995 \
    --batch-size 64 \
    --hidden-sizes 256 256 128 \
    --visualize
```

### Playing with Trained Model

**Watch the trained agent play:**
```bash
python play.py --mode agent --model saved_models/dqn_2048_final.pth --episodes 10
```

**Play manually:**
```bash
python play.py --mode manual
```

**Watch random agent (for testing):**
```bash
python play.py --mode random --episodes 5
```

## 📁 Project Structure

```
2048-Reinforcement-Learning/
├── src/
│   ├── game/
│   │   ├── __init__.py
│   │   └── game_2048.py          # 2048 game logic implementation
│   ├── agent/
│   │   ├── __init__.py
│   │   └── dqn_agent.py          # DQN agent with replay buffer
│   ├── ui/
│   │   ├── __init__.py
│   │   └── pygame_ui.py          # Pygame visualization
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # Logging utilities
│       └── plotter.py            # Plotting utilities
├── train.py                      # Training script
├── play.py                       # Playing/testing script
├── logs/                         # Training and playing logs
├── plots/                        # Generated plots
├── game_states/                  # Saved game states
├── saved_models/                 # Trained model checkpoints
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧠 Deep Q-Network Theory

### Reinforcement Learning Basics

Reinforcement Learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The agent receives:
- **State (s)**: Current situation/observation
- **Action (a)**: Choice the agent makes
- **Reward (r)**: Feedback from the environment
- **Next State (s')**: Resulting situation after action

The goal is to learn a **policy** π that maximizes cumulative reward over time.

### Q-Learning

Q-Learning is a model-free RL algorithm that learns the value of taking action `a` in state `s`:

```
Q(s, a) = Expected cumulative reward of taking action a in state s
```

The optimal Q-function satisfies the **Bellman equation**:

```
Q*(s, a) = r + γ · max[Q*(s', a')]
```

Where:
- `r` is the immediate reward
- `γ` (gamma) is the discount factor (0 < γ < 1)
- `s'` is the next state
- `a'` are possible actions in the next state

### Deep Q-Network (DQN)

Traditional Q-Learning uses a table to store Q-values, which doesn't scale to large state spaces. **DQN** uses a neural network to approximate Q-values:

```
Q(s, a; θ) ≈ Q*(s, a)
```

Where θ represents the neural network parameters.

### Key Innovations in DQN

#### 1. Experience Replay
Instead of learning from experiences sequentially, DQN stores experiences in a **replay buffer**:

```python
buffer = [(s₁, a₁, r₁, s₁', done₁), (s₂, a₂, r₂, s₂', done₂), ...]
```

During training, we sample random mini-batches from the buffer. This:
- Breaks correlation between consecutive samples
- Allows reusing experiences multiple times
- Improves sample efficiency
- Stabilizes training

**Code Implementation:** See `ReplayBuffer` class in `src/agent/dqn_agent.py`

#### 2. Target Network
DQN uses two networks:
- **Q-Network (θ)**: Updated every step
- **Target Network (θ⁻)**: Updated periodically

The target network provides stable Q-value targets:

```
Target = r + γ · max[Q(s', a'; θ⁻)]
Loss = MSE(Q(s, a; θ), Target)
```

This prevents the "moving target" problem and stabilizes training.

**Code Implementation:** See `DQNAgent.__init__()` and `train_step()` in `src/agent/dqn_agent.py`

#### 3. Epsilon-Greedy Exploration
The agent balances exploration and exploitation:

```python
if random() < epsilon:
    action = random_action()  # Explore
else:
    action = argmax(Q(s, a))  # Exploit
```

Epsilon decays over time: `ε = max(ε_min, ε × decay_rate)`

**Code Implementation:** See `select_action()` in `src/agent/dqn_agent.py`

### DQN Algorithm

```
1. Initialize Q-network with random weights θ
2. Initialize target network θ⁻ = θ
3. Initialize replay buffer D
4. For episode = 1 to N:
    5. Reset environment, get initial state s
    6. For step = 1 to max_steps:
        7. Select action a using ε-greedy policy
        8. Execute action, observe reward r and next state s'
        9. Store transition (s, a, r, s', done) in D
        10. Sample random mini-batch from D
        11. Compute target: y = r + γ·max[Q(s', a'; θ⁻)]
        12. Update Q-network by minimizing loss: (y - Q(s, a; θ))²
        13. Every C steps, update target network: θ⁻ = θ
        14. Decay ε
        15. s = s'
```

## 🔧 Implementation Details

### Game State Representation

The 2048 board is represented as a 4×4 NumPy array:

```python
board = np.array([
    [2, 4, 8, 16],
    [0, 0, 2, 4],
    [0, 0, 0, 2],
    [0, 0, 0, 0]
])
```

For the neural network, the board is:
1. Flattened to a 16-dimensional vector
2. Normalized by dividing by 131072 (2^17, maximum possible tile value in practice)

### Neural Network Architecture

```
Input Layer:    16 neurons (flattened 4×4 board)
                ↓
Hidden Layer 1: 256 neurons + ReLU + Dropout(0.2)
                ↓
Hidden Layer 2: 256 neurons + ReLU + Dropout(0.2)
                ↓
Hidden Layer 3: 128 neurons + ReLU + Dropout(0.2)
                ↓
Output Layer:   4 neurons (Q-values for 4 actions)
```

**Code Implementation:** See `DQNNetwork` class in `src/agent/dqn_agent.py`

### Reward Function Design

The reward function encourages strategic play:

```python
def calculate_reward(score_gained):
    reward = 0
    
    # 1. Base reward: log₂(max_tile) × 2
    if max_tile > 0:
        reward += np.log2(max_tile) * 2
    
    # 2. Score bonus
    reward += score_gained * 0.1
    
    # 3. Corner bonus (highest tile in corner)
    if max_tile in corners:
        reward += 5.0
    
    # 4. Snake pattern bonus
    reward += calculate_snake_bonus()
    
    # 5. Empty cells bonus
    reward += len(empty_cells) * 0.5
    
    return reward
```

**Code Implementation:** See `calculate_reward()` in `src/game/game_2048.py`

### Training Process

The training loop follows this structure:

```python
for episode in range(num_episodes):
    state = game.reset()
    
    while not done:
        # 1. Select action (ε-greedy)
        action = agent.select_action(state)
        
        # 2. Execute action
        next_state, reward, done, info = game.step(action)
        
        # 3. Store experience
        agent.store_experience(state, action, reward, next_state, done)
        
        # 4. Train agent
        loss = agent.train_step()
        
        # 5. Update state
        state = next_state
```

**Code Implementation:** See `train_dqn()` in `train.py`

## 📊 Training

### Default Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Episodes | 1000 | Number of training episodes |
| Learning Rate | 0.0001 | Adam optimizer learning rate |
| Gamma (γ) | 0.99 | Discount factor |
| Epsilon Start | 1.0 | Initial exploration rate |
| Epsilon End | 0.01 | Final exploration rate |
| Epsilon Decay | 0.995 | Decay rate per episode |
| Batch Size | 64 | Mini-batch size for training |
| Buffer Capacity | 100,000 | Replay buffer size |
| Target Update | 10 | Update target network every N steps |
| Hidden Layers | [256, 256, 128] | Neural network architecture |

### Training Commands

**Quick training (1000 episodes, ~1-2 hours):**
```bash
python train.py --episodes 1000
```

**Long training (5000 episodes, ~5-10 hours):**
```bash
python train.py --episodes 5000 --save-freq 250
```

**Training with visualization (slower but informative):**
```bash
python train.py --episodes 1000 --visualize --viz-freq 20 --fps 10
```

**Custom hyperparameters:**
```bash
python train.py \
    --episodes 2000 \
    --lr 0.0001 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.995 \
    --batch-size 128 \
    --buffer-capacity 200000 \
    --target-update 20 \
    --hidden-sizes 512 256 128 \
    --save-freq 100 \
    --model-name dqn_custom
```

### Training Output

During training, you'll see output like:

```
Episode 100/1000 | Score:  1234 | Max Tile:   256 | Moves:  234 | Reward: 145.23 | Eps: 0.605 | Avg Score (100): 987.45

  ✓ Checkpoint saved at episode 100
    Best Score: 1456 | Best Max Tile: 512
```

### Saved Artifacts

Training automatically saves:
- **Models**: `saved_models/dqn_2048_ep{N}.pth` (every N episodes)
- **Training Logs**: `logs/training_log_ep{N}.json`
- **Statistics**: `logs/statistics_ep{N}.json`
- **Plots**:
  - `plots/scores_ep{N}.png`
  - `plots/max_tiles_ep{N}.png`
  - `plots/combined_ep{N}.png`
  - `plots/loss_ep{N}.png`

## 🎮 Playing

### Agent Mode

Watch your trained agent play:

```bash
python play.py --mode agent --model saved_models/dqn_2048_final.pth --episodes 10 --fps 5
```

Options:
- `--model`: Path to trained model
- `--episodes`: Number of episodes to play
- `--fps`: Visualization speed (frames per second)
- `--no-save-logs`: Don't save play logs

### Manual Mode

Play the game yourself:

```bash
python play.py --mode manual
```

Controls:
- **Arrow Keys** or **WASD**: Move tiles
- **ESC** or **Q**: Quit game

### Random Agent Mode

Watch a random agent (useful for baseline comparison):

```bash
python play.py --mode random --episodes 5 --fps 10
```

## 📈 Results and Visualization

### Training Plots

The system automatically generates several types of plots:

#### 1. Episode vs Score
Shows how the agent's score improves over training episodes.
- Raw scores (light blue)
- Moving average (red line)

#### 2. Episode vs Maximum Tile
Tracks the highest tile achieved in each episode.
- Y-axis in log scale (base 2)
- Shows progression: 128 → 256 → 512 → 1024 → 2048

#### 3. Combined Metrics
Four-panel view showing:
- Episode scores
- Maximum tiles
- Moves per episode
- Distribution of maximum tiles

#### 4. Training Loss
Shows the neural network training loss over time.
- Indicates learning progress
- Should generally decrease over time

#### 5. Training Summary
Comprehensive overview with:
- All metrics in one view
- Statistical summary panel
- Final training results

### Interpreting Results

**Good Training Signs:**
- ✓ Average score increasing over episodes
- ✓ Maximum tile progressing (256 → 512 → 1024 → 2048)
- ✓ Loss decreasing and stabilizing
- ✓ Number of moves per episode increasing

**Poor Training Signs:**
- ✗ Scores plateauing early
- ✗ Maximum tile stuck at low values (64, 128)
- ✗ Loss increasing or unstable
- ✗ Short episodes (few moves)

### Example Results

After training for 1000 episodes, typical results:

```
Average Score: 3,000 - 8,000
Best Score: 10,000 - 20,000
Average Max Tile: 512 - 1024
Best Max Tile: 1024 - 2048
Success Rate (2048 tile): 5% - 20%
```

## 🎛️ Hyperparameter Tuning

### Key Hyperparameters

#### Learning Rate (lr)
- **Range**: 0.00001 - 0.001
- **Default**: 0.0001
- **Effect**: 
  - Too high: Unstable training, oscillation
  - Too low: Very slow learning
- **Recommendation**: Start with 0.0001, increase if learning is slow

#### Gamma (γ)
- **Range**: 0.95 - 0.99
- **Default**: 0.99
- **Effect**: 
  - Higher: Values long-term rewards more
  - Lower: More myopic (short-term focus)
- **Recommendation**: 0.99 for 2048 (long-term strategy game)

#### Epsilon Decay
- **Range**: 0.99 - 0.999
- **Default**: 0.995
- **Effect**: 
  - Faster decay: Exploit earlier
  - Slower decay: Explore longer
- **Recommendation**: 0.995 for 1000 episodes, 0.999 for 5000+ episodes

#### Batch Size
- **Range**: 32 - 256
- **Default**: 64
- **Effect**: 
  - Larger: More stable gradients, slower training
  - Smaller: Faster training, more noise
- **Recommendation**: 64 or 128 for good balance

#### Network Architecture
- **Default**: [256, 256, 128]
- **Alternatives**:
  - Smaller: [128, 128] - Faster, less capacity
  - Larger: [512, 256, 128] - More capacity, slower
- **Recommendation**: Start with default, increase if underfitting

### Tuning Strategy

1. **Start with defaults** - Get baseline performance
2. **One at a time** - Change one hyperparameter at a time
3. **Monitor metrics** - Watch loss, scores, max tiles
4. **Iterate** - Keep changes that improve performance

### Recommended Configurations

**Fast Training (good for testing):**
```bash
python train.py --episodes 500 --lr 0.0005 --epsilon-decay 0.99 --batch-size 32
```

**Balanced Training (recommended):**
```bash
python train.py --episodes 2000 --lr 0.0001 --epsilon-decay 0.995 --batch-size 64
```

**Deep Training (best performance):**
```bash
python train.py --episodes 5000 --lr 0.00005 --epsilon-decay 0.999 --batch-size 128 --hidden-sizes 512 256 128
```

## 💻 Code Explanation

### 1. Game Logic (`src/game/game_2048.py`)

#### Class: `Game2048`
Main game engine implementing 2048 rules.

**Key Methods:**

```python
def reset(self) -> np.ndarray:
    """Reset game to initial state."""
    # Clears board, adds two random tiles
    # Returns initial state
```

```python
def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
    """Execute one game step."""
    # Takes action (0=left, 1=right, 2=up, 3=down)
    # Returns (next_state, reward, done, info)
```

```python
def move_left(self) -> Tuple[bool, int]:
    """Move all tiles left."""
    # Uses _merge_line() to combine tiles
    # Returns (whether board changed, score gained)
```

```python
def calculate_reward(self, score_gained: int) -> float:
    """Calculate reward for current state."""
    # Implements: log₂(max_tile) × 2 + bonuses
    # Encourages corner strategy and snake pattern
```

**Reward Components:**
1. Base: `np.log2(max_tile) * 2`
2. Score: `score_gained * 0.1`
3. Corner: `5.0` if max tile in corner
4. Snake: Bonus for monotonic rows
5. Empty cells: `num_empty * 0.5`

### 2. DQN Agent (`src/agent/dqn_agent.py`)

#### Class: `ReplayBuffer`
Stores and samples experiences for training.

```python
def push(self, state, action, reward, next_state, done):
    """Store one experience."""
    self.buffer.append((state, action, reward, next_state, done))
```

```python
def sample(self, batch_size):
    """Sample random batch for training."""
    return random.sample(self.buffer, batch_size)
```

#### Class: `DQNNetwork`
Neural network for Q-value approximation.

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through network."""
    x = x / 131072.0  # Normalize input
    return self.network(x)  # Output Q-values for each action
```

#### Class: `DQNAgent`
Main agent implementing DQN algorithm.

**Key Methods:**

```python
def select_action(self, state, valid_actions, training=True):
    """Select action using ε-greedy policy."""
    if training and random() < epsilon:
        return random_action()  # Explore
    else:
        return argmax(Q(state))  # Exploit
```

```python
def train_step(self):
    """Perform one training step."""
    # 1. Sample batch from replay buffer
    # 2. Compute Q(s, a) from Q-network
    # 3. Compute target = r + γ·max[Q(s', a')] from target network
    # 4. Compute loss = MSE(Q(s, a), target)
    # 5. Backpropagate and update Q-network
    # 6. Update target network periodically
```

**Training Loop:**
1. Sample batch: `states, actions, rewards, next_states, dones = buffer.sample(batch_size)`
2. Current Q: `Q(s, a) = q_network(states).gather(1, actions)`
3. Target Q: `target = rewards + (1 - dones) * gamma * target_network(next_states).max()`
4. Loss: `loss = MSE(Q(s, a), target)`
5. Update: `optimizer.step()`

### 3. UI (`src/ui/pygame_ui.py`)

#### Class: `GameUI`
Pygame-based visualization.

```python
def draw_board(self, board: np.ndarray):
    """Draw the game board."""
    # Renders 4×4 grid with color-coded tiles
    # Tile colors based on value (2=light, 2048=gold)
```

```python
def update(self, board, score, max_tile, moves, episode, mode, fps):
    """Update display with current game state."""
    # Handles pygame events
    # Draws info panel and board
    # Controls FPS
```

#### Class: `ManualGameUI`
Extended UI for manual play with keyboard controls.

```python
def get_action_from_key(self):
    """Get action from keyboard input."""
    # Arrow keys or WASD → actions
    # ESC/Q → quit
```

### 4. Logging (`src/utils/logger.py`)

#### Class: `GameLogger`
Tracks training and playing sessions.

```python
def start_episode(self, episode, mode):
    """Start logging new episode."""
    # Initializes episode data structure
```

```python
def log_step(self, action, reward, state, score, max_tile):
    """Log single step."""
    # Records action, reward, state
```

```python
def end_episode(self, score, max_tile, additional_info):
    """End episode and save logs."""
    # Computes episode statistics
    # Adds to training/playing log
```

**Saved Data:**
- All moves and rewards per episode
- Final scores and max tiles
- Timestamps
- Agent statistics (epsilon, buffer size, etc.)

### 5. Plotting (`src/utils/plotter.py`)

#### Class: `TrainingPlotter`
Creates visualizations of training progress.

```python
def plot_episode_scores(self, scores, window_size=100):
    """Plot scores over episodes."""
    # Raw scores + moving average
    # Saves to plots/ directory
```

```python
def create_training_summary(self, scores, max_tiles, moves, losses):
    """Create comprehensive training summary."""
    # 6-panel plot with all metrics
    # Includes statistics summary
```

### 6. Training Script (`train.py`)

Main training loop:

```python
def train_dqn(episodes, hyperparameters...):
    # Initialize game, agent, logger, plotter
    
    for episode in range(episodes):
        state = game.reset()
        
        while not done:
            # Select action
            action = agent.select_action(state, valid_actions)
            
            # Execute action
            next_state, reward, done, info = game.step(action)
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            
            # Log and visualize
            logger.log_step(...)
            if visualize:
                ui.update(...)
        
        # Save checkpoints
        if episode % save_freq == 0:
            agent.save(...)
            logger.save_training_log(...)
            plotter.create_plots(...)
```

### 7. Play Script (`play.py`)

Testing and manual play:

```python
def play_agent(model_path, episodes):
    # Load trained model
    agent.load(model_path)
    
    for episode in range(episodes):
        state = game.reset()
        
        while not done:
            # Greedy action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = game.step(action)
            
            # Visualize
            ui.update(...)
```

## 🎓 Learning Resources

### Reinforcement Learning
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [Deep RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- [OpenAI Spinning Up](https://spinningup.openai.com/)

### DQN Papers
- [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Human-level control through deep reinforcement learning (2015)](https://www.nature.com/articles/nature14236) - Nature DQN paper

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Double DQN, Dueling DQN implementations
- Prioritized experience replay
- Multi-step returns (n-step DQN)
- Rainbow DQN (combining all improvements)
- Web-based UI
- Distributed training

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **CodeLanderV** - [GitHub](https://github.com/CodeLanderV)

## 🙏 Acknowledgments

- Original DQN paper by DeepMind
- 2048 game by Gabriele Cirulli
- PyTorch team for the deep learning framework
- Pygame community for the visualization library

---

**Happy Training! 🚀**

For questions or issues, please open an issue on GitHub.
