# 🎮 2048 Reinforcement Learning# 2048 Reinforcement Learning



A clean, well-documented reinforcement learning framework for training AI agents to play 2048. This project features multiple RL algorithms, comprehensive documentation, and professional code quality with **zero redundancy**.A comprehensive reinforcement learning framework for training AI agents to play the 2048 game. This project implements multiple state-of-the-art RL algorithms with professional tooling for training, evaluation, and hyperparameter tuning.



![Python](https://img.shields.io/badge/Python-3.10%2B-blue)![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

![License](https://img.shields.io/badge/License-MIT-green)![License](https://img.shields.io/badge/License-MIT-green)



---## Table of Contents



## ✨ Features- [Features](#features)

- [Installation](#installation)

- **🤖 4 RL Algorithms**: DQN, Double DQN, MCTS, Policy Gradient- [Quick Start](#quick-start)

- **⚙️ Simple Configuration**: Edit one file (`2048RL.py`) to control everything- [Project Architecture](#project-architecture)

- **📊 Live Visualization**: Real-time training plots + Pygame game window- [Algorithms](#algorithms)

- **📝 Comprehensive Docs**: Every function, class, and parameter explained- [Evaluation Metrics](#evaluation-metrics)

- **🧹 Clean Code**: Refactored, documented, zero duplicate code- [Workflow](#workflow)

- **🎯 Research-Proven Hyperparameters**: Optimized defaults that actually work- [Configuration](#configuration)

- [Advanced Usage](#advanced-usage)

---- [Training Tips](#training-tips)

- [Contributing](#contributing)

## 🚀 Quick Start- [License](#license)



### Installation## Features



```bash- **Multiple RL Algorithms**: DQN, Double DQN, Policy Gradient (REINFORCE), and MCTS

# Clone repository- **Centralized Configuration**: Easy hyperparameter tuning via single config file

git clone https://github.com/CodeLanderV/2048-Reinforcement-Learning.git- **Real-time Visualization**: Pygame UI and live training plots

cd 2048-Reinforcement-Learning- **Training Utilities**: Automatic timing, logging, and checkpointing

- **Professional Structure**: Modular codebase with clear separation of concerns

# Install dependencies- **Evaluation System**: Comprehensive logging and performance tracking

pip install -r requirements.txt

```## Installation



**Requirements**: Python 3.10+, PyTorch 2.0+, NumPy, Pygame, Matplotlib### Prerequisites



### Train Your First Agent (5 minutes)- Python 3.10 or higher

- pip package manager

```bash

# Train DQN for 100 episodes (quick test)### Install Dependencies

python 2048RL.py train --algorithm dqn --episodes 100

```bash

# Train for real (2000 episodes, ~2 hours)pip install -r requirements.txt

python 2048RL.py train --algorithm dqn --episodes 2000```



# Train without UI (faster)Required packages:

python 2048RL.py train --algorithm dqn --episodes 2000 --no-ui- `torch >= 2.0.0` - Neural network training

```- `numpy >= 1.24.0` - Numerical computations

- `pygame >= 2.5.0` - Game visualization

### Watch Your Agent Play- `matplotlib >= 3.7.0` - Training plots



```bash## Quick Start

# Play with trained model

python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 5### Train an Agent



# Or use simplified playerThe easiest way to train is using the central control panel:

python play.py

``````bash

# Train with default settings (DQN, 2000 episodes)

---python 2048RL.py train



## 📁 Project Structure# Train with custom episode count

python 2048RL.py train --episodes 5000

```

2048-Reinforcement-Learning/# View current configuration

│python 2048RL.py config

├── 2048RL.py              ⭐ MAIN: Train/play all algorithms (fully documented)```

├── play.py                Simple model player

├── README.md              This file### Watch a Trained Agent

├── REFACTORING_SUMMARY.md Code quality improvements

├── FILE_DOCUMENTATION.md  ⭐ Detailed explanation of every file```bash

├── requirements.txt       Python dependencies# Play with the trained model

│python 2048RL.py play

├── src/                   Core implementation

│   ├── environment.py     Gym-style RL interface (documented)# Play specific model with custom settings

│   ├── utils.py           Training utilities (documented)python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 10

│   ├── game/```

│   │   ├── board.py       2048 game logic

│   │   └── ui.py          Pygame visualization### Customize Training

│   └── agents/

│       ├── dqn/           Deep Q-NetworkEdit the `CONFIG` dictionary in `2048RL.py`:

│       ├── double_dqn/    Double DQN (reduces overestimation)

│       ├── mcts/          Monte Carlo Tree Search (planning)```python

│       └── policy_gradient/ Policy-based RLCONFIG = {

│    "algorithm": "dqn",        # Algorithm to use

├── models/                Saved model checkpoints    "episodes": 2000,          # Training episodes

│   ├── DQN/    "enable_ui": True,         # Show game window

│   └── DoubleDQN/    "enable_plots": True,      # Show training graphs

│    

└── evaluations/           Training logs    "dqn": {

    └── training_log.txt        "learning_rate": 1e-4,

```        "batch_size": 128,

        "epsilon_decay": 50000,

---        "hidden_dims": (256, 256),

        # ... more settings

## 🤖 Algorithms    }

}

| Algorithm | Type | Description | Best For |```

|-----------|------|-------------|----------|

| **DQN** | Value-based | Deep Q-Network with experience replay | General purpose, stable |## Project Architecture

| **Double DQN** | Value-based | Reduces Q-value overestimation bias | Higher scores, less stuck |

| **MCTS** | Planning | Monte Carlo Tree Search (no learning) | Baseline comparison |### Directory Structure

| **Policy Gradient** | Policy-based | Direct policy optimization | (Not yet implemented) |

```

### Quick Algorithm Comparison2048-Reinforcement-Learning/

├── src/                           # Source code package

```bash│   ├── game/                      # Game logic

# Train all algorithms and compare│   │   ├── board.py              # Core 2048 mechanics

python 2048RL.py train --algorithm dqn --episodes 2000│   │   └── ui.py                 # Pygame visualization

python 2048RL.py train --algorithm double-dqn --episodes 2000│   ├── agents/                    # RL agent implementations

python 2048RL.py train --algorithm mcts --episodes 50│   │   ├── dqn/                  # Deep Q-Network

│   │   │   ├── network.py        # DQN architecture

# View results│   │   │   └── agent.py          # DQN agent + replay buffer

cat evaluations/training_log.txt│   │   ├── double_dqn/           # Double DQN

```│   │   │   ├── network.py

│   │   │   └── agent.py

---│   │   ├── policy_gradient/      # Policy Gradient (REINFORCE)

│   │   │   ├── network.py

## ⚙️ Configuration│   │   │   └── agent.py

│   │   └── mcts/                 # Monte Carlo Tree Search

All settings in one place: **`2048RL.py`** → `CONFIG` dictionary│   │       └── agent.py

│   ├── environment.py             # Gym-style game wrapper

### Key Settings│   └── utils.py                   # Training timer & logger

├── scripts/                       # Training scripts

```python│   └── train_dqn.py              # DQN training with full pipeline

CONFIG = {├── models/                        # Saved model checkpoints

    # What to train├── evaluations/                   # Training logs and metrics

    "algorithm": "dqn",        # "dqn", "double-dqn", "mcts"│   └── training_log.txt          # Consolidated training results

    "episodes": 2000,          # How many games├── 2048RL.py                      # Central control panel

    ├── play.py                        # Model evaluation script

    # Visualization└── requirements.txt               # Python dependencies

    "enable_ui": True,         # Show game window```

    "enable_plots": True,      # Show training graphs

    ### Module Overview

    # DQN Hyperparameters (research-proven defaults)

    "dqn": {**`src/game/board.py`**

        "learning_rate": 1e-4,      # How fast to learn- Implements core 2048 game mechanics

        "epsilon_end": 0.1,         # Keep 10% exploration (prevents getting stuck)- Methods: `step()`, `reset()`, `get_valid_actions()`

        "epsilon_decay": 100000,    # Explore for 100k steps before exploiting- Handles tile merging, move validation, score tracking

        "hidden_dims": (256, 256),  # Neural network size

    },**`src/game/ui.py`**

}- Pygame-based visualization

```- Real-time rendering during training

- User input handling for manual play

### Hyperparameter Guide

**`src/environment.py`**

| Parameter | What It Does | If Too High | If Too Low |- Gym-style environment wrapper

|-----------|--------------|-------------|------------|- Provides standardized interface: `reset()`, `step()`, `render()`

| `learning_rate` | Learning speed | Unstable, diverges | Too slow |- Handles reward shaping and state normalization

| `epsilon_end` | Final exploration % | Keeps playing randomly | Gets stuck repeating moves |

| `epsilon_decay` | How long to explore | Exploits too early | Never learns to exploit |**`src/agents/*/agent.py`**

| `batch_size` | Training samples/update | Uses more memory | More variance |- Agent implementations with consistent interface

- Methods: `select_action()`, `act_greedy()`, `save()`, `load()`

**💡 Pro Tip**: Our defaults are optimized from research papers. Start with these!- Algorithm-specific training logic



---**`src/utils.py`**

- `TrainingTimer`: Tracks training duration with human-readable output

## 📊 Understanding Training Output- `EvaluationLogger`: Logs training results to `evaluations/training_log.txt`



### Console Output## Algorithms



```### 1. DQN (Deep Q-Network)

Ep  100 | Reward:   45.30 | Score:   312 | Tile:   64 | ε: 0.800 | Time: 0:05:23

Ep  200 | Reward:   78.15 | Score:   528 | Tile:  128 | ε: 0.600 | Time: 0:10:45**Type**: Value-based reinforcement learning

Ep  500 | Reward:  142.50 | Score:  1250 | Tile:  256 | ε: 0.300 | Time: 0:25:12

```**Key Concepts**:

- Learns Q-function Q(s,a) estimating expected cumulative reward

**What to watch:**- Uses neural network to approximate Q-values

- **Reward** should increase over time- Experience replay buffer for stable training

- **Score** tracks game performance- Separate target network updated periodically

- **Max Tile** indicates mastery (128 → 256 → 512 → 1024)

- **ε (epsilon)** shows exploration → exploitation transition**Architecture**:

- Training saves checkpoints every 100 episodes```

Input (16 values) → FC(256) → ReLU → FC(256) → ReLU → FC(4 actions)

### Training Plots```



**Top plot (Rewards):** Shows learning progress. Should trend upward.**Training Process**:

1. Agent selects action using epsilon-greedy policy

**Bottom plot (Score & Tiles):** Game performance metrics. Higher = better agent.2. Environment returns next state and reward

3. Transition stored in replay buffer

Close the plot window to stop training early.4. Sample random batch from buffer

5. Compute TD error: `Q(s,a) - (r + γ * max_a' Q(s',a'))`

---6. Update network via gradient descent

7. Periodically copy weights to target network

## 🎯 Training Tips

**Hyperparameters** (defaults):

### Getting Good Results- Learning rate: 1e-4

- Discount factor (γ): 0.99

1. **Start with 2000 episodes** (minimum for DQN)- Batch size: 128

2. **Watch the first 100 episodes** - should see improvement- Replay buffer: 100,000 transitions

3. **Check epsilon decay** - should drop from 1.0 → 0.1 gradually- Epsilon decay: 1.0 → 0.05 over 50,000 steps

4. **Monitor for "stuck" behavior** - agent repeating invalid moves means epsilon_end is too low- Target network update: Every 1,000 steps



### Common Issues & Fixes### 2. Double DQN



| Problem | Solution |**Type**: Improved value-based RL

|---------|----------|

| **Agent gets stuck** | Increase `epsilon_end` to 0.15 |**Key Innovation**: Reduces Q-value overestimation bias

| **Low scores (< 500)** | Train longer (3000 episodes) or increase `epsilon_decay` |

| **Training too slow** | Use `--no-ui` flag, reduce checkpoint frequency |**Difference from DQN**:

| **Unstable training** | Lower `learning_rate` to 5e-5 |- Uses policy network to **select** best action

- Uses target network to **evaluate** selected action

### Algorithm-Specific Advice- Update formula: `Q(s,a) - (r + γ * Q_target(s', argmax_a' Q_policy(s',a')))`



**DQN:****Benefits**:

- Good all-around choice- More accurate value estimates

- Needs 2000+ episodes- Better performance on complex tasks

- Watch for epsilon decay working properly- Same computational cost as DQN



**Double DQN:**### 3. Policy Gradient (REINFORCE)

- Better than DQN at avoiding "stuck" behavior

- More stable Q-value estimates**Type**: Policy-based reinforcement learning

- Try this if DQN gets stuck

**Key Concepts**:

**MCTS:**- Directly learns policy π(a|s) outputting action probabilities

- No training needed (pure planning)- No Q-function or value estimation

- Slower per move (runs simulations)- Updates after full episode using Monte Carlo returns

- Great baseline to compare learned agents against

**Architecture**:

---```

Input (16 values) → FC(256) → ReLU → FC(256) → ReLU → FC(4 actions) → Softmax

## 📖 Documentation```



- **`README.md`** (this file) - Quick start & overview**Training Process**:

- **`FILE_DOCUMENTATION.md`** ⭐ - Detailed explanation of every file in the project1. Play full episode, store (state, action, reward) at each step

- **`REFACTORING_SUMMARY.md`** - Code quality improvements made2. Compute returns: G_t = Σ(γ^k * r_{t+k})

- **Inline comments** - Every function and complex logic explained3. Compute policy gradient: ∇J = Σ ∇log π(a_t|s_t) * G_t

4. Update policy parameters via gradient ascent

### Learning the Codebase

**Characteristics**:

1. **Start here**: `FILE_DOCUMENTATION.md` - explains every file's purpose- High variance (single episode samples)

2. **Read**: `2048RL.py` - heavily commented main file- No replay buffer

3. **Explore**: `src/environment.py` - see how RL interface works- Better for stochastic environments

4. **Deep dive**: `src/agents/dqn/agent.py` - DQN implementation- Can learn stochastic policies



---### 4. MCTS (Monte Carlo Tree Search)



## 🔬 Evaluation & Metrics**Type**: Planning algorithm (no learning)



### View Training History**Key Concepts**:

- Builds search tree via simulations

```bash- Uses UCB1 for exploration-exploitation balance

# See all training sessions- Deterministic, no neural networks

cat evaluations/training_log.txt

```**Algorithm**:

1. **Selection**: Traverse tree using UCB1 until leaf node

### Sample Output2. **Expansion**: Add child nodes for unexplored actions

3. **Simulation**: Play random game to terminal state

```4. **Backpropagation**: Update visit counts and values

════════════════════════════════════════════════════════════════════════════════

Training Session: 2025-10-14 15:30:00**UCB1 Formula**:

════════════════════════════════════════════════════════════════════════════════```

Algorithm:          DQNUCB(node) = Q(node) + c * sqrt(ln(N_parent) / N_node)

Episodes:           2000```

Training Time:      2:15:30

Final Avg Reward:   145.32**Characteristics**:

Best Max Tile:      512- No training required

Best Score:         5234- Slower per move (runs simulations)

Model Saved:        models/DQN/dqn_final.pth- Deterministic given same state

Notes:              LR=0.0001, ε_end=0.1, ε_decay=100000- Good baseline for comparison

════════════════════════════════════════════════════════════════════════════════

```## Evaluation Metrics



### Key Metrics### Training Metrics



- **Final Avg Reward**: Average over last 100 episodes**Episode Reward**

- **Best Max Tile**: Highest tile reached (128, 256, 512, 1024, 2048)- Total reward accumulated in one episode

- **Best Score**: Highest game score achieved- Primary optimization target

- **Training Time**: Total duration- Tracked in real-time during training



---**Moving Average Reward**

- Average over last 50 episodes

## 💻 Command Line Reference- Smooths out variance for trend analysis

- Used for early stopping detection

### Training Commands

**Max Tile Achieved**

```bash- Highest tile reached (64, 128, 256, 512, 1024, 2048, etc.)

# Basic training- Indicates game mastery level

python 2048RL.py train --algorithm dqn --episodes 2000- Key performance indicator



# All options**Final Score**

python 2048RL.py train \- Game score at episode termination

    --algorithm double-dqn \- Sum of all merged tile values

    --episodes 3000 \- Correlates with max tile and strategy quality

    --no-ui \

    --no-plots**Training Loss**

```- For value-based methods (DQN, Double DQN): TD error magnitude

- For policy gradient: Negative log-likelihood weighted by returns

### Playing Commands- Indicates learning progress



```bash**Epsilon Value** (DQN/Double DQN only)

# Play with default model- Current exploration rate

python 2048RL.py play- Decays from 1.0 to 0.05

- Shows exploration-exploitation transition

# Custom model and episodes

python 2048RL.py play \### Evaluation Metrics

    --model models/DQN/dqn_ep1000.pth \

    --episodes 10 \All metrics logged to `evaluations/training_log.txt`:

    --no-ui  # Run without visualization

``````

Algorithm:          DQN

### HelpEpisodes:           2000

Training Time:      2h 15m 30s

```bashFinal Avg Reward:   150.25

python 2048RL.py --helpBest Max Tile:      512

python 2048RL.py train --helpBest Score:         5234

python 2048RL.py play --helpModel Saved:        models/DQN/dqn_2048_final.pth

``````



---**Comparison Metrics**:

- Average score across evaluation episodes

## 🧪 Example Workflow- Consistency (standard deviation of scores)

- Success rate (% reaching 512+ tile)

### Complete Training & Evaluation- Training efficiency (time to convergence)



```bash## Workflow

# 1. Train DQN

python 2048RL.py train --algorithm dqn --episodes 2000### Complete Training Pipeline



# 2. Watch it play```

python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 51. Configuration

   ├─ Edit CONFIG in 2048RL.py

# 3. Train Double DQN for comparison   ├─ Set algorithm, episodes, hyperparameters

python 2048RL.py train --algorithm double-dqn --episodes 2000   └─ Choose UI/plot options



# 4. Compare results2. Initialization

cat evaluations/training_log.txt   ├─ Load game environment (src/environment.py)

   ├─ Initialize agent (src/agents/*/agent.py)

# 5. Watch best model   ├─ Create replay buffer (DQN/Double DQN)

python 2048RL.py play --model models/DoubleDQN/double_dqn_final.pth   ├─ Setup training timer (src/utils.py)

```   └─ Initialize evaluation logger



---3. Training Loop (per episode)

   ├─ Reset environment → Initial state

## 🛠️ Advanced Usage   ├─ WHILE not done:

   │  ├─ Agent selects action (epsilon-greedy or policy)

### Modify Hyperparameters   │  ├─ Environment executes action

   │  ├─ Receive reward, next_state, done

Edit `CONFIG` in `2048RL.py`:   │  ├─ Store transition in buffer (if applicable)

   │  ├─ Update agent (every N steps)

```python   │  └─ Render UI (if enabled)

"dqn": {   ├─ Log episode metrics

    "learning_rate": 5e-4,      # Faster learning   ├─ Update plots

    "epsilon_end": 0.15,        # More exploration   └─ Save checkpoint (every 100 episodes)

    "epsilon_decay": 150000,    # Slower decay

    "hidden_dims": (512, 512),  # Bigger network4. Optimization (per update)

}   ├─ Sample batch from replay buffer

```   ├─ Compute loss (TD error or policy gradient)

   ├─ Backpropagate gradients

### Use Different Network Sizes   ├─ Clip gradients (prevent instability)

   ├─ Update network parameters

```python   └─ Update target network (if applicable)

"hidden_dims": (128, 128),    # Small & fast

"hidden_dims": (256, 256),    # Default (recommended)5. Completion

"hidden_dims": (512, 512),    # Large (more capacity)   ├─ Stop training timer

```   ├─ Save final model

   ├─ Log results to evaluations/training_log.txt

### Checkpoint Management   └─ Display summary statistics

```

Models saved to `models/<ALGORITHM>/`:

- `dqn_ep100.pth`, `dqn_ep200.pth`, ... (every 100 episodes)### Data Flow

- `dqn_final.pth` (final model)

```

Load specific checkpoint:Game State (4x4 board)

```bash    ↓

python 2048RL.py play --model models/DQN/dqn_ep1500.pthEnvironment.to_normalized_state()

```    ↓

Flattened vector (16 values, normalized 0-1)

---    ↓

Neural Network (agent)

## 🤝 Contributing    ↓

Q-values [4] or Action Probs [4]

We welcome contributions! Areas of interest:    ↓

Action Selection (epsilon-greedy or sample)

- **Algorithms**: Implement A3C, PPO, Rainbow DQN    ↓

- **Features**: Prioritized experience replay, curriculum learningEnvironment.step(action)

- **Visualization**: TensorBoard integration, better plots    ↓

- **Testing**: Unit tests, integration testsNext State, Reward, Done, Info

- **Documentation**: More examples, tutorials    ↓

Replay Buffer (DQN/Double DQN)

---    ↓

Batch Sampling

## 📜 License    ↓

Loss Computation

MIT License - see `LICENSE` file for details.    ↓

Gradient Descent

---    ↓

Parameter Update

## 🙏 Acknowledgments```



- **DeepMind** for DQN paper ([Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602))### File Interactions

- **van Hasselt et al.** for Double DQN ([Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461))

- **2048 Game** by Gabriele Cirulli```

2048RL.py (entry point)

---    ↓

Imports: src.agents.dqn.agent, src.environment, src.utils

## 📞 Support    ↓

Creates: GameEnvironment (wraps src.game.board + src.game.ui)

- **Issues**: [GitHub Issues](https://github.com/CodeLanderV/2048-Reinforcement-Learning/issues)    ↓

- **Documentation**: See `FILE_DOCUMENTATION.md` for detailed file explanationsCreates: DQNAgent (loads src.agents.dqn.network)

- **Training Problems**: Check "Training Tips" section above    ↓

Training Loop:

---    - environment.reset() → board.reset() → ui.draw()

    - agent.select_action() → network.forward()

**⭐ Star this repo if you find it useful!**    - environment.step() → board.step() → reward calculation

    - agent.optimize_model() → loss computation → backprop

**📖 Read `FILE_DOCUMENTATION.md` for detailed codebase documentation**    ↓

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
