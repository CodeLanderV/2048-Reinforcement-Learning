# 📚 Complete File Documentation

**A comprehensive guide explaining every single file in the 2048 Reinforcement Learning project.**

This document provides detailed explanations of:
- What each file does
- How files interact with each other
- Key functions and classes
- When to use/modify each file

---

## 📖 Table of Contents

- [Core Files](#core-files)
- [Source Code (`src/`)](#source-code-src)
- [Agent Implementations (`src/agents/`)](#agent-implementations-srcagents)
- [Game Logic (`src/game/`)](#game-logic-srcgame)
- [Data Files](#data-files)
- [Documentation Files](#documentation-files)
- [File Interaction Diagram](#file-interaction-diagram)

---

## Core Files

### `2048RL.py` ⭐ MAIN ENTRY POINT

**Purpose**: Central control panel for all training and playing operations.

**What it does:**
- Provides unified interface for training all algorithms
- Handles command-line arguments (train/play commands)
- Contains `CONFIG` dictionary for all hyperparameters
- Implements training functions for DQN, Double DQN, and MCTS
- Includes play mode for watching trained models

**Key Components:**

1. **CONFIG Dictionary** (Lines 1-80)
   ```python
   CONFIG = {
       "algorithm": "dqn",
       "episodes": 2000,
       "dqn": { ... },
       "double_dqn": { ... },
       "mcts": { ... }
   }
   ```
   - **Purpose**: Centralized configuration for all algorithms
   - **When to edit**: Changing hyperparameters, algorithm selection, training duration
   - **Key settings**: `learning_rate`, `epsilon_decay`, `hidden_dims`

2. **`train_dqn_variant(algorithm)`** (Lines ~120-350)
   - **Purpose**: Unified training function for DQN and Double DQN
   - **How it works**:
     1. Loads appropriate agent class based on `algorithm` parameter
     2. Creates environment and initializes agent
     3. Runs training loop (episodes → steps → optimize)
     4. Saves checkpoints every 100 episodes
     5. Generates live plots and logs results
   - **Called by**: Command line `train` command

3. **`train_mcts()`** (Lines ~350-450)
   - **Purpose**: Run MCTS simulations (planning, no learning)
   - **Difference**: No neural network, no training, just tree search
   - **Use case**: Baseline comparison for learned agents

4. **`play_model()`** (Lines ~500-600)
   - **Purpose**: Load trained model and watch it play
   - **Auto-detects**: Algorithm type from model path
   - **Features**: Disables exploration (epsilon=0), shows game statistics

5. **`main()`** (Lines ~600-end)
   - **Purpose**: Command-line argument parser
   - **Commands**: `train`, `play`
   - **Flags**: `--algorithm`, `--episodes`, `--no-ui`, `--no-plots`

**When to use:**
- ✅ Training any algorithm: `python 2048RL.py train --algorithm dqn`
- ✅ Watching models play: `python 2048RL.py play`
- ✅ Changing hyperparameters: Edit `CONFIG` dictionary

**When to modify:**
- To tune hyperparameters (edit `CONFIG`)
- To add new algorithm (add new training function)
- To change training loop behavior

**Dependencies:**
- Imports: `src.environment`, `src.agents.*`, `src.utils`
- External: `matplotlib`, `numpy`, `torch`

---

### `play.py`

**Purpose**: Simplified script for playing trained models without command-line arguments.

**What it does:**
- Hardcoded to load `models/DQN/dqn_final.pth`
- Plays one game with visualization
- Shows detailed statistics (score, tile, steps, valid/invalid moves)

**Key Components:**

1. **`play_model(model_path, episodes, use_ui)`**
   - Loads DQN agent from file
   - Runs episode with greedy action selection (no exploration)
   - Prints step-by-step debug info every 10 steps

**When to use:**
- Quick testing of trained models
- Debugging agent behavior with detailed output
- Simple script to share with others

**When to modify:**
- Change model path to test different checkpoints
- Add/remove debug output
- Test with different number of episodes

**Differences from `2048RL.py play`:**
- Simpler (no argument parsing)
- More debug output
- Hardcoded paths

---

### `requirements.txt`

**Purpose**: Python package dependencies.

**Contents:**
```
torch>=2.0.0
numpy>=1.24.0
pygame>=2.5.0
matplotlib>=3.7.0
```

**When to use:**
- Initial setup: `pip install -r requirements.txt`
- After pulling updates from git
- Setting up new environment

**When to modify:**
- Adding new dependencies
- Upgrading package versions
- Fixing compatibility issues

---

## Source Code (`src/`)

### `src/__init__.py`

**Purpose**: Makes `src/` a Python package.

**Contents**: Empty file (just marks directory as importable)

**Why it exists:** Allows `from src.agents import ...` imports

---

### `src/environment.py` ⭐ RL INTERFACE

**Purpose**: Gym-style Reinforcement Learning environment wrapper for 2048 game.

**What it does:**
- Wraps game logic in standard RL interface (reset, step, render)
- Handles state representation (log2-normalized 16D vector)
- Calculates rewards (score gains + invalid move penalties)
- Manages pygame UI (if enabled)

**Key Classes:**

1. **`EnvironmentConfig`** (Dataclass)
   ```python
   @dataclass
   class EnvironmentConfig:
       seed: Optional[int] = None
       invalid_move_penalty: float = -10.0
       enable_ui: bool = False
   ```
   - **Purpose**: Configuration for environment behavior
   - **When to modify**: Change penalty value, toggle UI

2. **`StepResult`** (Dataclass)
   ```python
   @dataclass
   class StepResult:
       state: np.ndarray      # Next state (16D)
       reward: float          # Immediate reward
       done: bool             # Episode over?
       info: Dict             # Game statistics
   ```
   - **Purpose**: Return value from `step()` method
   - **Standard RL interface**: Matches OpenAI Gym

3. **`GameEnvironment`** (Main Class)

   **Key Methods:**

   - **`reset() → state`**
     - Starts new game
     - Returns initial state (16D vector)
     - Called at beginning of each episode

   - **`step(action) → StepResult`**
     - Executes action (0=up, 1=down, 2=left, 3=right)
     - Returns (next_state, reward, done, info)
     - Core RL interaction method
     - **Reward calculation**:
       ```python
       reward = score_gained
       if not moved:  # Invalid move
           reward += invalid_move_penalty  # e.g., -10
       ```

   - **`get_state() → Dict`**
     - Returns full game info (board, score, max_tile, empty_cells)
     - Used for logging and debugging

   - **`_build_state(board) → np.ndarray`**
     - **Critical method**: Converts raw board to neural network input
     - **Transformation**:
       ```
       Board: [[2, 4], [8, 0]]  →  Log2: [[1, 2], [3, 0]]  →  Flatten: [1, 2, 3, 0, ...]
       ```
     - **Why log2?** Keeps values in range 0-11 instead of 0-2048 (easier to learn)

**State Representation:**
```python
# Raw board (4x4)
[[  2,   4,   8,   0],
 [ 16,  32,   0,   0],
 [  0,   0,   0,   0],
 [  0,   0,   0,   0]]

# After log2 normalization (4x4)
[[1, 2, 3, 0],
 [4, 5, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]]

# Flattened for neural network (16D vector)
[1, 2, 3, 0, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**When to use:**
- Creating environment: `env = GameEnvironment(EnvironmentConfig(enable_ui=True))`
- In training loop: `state = env.reset()` → `result = env.step(action)`

**When to modify:**
- Change reward structure (e.g., bonus for merging high tiles)
- Modify state representation (e.g., add more features)
- Adjust invalid move penalty

**Dependencies:**
- Imports: `src.game.board`, `src.game.ui`
- External: `numpy`

---

### `src/utils.py` ⭐ TRAINING UTILITIES

**Purpose**: Helper classes for training (timer, logger).

**What it does:**
- Tracks training duration with human-readable format
- Logs training results to file for comparison
- Provides formatted output for evaluation

**Key Classes:**

1. **`TrainingTimer`**

   **Purpose**: Track and format training duration.

   **Methods:**
   - `start()` - Begin timing
   - `stop()` - End timing
   - `elapsed()` - Get seconds as float
   - `elapsed_str()` - Get formatted string ("2:15:30")

   **Usage:**
   ```python
   timer = TrainingTimer().start()
   # ... train model ...
   timer.stop()
   print(f"Took: {timer.elapsed_str()}")  # "2:15:30"
   ```

2. **`EvaluationLogger`**

   **Purpose**: Log training sessions to file for comparison.

   **Methods:**
   - `log_training(...)` - Append training session to `evaluations/training_log.txt`
   - `log_evaluation(...)` - Append evaluation results
   - `get_summary()` - Return full log file content

   **Log Format:**
   ```
   ════════════════════════════════════════════════════════════════
   Training Session: 2025-10-14 15:30:00
   ════════════════════════════════════════════════════════════════
   Algorithm:          DQN
   Episodes:           2000
   Training Time:      2:15:30
   Final Avg Reward:   145.32
   Best Max Tile:      512
   Best Score:         5234
   Model Saved:        models/DQN/dqn_final.pth
   Notes:              LR=0.0001, ε_end=0.1
   ════════════════════════════════════════════════════════════════
   ```

   **Usage:**
   ```python
   logger = EvaluationLogger()
   logger.log_training(
       algorithm="DQN",
       episodes=2000,
       final_avg_reward=145.3,
       max_tile=512,
       final_score=5234,
       training_time="2:15:30",
       model_path="models/DQN/dqn_final.pth",
       notes="LR=1e-4"
   )
   ```

**When to use:**
- Every training session (automatically called by `2048RL.py`)
- Comparing different hyperparameters
- Tracking training history

**When to modify:**
- Add more metrics to log
- Change log format
- Add CSV export functionality

---

## Agent Implementations (`src/agents/`)

### Directory Structure

```
src/agents/
├── dqn/
│   ├── __init__.py
│   ├── network.py         # DQN neural network
│   └── agent.py           # DQN agent + replay buffer
├── double_dqn/
│   ├── __init__.py
│   ├── network.py         # Double DQN network (similar to DQN)
│   └── agent.py           # Double DQN agent
├── mcts/
│   ├── __init__.py
│   └── agent.py           # MCTS tree search (no network)
└── policy_gradient/
    ├── __init__.py
    ├── network.py         # Policy network
    └── agent.py           # Policy gradient agent (stub)
```

---

### `src/agents/dqn/network.py`

**Purpose**: Define neural network architecture for DQN.

**What it does:**
- Implements Q-network that maps states to Q-values for each action
- Simple feedforward architecture: Input → Hidden → Hidden → Output

**Key Classes:**

1. **`DQNModelConfig`** (Dataclass)
   ```python
   @dataclass
   class DQNModelConfig:
       input_dim: int = 16            # Flattened board
       output_dim: int = 4            # 4 actions
       hidden_dims: tuple = (256, 256)  # Hidden layer sizes
   ```

2. **`DQNModel`** (nn.Module)

   **Architecture:**
   ```
   Input (16) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(4)
   ```

   **Forward Pass:**
   ```python
   def forward(self, state):
       # state: [batch_size, 16]
       x = F.relu(self.fc1(state))
       x = F.relu(self.fc2(x))
       q_values = self.fc3(x)
       # output: [batch_size, 4] Q-values for each action
       return q_values
   ```

**When to modify:**
- Change network size: `hidden_dims = (512, 512)`
- Add more layers: `hidden_dims = (256, 256, 256)`
- Try different activations (LeakyReLU, ELU)

---

### `src/agents/dqn/agent.py` ⭐ DQN IMPLEMENTATION

**Purpose**: Complete DQN agent with experience replay and training logic.

**What it does:**
- Implements Deep Q-Network algorithm
- Manages policy network and target network
- Handles experience replay buffer
- Implements epsilon-greedy exploration
- Provides training (optimize_model) and inference methods

**Key Classes:**

1. **`AgentConfig`** (Dataclass)
   ```python
   @dataclass
   class AgentConfig:
       gamma: float = 0.99                    # Discount factor
       batch_size: int = 128                  # Training batch size
       learning_rate: float = 1e-4            # Adam optimizer LR
       epsilon_start: float = 1.0             # Initial exploration
       epsilon_end: float = 0.1               # Final exploration
       epsilon_decay: int = 100000            # Decay steps
       target_update_interval: int = 1000     # Target net sync
       replay_buffer_size: int = 100000       # Buffer capacity
       gradient_clip: float = 5.0             # Gradient clipping
   ```

2. **`Transition`** (NamedTuple)
   ```python
   Transition = namedtuple('Transition', 
       ['state', 'action', 'reward', 'next_state', 'done'])
   ```
   - Stores single experience tuple
   - Used in replay buffer

3. **`ReplayBuffer`**

   **Purpose**: Store and sample past experiences for training.

   **Methods:**
   - `push(state, action, reward, next_state, done)` - Add experience
   - `sample(batch_size)` - Randomly sample batch
   - `__len__()` - Current buffer size

   **Why it exists:** Breaks correlation between consecutive samples (improves stability)

4. **`DQNAgent`** (Main Agent Class)

   **Key Methods:**

   - **`select_action(state) → int`**
     - **Epsilon-greedy action selection**
     ```python
     if random() < epsilon:
         return random_action()  # Explore
     else:
         return argmax(Q(state))  # Exploit
     ```
     - Epsilon decays over time: 1.0 → 0.1
     - Used during training

   - **`act_greedy(state) → int`**
     - Always picks best action (no exploration)
     - Used during evaluation/playing

   - **`store_transition(...)`**
     - Stores experience in replay buffer
     - Called after each environment step

   - **`can_optimize() → bool`**
     - Check if buffer has enough samples to start training
     - Returns True if `len(buffer) >= batch_size`

   - **`optimize_model()`** ⭐ CORE TRAINING LOGIC
     ```python
     def optimize_model(self):
         # 1. Sample batch from replay buffer
         batch = self.replay_buffer.sample(self.batch_size)
         
         # 2. Compute Q(s,a) for current states
         current_q = self.policy_net(states).gather(1, actions)
         
         # 3. Compute target: r + γ * max Q_target(s',a')
         with torch.no_grad():
             next_q = self.target_net(next_states).max(1)[0]
             target_q = rewards + gamma * next_q * (1 - dones)
         
         # 4. Compute loss (MSE between current and target)
         loss = F.mse_loss(current_q, target_q)
         
         # 5. Backpropagation
         self.optimizer.zero_grad()
         loss.backward()
         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
         self.optimizer.step()
         
         # 6. Update target network periodically
         if steps % target_update_interval == 0:
             self.target_net.load_state_dict(self.policy_net.state_dict())
     ```

   - **`save(path, episode)`**
     - Saves model checkpoint with metadata
     - Includes: network weights, optimizer state, epsilon, episode

   - **`load(path)`**
     - Loads trained model from file
     - Restores epsilon and training state

**DQN Algorithm Flow:**
```
1. Observe state s
2. Select action a (epsilon-greedy)
3. Execute a, observe reward r and next state s'
4. Store (s, a, r, s', done) in replay buffer
5. Sample random batch from buffer
6. Compute loss: (Q(s,a) - (r + γ * max Q_target(s',a')))²
7. Update policy network via gradient descent
8. Every N steps, copy policy_net → target_net
```

**When to use:**
- Created automatically by `2048RL.py`
- Can instantiate directly for custom training loops

**When to modify:**
- Change Q-learning algorithm (e.g., add prioritized replay)
- Modify epsilon decay schedule
- Add dueling network architecture

---

### `src/agents/double_dqn/`

**Purpose**: Double DQN implementation (reduces Q-value overestimation).

**Files:**
- `network.py` - Same as DQN network
- `agent.py` - Similar to DQN but different target calculation

**Key Difference from DQN:**

**DQN Target:**
```python
target = reward + gamma * max(Q_target(next_state))
```

**Double DQN Target:**
```python
best_action = argmax(Q_policy(next_state))    # Policy net picks action
target = reward + gamma * Q_target(next_state, best_action)  # Target net evaluates
```

**Why it's better:**
- DQN overestimates Q-values (max operator is biased)
- Double DQN decouples action selection from evaluation
- Results in more accurate value estimates

**When to use:**
- If DQN agent gets stuck or has unstable training
- Generally performs better than vanilla DQN
- Same computational cost as DQN

---

### `src/agents/mcts/agent.py`

**Purpose**: Monte Carlo Tree Search implementation (planning algorithm, no learning).

**What it does:**
- Builds search tree via simulations
- Uses UCB1 for exploration-exploitation
- Picks best action based on visit counts

**Key Classes:**

1. **`MCTSConfig`** (Dataclass)
   ```python
   @dataclass
   class MCTSConfig:
       simulations: int = 100          # Simulations per move
       exploration_constant: float = 1.41  # UCB exploration (√2)
   ```

2. **`MCTSNode`**
   - Represents node in search tree
   - Stores: board state, parent, children, visits, value

   **Methods:**
   - `best_child(c)` - Select child with highest UCB score
   - `expand(action)` - Add child node

3. **`MCTSAgent`**

   **Key Method:**
   - **`select_action(state, board) → int`**
     ```python
     def select_action(self, state, board):
         root = MCTSNode(board)
         
         for _ in range(simulations):
             # 1. Selection: Traverse tree using UCB
             node = select_best_leaf(root)
             
             # 2. Expansion: Add child node
             if not fully_expanded(node):
                 node = expand_child(node)
             
             # 3. Simulation: Play random game to end
             reward = simulate_random_playthrough(node.board)
             
             # 4. Backpropagation: Update node values
             backpropagate(node, reward)
         
         # Pick most visited child
         return most_visited_action(root)
     ```

   **UCB1 Formula:**
   ```python
   UCB(node) = Q(node) + c * sqrt(ln(parent_visits) / node_visits)
   ```
   - First term: Exploitation (pick high-value nodes)
   - Second term: Exploration (try less-visited nodes)

**When to use:**
- Baseline comparison (no training required)
- Understanding optimal play patterns
- Debugging learned agents

**Characteristics:**
- **No learning**: Performance stays constant
- **Deterministic**: Same state → same action
- **Slower**: Runs simulations for each move
- **Memory intensive**: Builds tree structure

---

### `src/agents/policy_gradient/`

**Purpose**: Policy Gradient (REINFORCE) implementation.

**Status**: Stub (not fully implemented)

**Planned Architecture:**
- Policy network outputs action probabilities
- Trains after full episode using Monte Carlo returns
- No replay buffer (on-policy algorithm)

**When to implement:**
- Learning stochastic policies
- Alternative to value-based methods
- Continuous action spaces (future)

---

## Game Logic (`src/game/`)

### `src/game/board.py`

**Purpose**: Core 2048 game mechanics.

**What it does:**
- Implements game rules (merging tiles, valid moves)
- Tracks game state (board, score, game over)
- Provides game loop interface

**Key Classes:**

1. **`BoardConfig`** (Dataclass)
   ```python
   @dataclass
   class BoardConfig:
       size: int = 4        # Board size (4x4)
       seed: Optional[int] = None  # Random seed
   ```

2. **`StepResult`** (NamedTuple)
   ```python
   StepResult = namedtuple('StepResult',
       ['board', 'moved', 'score_gain'])
   ```

3. **`GameBoard`** (Main Class)

   **Key Methods:**

   - **`reset() → board`**
     - Clear board and add two random tiles
     - Returns initial 4x4 board

   - **`step(direction) → StepResult`**
     - **Core game logic**: Execute one move
     - **Directions**: "up", "down", "left", "right"
     - **Returns**: New board, whether moved, score gained
     
     **Algorithm:**
     ```python
     def step(self, direction):
         # 1. Rotate board based on direction
         rotated = rotate_for_direction(self.grid, direction)
         
         # 2. Slide tiles left (merge logic)
         new_board, moved, score = slide_and_merge(rotated)
         
         # 3. Rotate back
         self.grid = rotate_back(new_board, direction)
         
         # 4. Add random tile if board changed
         if moved:
             add_random_tile(self.grid)
             self.score += score
         
         return StepResult(self.grid, moved, score)
     ```

   - **`is_game_over() → bool`**
     - Check if any valid moves remain
     - Tests all 4 directions

   - **`get_valid_actions() → List[str]`**
     - Returns list of valid moves
     - Used by agents to avoid invalid moves

   - **`clone() → GameBoard`**
     - Deep copy of board (used by MCTS)

   - **`to_normalized_state() → np.ndarray`**
     - Convert to log2-normalized state for neural networks

**Merge Logic Example:**
```python
# Before move right: [2, 2, 4, 0]
# After sliding:     [0, 0, 4, 4]  → merge → [0, 0, 0, 8]
# Score gain: 4 + 8 = 12
```

**When to modify:**
- Change board size (e.g., 5x5 grid)
- Modify merge rules
- Add special tiles or power-ups

---

### `src/game/ui.py`

**Purpose**: Pygame-based visualization of 2048 game.

**What it does:**
- Renders game board in real-time
- Displays score and max tile
- Handles user input (arrow keys, R for restart, Q for quit)

**Key Classes:**

1. **`UIConfig`** (Dataclass)
   ```python
   @dataclass
   class UIConfig:
       window_size: int = 600
       grid_size: int = 4
       fps: int = 60
       colors: Dict = ...  # Tile colors
   ```

2. **`GameUI`**

   **Key Methods:**

   - **`draw()`**
     - Renders current game state
     - Shows tiles with colors based on value
     - Displays score and max tile

   - **`handle_events() → Optional[str]`**
     - Processes keyboard input
     - Returns: "up"/"down"/"left"/"right", "restart", "quit", or None
     - **Keys**:
       - Arrow keys: Move
       - R: Restart game
       - Q or close window: Quit

   - **`close()`**
     - Clean up pygame resources

**Tile Colors:**
```python
COLORS = {
    0: (205, 193, 180),     # Empty
    2: (238, 228, 218),     # 2
    4: (237, 224, 200),     # 4
    8: (242, 177, 121),     # 8
    16: (245, 149, 99),     # 16
    # ... up to 2048
}
```

**When to modify:**
- Change visual appearance (colors, fonts, layout)
- Add animations (tile sliding, merging)
- Add sound effects
- Support touch input

---

## Data Files

### `models/`

**Purpose**: Directory for saving trained model checkpoints.

**Structure:**
```
models/
├── DQN/
│   ├── dqn_ep100.pth
│   ├── dqn_ep200.pth
│   ├── ...
│   └── dqn_final.pth
└── DoubleDQN/
    └── double_dqn_final.pth
```

**Checkpoint Contents:**
```python
{
    'episode': 2000,
    'model_state_dict': ...,      # Network weights
    'optimizer_state_dict': ...,  # Optimizer state
    'epsilon': 0.1,               # Current exploration rate
    'config': { ... }             # Hyperparameters
}
```

**File Naming:**
- `<algo>_ep<episode>.pth` - Periodic checkpoint
- `<algo>_final.pth` - Final model after training complete

---

### `evaluations/`

**Purpose**: Directory for training logs.

**Files:**
- `training_log.txt` - Consolidated log of all training sessions

**Log Format:** See `src/utils.py` → `EvaluationLogger` section above

**Usage:**
```bash
# View training history
cat evaluations/training_log.txt

# Find best performing model
grep "Best Score" evaluations/training_log.txt
```

---

## Documentation Files

### `README.md`

**Purpose**: Main project documentation (quick start, usage, API).

**Sections:**
- Quick Start
- Installation
- Training commands
- Configuration guide
- Training tips
- Algorithm explanations

**When to update:**
- Adding new features
- Changing API
- Adding examples

---

### `FILE_DOCUMENTATION.md` (This File)

**Purpose**: Detailed explanation of every file in the project.

**Audience:**
- New contributors
- Students learning the codebase
- Anyone wanting deep understanding

---

### `REFACTORING_SUMMARY.md`

**Purpose**: Documents code quality improvements made.

**Contents:**
- Files removed
- Code duplication eliminated
- Documentation added
- Before/after comparisons

---

### `LICENSE`

**Purpose**: MIT License for the project.

---

## File Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         2048RL.py (MAIN)                        │
│  - Parse command line arguments                                 │
│  - Read CONFIG dictionary                                       │
│  - Call train/play functions                                    │
└────────┬─────────────────────────────────────────┬──────────────┘
         │                                         │
         ├─ train_dqn_variant(algorithm)          ├─ play_model(path)
         │                                         │
         │                                         │
    ┌────▼────────┐                          ┌────▼────────┐
    │ Environment │                          │    Agent    │
    │             │                          │             │
    │ - reset()   │◄────────────────────────►│ - load()    │
    │ - step()    │  state, reward, done     │ - greedy()  │
    └─────┬───────┘                          └─────────────┘
          │
          ├────────────────┐
          │                │
    ┌─────▼─────┐    ┌─────▼─────┐
    │ GameBoard │    │  GameUI   │
    │           │    │           │
    │ - step()  │    │ - draw()  │
    │ - merge() │    │ - events()│
    └───────────┘    └───────────┘

┌───────────────────────────────────────────────────────────────────┐
│                    Training Loop Flow                             │
│                                                                   │
│  1. CONFIG → Initialize Agent + Environment                      │
│  2. FOR each episode:                                             │
│     ├─ state = env.reset()                                       │
│     ├─ WHILE not done:                                           │
│     │  ├─ action = agent.select_action(state)                   │
│     │  ├─ result = env.step(action)                             │
│     │  ├─ agent.store_transition(...)                           │
│     │  ├─ agent.optimize_model()  # Train network               │
│     │  └─ state = result.state                                  │
│     ├─ Log metrics (score, tile, reward)                        │
│     ├─ Update plots                                              │
│     └─ Save checkpoint (every 100 episodes)                      │
│  3. Save final model                                             │
│  4. EvaluationLogger.log_training(...)                          │
└───────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference: Which File to Modify?

| Task | File to Modify |
|------|----------------|
| Change hyperparameters | `2048RL.py` → `CONFIG` |
| Implement new algorithm | Add to `src/agents/<new>/` |
| Change reward structure | `src/environment.py` → `step()` |
| Modify game rules | `src/game/board.py` → `step()` |
| Change visualization | `src/game/ui.py` |
| Add training metrics | `src/utils.py` → `EvaluationLogger` |
| Change network architecture | `src/agents/*/network.py` |
| Modify training loop | `2048RL.py` → `train_dqn_variant()` |

---

## File Size & Complexity Overview

| File | Lines | Complexity | Priority to Understand |
|------|-------|------------|------------------------|
| `2048RL.py` | 600 | Medium | ⭐⭐⭐⭐⭐ Essential |
| `src/environment.py` | 200 | Low-Med | ⭐⭐⭐⭐ Very Important |
| `src/agents/dqn/agent.py` | 300 | High | ⭐⭐⭐⭐ Very Important |
| `src/agents/dqn/network.py` | 50 | Low | ⭐⭐⭐ Important |
| `src/game/board.py` | 250 | Medium | ⭐⭐⭐ Important |
| `src/game/ui.py` | 150 | Low | ⭐⭐ Helpful |
| `src/utils.py` | 150 | Low | ⭐⭐ Helpful |
| `play.py` | 100 | Low | ⭐ Optional |

---

## Summary

**Start with these files to understand the project:**

1. **`README.md`** - Overview and quick start
2. **`2048RL.py`** - See how everything connects
3. **`src/environment.py`** - Understand RL interface
4. **`src/agents/dqn/agent.py`** - Deep dive into DQN

**Then explore:**
- `src/game/board.py` - Game mechanics
- `src/agents/double_dqn/` - Algorithm comparison
- `src/utils.py` - Helpful utilities

**For visual learners:**
- Run training with UI enabled
- Watch `src/game/ui.py` render the game
- See live training plots from `matplotlib`

---

**📖 Happy learning! This documentation explains every file in detail.**

**💡 Still have questions? Check inline comments in each file - every function is documented!**
