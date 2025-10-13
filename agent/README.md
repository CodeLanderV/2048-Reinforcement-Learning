# MCTS Agent for 2048

## What is Monte Carlo Tree Search (MCTS)?

MCTS is a search algorithm that makes decisions by:
1. **Simulating** many random games from the current position
2. **Evaluating** which action leads to the best outcomes
3. **Choosing** the action with highest average score

**No training needed!** It works immediately by looking ahead.

## How It Works

### For each move:
```
1. Look at available actions (up, down, left, right)
2. For each action:
   a. Simulate 50 random games starting with that action
   b. Track average score achieved
3. Pick the action with best average score
```

### Example:
```
Current board state
Action UP:   Simulate 50 games → Average score: 1200
Action RIGHT: Simulate 50 games → Average score: 1500  ← Best!
Action DOWN: Simulate 50 games → Average score: 1100
Action LEFT: Simulate 50 games → Average score: 1300

Choose: RIGHT
```

## Files

- **`mcts_agent.py`** - MCTS implementation
  - `MCTSAgent` class
  - `choose_action()` - Pick best move
  - `play_game()` - Play full game

## Usage

### Basic Usage:
```python
from mcts_agent import MCTSAgent

# Create agent
agent = MCTSAgent(num_simulations=50, max_depth=10)

# Play a game
final_score, highest_tile = agent.play_game()
```

### With Custom Environment:
```python
import sys
sys.path.append('../game')
from Environment import Game2048Environment
from mcts_agent import MCTSAgent

# Create environment and agent
env = Game2048Environment()
agent = MCTSAgent(num_simulations=50)

# Play episode
state = env.reset()
done = False

while not done:
    # Agent chooses action
    action = agent.choose_action(env.game)
    
    # Take action
    state, reward, done, info = env.step(action)
    print(f"Score: {info['score']}")
```

## Parameters

### `num_simulations` (default: 50)
- Number of random games to simulate per action
- **Higher** = Better decisions, but slower
- **Lower** = Faster, but less accurate

Recommended values:
- Fast: 20-30 simulations
- Balanced: 50 simulations
- Strong: 100+ simulations

### `max_depth` (default: 10)
- Maximum moves per simulation
- **Higher** = Looks further ahead
- **Lower** = Faster simulations

Recommended: 8-15 moves

## Performance

Expected performance (50 simulations):
- **Average Score**: 3,000 - 8,000
- **Highest Tile**: Usually 256-512
- **Occasionally**: Reaches 1024
- **Speed**: ~1-2 seconds per move

## Testing

### Run the agent:
```bash
cd agent
python mcts_agent.py
```

### Expected output:
```
Testing MCTS Agent for 2048
==================================================
Playing one game...

Starting MCTS game...
Simulations per move: 30
Max depth per simulation: 8
==================================================
Move 10: Score=120, Max Tile=32
Move 20: Score=380, Max Tile=64
...
==================================================
Game Over!
Final Score: 2456
Highest Tile: 256
Total Moves: 145
```

## Strengths

✅ **No training required** - Works immediately  
✅ **Reasonable performance** - Decent scores  
✅ **Easy to understand** - Simple algorithm  
✅ **Good baseline** - Compare other agents against this  
✅ **Explainable** - Can see why it chose each action  

## Weaknesses

❌ **Slow** - Takes 1-2 seconds per move  
❌ **Limited lookahead** - Only sees a few moves ahead  
❌ **No learning** - Doesn't improve with experience  
❌ **Random simulations** - Doesn't use optimal play in rollouts  

## Improvements (Future)

1. **Better rollout policy** - Use smarter moves in simulations
2. **UCB selection** - Full MCTS tree with UCB1
3. **Parallel simulations** - Run simulations in parallel
4. **Learned value function** - Combine with neural network

## Comparison with Other Methods

| Method | Training? | Speed | Score |
|--------|-----------|-------|-------|
| Random | No | Fast | ~300 |
| **MCTS** | **No** | **Slow** | **~5000** |
| DQN | Yes | Fast | ~10000 |
| PPO | Yes | Fast | ~15000 |

MCTS is a great **baseline** to start with!
