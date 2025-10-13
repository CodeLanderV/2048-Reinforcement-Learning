# Environment Architecture

## Overview

The Environment acts as a bridge between the RL Agent and the Game Logic.

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Agent     │ ◄─────► │ Environment  │ ◄─────► │ game_logic  │
│  (Brain)    │         │   (Wrapper)  │         │  (Game)     │
└─────────────┘         └──────────────┘         └─────────────┘
    "What do              "Translate               "Here's what
     I do next?"          & standardize"           happened"
```

## Responsibilities

### Environment's Job:
1. **Wrap** the game in a standard interface
2. **Translate** game state for the agent
3. **Calculate** rewards
4. **Track** episode information

### Interface:
```python
env.reset()           # Start new game → returns state
env.step(action)      # Take action → returns (state, reward, done, info)
env.get_state()       # Get current state
env.render()          # (Optional) Display game
```

## Design Decisions

### State Representation
- **Format**: Raw board values (4x4 numpy array)
- **Values**: `[0, 2, 4, 8, 16, ..., 2048, ...]`
- **Shape**: `(4, 4)` or flattened `(16,)`

### Action Space
- **0** = Up
- **1** = Right
- **2** = Down
- **3** = Left

### Reward Function
- **Reward** = Score increase from the move
- **Invalid move** = 0 reward (no penalty, just no progress)
- **Game over** = No additional penalty

### Example Flow:
```python
env = Game2048Environment()
state = env.reset()  # [[0, 2, 0, 0], [0, 0, 2, 0], ...]

# Agent takes action
action = agent.choose_action(state)  # e.g., 0 (up)
next_state, reward, done, info = env.step(action)

# reward = score increase
# done = True if game over
# info = {'score': 128, 'highest_tile': 128}
```

## No OpenAI Gym Dependency

We're building a **custom, lightweight environment** without Gym to:
- Keep it simple
- Avoid unnecessary dependencies
- Have full control
- Easy to understand

Later, if needed, we can add Gym compatibility with a simple wrapper.
