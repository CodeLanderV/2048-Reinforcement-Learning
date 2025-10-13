# 2048 Game Implementation

This directory contains a complete, playable implementation of the 2048 game using Pygame.

## Files

- **`game_logic.py`** - Core game mechanics (board, moves, scoring)
  - Pure Python/NumPy implementation
  - No UI dependencies
  - Can be used independently for RL training

- **`game_ui.py`** - Pygame rendering and user interface
  - Visual representation of the game
  - Keyboard input handling
  - Game loop implementation

- **`constants.py`** - Game constants
  - Colors, sizes, fonts
  - Configuration values

- **`play.py`** - Main entry point to run the game

## How to Play

### Run the game:
```bash
cd game
python play.py
```

### Controls:
- **Arrow Keys** - Move tiles (Up, Down, Left, Right)
- **R** - Restart game
- **ESC** - Quit

## Game Rules

1. Use arrow keys to move tiles
2. When two tiles with the same number touch, they merge into one
3. After each move, a new tile (2 or 4) appears
4. Goal: Create a tile with the number 2048
5. Game over when no more moves are possible

## Architecture

### Separation of Concerns

```
game_logic.py (Model)
    ↓
game_ui.py (View + Controller)
    ↓
play.py (Entry point)
```

This separation allows the game logic to be reused for:
- RL agent training (using `game_logic.py` directly)
- Testing without UI
- Different rendering backends

## For RL Integration

The `Game2048` class in `game_logic.py` provides:

- `reset()` - Start new game
- `move(direction)` - Execute move (0=up, 1=right, 2=down, 3=left)
- `get_board()` - Get current board state
- `get_score()` - Get current score
- `is_game_over()` - Check if game ended
- `get_available_moves()` - Get valid moves

This makes it easy to wrap in an OpenAI Gym environment later.

## Dependencies

- Python 3.8+
- NumPy
- Pygame

Install:
```bash
pip install numpy pygame
```
