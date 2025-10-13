# 2048 Game Details

## What is 2048?

2048 is a single-player sliding tile puzzle video game written by Italian web developer Gabriele Cirulli and published on GitHub in 2014.

## Game Mechanics

### Objective
Combine numbered tiles to create a tile with the number 2048.

### How to Play
1. Use arrow keys to slide all tiles in one direction (up, down, left, right)
2. When two tiles with the same number touch, they merge into one
3. After each move, a new tile (2 or 4) appears in an empty spot
4. Keep combining tiles to reach 2048!

### Scoring
- Score increases by the value of merged tiles
- Example: Merging two 4s → +8 points
- Higher merges → Higher scores

### Game Over
The game ends when:
- No empty spaces remain AND
- No adjacent tiles can be merged

## Strategy Tips

1. **Keep your highest tile in a corner** - Prevents it from blocking merges
2. **Build in one direction** - Focus moves in 2-3 directions primarily
3. **Plan ahead** - Think about consequences of each move
4. **Maintain options** - Keep some empty spaces when possible

## Difficulty

- **Easy to learn** - Simple rules, intuitive gameplay
- **Hard to master** - Requires strategy and planning
- **Highly addictive** - "Just one more game!"

## Fun Facts

- Created in a single weekend
- Became viral with millions of players
- Spawned countless clones and variants
- Maximum theoretical tile: 131,072 (2^17)
- Perfect play can reach 2048 almost every time

## Why It's Good for RL

1. **Discrete actions** - Only 4 possible moves
2. **Clear objective** - Maximize score / reach 2048
3. **Observable state** - Complete information visible
4. **Deterministic** (mostly) - Same action → predictable result
5. **Challenging** - Requires strategy, not just random moves
6. **Quick episodes** - Fast feedback for learning


