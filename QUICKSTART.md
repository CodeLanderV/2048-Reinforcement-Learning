# 2048 DQN Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training

### Quick Training (100 episodes, ~10-15 minutes)
```bash
python train.py --episodes 100
```

### Standard Training (1000 episodes, ~1-2 hours)
```bash
python train.py --episodes 1000 --save-freq 100
```

### Training with Visualization
```bash
python train.py --episodes 500 --visualize --viz-freq 10 --fps 10
```

### Advanced Training with Custom Hyperparameters
```bash
python train.py \
    --episodes 2000 \
    --lr 0.0001 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.995 \
    --batch-size 64 \
    --buffer-capacity 100000 \
    --target-update 10 \
    --hidden-sizes 256 256 128 \
    --save-freq 100 \
    --visualize \
    --model-name my_dqn_model
```

## Playing

### Watch Trained Agent
```bash
python play.py --mode agent --model saved_models/dqn_2048_final.pth --episodes 10 --fps 5
```

### Play Manually
```bash
python play.py --mode manual
```

Controls:
- Arrow Keys or WASD: Move tiles
- ESC or Q: Quit

### Watch Random Agent (Baseline)
```bash
python play.py --mode random --episodes 5 --fps 10
```

## File Organization

### Saved Models
- Located in: `saved_models/`
- Naming: `dqn_2048_ep{episode}.pth` or `dqn_2048_final.pth`

### Training Logs
- Located in: `logs/`
- Files: `training_log_ep{N}.json`, `statistics_ep{N}.json`

### Plots
- Located in: `plots/`
- Files: `scores_ep{N}.png`, `max_tiles_ep{N}.png`, `combined_ep{N}.png`, `loss_ep{N}.png`

### Game States
- Located in: `game_states/`
- Auto-saved during manual play

## Common Use Cases

### 1. Quick Test Run
```bash
python train.py --episodes 50 --visualize
```

### 2. Overnight Training
```bash
python train.py --episodes 5000 --save-freq 250 --model-name overnight_run
```

### 3. Hyperparameter Tuning
Try different configurations:

**Configuration A: More Exploration**
```bash
python train.py --episodes 1000 --epsilon-decay 0.999 --model-name config_a
```

**Configuration B: Larger Network**
```bash
python train.py --episodes 1000 --hidden-sizes 512 256 128 --model-name config_b
```

**Configuration C: Larger Batch**
```bash
python train.py --episodes 1000 --batch-size 128 --buffer-capacity 200000 --model-name config_c
```

### 4. Resume Training
To continue from a checkpoint, modify `train.py` to load the model before training:
```python
agent.load("saved_models/dqn_2048_ep500.pth")
```

## Monitoring Training

### Real-time Monitoring
Use `--visualize` flag to watch the agent play during training.

### Check Progress
Look at the console output:
```
Episode 100/1000 | Score: 1234 | Max Tile: 256 | Moves: 234 | Reward: 145.23 | Eps: 0.605 | Avg Score (100): 987.45
```

Key metrics:
- **Score**: Game score (higher is better)
- **Max Tile**: Highest tile achieved (target: 2048)
- **Moves**: Number of moves in episode (more moves = longer game)
- **Eps**: Current exploration rate (decreases over time)
- **Avg Score**: Average score over last 100 episodes (should increase)

### View Plots
Check the `plots/` directory periodically:
- `scores_ep{N}.png`: Score progression
- `max_tiles_ep{N}.png`: Highest tiles achieved
- `combined_ep{N}.png`: All metrics together
- `training_summary_final.png`: Complete training overview

## Troubleshooting

### Out of Memory
Reduce batch size or buffer capacity:
```bash
python train.py --batch-size 32 --buffer-capacity 50000
```

### Training Too Slow
- Disable visualization: Remove `--visualize` flag
- Reduce save frequency: `--save-freq 200`
- Use GPU if available (automatic with CUDA-enabled PyTorch)

### Poor Performance
- Train longer: Increase episodes
- Adjust learning rate: Try `--lr 0.0005` or `--lr 0.00005`
- Slower exploration decay: `--epsilon-decay 0.999`

### Pygame Issues
If running on a server without display:
- Don't use `--visualize` flag
- Use SSH with X11 forwarding if needed

## Expected Results

### After 100 Episodes
- Average Score: 500-1500
- Max Tile: 64-256
- Model starting to learn basic strategies

### After 500 Episodes
- Average Score: 2000-5000
- Max Tile: 256-512
- Model using corner strategy

### After 1000 Episodes
- Average Score: 3000-8000
- Max Tile: 512-1024
- Model playing consistently well

### After 2000+ Episodes
- Average Score: 5000-15000
- Max Tile: 1024-2048
- Model achieving 2048 tile occasionally (5-20% of games)

## Tips for Best Results

1. **Train for many episodes**: 1000+ episodes recommended
2. **Use visualization sparingly**: Only for monitoring, slows training
3. **Save frequently**: Use `--save-freq 100` to avoid losing progress
4. **Monitor plots**: Check plots directory to see if learning is happening
5. **Try different hyperparameters**: Experiment to find what works best
6. **Be patient**: Good 2048 agents take time to train

## Next Steps

1. Train a model with your preferred configuration
2. Watch the trained agent play
3. Compare with manual play
4. Experiment with hyperparameters
5. Try to achieve the 2048 tile!

For detailed documentation, see the main [README.md](README.md).
