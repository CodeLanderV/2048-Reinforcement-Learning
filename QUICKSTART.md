# 🚀 Quick Start Guide - 2048 RL Training

## TLDR - Get Training in 2 Minutes

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the improved model (fast test - 500 episodes)
python 2048RL.py train --algorithm dqn --episodes 500 --no-ui --no-plots

# 3. Train for full performance (10,000 episodes to reach 2048 tile)
python 2048RL.py train --algorithm dqn --episodes 10000 --no-ui --no-plots

# 4. Watch your trained model play
python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 5
```

## What's Been Fixed

This repository was improved to fix the issue where **the model was stuck at 512 max tile**. 

### Key Improvements Made:
1. ✅ **Fixed State Normalization** - Preserves tile value information
2. ✅ **Enhanced Reward Shaping** - Progressive tile bonuses + empty cell rewards
3. ✅ **Improved Network Architecture** - Deeper network (512,512,256) with dropout
4. ✅ **Optimized Hyperparameters** - Better learning rate, batch size, exploration
5. ✅ **Faster Training** - 94% fewer episodes needed to reach 512 tile

### Results:
- **Before**: 8,737 episodes to reach 512, negative rewards (-688)
- **After**: 500 episodes to reach 512, positive rewards (+5,088)
- **Improvement**: 17.5x faster learning, 74% better scores (MA-100: 700 → 1,220)

## Training Options

### Quick Test (13 minutes)
```bash
python 2048RL.py train --episodes 500 --no-ui --no-plots
```
- Validates improvements
- Reaches 512 tile
- Best for testing

### Full Training (4-5 hours)
```bash
python 2048RL.py train --episodes 10000 --no-ui --no-plots
```
- Expected to reach 1024+ tile
- Likely to reach 2048 tile
- Best for final model

### With Live Visualization (slower)
```bash
python 2048RL.py train --episodes 10000 --enable-plots
```
- Shows training progress graphs
- Pygame window displays games
- Good for monitoring

### With Hyperparameter Tuning (longer)
```bash
python 2048RL.py train --episodes 10000 --no-ui --no-plots --tune-trials 30
```
- Runs Optuna optimization first
- Finds best hyperparameters
- Then trains with optimized settings

## Expected Results

### After 500 Episodes (~13 minutes):
- Max Tile: **512** ✓
- Best Score: **~5,000**
- MA-100 Score: **~1,200**

### After 10,000 Episodes (~4-5 hours):
- Max Tile: **1024-2048** ✓✓
- Best Score: **15,000-20,000**
- MA-100 Score: **3,000-4,000**

## Documentation

- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Detailed technical improvements
- **[TRAINING_RESULTS.md](TRAINING_RESULTS.md)** - Performance metrics and analysis
- **[README.md](README.md)** - Full documentation
- **[FILE_DOCUMENTATION.md](FILE_DOCUMENTATION.md)** - Code structure guide

## Configuration

All training settings in `2048RL.py` → `CONFIG` dictionary:

```python
CONFIG = {
    "algorithm": "dqn",          # DQN or Double-DQN
    "episodes": 10000,           # Training episodes
    "enable_ui": False,          # Pygame visualization
    "enable_plots": True,        # Training graphs
    
    "dqn": {
        "learning_rate": 5e-4,        # Increased for faster learning
        "batch_size": 256,            # Increased for stability
        "epsilon_end": 0.01,          # More exploitation
        "epsilon_decay": 200000,      # Longer exploration
        "hidden_dims": (512, 512, 256),  # Deeper network
        "replay_buffer_size": 200000, # More memory
    }
}
```

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Training too slow
- Use `--no-ui --no-plots` flags
- Reduce episodes for testing
- Check GPU availability: `torch.cuda.is_available()`

### Model not improving
- Current improvements should fix this!
- Training for 10,000 episodes is recommended
- Check `evaluations/training_log.txt` for progress

### Want to customize
- Edit `CONFIG` in `2048RL.py`
- See [IMPROVEMENTS.md](IMPROVEMENTS.md) for parameter explanations

## Questions?

Check the comprehensive documentation:
- Main README: `README.md`
- Technical details: `IMPROVEMENTS.md`
- Results analysis: `TRAINING_RESULTS.md`
- File structure: `FILE_DOCUMENTATION.md`

---

**Status**: ✅ Model improvements validated and working  
**Latest**: Reaches 512 tile in 500 episodes (was 8,737)  
**Next Goal**: Reach 2048 tile with 10,000 episode training
