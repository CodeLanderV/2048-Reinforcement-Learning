# Quick Reference Guide - 2048 RL Training

## Training Commands

### Basic Training (Recommended)
```bash
# Automatically tunes hyperparameters, then trains
python 2048RL.py train --algorithm dqn --episodes 3000
```

### Fast Training (No UI/Plots)
```bash
python 2048RL.py train --algorithm dqn --episodes 3000 --no-ui --no-plots
```

### Custom Hyperparameter Tuning
```bash
# More trials = better optimization (but slower)
python 2048RL.py train --algorithm dqn --tune-trials 50 --tune-episodes 150

# Fewer trials = faster (but may not find optimal settings)
python 2048RL.py train --algorithm dqn --tune-trials 20 --tune-episodes 100
```

### Train All Algorithms
```bash
python 2048RL.py train --algorithm dqn --episodes 3000
python 2048RL.py train --algorithm double-dqn --episodes 3000
python 2048RL.py train --algorithm reinforce --episodes 3000
python 2048RL.py train --algorithm mcts --episodes 50  # No tuning for MCTS
```

---

## What Happens During Training

### Phase 1: Hyperparameter Tuning (Automatic)
- Runs 30 quick trials (200 episodes each)
- Tests different learning rates, batch sizes, network sizes
- Finds best configuration
- **Time:** 30-60 minutes
- **Output:** `evaluations/optuna_{algorithm}_{timestamp}.json`

### Phase 2: Full Training
- Uses best hyperparameters from tuning
- Trains for full 3000 episodes
- Saves checkpoints every 100 episodes
- **Time:** 3-6 hours (depending on hardware)
- **Output:** 
  - Model: `models/{Algorithm}/{algorithm}_final.pth`
  - Plot: `evaluations/{Algorithm}_training_plot.png`
  - Logs: `evaluations/logs.txt`

---

## Command Line Options

### Required
- `train` or `play` - Command to execute
- `--algorithm {dqn,double-dqn,mcts,reinforce}` - Which algorithm

### Optional (Training)
- `--episodes N` - Number of training episodes (default: 3000)
- `--no-ui` - Disable pygame window (faster)
- `--no-plots` - Disable live matplotlib plots (faster)
- `--tune-trials N` - Number of hyperparameter trials (default: 30)
- `--tune-episodes N` - Episodes per tuning trial (default: 200)

### Optional (Playing)
- `--model PATH` - Path to trained model
- `--episodes N` - Number of games to play (default: 1)
- `--no-ui` - Disable pygame window

---

## Files Generated

### Training Outputs
```
evaluations/
├── logs.txt                          # All console output with timestamps
├── training_log.txt                  # Training metrics summary
├── DQN_training_plot.png             # Reward/score curves
├── Double_DQN_training_plot.png
├── REINFORCE_training_plot.png
├── MCTS_performance_plot.png
└── optuna_dqn_20251015_143215.json   # Best hyperparameters found

models/
├── DQN/
│   ├── dqn_ep100.pth                 # Checkpoint at episode 100
│   ├── dqn_ep200.pth                 # Checkpoint at episode 200
│   ├── ...
│   └── dqn_final.pth                 # Final trained model
├── DoubleDQN/
│   └── double_dqn_final.pth
└── REINFORCE/
    └── reinforce_final.pth
```

---

## Checking Results

### View Training Progress
```bash
# Watch logs in real-time
Get-Content evaluations\logs.txt -Wait

# View last 100 lines
Get-Content evaluations\logs.txt -Tail 100

# View training summary
Get-Content evaluations\training_log.txt
```

### View Best Hyperparameters
```bash
# View tuning results
Get-Content evaluations\optuna_dqn_*.json | ConvertFrom-Json

# Or use Python
python -c "import json; print(json.dumps(json.load(open('evaluations/optuna_dqn_20251015_143215.json')), indent=2))"
```

### View Training Plots
```bash
# Open plot images
Start-Process evaluations\DQN_training_plot.png
```

---

## Playing Trained Models

### Play with Best Model
```bash
python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 5
```

### Play Without UI (Console Only)
```bash
python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 10 --no-ui
```

### Compare Models
```bash
python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 10
python 2048RL.py play --model models/DoubleDQN/double_dqn_final.pth --episodes 10
python 2048RL.py play --model models/REINFORCE/reinforce_final.pth --episodes 10
```

---

## Typical Training Times

### DQN / Double-DQN
- **Tuning:** 45-60 min (30 trials × 200 episodes)
- **Training:** 4-6 hours (3000 episodes)
- **Total:** ~5-7 hours

### REINFORCE
- **Tuning:** 30-45 min (30 trials × 200 episodes)
- **Training:** 3-5 hours (3000 episodes)
- **Total:** ~4-6 hours

### MCTS
- **No tuning** (deterministic algorithm)
- **Simulation:** 1-2 hours (50 episodes, much slower per episode)

*Times vary based on:*
- CPU/GPU speed
- UI enabled/disabled
- Network architecture size
- Batch size

---

## Troubleshooting

### Training is slow
**Solutions:**
- Add `--no-ui --no-plots` flags
- Reduce `--tune-trials` (e.g., 20 instead of 30)
- Reduce `--tune-episodes` (e.g., 150 instead of 200)
- Use GPU if available (automatically detected)

### Out of memory
**Solutions:**
- Close other programs
- Reduce batch size in CONFIG (edit 2048RL.py)
- Use smaller network architecture

### Want to resume training
**Current limitation:** Not supported yet
**Workaround:** Load checkpoint and manually continue training in code

### Tuning results not good
**Solutions:**
- Increase `--tune-trials` (e.g., 50 or 100)
- Run multiple tuning sessions and pick best
- Manually adjust hyperparameters in CONFIG

---

## Tips for Best Results

1. **First Time:** Run with defaults
   ```bash
   python 2048RL.py train --algorithm dqn --episodes 3000
   ```

2. **Fast Iteration:** Use fewer trials during development
   ```bash
   python 2048RL.py train --algorithm dqn --tune-trials 10 --episodes 1000
   ```

3. **Final Run:** Use more trials for publication-quality results
   ```bash
   python 2048RL.py train --algorithm dqn --tune-trials 100 --episodes 5000
   ```

4. **Compare Algorithms:** Train all with same settings
   ```bash
   for algo in dqn double-dqn reinforce; do
       python 2048RL.py train --algorithm $algo --episodes 3000
   done
   ```

5. **Save Everything:** All outputs auto-saved to `evaluations/` and `models/`

---

## Algorithm Characteristics

### DQN (Deep Q-Network)
- **Best for:** Stable, reliable performance
- **Speed:** Medium
- **Sample efficiency:** High (uses replay buffer)
- **Typical max tile:** 2048-4096

### Double-DQN
- **Best for:** Avoiding overestimation
- **Speed:** Medium
- **Sample efficiency:** High
- **Typical max tile:** 2048-4096 (slightly better than DQN)

### REINFORCE
- **Best for:** Learning stochastic policies
- **Speed:** Fast per episode
- **Sample efficiency:** Low (on-policy)
- **Typical max tile:** 1024-2048

### MCTS
- **Best for:** No learning, pure planning
- **Speed:** Very slow (tree search per move)
- **Sample efficiency:** N/A (no learning)
- **Typical max tile:** 512-1024

---

## Quick Start Checklist

- [ ] Install Optuna: `pip install optuna`
- [ ] Choose algorithm: dqn, double-dqn, or reinforce
- [ ] Run training: `python 2048RL.py train --algorithm dqn --episodes 3000`
- [ ] Wait for tuning (~1 hour) + training (~5 hours)
- [ ] Check results: `evaluations/*.png` and `evaluations/logs.txt`
- [ ] Play model: `python 2048RL.py play --model models/DQN/dqn_final.pth`
- [ ] Compare with other algorithms

**Estimated total time:** 6-7 hours for one complete training run

---

**Need help?** Check `CHANGES_SUMMARY.md` for detailed documentation.
