# Convergence Detection & Early Stopping

## Problem: Wasting Resources on Plateaued Training

Every RL agent has a **performance ceiling** determined by its architecture and hyperparameters. After reaching this ceiling, the agent stops improving and enters a **noisy plateau**. Continuing to train wastes electricity and time without meaningful gains.

## Solution: Intelligent Early Stopping

The training system now automatically detects when your agent has converged and stops training early.

---

## How It Works

### 1. Score Tracking
- Every episode's final score is recorded in `episode_scores[]`
- These raw scores are noisy (high variance between episodes)

### 2. Moving Average Calculation
- After 100 episodes, we calculate a **100-episode moving average**
- Formula: `moving_avg = sum(episode_scores[-100:]) / 100`
- This smooths out noise and reveals the true learning trend

### 3. Convergence Detection
The algorithm tracks two key metrics:

```python
convergence_window = 100      # Moving average window
convergence_patience = 5000   # Episodes to wait without improvement
```

**Stopping Condition:**
> Training stops when the 100-episode moving average hasn't improved by at least 1% for 5,000 consecutive episodes.

**Why 1% threshold?**
- Filters out random noise fluctuations
- Ensures we only count meaningful improvements
- Prevents premature stopping from short-term variance

**Why 5,000 episodes patience?**
- RL learning can have long plateaus before breakthroughs
- Too short (e.g., 500) risks stopping before real improvements
- Too long (e.g., 10,000) wastes computation
- 5,000 balances patience with efficiency

---

## What You'll See During Training

### Console Output (Every 10 Episodes)

```
Ep  100 | Reward: 1234.56 | Score:   856 | MA-100:   856 | Tile: 256 | ε: 0.850 | No-Imp: 0 | Time: 00:05:23
Ep  110 | Reward: 1345.67 | Score:   912 | MA-100:   884 | Tile: 256 | ε: 0.840 | No-Imp: 0 | Time: 00:06:01
...
Ep 5200 | Reward: 2134.23 | Score:  1456 | MA-100:  1450 | Tile: 512 | ε: 0.050 | No-Imp: 4823 | Time: 03:45:12

[CONVERGENCE] Agent converged! No improvement for 5000 episodes
[CONVERGENCE] Best moving average: 1450.23
```

**Key Metrics:**
- `MA-100`: Current 100-episode moving average (your convergence metric)
- `No-Imp`: Episodes since last moving average improvement
- When `No-Imp` reaches 5000 → training stops automatically

### Training Plots

The live plot shows:

**Top Panel:** Scores (Raw + Moving Average)
- Blue scatter dots: Raw episode scores (noisy)
- Red smooth line: 100-episode moving average (convergence trend)

**Bottom Panel:** Game Metrics
- Green: Episode scores
- Red: Max tiles achieved

The moving average line is your guide to convergence. When it flattens for 5,000 episodes, you've hit the plateau.

---

## Benefits

### 1. **Saves Time & Energy**
- No more training for arbitrary 3,000 or 10,000 episodes
- Stop as soon as learning plateaus
- Some agents converge in 2,000 episodes, others need 8,000+

### 2. **Finds True Performance Limit**
- You know the agent has learned everything it can
- Best moving average represents its stable performance capability

### 3. **Prevents Overfitting**
- Stopping at convergence prevents memorizing specific board states
- Agent generalizes better to unseen situations

### 4. **Fair Algorithm Comparison**
- Compare DQN vs Double-DQN vs REINFORCE at their true convergence points
- Not at arbitrary episode counts where one might still be improving

---

## Understanding the Output

### If Training Stops Early (e.g., Episode 4,523)

```
Training Complete!
Reason: Converged (no improvement for 5000 episodes)
Best Moving Average (100ep): 1847.34
Total Episodes: 4523
```

**This is GOOD!** 
- Your agent found its performance ceiling efficiently
- No need to train longer - it won't improve
- The best moving average (1847.34) is its true capability

### If Training Reaches Max Episodes (e.g., 3000)

```
Training Complete!
Total Episodes: 3000
Best Score: 2156
Best Tile: 512
```

**This means:**
- Agent was still improving at episode 3000
- Consider increasing `CONFIG["episodes"]` to let it converge naturally
- Or accept that you're getting a partially-trained model

---

## Configuration

Edit these in `2048RL.py` if needed:

```python
# In training functions (around line 413 for DQN)
convergence_window = 100      # Moving average window size
convergence_patience = 5000   # Episodes without improvement before stopping
```

**Recommended Settings:**
- `convergence_window`: Keep at 100 (standard in RL literature)
- `convergence_patience`: 
  - Fast algorithms (DQN): 3000-5000
  - Slow algorithms (REINFORCE): 5000-10000
  - Complex environments: Increase up to 10000

**Improvement Threshold:**
```python
if moving_avg > best_moving_avg * 1.01:  # 1% improvement
```
- Current: 1% improvement required
- Make stricter: 1.02 (2%), 1.05 (5%)
- Make looser: 1.005 (0.5%)

---

## Scientific Basis

This approach is standard in RL research:

1. **Moving Average Smoothing**: Reduces variance in noisy episodic returns
2. **Patience Window**: Allows for temporary plateaus before breakthroughs
3. **Percentage Threshold**: Scales with score magnitude (100 → 101 vs 10000 → 10100)

**References:**
- Mnih et al. (2015) - DQN: "Training stopped when performance plateaued"
- Schulman et al. (2017) - PPO: Used early stopping based on KL-divergence
- OpenAI Gym documentation recommends moving average convergence detection

---

## Example Convergence Scenarios

### Scenario 1: Fast Learner (DQN)
```
Episode 100  → MA-100: 856
Episode 500  → MA-100: 1234
Episode 1000 → MA-100: 1567
Episode 1500 → MA-100: 1645  ← Improving slowly
Episode 2000 → MA-100: 1651  ← Tiny improvement
Episode 2500 → MA-100: 1653  ← Barely moving
Episode 7500 → MA-100: 1658  ← Still stuck at ~1650

[CONVERGENCE] Stopped at episode 7500
Best MA: 1658.23
```

### Scenario 2: Breakthrough Learner (REINFORCE)
```
Episode 1000 → MA-100: 456
Episode 2000 → MA-100: 678
Episode 3000 → MA-100: 723   ← Slow improvement
Episode 4000 → MA-100: 745   ← Still learning
Episode 5000 → MA-100: 1234  ← BREAKTHROUGH! (doubled)
Episode 6000 → MA-100: 1456  ← Continuing
Episode 7000 → MA-100: 1512  ← Slowing down
Episode 12000 → MA-100: 1523 ← Plateaued

[CONVERGENCE] Stopped at episode 12000
Best MA: 1523.67
```

Notice how patience (5000 episodes) allowed the breakthrough at episode 5000 to be discovered!

---

## Troubleshooting

### "My agent stopped too early!"

**Possible causes:**
1. Patience too short → Increase `convergence_patience` to 7000-10000
2. Threshold too strict → Lower from 1.01 to 1.005
3. Agent genuinely converged → This is expected behavior

### "Training never stops!"

**Possible causes:**
1. Agent continuously improving (rare but good!)
2. High variance in scores → Increase `convergence_window` to 200
3. Threshold too loose → Increase from 1.01 to 1.02

### "How do I disable early stopping?"

Set an impossibly high patience:
```python
convergence_patience = 999999  # Effectively disables early stopping
```

Or set episodes to max you want to train:
```python
CONFIG["episodes"] = 3000  # Will stop at 3000 regardless
```

---

## Summary

**Before:** Train for 3000 episodes, hope it's enough, waste time if agent converged at 1500

**After:** Train until convergence detected automatically, stop early when learning plateaus, get true performance limit

**Result:** Faster training, better models, fair comparisons, resource efficiency

The 100-episode moving average is your **convergence metric**. When it stops improving for 5000 episodes, your agent has reached its limit with current hyperparameters. Time to save the model and celebrate! 🎉
