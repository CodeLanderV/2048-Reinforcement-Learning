# Archived Algorithms

## Overview

MCTS and REINFORCE algorithms have been archived to streamline the codebase and focus on the most effective deep learning approaches (DQN and Double-DQN).

The code has been **preserved but commented out** in `2048RL.py` for future reference.

---

## Archived Algorithms

### 1. MCTS (Monte Carlo Tree Search)

**Status:** Archived (commented out in `2048RL.py` around line 645)

**Why Archived:**
- Planning-only algorithm (doesn't learn from experience)
- Doesn't save/load models
- Very slow for real-time training
- Better suited as a baseline comparison, not production use

**Code Location:** `src/agents/mcts/` folder still exists

**To Re-enable:**
1. Uncomment `train_mcts()` function in `2048RL.py` (lines ~645-770)
2. Update `main()` function to call `train_mcts()` when requested (line ~1295)
3. Add `'mcts'` back to argument parser choices (line ~1208)

---

### 2. REINFORCE (Policy Gradient)

**Status:** Archived (commented out in `2048RL.py` around line 776)

**Why Archived:**
- On-policy learning (less sample efficient than DQN)
- High variance in training
- Slower convergence compared to value-based methods
- DQN/Double-DQN outperform for this task

**Code Location:** `src/agents/reinforce/` folder still exists

**To Re-enable:**
1. Uncomment `train_reinforce()` function in `2048RL.py` (lines ~776-1183)
2. Update `main()` function to call `train_reinforce()` when requested (line ~1297)
3. Add `'reinforce'` back to argument parser choices (line ~1208)

---

## Current Active Algorithms

### DQN (Deep Q-Network)
- **Status:** Active ✅
- **Use:** `python 2048RL.py train --algorithm dqn`
- **Best For:** General-purpose RL, good sample efficiency

### Double-DQN
- **Status:** Active ✅
- **Use:** `python 2048RL.py train --algorithm double-dqn`
- **Best For:** Reduces Q-value overestimation, more stable than DQN

---

## If You Try to Use Archived Algorithms

Running:
```bash
python 2048RL.py train --algorithm mcts
```

Will show:
```
error: argument --algorithm/-a: invalid choice: 'mcts' (choose from 'dqn', 'double-dqn')
```

The archived algorithms are hidden from the CLI to prevent accidental use.

---

## Configuration Files

### Active Config Keys
```python
CONFIG = {
    "dqn": { ... },           # Active
    "double_dqn": { ... },    # Active
    "mcts": { ... },          # Archived but config preserved
    "reinforce": { ... },     # Archived but config preserved
}
```

The MCTS and REINFORCE config sections are still in `2048RL.py` but unused. They're kept for reference if you re-enable these algorithms.

---

## Agent Folders

These folders still exist but are not used by the active training pipeline:

```
src/agents/
    ├── dqn/           # Active
    ├── double_dqn/    # Active
    ├── mcts/          # Archived (code exists but unused)
    └── reinforce/     # Archived (code exists but unused)
```

You can safely delete `src/agents/mcts/` and `src/agents/reinforce/` if you're certain you won't need them.

---

## Re-enabling Steps (Detailed)

### To Re-enable MCTS:

1. **Uncomment training function:**
   - File: `2048RL.py`
   - Lines: ~645-770
   - Remove `#` from all lines in `train_mcts()` function

2. **Update main() routing:**
   - File: `2048RL.py`
   - Line: ~1295
   - Change:
     ```python
     # From:
     print("[ERROR] MCTS algorithm has been archived...")
     
     # To:
     train_mcts()
     ```

3. **Add to CLI choices:**
   - File: `2048RL.py`
   - Line: ~1208
   - Change:
     ```python
     choices=['dqn', 'double-dqn', 'mcts'],  # Add 'mcts'
     ```

4. **Test:**
   ```bash
   python 2048RL.py train --algorithm mcts --episodes 10
   ```

### To Re-enable REINFORCE:

Follow the same steps as MCTS but for:
- Function lines: ~776-1183
- Main() call line: ~1297
- CLI choice: add `'reinforce'`

---

## Performance Comparison (Why We Archived)

From previous training runs:

| Algorithm   | Avg Score | Max Tile | Training Time | Sample Efficiency |
|-------------|-----------|----------|---------------|-------------------|
| **DQN**         | ~1500     | 512-1024 | 2-3 hours     | ⭐⭐⭐⭐            |
| **Double-DQN**  | ~1600     | 512-1024 | 2-3 hours     | ⭐⭐⭐⭐⭐          |
| MCTS        | ~800      | 256-512  | 5+ hours      | N/A (no learning) |
| REINFORCE   | ~600      | 256      | 4-6 hours     | ⭐⭐               |

DQN and Double-DQN are significantly more effective for this task.

---

## Summary

- **MCTS** and **REINFORCE** are archived but not deleted
- All code is preserved in comments for future reference
- **DQN** and **Double-DQN** are the recommended production algorithms
- Re-enabling archived algorithms is straightforward if needed
- Agent implementation code still exists in `src/agents/` folders

The codebase now focuses on the algorithms that actually work well for 2048! 🎯
