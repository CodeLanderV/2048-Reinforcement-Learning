# 📊 Advanced Plotting & Visualization System

## Overview

The refactored plotting system provides comprehensive visualization capabilities for analyzing 2048 RL training progress. It replaces the basic real-time plots with a sophisticated multi-panel dashboard and historical analysis tools.

## ✨ Key Features

### 1. **Enhanced Real-Time Training Plots**
During training, you get a **3-panel interactive dashboard**:

#### Top Panel: Score Progression
- Raw scores (scatter plot, semi-transparent)
- 100-episode moving average (red line) - the convergence metric
- Best MA-100 (gold dashed line) - your best performance
- Score milestones (1000, 2000, 5000, 10000) as reference lines

#### Bottom Left: Tile Distribution
- Histogram showing how many times each max tile was achieved
- Color-coded bars (darker = higher tiles)
- Frequency labels on each bar

#### Bottom Right: Convergence Tracking
- Episodes since last improvement (orange filled area)
- Patience threshold (red dashed line at 1000 episodes)
- Current status indicator (shows exact count)

### 2. **Historical Training Analysis**
Parse and visualize **all past training sessions** from `evaluations/training_log.txt`:

- **Training duration trends** - Episodes over time
- **Best score progression** - Color-coded by algorithm
- **Max tile achievements** - Logarithmic scale with milestones
- **Algorithm performance summary** - Average scores and tiles

### 3. **Comprehensive Dashboard Generation**
Combine multiple data sources into a single publication-quality figure:
- Live training plots
- Historical session data
- Optuna hyperparameter tuning results (future)

---

## 🚀 Usage

### During Training (Automatic)

When you start training with plots enabled:

```bash
python 2048RL.py train --algorithm dqn --episodes 2000 --enable-plots
```

The new `TrainingPlotter` automatically displays:
- Real-time updates every 5 episodes
- Auto-saves final plot to `evaluations/DQN_training_plot.png`
- Supports interactive early-stopping (close window to stop)

### After Training (Historical Analysis)

#### Generate Comprehensive Dashboard

```bash
python plot_training_results.py --dashboard
```
**Output:** `evaluations/training_dashboard.png`

#### Plot All Historical Sessions

```bash
python plot_training_results.py --history
```
**Output:** `evaluations/all_sessions_comparison.png`

Shows 4-panel comparison:
1. Episodes trained per session
2. Best scores with algorithm color-coding
3. Max tiles (log scale)
4. Algorithm performance averages

#### Plot Latest Session Only

```bash
python plot_training_results.py --latest
```

#### Plot Specific Session

```bash
python plot_training_results.py --session 5    # 6th session (0-indexed)
python plot_training_results.py --session -1   # Last session
python plot_training_results.py --session -2   # Second-to-last
```

### Custom Output Options

#### High-Resolution for Publications

```bash
python plot_training_results.py --dashboard --dpi 300 --output paper_figure.pdf
```

#### Interactive Display (Don't Save)

```bash
python plot_training_results.py --history --show
```

#### Custom Paths

```bash
python plot_training_results.py --history \
    --log-file my_experiments/training_log.txt \
    --output results/my_plot.png \
    --dpi 200
```

---

## 📈 What Changed from Old System

### Before (Old System)
- **2 basic plots**: Raw scores + moving average, Score/Tile overlay
- **Limited info**: Just episode numbers and values
- **No history**: Only current training session
- **Manual analysis**: Had to parse logs manually

### After (New System)
- **3 comprehensive plots**: Score progression, tile distribution, convergence tracking
- **Rich information**: Milestones, best performance, patience tracking
- **Full history**: Analyze all past training sessions
- **Automated analysis**: Parse logs automatically, generate comparisons
- **Better visuals**: Color-coded, labeled, publication-quality

### Side-by-Side Comparison

| Feature | Old System | New System |
|---------|-----------|------------|
| Real-time panels | 2 | 3 |
| Convergence tracking | ❌ | ✅ |
| Tile distribution | ❌ | ✅ |
| Historical analysis | ❌ | ✅ |
| Multi-session comparison | ❌ | ✅ |
| Algorithm comparison | ❌ | ✅ |
| Milestone markers | ❌ | ✅ |
| Best performance indicator | ❌ | ✅ |
| Patience visualization | ❌ | ✅ |
| Publication-quality | ❌ | ✅ |
| CLI tool | ❌ | ✅ |

---

## 🔧 Technical Details

### New Files Created

#### `src/plotting.py`
- **`TrainingPlotter`**: Real-time training visualization class
  - 3-panel dashboard with auto-refresh
  - Convergence tracking integration
  - Smart layout with GridSpec
  
- **`TrainingAnalyzer`**: Historical log parser and visualizer
  - Regex-based parsing of `training_log.txt`
  - 4-panel session comparison plots
  - Algorithm performance summaries
  
- **`ResultsVisualizer`**: Dashboard generator
  - Combines multiple data sources
  - Comprehensive training overview

#### `plot_training_results.py`
- Standalone CLI tool for post-training analysis
- Multiple plot modes (history, latest, session, dashboard)
- Flexible output options (format, resolution, path)
- Interactive display mode

### Integration Points

#### In `2048RL.py` (`train_dqn_variant` function):

**Old Code:**
```python
if CONFIG["enable_plots"]:
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ... training loop ...

if CONFIG["enable_plots"] and episode % 5 == 0:
    _update_training_plot(ax1, ax2, rewards, scores, tiles, ma, algo)
    plt.pause(0.01)
```

**New Code:**
```python
plotter = None
if CONFIG["enable_plots"]:
    plotter = TrainingPlotter(algo_name=algo_name, refresh_interval=5)

# ... training loop ...

if plotter and episode % 5 == 0:
    plotter.update(episode, score, max_tile, reward, moving_avg)
    plotter.refresh()
```

### Data Sources

The system automatically reads and visualizes:

1. **`evaluations/training_log.txt`**
   - Session timestamps
   - Algorithm names
   - Episode counts
   - Final average rewards
   - Best scores and max tiles
   - Hyperparameter notes

2. **Live Training Data** (during training)
   - Episode numbers
   - Scores
   - Max tiles
   - Rewards
   - Moving averages (MA-100)

3. **Saved PNG Plots** (from previous sessions)
   - `DQN_training_plot.png`
   - `Double_DQN_training_plot.png`

---

## 📊 Plot Examples

### Real-Time Training Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│ DQN Training Progress - Score & Moving Average              │
│ • Blue scatter: Raw episode scores                          │
│ • Red line: MA-100 (convergence metric)                    │
│ • Gold dash: Best MA-100 achieved                          │
│ • Gray dotted: Score milestones (1000, 2000, 5000)        │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────┬──────────────────────────────┐
│ Max Tile Distribution        │ Convergence Tracking          │
│ Bar chart showing:           │ Orange area: Episodes w/o     │
│ - Tile 64: 150 times         │ improvement                   │
│ - Tile 128: 85 times         │ Red line: Patience = 1000     │
│ - Tile 256: 12 times         │ Current: 347 episodes         │
│ - Tile 512: 2 times          │                               │
└──────────────────────────────┴──────────────────────────────┘
```

### Historical Sessions Comparison

```
┌──────────────────────────────┬──────────────────────────────┐
│ Training Duration (Episodes) │ Best Score Progression        │
│ Bar chart: Episodes/session  │ Scatter: Scores over time    │
│ Color: Uniform steel blue    │ Color: By algorithm          │
│                              │ Line: Trend connection        │
└──────────────────────────────┴──────────────────────────────┘

┌──────────────────────────────┬──────────────────────────────┐
│ Max Tile Progression         │ Algorithm Performance Summary │
│ Log scale (2, 4, 8, ..., 512)│ Dual bar chart:              │
│ Purple line with markers     │ - Blue: Avg scores           │
│ Milestone lines at powers-2  │ - Coral: Avg tiles           │
└──────────────────────────────┴──────────────────────────────┘
```

---

## 🎯 Recommendations

### For Training

1. **Keep plots enabled** during initial experiments:
   ```bash
   python 2048RL.py train --algorithm dqn --episodes 500 --enable-plots
   ```

2. **Disable for long runs** to avoid GUI overhead:
   ```bash
   python 2048RL.py train --algorithm dqn --episodes 10000 --no-ui --no-plots
   ```

3. **Monitor convergence** using the bottom-right panel:
   - Orange area approaching red line = nearly converged
   - Orange area resetting to 0 = still improving

### For Analysis

1. **After each training session**, generate dashboard:
   ```bash
   python plot_training_results.py --dashboard
   ```

2. **Before tuning hyperparameters**, review historical performance:
   ```bash
   python plot_training_results.py --history --show
   ```

3. **For presentations/papers**, use high-DPI exports:
   ```bash
   python plot_training_results.py --dashboard --dpi 300 --format pdf
   ```

---

## 🐛 Troubleshooting

### Issue: Plots not showing during training

**Solution:** Make sure you're running with `--enable-plots`:
```bash
python 2048RL.py train --algorithm dqn --episodes 500 --enable-plots
```

### Issue: Historical plots are empty

**Solution:** Check if `evaluations/training_log.txt` exists and has data:
```bash
cat evaluations/training_log.txt
```

### Issue: "No training sessions found"

**Solution:** Train at least once to generate log data:
```bash
python 2048RL.py train --algorithm dqn --episodes 100
```

### Issue: Plot window closes immediately

**Solution:** Use `--show` flag for interactive display:
```bash
python plot_training_results.py --history --show
```

---

## 🔮 Future Enhancements

1. **Optuna Integration**: Visualize hyperparameter tuning results
2. **Loss Curves**: Add training loss and Q-value plots
3. **Per-Episode Metrics**: Detailed drill-down for individual episodes
4. **Comparative Analysis**: Side-by-side model comparison
5. **Interactive Dashboard**: Web-based real-time monitoring
6. **Tile Heatmaps**: Visualize board state distribution
7. **Action Statistics**: Frequency of moves (UP, DOWN, LEFT, RIGHT)

---

## 📝 Summary

The new plotting system provides:

✅ **Better real-time visualization** - 3-panel dashboard with comprehensive metrics  
✅ **Historical analysis tools** - Compare all training sessions  
✅ **Automated log parsing** - Extract insights from training_log.txt  
✅ **Publication-quality outputs** - High-DPI, multiple formats  
✅ **Flexible CLI tool** - Easy post-training analysis  
✅ **Convergence tracking** - Visual feedback on training progress  

Use `python plot_training_results.py --help` for full CLI documentation.
