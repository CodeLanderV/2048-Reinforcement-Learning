"""
═══════════════════════════════════════════════════════════════════════════════
Advanced Plotting System for 2048 RL Training
═══════════════════════════════════════════════════════════════════════════════

Comprehensive visualization utilities for analyzing training progress, comparing
models, and generating publication-quality plots.

FEATURES:
    - Real-time training plots (live updates during training)
    - Post-training analysis plots (comprehensive metrics)
    - Multi-run comparison plots (compare different algorithms/hyperparameters)
    - Historical log parsing (visualize old training sessions)
    - Publication-quality exports (high-DPI PNG/PDF)

USAGE:
    # During training (real-time)
    plotter = TrainingPlotter(algo_name="DQN")
    for episode in range(episodes):
        # ... training code ...
        plotter.update(episode, score, max_tile, reward, moving_avg)
        if episode % 10 == 0:
            plotter.refresh()  # Update display
    
    plotter.save("evaluations/training_plot.png")
    
    # After training (comprehensive analysis)
    analyzer = TrainingAnalyzer("evaluations/training_log.txt")
    analyzer.plot_session_summary(session_id=-1)  # Latest session
    analyzer.plot_all_sessions_comparison()
    
    # Parse existing plots from evaluations directory
    visualizer = ResultsVisualizer("evaluations")
    visualizer.create_dashboard()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import re
from datetime import datetime
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# Real-Time Training Plotter
# ═══════════════════════════════════════════════════════════════════════════

class TrainingPlotter:
    """
    Real-time plotting during training with auto-refresh and smart layout.
    
    Displays:
        - Top: Score progression (raw + moving average + milestones)
        - Bottom Left: Max tile distribution (histogram)
        - Bottom Right: Convergence tracking (episodes since improvement)
    
    Example:
        plotter = TrainingPlotter("DQN", refresh_interval=10)
        for ep in range(1000):
            score, tile = train_episode()
            plotter.update(ep, score, tile)
            if ep % 10 == 0:
                plotter.refresh()
        plotter.save("training_plot.png")
    """
    
    def __init__(
        self,
        algo_name: str = "DQN",
        refresh_interval: int = 5,
        ma_window: int = 100,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Initialize real-time training plotter.
        
        Args:
            algo_name: Algorithm name for plot title
            refresh_interval: Episodes between plot refreshes
            ma_window: Window size for moving average (100 for convergence metric)
            figsize: Figure size (width, height) in inches
        """
        self.algo_name = algo_name
        self.refresh_interval = refresh_interval
        self.ma_window = ma_window
        
        # Data storage
        self.episodes: List[int] = []
        self.scores: List[float] = []
        self.tiles: List[int] = []
        self.rewards: List[float] = []
        self.moving_avgs: List[float] = []
        self.best_ma: float = 0.0
        self.episodes_since_improvement: int = 0
        
        # Setup figure with advanced layout
        self.fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=self.fig, height_ratios=[2, 1])
        
        # Top row: Score progression (full width)
        self.ax_score = self.fig.add_subplot(gs[0, :])
        
        # Bottom left: Tile distribution
        self.ax_tiles = self.fig.add_subplot(gs[1, 0])
        
        # Bottom right: Convergence tracker
        self.ax_convergence = self.fig.add_subplot(gs[1, 1])
        
        # Enable interactive mode
        plt.ion()
        plt.show()
    
    def update(
        self,
        episode: int,
        score: float,
        max_tile: int,
        reward: float = 0.0,
        moving_avg: Optional[float] = None
    ):
        """
        Add new episode data point.
        
        Args:
            episode: Episode number
            score: Final score
            max_tile: Highest tile achieved
            reward: Total episode reward
            moving_avg: Pre-computed moving average (or None to auto-calculate)
        """
        self.episodes.append(episode)
        self.scores.append(score)
        self.tiles.append(max_tile)
        self.rewards.append(reward)
        
        # Calculate/store moving average
        if moving_avg is not None:
            self.moving_avgs.append(moving_avg)
        elif len(self.scores) >= self.ma_window:
            ma = np.mean(self.scores[-self.ma_window:])
            self.moving_avgs.append(ma)
            
            # Track best MA for convergence
            if ma > self.best_ma * 1.01:  # 1% improvement threshold
                self.best_ma = ma
                self.episodes_since_improvement = 0
            else:
                self.episodes_since_improvement += 1
    
    def refresh(self):
        """Redraw all plots with latest data."""
        if not self.episodes:
            return
        
        # Clear all axes
        self.ax_score.clear()
        self.ax_tiles.clear()
        self.ax_convergence.clear()
        
        # ─── Plot 1: Score Progression ───
        self._plot_scores()
        
        # ─── Plot 2: Tile Distribution ───
        self._plot_tile_distribution()
        
        # ─── Plot 3: Convergence Tracking ───
        self._plot_convergence()
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _plot_scores(self):
        """Plot score progression with moving average and milestones."""
        episodes = np.array(self.episodes)
        scores = np.array(self.scores)
        
        # Raw scores (scatter, semi-transparent)
        self.ax_score.scatter(
            episodes, scores,
            alpha=0.3, s=15, color='steelblue',
            label='Episode Score'
        )
        
        # Plot moving average if available
        if len(self.moving_avgs) > 1:
            # Ensure we only plot episodes for which we have a moving average
            ma_start_index = len(self.episodes) - len(self.moving_avgs)
            ma_episodes = self.episodes[ma_start_index:]
            self.ax_score.plot(ma_episodes, self.moving_avgs, color=self.ma_color, label=f"MA-{self.ma_window}", linewidth=1.5)
            
            # Highlight best MA
            if self.best_ma > 0:
                self.ax_score.axhline(
                    y=self.best_ma, color='gold',
                    linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Best MA: {self.best_ma:.0f}'
                )
        
        # Score milestones
        milestones = [1000, 2000, 5000, 10000]
        for milestone in milestones:
            if np.max(scores) > milestone * 0.8:
                self.ax_score.axhline(
                    y=milestone, color='gray',
                    linestyle=':', linewidth=1, alpha=0.5
                )
                self.ax_score.text(
                    0.02, milestone, f'{milestone}',
                    transform=self.ax_score.get_yaxis_transform(),
                    fontsize=8, color='gray', alpha=0.7
                )
        
        self.ax_score.set_xlabel('Episode', fontsize=11)
        self.ax_score.set_ylabel('Score', fontsize=11)
        self.ax_score.set_title(
            f'{self.algo_name} Training Progress - Score & Moving Average',
            fontsize=13, fontweight='bold'
        )
        self.ax_score.legend(loc='upper left', fontsize=9)
        self.ax_score.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_tile_distribution(self):
        """Plot histogram of max tiles achieved."""
        tiles = np.array(self.tiles)
        
        # Get unique tiles and their frequencies
        unique_tiles, counts = np.unique(tiles, return_counts=True)
        
        # Create bar chart
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(unique_tiles)))
        bars = self.ax_tiles.bar(
            range(len(unique_tiles)), counts,
            color=colors, alpha=0.8, edgecolor='black'
        )
        
        # Set x-axis labels to tile values
        self.ax_tiles.set_xticks(range(len(unique_tiles)))
        self.ax_tiles.set_xticklabels(unique_tiles, rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.ax_tiles.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=8
            )
        
        self.ax_tiles.set_xlabel('Max Tile', fontsize=10)
        self.ax_tiles.set_ylabel('Frequency', fontsize=10)
        self.ax_tiles.set_title(
            'Max Tile Distribution',
            fontsize=11, fontweight='bold'
        )
        self.ax_tiles.grid(True, axis='y', alpha=0.3)
    
    def _plot_convergence(self):
        """Plot episodes since last improvement (convergence tracking)."""
        if len(self.moving_avgs) <= 1:
            return
        
        # Calculate episodes since improvement over time
        convergence_history = []
        best_so_far = 0.0
        episodes_no_improve = 0
        
        for ma in self.moving_avgs:
            if ma > best_so_far * 1.01:
                best_so_far = ma
                episodes_no_improve = 0
            else:
                episodes_no_improve += 1
            convergence_history.append(episodes_no_improve)
        
        # Get episodes that correspond to moving averages
        ma_start_index = len(self.episodes) - len(self.moving_avgs)
        ma_episodes = np.array(self.episodes[ma_start_index:])
        
        # Plot convergence metric
        self.ax_convergence.fill_between(
            ma_episodes, 0, convergence_history,
            color='orange', alpha=0.4, label='No Improvement Count'
        )
        self.ax_convergence.plot(
            ma_episodes, convergence_history,
            color='darkorange', linewidth=2
        )
        
        # Add patience threshold line
        patience = 1000  # From CONFIG
        self.ax_convergence.axhline(
            y=patience, color='red',
            linestyle='--', linewidth=2, alpha=0.7,
            label=f'Patience Threshold: {patience}'
        )
        
        # Current status
        current_text = f'Current: {self.episodes_since_improvement} episodes'
        self.ax_convergence.text(
            0.95, 0.95, current_text,
            transform=self.ax_convergence.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        self.ax_convergence.set_xlabel('Episode', fontsize=10)
        self.ax_convergence.set_ylabel('Episodes Without Improvement', fontsize=10)
        self.ax_convergence.set_title(
            'Convergence Tracking',
            fontsize=11, fontweight='bold'
        )
        self.ax_convergence.legend(loc='upper left', fontsize=8)
        self.ax_convergence.grid(True, alpha=0.3)
    
    def save(self, filepath: str, dpi: int = 150):
        """
        Save current plot to file.
        
        Args:
            filepath: Output file path (PNG recommended)
            dpi: Image resolution (150 for screen, 300 for publication)
        """
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"[PLOT] Saved: {filepath}")
    
    def close(self):
        """Close the figure and clean up."""
        plt.ioff()
        plt.close(self.fig)


# ═══════════════════════════════════════════════════════════════════════════
# Historical Training Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class TrainingAnalyzer:
    """
    Parse and visualize historical training sessions from logs.txt.
    
    Reads evaluations/training_log.txt and creates comprehensive
    analysis plots comparing different training runs.
    
    Example:
        analyzer = TrainingAnalyzer("evaluations/training_log.txt")
        analyzer.plot_all_sessions_comparison()
        analyzer.plot_hyperparameter_trends()
    """
    
    def __init__(self, log_file: str = "evaluations/training_log.txt"):
        """
        Initialize analyzer with training log file.
        
        Args:
            log_file: Path to training_log.txt
        """
        self.log_file = Path(log_file)
        self.sessions = self._parse_log_file()
    
    def _parse_log_file(self) -> List[Dict]:
        """Parse training log file into structured data."""
        if not self.log_file.exists():
            print(f"[WARNING] Log file not found: {self.log_file}")
            return []
        
        sessions = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by session separator
        session_blocks = content.split('=' * 80)
        
        for block in session_blocks:
            if 'Training Session:' not in block:
                continue
            
            session = {}
            
            # Extract fields using regex
            patterns = {
                'timestamp': r'Training Session: (.+)',
                'algorithm': r'Algorithm:\s+(.+)',
                'episodes': r'Episodes:\s+(\d+)',
                'training_time': r'Training Time:\s+(.+)',
                'final_avg_reward': r'Final Avg Reward:\s+([\d.]+)',
                'max_tile': r'Best Max Tile:\s+(\d+)',
                'best_score': r'Best Score:\s+(\d+)',
                'model_path': r'Model Saved:\s+(.+)',
                'notes': r'Notes:\s+(.+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, block)
                if match:
                    value = match.group(1).strip()
                    # Convert numeric fields
                    if key in ['episodes', 'max_tile', 'best_score']:
                        session[key] = int(value)
                    elif key == 'final_avg_reward':
                        session[key] = float(value)
                    else:
                        session[key] = value
            
            if session:
                sessions.append(session)
        
        print(f"[INFO] Parsed {len(sessions)} training sessions from log")
        return sessions
    
    def plot_all_sessions_comparison(self, save_path: Optional[str] = None):
        """
        Create comprehensive comparison plot of all training sessions.
        
        Shows:
            - Training duration trends
            - Best scores over time
            - Max tiles achieved
            - Algorithm performance comparison
        """
        if not self.sessions:
            print("[WARNING] No sessions to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        sessions = self.sessions[-20:]  # Last 20 sessions
        timestamps = [s.get('timestamp', 'Unknown') for s in sessions]
        episodes = [s.get('episodes', 0) for s in sessions]
        scores = [s.get('best_score', 0) for s in sessions]
        tiles = [s.get('max_tile', 0) for s in sessions]
        algorithms = [s.get('algorithm', 'Unknown') for s in sessions]
        
        # Simplify timestamps for x-axis
        x_labels = [ts.split()[0] if ts != 'Unknown' else f'Session {i}'
                    for i, ts in enumerate(timestamps)]
        x = np.arange(len(sessions))
        
        # Plot 1: Episode counts
        ax1.bar(x, episodes, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Session', fontsize=10)
        ax1.set_ylabel('Episodes Trained', fontsize=10)
        ax1.set_title('Training Duration (Episodes)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x[::2])
        ax1.set_xticklabels(x_labels[::2], rotation=45, ha='right', fontsize=8)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: Best scores
        colors_by_algo = {'DQN': 'blue', 'Double DQN': 'red', 'MCTS': 'green'}
        colors = [colors_by_algo.get(algo, 'gray') for algo in algorithms]
        ax2.scatter(x, scores, c=colors, s=100, alpha=0.7)
        ax2.plot(x, scores, 'k--', alpha=0.3)
        ax2.set_xlabel('Session', fontsize=10)
        ax2.set_ylabel('Best Score', fontsize=10)
        ax2.set_title('Best Score Progression', fontsize=12, fontweight='bold')
        ax2.set_xticks(x[::2])
        ax2.set_xticklabels(x_labels[::2], rotation=45, ha='right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Add legend for algorithms
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=algo)
                          for algo, color in colors_by_algo.items()
                          if algo in algorithms]
        ax2.legend(handles=legend_elements, loc='upper left')
        
        # Plot 3: Max tiles (logarithmic)
        ax3.plot(x, tiles, 'o-', color='purple', linewidth=2, markersize=8)
        ax3.set_yscale('log', base=2)
        ax3.set_xlabel('Session', fontsize=10)
        ax3.set_ylabel('Max Tile (log scale)', fontsize=10)
        ax3.set_title('Max Tile Progression', fontsize=12, fontweight='bold')
        ax3.set_xticks(x[::2])
        ax3.set_xticklabels(x_labels[::2], rotation=45, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, which='both')
        
        # Add milestone lines
        for milestone in [128, 256, 512, 1024, 2048]:
            if any(t >= milestone for t in tiles):
                ax3.axhline(y=milestone, color='gray', linestyle=':', alpha=0.5)
                ax3.text(0, milestone, f'{milestone}', fontsize=8, color='gray')
        
        # Plot 4: Algorithm performance summary
        algo_stats = defaultdict(lambda: {'scores': [], 'tiles': []})
        for session in sessions:
            algo = session.get('algorithm', 'Unknown')
            algo_stats[algo]['scores'].append(session.get('best_score', 0))
            algo_stats[algo]['tiles'].append(session.get('max_tile', 0))
        
        algo_names = list(algo_stats.keys())
        avg_scores = [np.mean(algo_stats[algo]['scores']) for algo in algo_names]
        avg_tiles = [np.mean(algo_stats[algo]['tiles']) for algo in algo_names]
        
        ax4_twin = ax4.twinx()
        bar_width = 0.35
        x_algo = np.arange(len(algo_names))
        
        bars1 = ax4.bar(x_algo - bar_width/2, avg_scores, bar_width,
                        label='Avg Score', color='steelblue', alpha=0.7)
        bars2 = ax4_twin.bar(x_algo + bar_width/2, avg_tiles, bar_width,
                             label='Avg Tile', color='coral', alpha=0.7)
        
        ax4.set_xlabel('Algorithm', fontsize=10)
        ax4.set_ylabel('Average Score', fontsize=10, color='steelblue')
        ax4_twin.set_ylabel('Average Max Tile', fontsize=10, color='coral')
        ax4.set_title('Algorithm Performance Summary', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_algo)
        ax4.set_xticklabels(algo_names, rotation=0)
        ax4.tick_params(axis='y', labelcolor='steelblue')
        ax4_twin.tick_params(axis='y', labelcolor='coral')
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[PLOT] Saved: {save_path}")
        else:
            plt.show()
        
        return fig


# ═══════════════════════════════════════════════════════════════════════════
# Results Visualizer (Dashboard)
# ═══════════════════════════════════════════════════════════════════════════

class ResultsVisualizer:
    """
    Create comprehensive dashboard from all evaluation data.
    
    Combines:
        - Live training plots (PNG files)
        - Historical session data (training_log.txt)
        - Optuna hyperparameter tuning results (JSON files)
    
    Example:
        viz = ResultsVisualizer("evaluations")
        viz.create_dashboard("dashboard.png")
    """
    
    def __init__(self, eval_dir: str = "evaluations"):
        """
        Initialize visualizer with evaluations directory.
        
        Args:
            eval_dir: Directory containing logs, plots, and results
        """
        self.eval_dir = Path(eval_dir)
        self.analyzer = TrainingAnalyzer(self.eval_dir / "training_log.txt")
    
    def create_dashboard(self, save_path: str = "evaluations/dashboard.png"):
        """
        Generate comprehensive training dashboard.
        
        Args:
            save_path: Where to save dashboard image
        """
        print("[INFO] Generating training dashboard...")
        self.analyzer.plot_all_sessions_comparison(save_path)
        print(f"[SUCCESS] Dashboard saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_session(
    scores: List[float],
    tiles: List[int],
    rewards: List[float],
    algo_name: str = "DQN",
    save_path: Optional[str] = None
):
    """
    Quick plot of training session data (for external use).
    
    Args:
        scores: List of episode scores
        tiles: List of max tiles
        rewards: List of episode rewards
        algo_name: Algorithm name
        save_path: Where to save plot (or None to display)
    """
    plotter = TrainingPlotter(algo_name=algo_name)
    
    for i, (score, tile, reward) in enumerate(zip(scores, tiles, rewards)):
        plotter.update(i, score, tile, reward)
    
    plotter.refresh()
    
    if save_path:
        plotter.save(save_path)
    else:
        plt.show()
    
    plotter.close()


def analyze_training_history(log_file: str = "evaluations/training_log.txt"):
    """
    Quick analysis of historical training sessions.
    
    Args:
        log_file: Path to training log
    """
    analyzer = TrainingAnalyzer(log_file)
    analyzer.plot_all_sessions_comparison()


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("2048 RL Training Visualizer")
    print("=" * 80)
    
    # Analyze historical sessions
    analyze_training_history()
