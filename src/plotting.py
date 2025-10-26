"""
═══════════════════════════════════════════════════════════════════════════════
Training and Evaluation Plotting System
═══════════════════════════════════════════════════════════════════════════════

This module provides comprehensive plotting for:
    TRAINING METRICS:
        - Episode vs Max Tile
        - Episode vs Max Score  
        - Episode vs Reward
        - Episode vs Total Steps
        - DQN Loss over time
    
    EVALUATION METRICS:
        - Highest tile achieved distribution
        - Win rate (% reaching 2048)
        - Average and max scores

USAGE:
    # Training
    plotter = TrainingPlotter(algo_name="DQN", save_dir="evaluations")
    
    for episode in range(episodes):
        # ... train ...
        plotter.update(episode, score, max_tile, reward, steps, loss)
        if episode % 10 == 0:
            plotter.refresh()
    
    plotter.save("final_plot.png")
    plotter.close()
    
    # Evaluation
    eval_plotter = EvaluationPlotter()
    for game in games:
        # ... play ...
        eval_plotter.add_game(score, max_tile, reached_2048)
    eval_plotter.plot_and_save("evaluation.png")
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict


class TrainingPlotter:
    """
    Real-time training metrics plotter with automatic saving.
    
    Tracks and visualizes:
        - Episode vs Max Tile (highest tile each episode)
        - Episode vs Score (game score each episode)
        - Episode vs Reward (total reward per episode)
        - Episode vs Steps (moves per episode)
        - Training Loss (DQN loss over time)
    """
    
    def __init__(
        self,
        algo_name: str = "DQN",
        save_dir: str = "evaluations",
        ma_window: int = 100,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Initialize training plotter.
        
        Args:
            algo_name: Algorithm name for plot titles
            save_dir: Directory to save plots
            ma_window: Moving average window size
            figsize: Figure size (width, height)
        """
        self.algo_name = algo_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.ma_window = ma_window
        
        # Data storage
        self.episodes: List[int] = []
        self.scores: List[float] = []
        self.max_tiles: List[int] = []
        self.rewards: List[float] = []
        self.steps: List[int] = []
        self.losses: List[float] = []
        
        # Setup interactive plot
        plt.ion()
        self.fig, self.axes = plt.subplots(3, 2, figsize=figsize)
        self.fig.suptitle(f'{algo_name} Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Configure subplots
        self.ax_tile = self.axes[0, 0]
        self.ax_score = self.axes[0, 1]
        self.ax_reward = self.axes[1, 0]
        self.ax_steps = self.axes[1, 1]
        self.ax_loss = self.axes[2, 0]
        self.ax_summary = self.axes[2, 1]
        
        # Initial empty plots
        self._setup_axes()
    
    def _setup_axes(self):
        """Configure axis labels and titles."""
        # Max Tile
        self.ax_tile.set_title('Episode vs Max Tile', fontsize=12, fontweight='bold')
        self.ax_tile.set_xlabel('Episode')
        self.ax_tile.set_ylabel('Max Tile')
        self.ax_tile.grid(True, alpha=0.3)
        
        # Score
        self.ax_score.set_title('Episode vs Score', fontsize=12, fontweight='bold')
        self.ax_score.set_xlabel('Episode')
        self.ax_score.set_ylabel('Score')
        self.ax_score.grid(True, alpha=0.3)
        
        # Reward
        self.ax_reward.set_title('Episode vs Total Reward', fontsize=12, fontweight='bold')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.grid(True, alpha=0.3)
        
        # Steps
        self.ax_steps.set_title('Episode vs Total Steps', fontsize=12, fontweight='bold')
        self.ax_steps.set_xlabel('Episode')
        self.ax_steps.set_ylabel('Steps')
        self.ax_steps.grid(True, alpha=0.3)
        
        # Loss
        self.ax_loss.set_title('Training Loss over Time', fontsize=12, fontweight='bold')
        self.ax_loss.set_xlabel('Training Step')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Summary stats
        self.ax_summary.set_title('Training Summary', fontsize=12, fontweight='bold')
        self.ax_summary.axis('off')
    
    def update(
        self,
        episode: int,
        score: float,
        max_tile: int,
        reward: float,
        steps: int,
        loss: Optional[float] = None
    ):
        """
        Add new data point.
        
        Args:
            episode: Episode number
            score: Final score
            max_tile: Highest tile reached
            reward: Total reward
            steps: Number of steps/moves
            loss: Training loss (optional, can be None)
        """
        self.episodes.append(episode)
        self.scores.append(score)
        self.max_tiles.append(max_tile)
        self.rewards.append(reward)
        self.steps.append(steps)
        if loss is not None:
            self.losses.append(loss)
    
    def refresh(self):
        """Update plot display with latest data."""
        if not self.episodes:
            return
        
        # Clear all axes
        for ax in [self.ax_tile, self.ax_score, self.ax_reward, 
                   self.ax_steps, self.ax_loss, self.ax_summary]:
            ax.clear()
        
        # Reconfigure after clearing
        self._setup_axes()
        
        episodes = np.array(self.episodes)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 1: Max Tile
        # ─────────────────────────────────────────────────────────────────
        self.ax_tile.scatter(episodes, self.max_tiles, alpha=0.4, s=10, 
                            color='red', label='Max Tile')
        if len(self.max_tiles) >= self.ma_window:
            ma_tiles = self._moving_average(self.max_tiles, self.ma_window)
            self.ax_tile.plot(episodes[self.ma_window-1:], ma_tiles, 
                             color='darkred', linewidth=2, label=f'MA-{self.ma_window}')
        self.ax_tile.legend(loc='lower right')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 2: Score
        # ─────────────────────────────────────────────────────────────────
        self.ax_score.scatter(episodes, self.scores, alpha=0.4, s=10, 
                             color='blue', label='Score')
        if len(self.scores) >= self.ma_window:
            ma_scores = self._moving_average(self.scores, self.ma_window)
            self.ax_score.plot(episodes[self.ma_window-1:], ma_scores, 
                              color='darkblue', linewidth=2, label=f'MA-{self.ma_window}')
        self.ax_score.legend(loc='lower right')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 3: Reward
        # ─────────────────────────────────────────────────────────────────
        self.ax_reward.scatter(episodes, self.rewards, alpha=0.4, s=10, 
                              color='green', label='Reward')
        if len(self.rewards) >= self.ma_window:
            ma_rewards = self._moving_average(self.rewards, self.ma_window)
            self.ax_reward.plot(episodes[self.ma_window-1:], ma_rewards, 
                               color='darkgreen', linewidth=2, label=f'MA-{self.ma_window}')
        self.ax_reward.legend(loc='lower right')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 4: Steps
        # ─────────────────────────────────────────────────────────────────
        self.ax_steps.scatter(episodes, self.steps, alpha=0.4, s=10, 
                             color='orange', label='Steps')
        if len(self.steps) >= self.ma_window:
            ma_steps = self._moving_average(self.steps, self.ma_window)
            self.ax_steps.plot(episodes[self.ma_window-1:], ma_steps, 
                              color='darkorange', linewidth=2, label=f'MA-{self.ma_window}')
        self.ax_steps.legend(loc='lower right')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 5: Loss (if available)
        # ─────────────────────────────────────────────────────────────────
        if self.losses:
            # Plot loss with smoothing for readability
            loss_steps = np.arange(len(self.losses))
            self.ax_loss.plot(loss_steps, self.losses, alpha=0.3, 
                             color='purple', linewidth=0.5)
            if len(self.losses) >= 100:
                ma_loss = self._moving_average(self.losses, min(100, len(self.losses)))
                self.ax_loss.plot(loss_steps[99:], ma_loss, 
                                 color='purple', linewidth=2, label='MA-100')
                self.ax_loss.legend(loc='upper right')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 6: Summary Statistics
        # ─────────────────────────────────────────────────────────────────
        summary_text = self._generate_summary()
        self.ax_summary.text(0.1, 0.9, summary_text, transform=self.ax_summary.transAxes,
                            fontsize=11, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Update display
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.pause(0.01)
    
    def _moving_average(self, data: List, window: int) -> np.ndarray:
        """Calculate moving average."""
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def _generate_summary(self) -> str:
        """Generate summary statistics text."""
        if not self.episodes:
            return "No data yet"
        
        recent_n = min(100, len(self.episodes))
        recent_scores = self.scores[-recent_n:]
        recent_tiles = self.max_tiles[-recent_n:]
        recent_rewards = self.rewards[-recent_n:]
        
        summary = f"""
Episodes: {len(self.episodes)}
        
Recent {recent_n} Episodes:
  Avg Score:   {np.mean(recent_scores):>8.1f}
  Max Score:   {np.max(recent_scores):>8.0f}
  Avg Tile:    {np.mean(recent_tiles):>8.0f}
  Max Tile:    {np.max(recent_tiles):>8.0f}
  Avg Reward:  {np.mean(recent_rewards):>8.1f}

Overall Best:
  Score:       {np.max(self.scores):>8.0f}
  Tile:        {np.max(self.max_tiles):>8.0f}
"""
        return summary.strip()
    
    def save(self, filename: str = None, dpi: int = 150):
        """
        Save plot to file.
        
        Args:
            filename: Output filename (default: auto-generated)
            dpi: Resolution in dots per inch
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.algo_name.replace(' ', '_')}_training_{timestamp}.png"
        
        save_path = self.save_dir / filename
        self.fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        return str(save_path)
    
    def close(self):
        """Close plot window and cleanup."""
        plt.ioff()
        plt.close(self.fig)


class EvaluationPlotter:
    """
    Evaluation session plotter for model performance analysis.
    
    Visualizes:
        - Highest tile achieved distribution
        - Win rate (games reaching 2048)
        - Score distribution (average and max)
    """
    
    def __init__(self):
        """Initialize evaluation plotter."""
        self.scores: List[int] = []
        self.max_tiles: List[int] = []
        self.wins: List[bool] = []  # Whether 2048 was reached
    
    def add_game(self, score: int, max_tile: int, reached_2048: bool = False):
        """
        Add evaluation game result.
        
        Args:
            score: Final game score
            max_tile: Highest tile reached
            reached_2048: Whether 2048 tile was achieved
        """
        self.scores.append(score)
        self.max_tiles.append(max_tile)
        self.wins.append(reached_2048 or max_tile >= 2048)
    
    def get_metrics(self) -> dict:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary with avg_score, max_score, avg_tile, max_tile, 
            win_rate, tile_distribution
        """
        if not self.scores:
            return {}
        
        # Calculate tile distribution
        tile_counts = defaultdict(int)
        for tile in self.max_tiles:
            tile_counts[tile] += 1
        
        return {
            'num_games': len(self.scores),
            'avg_score': float(np.mean(self.scores)),
            'max_score': int(np.max(self.scores)),
            'avg_tile': float(np.mean(self.max_tiles)),
            'max_tile': int(np.max(self.max_tiles)),
            'win_rate': float(np.mean(self.wins) * 100),
            'tile_distribution': dict(tile_counts)
        }
    
    def plot_and_save(self, filename: str, save_dir: str = "evaluations"):
        """
        Create and save evaluation plots.
        
        Args:
            filename: Output filename
            save_dir: Directory to save plot
        """
        if not self.scores:
            print("No evaluation data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Evaluation Results', fontsize=16, fontweight='bold')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 1: Score Distribution
        # ─────────────────────────────────────────────────────────────────
        ax = axes[0, 0]
        ax.hist(self.scores, bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(self.scores), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.scores):.0f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 2: Tile Distribution
        # ─────────────────────────────────────────────────────────────────
        ax = axes[0, 1]
        tile_counts = defaultdict(int)
        for tile in self.max_tiles:
            tile_counts[tile] += 1
        
        tiles = sorted(tile_counts.keys())
        counts = [tile_counts[t] for t in tiles]
        
        bars = ax.bar(range(len(tiles)), counts, color='red', alpha=0.7, 
                     edgecolor='black')
        ax.set_xticks(range(len(tiles)))
        ax.set_xticklabels([str(t) for t in tiles], rotation=45)
        ax.set_xlabel('Max Tile')
        ax.set_ylabel('Games')
        ax.set_title('Highest Tile Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight 2048+ tiles
        for i, tile in enumerate(tiles):
            if tile >= 2048:
                bars[i].set_color('gold')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 3: Win Rate
        # ─────────────────────────────────────────────────────────────────
        ax = axes[1, 0]
        win_rate = np.mean(self.wins) * 100
        colors = ['green' if win_rate >= 50 else 'orange' if win_rate >= 25 else 'red', 'lightgray']
        ax.pie([win_rate, 100 - win_rate], labels=['Wins (2048+)', 'Losses'], 
               autopct='%1.1f%%', startangle=90, colors=colors)
        ax.set_title(f'Win Rate: {win_rate:.1f}%')
        
        # ─────────────────────────────────────────────────────────────────
        # Plot 4: Summary Statistics
        # ─────────────────────────────────────────────────────────────────
        ax = axes[1, 1]
        ax.axis('off')
        
        metrics = self.get_metrics()
        summary_text = f"""
EVALUATION SUMMARY

Total Games:     {metrics['num_games']}

Score Statistics:
  Average:       {metrics['avg_score']:.2f}
  Maximum:       {metrics['max_score']}
  
Tile Statistics:
  Average:       {metrics['avg_tile']:.0f}
  Maximum:       {metrics['max_tile']}
  
Win Rate:        {metrics['win_rate']:.1f}%
(2048 reached)
"""
        ax.text(0.1, 0.9, summary_text.strip(), transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(save_path)
