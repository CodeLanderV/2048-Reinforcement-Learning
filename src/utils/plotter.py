"""
Plotting utilities for visualizing training progress.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional
from datetime import datetime


class TrainingPlotter:
    """
    Create plots for training metrics.
    """
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize plotter.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_episode_scores(self, scores: List[int], window_size: int = 100, 
                           save: bool = True, filename: str = None):
        """
        Plot episode scores over time.
        
        Args:
            scores: List of episode scores
            window_size: Window size for moving average
            save: Whether to save the plot
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(scores) + 1)
        
        # Plot raw scores
        ax.plot(episodes, scores, alpha=0.3, color='blue', label='Raw Score')
        
        # Plot moving average
        if len(scores) >= window_size:
            moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = range(window_size, len(scores) + 1)
            ax.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2, 
                   label=f'{window_size}-Episode Moving Average')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Training Progress: Episode vs Score', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"episode_scores_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()
        return fig
    
    def plot_episode_max_tiles(self, max_tiles: List[int], window_size: int = 100,
                              save: bool = True, filename: str = None):
        """
        Plot maximum tile achieved per episode.
        
        Args:
            max_tiles: List of maximum tiles per episode
            window_size: Window size for moving average
            save: Whether to save the plot
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(max_tiles) + 1)
        
        # Plot raw max tiles
        ax.plot(episodes, max_tiles, alpha=0.3, color='green', label='Max Tile')
        
        # Plot moving average
        if len(max_tiles) >= window_size:
            moving_avg = np.convolve(max_tiles, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = range(window_size, len(max_tiles) + 1)
            ax.plot(moving_avg_episodes, moving_avg, color='orange', linewidth=2,
                   label=f'{window_size}-Episode Moving Average')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Maximum Tile', fontsize=12)
        ax.set_title('Training Progress: Episode vs Maximum Tile', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to log scale for better visualization
        ax.set_yscale('log', base=2)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"episode_max_tiles_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()
        return fig
    
    def plot_combined_metrics(self, scores: List[int], max_tiles: List[int], 
                             moves: List[int], window_size: int = 100,
                             save: bool = True, filename: str = None):
        """
        Plot multiple metrics in a combined view.
        
        Args:
            scores: List of episode scores
            max_tiles: List of maximum tiles
            moves: List of moves per episode
            window_size: Window size for moving average
            save: Whether to save the plot
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(scores) + 1)
        
        # Plot 1: Scores
        ax1 = axes[0, 0]
        ax1.plot(episodes, scores, alpha=0.3, color='blue', label='Score')
        if len(scores) >= window_size:
            moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = range(window_size, len(scores) + 1)
            ax1.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2,
                    label=f'{window_size}-Ep MA')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Episode vs Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Max Tiles
        ax2 = axes[0, 1]
        ax2.plot(episodes, max_tiles, alpha=0.3, color='green', label='Max Tile')
        if len(max_tiles) >= window_size:
            moving_avg = np.convolve(max_tiles, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = range(window_size, len(max_tiles) + 1)
            ax2.plot(moving_avg_episodes, moving_avg, color='orange', linewidth=2,
                    label=f'{window_size}-Ep MA')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Maximum Tile')
        ax2.set_title('Episode vs Maximum Tile')
        ax2.set_yscale('log', base=2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Moves
        ax3 = axes[1, 0]
        ax3.plot(episodes, moves, alpha=0.3, color='purple', label='Moves')
        if len(moves) >= window_size:
            moving_avg = np.convolve(moves, np.ones(window_size)/window_size, mode='valid')
            moving_avg_episodes = range(window_size, len(moves) + 1)
            ax3.plot(moving_avg_episodes, moving_avg, color='brown', linewidth=2,
                    label=f'{window_size}-Ep MA')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Moves per Episode')
        ax3.set_title('Episode vs Moves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Max Tile Distribution
        ax4 = axes[1, 1]
        unique_tiles, counts = np.unique(max_tiles, return_counts=True)
        ax4.bar(range(len(unique_tiles)), counts, tick_label=[int(t) for t in unique_tiles])
        ax4.set_xlabel('Maximum Tile Reached')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Maximum Tiles')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"combined_metrics_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()
        return fig
    
    def plot_loss_curve(self, losses: List[float], window_size: int = 100,
                       save: bool = True, filename: str = None):
        """
        Plot training loss over time.
        
        Args:
            losses: List of loss values
            window_size: Window size for moving average
            save: Whether to save the plot
            filename: Output filename
        """
        if not losses:
            print("No loss data to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(1, len(losses) + 1)
        
        # Plot raw losses
        ax.plot(steps, losses, alpha=0.2, color='red', label='Loss')
        
        # Plot moving average
        if len(losses) >= window_size:
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            moving_avg_steps = range(window_size, len(losses) + 1)
            ax.plot(moving_avg_steps, moving_avg, color='darkred', linewidth=2,
                   label=f'{window_size}-Step Moving Average')
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"loss_curve_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()
        return fig
    
    def create_training_summary(self, scores: List[int], max_tiles: List[int],
                               moves: List[int], losses: List[float],
                               save: bool = True, filename: str = None):
        """
        Create a comprehensive training summary with all metrics.
        
        Args:
            scores: Episode scores
            max_tiles: Episode max tiles
            moves: Episode moves
            losses: Training losses
            save: Whether to save
            filename: Output filename
        """
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        episodes = range(1, len(scores) + 1)
        window_size = min(100, len(scores) // 10) if len(scores) > 10 else 1
        
        # Plot 1: Scores
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(episodes, scores, alpha=0.3, color='blue')
        if len(scores) >= window_size:
            ma = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size, len(scores) + 1), ma, color='red', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Episode Scores')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Max Tiles
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(episodes, max_tiles, alpha=0.3, color='green')
        if len(max_tiles) >= window_size:
            ma = np.convolve(max_tiles, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size, len(max_tiles) + 1), ma, color='orange', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Max Tile')
        ax2.set_title('Maximum Tiles')
        ax2.set_yscale('log', base=2)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Moves
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(episodes, moves, alpha=0.3, color='purple')
        if len(moves) >= window_size:
            ma = np.convolve(moves, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(range(window_size, len(moves) + 1), ma, color='brown', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Moves')
        ax3.set_title('Moves per Episode')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Loss
        ax4 = fig.add_subplot(gs[1, 1])
        if losses:
            loss_window = min(100, len(losses) // 10) if len(losses) > 10 else 1
            if len(losses) >= loss_window:
                ma = np.convolve(losses, np.ones(loss_window)/loss_window, mode='valid')
                ax4.plot(range(loss_window, len(losses) + 1), ma, color='darkred', linewidth=2)
            else:
                ax4.plot(range(1, len(losses) + 1), losses, color='red')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Max Tile Distribution
        ax5 = fig.add_subplot(gs[2, 0])
        unique_tiles, counts = np.unique(max_tiles, return_counts=True)
        ax5.bar(range(len(unique_tiles)), counts, tick_label=[int(t) for t in unique_tiles])
        ax5.set_xlabel('Max Tile')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Max Tile Distribution')
        ax5.grid(True, alpha=0.3, axis='y')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 6: Statistics Summary
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        stats_text = f"""
        Training Statistics
        ═══════════════════════════
        Total Episodes: {len(scores)}
        
        Scores:
          Mean: {np.mean(scores):.2f}
          Best: {np.max(scores)}
          Std: {np.std(scores):.2f}
        
        Max Tiles:
          Mean: {np.mean(max_tiles):.2f}
          Best: {int(np.max(max_tiles))}
          Worst: {int(np.min(max_tiles))}
        
        Moves:
          Mean: {np.mean(moves):.2f}
          Best: {int(np.max(moves))}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.suptitle('Training Summary', fontsize=16, fontweight='bold')
        
        if save:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"training_summary_{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Training summary saved to {filepath}")
        
        plt.close()
        return fig
