"""
═══════════════════════════════════════════════════════════════════════════════
Post-Training Plot Generator
═══════════════════════════════════════════════════════════════════════════════

Generates comprehensive training plots from saved metrics JSON files.

USAGE:
    # Command line
    python -m src.plot_from_logs evaluations/DQN_metrics_20251026_123456.json
    
    # Python
    from src.plot_from_logs import plot_training_metrics
    plot_training_metrics("evaluations/DQN_metrics.json", output="plots/training.png")
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List


def calculate_moving_average(data: List[float], window: int = 100) -> np.ndarray:
    """Calculate moving average with given window size."""
    if len(data) < window:
        return np.array([])
    
    data_array = np.array(data)
    cumsum = np.cumsum(data_array)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    return cumsum[window - 1:] / window


def plot_training_metrics(
    metrics_file: str,
    output_file: Optional[str] = None,
    ma_window: int = 100,
    figsize: tuple = (16, 12),
    dpi: int = 150
) -> str:
    """
    Generate comprehensive training plots from metrics JSON.
    
    Args:
        metrics_file: Path to metrics JSON file
        output_file: Output PNG path (default: auto-generated)
        ma_window: Moving average window size
        figsize: Figure size (width, height)
        dpi: Resolution in dots per inch
        
    Returns:
        Path to saved plot
    """
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    metrics = data['metrics']
    
    algo_name = metadata['algorithm']
    
    # Extract data
    episodes = np.array(metrics['episodes'])
    scores = np.array(metrics['scores'])
    max_tiles = np.array(metrics['max_tiles'])
    rewards = np.array(metrics['rewards'])
    steps = np.array(metrics['steps'])
    epsilons = np.array(metrics['epsilons'])
    losses = [l for l in metrics['losses'] if l is not None]
    
    # Calculate moving averages
    score_ma = calculate_moving_average(scores.tolist(), ma_window)
    reward_ma = calculate_moving_average(rewards.tolist(), ma_window)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f'{algo_name} Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode vs Max Score
    ax = axes[0, 0]
    ax.scatter(episodes, scores, alpha=0.3, s=10, color='blue', label='Raw Score')
    if len(score_ma) > 0:
        ma_episodes = episodes[ma_window-1:]
        ax.plot(ma_episodes, score_ma, color='red', linewidth=2, label=f'MA-{ma_window}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Score per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Episode vs Max Tile
    ax = axes[0, 1]
    ax.scatter(episodes, max_tiles, alpha=0.4, s=10, color='green')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Max Tile')
    ax.set_title('Highest Tile per Episode')
    ax.set_yscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Episode vs Reward
    ax = axes[1, 0]
    ax.scatter(episodes, rewards, alpha=0.3, s=10, color='purple', label='Raw Reward')
    if len(reward_ma) > 0:
        ma_episodes = episodes[ma_window-1:]
        ax.plot(ma_episodes, reward_ma, color='red', linewidth=2, label=f'MA-{ma_window}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Episode vs Steps
    ax = axes[1, 1]
    ax.scatter(episodes, steps, alpha=0.3, s=10, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Training Loss (if available)
    ax = axes[2, 0]
    if losses:
        loss_episodes = [episodes[i] for i in range(len(episodes)) if metrics['losses'][i] is not None]
        ax.plot(loss_episodes, losses, color='red', linewidth=1, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training Loss')
    
    # Plot 6: Epsilon Decay
    ax = axes[2, 1]
    ax.plot(episodes, epsilons, color='blue', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (ε)')
    ax.set_title('Exploration Rate (Epsilon) Decay')
    ax.grid(True, alpha=0.3)
    
    # Add summary text
    summary_text = f"""
Training Summary:
Episodes: {len(episodes)}
Best Score: {metadata.get('best_score', max(scores))}
Best Tile: {metadata.get('best_tile', max(max_tiles))}
Avg Score: {np.mean(scores):.1f}
Final ε: {epsilons[-1]:.4f}
"""
    fig.text(0.02, 0.02, summary_text.strip(), 
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        metrics_path = Path(metrics_file)
        output_file = str(metrics_path.parent / f"{metrics_path.stem}_plot.png")
    
    Path(output_file).parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file


def plot_comparison(
    metrics_files: List[str],
    output_file: str = "evaluations/algorithm_comparison.png",
    ma_window: int = 100,
    figsize: tuple = (14, 8),
    dpi: int = 150
) -> str:
    """
    Compare multiple training runs on the same plots.
    
    Args:
        metrics_files: List of paths to metrics JSON files
        output_file: Output PNG path
        ma_window: Moving average window size
        figsize: Figure size (width, height)
        dpi: Resolution in dots per inch
        
    Returns:
        Path to saved comparison plot
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    for idx, metrics_file in enumerate(metrics_files):
        # Load metrics
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        algo_name = data['metadata']['algorithm']
        metrics = data['metrics']
        
        episodes = np.array(metrics['episodes'])
        scores = np.array(metrics['scores'])
        max_tiles = np.array(metrics['max_tiles'])
        rewards = np.array(metrics['rewards'])
        
        color = colors[idx % len(colors)]
        
        # Score comparison
        score_ma = calculate_moving_average(scores.tolist(), ma_window)
        if len(score_ma) > 0:
            ma_episodes = episodes[ma_window-1:]
            axes[0, 0].plot(ma_episodes, score_ma, color=color, 
                          linewidth=2, label=algo_name, alpha=0.8)
        
        # Max tile comparison
        axes[0, 1].scatter(episodes, max_tiles, alpha=0.3, s=5, 
                          color=color, label=algo_name)
        
        # Reward comparison
        reward_ma = calculate_moving_average(rewards.tolist(), ma_window)
        if len(reward_ma) > 0:
            ma_episodes = episodes[ma_window-1:]
            axes[1, 0].plot(ma_episodes, reward_ma, color=color,
                          linewidth=2, label=algo_name, alpha=0.8)
        
        # Best scores histogram
        axes[1, 1].hist(scores, bins=30, alpha=0.5, color=color, label=algo_name)
    
    # Configure plots
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title(f'Score Comparison (MA-{ma_window})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Max Tile')
    axes[0, 1].set_title('Max Tile Distribution')
    axes[0, 1].set_yscale('log', base=2)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title(f'Reward Comparison (MA-{ma_window})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.plot_from_logs <metrics_file.json> [output_file.png]")
        sys.exit(1)
    
    metrics_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    output = plot_training_metrics(metrics_file, output_file)
    print(f"Plot saved to: {output}")
