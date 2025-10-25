"""
═══════════════════════════════════════════════════════════════════════════════
Training and Testing Results Plotter
═══════════════════════════════════════════════════════════════════════════════

This script parses training.txt and testing.txt log files to generate
comprehensive performance plots.

USAGE:
    python plot_results.py                    # Plot both training and testing
    python plot_results.py --training-only    # Plot only training results
    python plot_results.py --testing-only     # Plot only testing results
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def parse_training_log(log_path="evaluations/training.txt"):
    """
    Parse training.txt to extract episode metrics.
    
    Returns:
        dict: {
            'episodes': list of episode numbers,
            'scores': list of scores (avg of last 50),
            'ma_100': list of MA-100 values,
            'max_tiles': list of max tiles achieved,
            'best_scores': list of best scores achieved,
            'convergence_metric': list of episodes_since_improvement
        }
    """
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"[WARNING] Training log not found: {log_path}")
        return None
    
    data = {
        'episodes': [],
        'scores': [],
        'ma_100': [],
        'max_tiles': [],
        'best_scores': [],
        'convergence_metric': [],
        'rewards': [],
        'epsilon': []
    }
    
    # Patterns to match
    # Ep 2620 | Reward: 1784.21 | Score:   1386 | MA-100:   1362 (best   1444, Δ= -82.1) | Tile:  128 | ε: 0.250 | No-Imp: 694 | Time: 3:44:48
    episode_pattern = re.compile(
        r'Ep\s+(\d+)\s+\|\s+Reward:\s+([\d.]+)\s+\|\s+Score:\s+([\d.]+)\s+\|\s+MA-100:\s+([\d.]+)\s+\(best\s+([\d.]+).*?\)\s+\|\s+Tile:\s+(\d+)\s+\|\s+ε:\s+([\d.]+)(?:\s+\|\s+No-Imp:\s+(\d+))?'
    )
    
    # [NEW BEST] Ep  899 | New best score: 4708 | Tile: 512
    best_pattern = re.compile(r'\[NEW BEST\]\s+Ep\s+(\d+)\s+\|\s+New best score:\s+(\d+)\s+\|\s+Tile:\s+(\d+)')
    
    print(f"[INFO] Parsing training log: {log_path}")
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match regular episode line
            match = episode_pattern.search(line)
            if match:
                episode = int(match.group(1))
                reward = float(match.group(2))
                score = float(match.group(3))
                ma_100 = float(match.group(4))
                best_ma = float(match.group(5))
                tile = int(match.group(6))
                epsilon = float(match.group(7))
                no_imp = int(match.group(8)) if match.group(8) else 0
                
                data['episodes'].append(episode)
                data['scores'].append(score)
                data['ma_100'].append(ma_100)
                data['max_tiles'].append(tile)
                data['convergence_metric'].append(no_imp)
                data['rewards'].append(reward)
                data['epsilon'].append(epsilon)
            
            # Match NEW BEST lines
            best_match = best_pattern.search(line)
            if best_match:
                episode = int(best_match.group(1))
                best_score = int(best_match.group(2))
                # Store as tuple for later plotting
                if not hasattr(data, 'best_score_events'):
                    data['best_score_events'] = []
                data['best_score_events'].append((episode, best_score))
    
    print(f"[INFO] Parsed {len(data['episodes'])} training episodes")
    return data if data['episodes'] else None


def parse_testing_log(log_path="evaluations/testing.txt"):
    """
    Parse testing.txt to extract attempt metrics.
    
    Returns:
        dict: {
            'attempts': list of attempt numbers,
            'scores': list of scores,
            'max_tiles': list of max tiles achieved,
            'avg_scores': list of cumulative average scores
        }
    """
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"[WARNING] Testing log not found: {log_path}")
        return None
    
    data = {
        'attempts': [],
        'scores': [],
        'max_tiles': [],
        'avg_scores': []
    }
    
    # Attempt #1 Result:
    #   Score: 1580
    #   Max Tile: 128
    #   Average Score: 1580.0
    attempt_pattern = re.compile(r'Attempt #(\d+) Result:')
    score_pattern = re.compile(r'Score:\s+(\d+)')
    max_tile_pattern = re.compile(r'Max Tile:\s+(\d+)')
    avg_score_pattern = re.compile(r'Average Score:\s+([\d.]+)')
    
    print(f"[INFO] Parsing testing log: {log_path}")
    
    current_attempt = None
    current_score = None
    current_tile = None
    current_avg = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for attempt number
            attempt_match = attempt_pattern.search(line)
            if attempt_match:
                current_attempt = int(attempt_match.group(1))
                continue
            
            # Check for score
            score_match = score_pattern.search(line)
            if score_match and current_attempt:
                current_score = int(score_match.group(1))
                continue
            
            # Check for max tile
            tile_match = max_tile_pattern.search(line)
            if tile_match and current_attempt:
                current_tile = int(tile_match.group(1))
                continue
            
            # Check for average score
            avg_match = avg_score_pattern.search(line)
            if avg_match and current_attempt:
                current_avg = float(avg_match.group(1))
                
                # We have all data for this attempt
                data['attempts'].append(current_attempt)
                data['scores'].append(current_score)
                data['max_tiles'].append(current_tile)
                data['avg_scores'].append(current_avg)
                
                # Reset
                current_attempt = None
                current_score = None
                current_tile = None
                current_avg = None
    
    print(f"[INFO] Parsed {len(data['attempts'])} testing attempts")
    return data if data['attempts'] else None


def plot_training_results(data, save_path="evaluations/training_plots.png"):
    """
    Create comprehensive training plots.
    
    Creates 4 subplots:
    1. Episode vs Max Tile Achieved
    2. Episode vs Best Score
    3. Episode vs Average Score (with MA-100)
    4. Episode vs Convergence Metric (No-Improvement count)
    """
    if not data:
        print("[WARNING] No training data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Performance Analysis', fontsize=16, fontweight='bold')
    
    episodes = data['episodes']
    
    # Plot 1: Max Tile Achieved
    ax1 = axes[0, 0]
    ax1.scatter(episodes, data['max_tiles'], alpha=0.5, s=20, c='blue', label='Max Tile')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Max Tile Achieved', fontsize=12)
    ax1.set_title('Episode vs Max Tile Achieved', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add horizontal lines for key milestones
    for tile in [128, 256, 512, 1024, 2048]:
        ax1.axhline(y=tile, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
        ax1.text(max(episodes)*0.02, tile, f'{tile}', fontsize=8, color='red')
    
    # Plot 2: Best Score Over Time
    ax2 = axes[0, 1]
    if hasattr(data, 'best_score_events') and data['best_score_events']:
        best_eps, best_scores = zip(*data['best_score_events'])
        ax2.plot(best_eps, best_scores, marker='o', linestyle='-', color='green', 
                linewidth=2, markersize=6, label='Best Score Events')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Best Score', fontsize=12)
    ax2.set_title('Episode vs Best Score Achieved', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Average Score and MA-100
    ax3 = axes[1, 0]
    ax3.plot(episodes, data['scores'], alpha=0.3, color='blue', linewidth=1, label='Avg Score (Last 50)')
    ax3.plot(episodes, data['ma_100'], color='red', linewidth=2.5, label='MA-100 (Convergence Metric)')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Episode vs Average Score (with MA-100)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Highlight best MA-100
    if data['ma_100']:
        best_ma = max(data['ma_100'])
        best_ma_idx = data['ma_100'].index(best_ma)
        best_ma_ep = episodes[best_ma_idx]
        ax3.scatter([best_ma_ep], [best_ma], color='gold', s=200, marker='*', 
                   zorder=5, label=f'Best MA-100: {best_ma:.0f}')
        ax3.legend()
    
    # Plot 4: Convergence Metric (No-Improvement Episodes)
    ax4 = axes[1, 1]
    ax4.plot(episodes, data['convergence_metric'], color='orange', linewidth=2, label='Episodes Since Improvement')
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Episodes Without Improvement', fontsize=12)
    ax4.set_title('Convergence Metric (Patience Counter)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1000, color='red', linestyle='--', linewidth=2, label='Convergence Threshold (1000)')
    ax4.legend()
    ax4.fill_between(episodes, 0, data['convergence_metric'], alpha=0.2, color='orange')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Training plots saved to: {save_path}")
    
    plt.show()


def plot_testing_results(data, save_path="evaluations/testing_plots.png"):
    """
    Create testing/challenge mode plots.
    
    Creates 2 subplots:
    1. Attempt vs Max Tile Achieved
    2. Attempt vs Score (with moving average)
    """
    if not data:
        print("[WARNING] No testing data to plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Testing/Challenge Mode Performance', fontsize=16, fontweight='bold')
    
    attempts = data['attempts']
    
    # Plot 1: Max Tile per Attempt
    ax1 = axes[0]
    ax1.plot(attempts, data['max_tiles'], marker='o', linestyle='-', color='purple', 
            linewidth=2, markersize=6, label='Max Tile')
    ax1.set_xlabel('Attempt Number', fontsize=12)
    ax1.set_ylabel('Max Tile Achieved', fontsize=12)
    ax1.set_title('Attempt vs Max Tile Achieved', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add horizontal lines for milestones
    for tile in [128, 256, 512, 1024, 2048]:
        if any(t >= tile for t in data['max_tiles']):
            ax1.axhline(y=tile, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Highlight best tile
    best_tile = max(data['max_tiles'])
    best_tile_idx = data['max_tiles'].index(best_tile)
    best_tile_attempt = attempts[best_tile_idx]
    ax1.scatter([best_tile_attempt], [best_tile], color='gold', s=300, marker='*', 
               zorder=5, label=f'Best: {best_tile}')
    ax1.legend()
    
    # Plot 2: Score Progress
    ax2 = axes[1]
    ax2.plot(attempts, data['scores'], marker='o', linestyle='-', color='blue', 
            linewidth=2, markersize=5, alpha=0.6, label='Score per Attempt')
    ax2.plot(attempts, data['avg_scores'], color='red', linewidth=3, 
            label='Cumulative Average', linestyle='--')
    ax2.set_xlabel('Attempt Number', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Attempt vs Score (with Average)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.fill_between(attempts, 0, data['scores'], alpha=0.2, color='blue')
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[SAVE] Testing plots saved to: {save_path}")
    
    plt.show()


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description="Plot training and testing results from log files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot both training and testing results
  python plot_results.py
  
  # Plot only training results
  python plot_results.py --training-only
  
  # Plot only testing results
  python plot_results.py --testing-only
  
  # Specify custom log paths
  python plot_results.py --training-log evaluations/training.txt
        """
    )
    
    parser.add_argument('--training-only', action='store_true',
                       help='Plot only training results')
    parser.add_argument('--testing-only', action='store_true',
                       help='Plot only testing results')
    parser.add_argument('--training-log', type=str, default='evaluations/training.txt',
                       help='Path to training log file')
    parser.add_argument('--testing-log', type=str, default='evaluations/testing.txt',
                       help='Path to testing log file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING & TESTING RESULTS PLOTTER")
    print("=" * 80)
    
    # Plot training results
    if not args.testing_only:
        print("\n[INFO] Generating training plots...")
        training_data = parse_training_log(args.training_log)
        if training_data:
            plot_training_results(training_data)
        else:
            print("[WARNING] No training data available to plot")
    
    # Plot testing results
    if not args.training_only:
        print("\n[INFO] Generating testing plots...")
        testing_data = parse_testing_log(args.testing_log)
        if testing_data:
            plot_testing_results(testing_data)
        else:
            print("[WARNING] No testing data available to plot")
    
    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
