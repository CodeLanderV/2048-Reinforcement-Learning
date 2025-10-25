#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
2048 RL Results Plotter - Comprehensive Visualization Tool
═══════════════════════════════════════════════════════════════════════════════

Generate comprehensive plots from training logs, live session data, and 
historical evaluation results.

USAGE:
    # Generate dashboard from all historical training sessions
    python plot_training_results.py --history

    # Plot latest training session only
    python plot_training_results.py --latest
    
    # Plot specific session by index
    python plot_training_results.py --session 5
    
    # Combine multiple plots into dashboard
    python plot_training_results.py --dashboard
    
    # Custom output file
    python plot_training_results.py --history --output results/my_plot.png
    
    # High resolution for publication
    python plot_training_results.py --dashboard --dpi 300 --output paper_figure.png

FEATURES:
    - Parse evaluations/training_log.txt for historical data
    - Generate comparison plots across sessions
    - Create comprehensive training dashboards
    - Export high-quality images (PNG, PDF)
    - Support for multiple algorithms (DQN, Double DQN, MCTS)
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.plotting import TrainingAnalyzer, ResultsVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from 2048 RL training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comprehensive dashboard
  python plot_training_results.py --dashboard
  
  # Plot all historical training sessions
  python plot_training_results.py --history --output evaluations/all_sessions.png
  
  # Plot latest session only  
  python plot_training_results.py --latest
  
  # High-DPI for publication
  python plot_training_results.py --dashboard --dpi 300 --output paper_figure.pdf
        """
    )
    
    # Input source
    parser.add_argument(
        '--log-file',
        type=str,
        default='evaluations/training_log.txt',
        help='Path to training log file (default: evaluations/training_log.txt)'
    )
    
    parser.add_argument(
        '--eval-dir',
        type=str,
        default='evaluations',
        help='Directory containing evaluation files (default: evaluations)'
    )
    
    # Plot type
    plot_type = parser.add_mutually_exclusive_group()
    plot_type.add_argument(
        '--history',
        action='store_true',
        help='Plot all historical training sessions comparison'
    )
    
    plot_type.add_argument(
        '--latest',
        action='store_true',
        help='Plot only the most recent training session'
    )
    
    plot_type.add_argument(
        '--session',
        type=int,
        metavar='N',
        help='Plot specific session by index (0-based, negative for counting from end)'
    )
    
    plot_type.add_argument(
        '--dashboard',
        action='store_true',
        default=False,
        help='Generate comprehensive dashboard (default if no options specified)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (default: evaluations/training_results.png)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Image resolution DPI (150 for screen, 300 for print, default: 150)'
    )
    
    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format (default: png)'
    )
    
    # Display options
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plot interactively instead of saving'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # If no plot type specified, default to dashboard
    if not (args.history or args.latest or args.session is not None or args.dashboard):
        args.dashboard = True
    
    # Print banner
    print("=" * 80)
    print("2048 RL TRAINING RESULTS PLOTTER")
    print("=" * 80)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default output based on plot type
        if args.history:
            output_path = f"{args.eval_dir}/all_sessions_comparison.{args.format}"
        elif args.latest:
            output_path = f"{args.eval_dir}/latest_session.{args.format}"
        elif args.session is not None:
            output_path = f"{args.eval_dir}/session_{args.session}.{args.format}"
        else:  # dashboard
            output_path = f"{args.eval_dir}/training_dashboard.{args.format}"
    
    # Execute plotting
    try:
        if args.dashboard:
            print("[INFO] Generating comprehensive training dashboard...")
            visualizer = ResultsVisualizer(args.eval_dir)
            visualizer.create_dashboard(output_path)
            
        elif args.history:
            print("[INFO] Plotting all historical training sessions...")
            analyzer = TrainingAnalyzer(args.log_file)
            
            if args.show:
                analyzer.plot_all_sessions_comparison(save_path=None)
                import matplotlib.pyplot as plt
                plt.show()
            else:
                analyzer.plot_all_sessions_comparison(save_path=output_path)
                print(f"[SUCCESS] Plot saved: {output_path}")
        
        elif args.latest:
            print("[INFO] Plotting latest training session...")
            analyzer = TrainingAnalyzer(args.log_file)
            
            if not analyzer.sessions:
                print("[ERROR] No training sessions found in log file")
                return 1
            
            latest = analyzer.sessions[-1]
            print(f"[INFO] Latest session: {latest.get('timestamp', 'Unknown')}")
            print(f"       Algorithm: {latest.get('algorithm', 'Unknown')}")
            print(f"       Episodes: {latest.get('episodes', 'Unknown')}")
            print(f"       Best Score: {latest.get('best_score', 'Unknown')}")
            print(f"       Max Tile: {latest.get('max_tile', 'Unknown')}")
            
            # For now, plot all sessions (future: implement single-session plot)
            if args.show:
                analyzer.plot_all_sessions_comparison(save_path=None)
                import matplotlib.pyplot as plt
                plt.show()
            else:
                analyzer.plot_all_sessions_comparison(save_path=output_path)
                print(f"[SUCCESS] Plot saved: {output_path}")
        
        elif args.session is not None:
            print(f"[INFO] Plotting session #{args.session}...")
            analyzer = TrainingAnalyzer(args.log_file)
            
            if not analyzer.sessions:
                print("[ERROR] No training sessions found in log file")
                return 1
            
            if args.session < 0:
                session_idx = len(analyzer.sessions) + args.session
            else:
                session_idx = args.session
            
            if session_idx < 0 or session_idx >= len(analyzer.sessions):
                print(f"[ERROR] Session index {args.session} out of range (0-{len(analyzer.sessions)-1})")
                return 1
            
            session = analyzer.sessions[session_idx]
            print(f"[INFO] Session: {session.get('timestamp', 'Unknown')}")
            print(f"       Algorithm: {session.get('algorithm', 'Unknown')}")
            print(f"       Episodes: {session.get('episodes', 'Unknown')}")
            print(f"       Best Score: {session.get('best_score', 'Unknown')}")
            print(f"       Max Tile: {session.get('max_tile', 'Unknown')}")
            
            # For now, plot all sessions (future: implement single-session plot)
            if args.show:
                analyzer.plot_all_sessions_comparison(save_path=None)
                import matplotlib.pyplot as plt
                plt.show()
            else:
                analyzer.plot_all_sessions_comparison(save_path=output_path)
                print(f"[SUCCESS] Plot saved: {output_path}")
        
        print("=" * 80)
        print("PLOTTING COMPLETE")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate plots: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
