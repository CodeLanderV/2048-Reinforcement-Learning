"""Main launcher for training 2048 agents - now uses src/ structure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a 2048 RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py dqn              # Train vanilla DQN
  python train.py double-dqn       # Train Double DQN  
  python train.py policy-gradient  # Train Policy Gradient (REINFORCE)
  python train.py mcts             # Run MCTS simulations

All algorithms launch with UI and live plotting enabled by default.
Close the plot window to stop training early.
        """,
    )
    parser.add_argument(
        "algorithm",
        choices=["dqn", "double-dqn", "policy-gradient", "mcts"],
        help="Algorithm to train",
    )
    parser.add_argument("--episodes", type=int, help="Number of episodes (default varies by algorithm)")
    args = parser.parse_args()

    print(f"Launching {args.algorithm.upper()} training...")
    print()

    if args.algorithm == "dqn":
        from scripts.train_dqn import train_dqn
        train_dqn(episodes=args.episodes or 2000)
    
    elif args.algorithm == "double-dqn":
        print("Double DQN training script not yet created in src/ structure")
        print("Use: python 2048RL.py train (and set algorithm to 'double-dqn' in CONFIG)")
    
    elif args.algorithm == "policy-gradient":
        print("Policy Gradient training script not yet created in src/ structure")
        print("Use: python 2048RL.py train (and set algorithm to 'policy-gradient' in CONFIG)")
    
    elif args.algorithm == "mcts":
        print("MCTS training script not yet created in src/ structure")
        print("Use: python 2048RL.py train (and set algorithm to 'mcts' in CONFIG)")


if __name__ == "__main__":
    main()
