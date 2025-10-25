"""
Training script for DQN agent on 2048 game.
Implements episodic training with hyperparameter tuning and visualization.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game.game_2048 import Game2048
from src.agent.dqn_agent import DQNAgent
from src.ui.pygame_ui import GameUI
from src.utils.logger import GameLogger
from src.utils.plotter import TrainingPlotter


def train_dqn(
    episodes: int = 1000,
    max_steps: int = 10000,
    learning_rate: float = 0.0001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_capacity: int = 100000,
    target_update_freq: int = 10,
    hidden_sizes: list = None,
    save_freq: int = 100,
    visualize: bool = True,
    viz_freq: int = 10,
    fps: int = 10,
    model_name: str = "dqn_2048"
):
    """
    Train DQN agent to play 2048.
    
    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Exploration decay rate
        batch_size: Training batch size
        buffer_capacity: Replay buffer capacity
        target_update_freq: Target network update frequency
        hidden_sizes: Hidden layer sizes
        save_freq: Model save frequency
        visualize: Whether to visualize training
        viz_freq: Visualization frequency (every N episodes)
        fps: Frames per second for visualization
        model_name: Name for saved model
    """
    print("=" * 70)
    print("2048 DQN TRAINING")
    print("=" * 70)
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {episodes}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
    print(f"  Batch size: {batch_size}")
    print(f"  Buffer capacity: {buffer_capacity}")
    print(f"  Target update frequency: {target_update_freq}")
    print(f"  Hidden layers: {hidden_sizes}")
    print(f"  Visualization: {'Enabled' if visualize else 'Disabled'}")
    print("=" * 70)
    print()
    
    # Set default hidden sizes
    if hidden_sizes is None:
        hidden_sizes = [256, 256, 128]
    
    # Initialize components
    game = Game2048()
    agent = DQNAgent(
        state_size=16,
        action_size=4,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    logger = GameLogger()
    plotter = TrainingPlotter()
    
    # Initialize UI if visualizing
    ui = None
    if visualize:
        try:
            ui = GameUI()
        except Exception as e:
            print(f"Warning: Could not initialize UI: {e}")
            print("Continuing without visualization...")
            visualize = False
    
    # Training loop
    print("Starting training...\n")
    best_score = 0
    best_max_tile = 0
    
    try:
        for episode in range(1, episodes + 1):
            # Reset environment
            state = game.reset()
            logger.start_episode(episode, mode='training')
            
            episode_reward = 0
            step = 0
            done = False
            
            # Determine if we should visualize this episode
            show_viz = visualize and (episode % viz_freq == 0 or episode == 1)
            
            # Episode loop
            while not done and step < max_steps:
                # Get valid actions
                valid_actions = game.get_valid_moves()
                
                if not valid_actions:
                    done = True
                    break
                
                # Select action
                action = agent.select_action(state, valid_actions, training=True)
                
                # Take step
                next_state, reward, done, info = game.step(action)
                
                # Store experience
                agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                loss = agent.train_step()
                
                # Log step
                logger.log_step(action, reward, state, game.score, game.max_tile)
                
                # Update state
                state = next_state
                episode_reward += reward
                step += 1
                
                # Visualize if enabled
                if show_viz and ui is not None:
                    should_continue = ui.update(
                        game.board,
                        game.score,
                        game.max_tile,
                        game.moves_count,
                        episode,
                        mode="Training",
                        fps=fps
                    )
                    if not should_continue:
                        print("\nVisualization closed. Continuing training without UI...")
                        visualize = False
                        ui = None
            
            # End episode
            logger.end_episode(
                game.score,
                game.max_tile,
                additional_info={
                    'epsilon': agent.epsilon,
                    'buffer_size': len(agent.replay_buffer),
                    'total_reward': episode_reward,
                    'steps': step
                }
            )
            
            # Update best scores
            if game.score > best_score:
                best_score = game.score
            if game.max_tile > best_max_tile:
                best_max_tile = game.max_tile
            
            # Print progress
            stats = logger.get_statistics()
            agent_stats = agent.get_stats()
            
            print(f"Episode {episode}/{episodes} | "
                  f"Score: {game.score:5d} | "
                  f"Max Tile: {game.max_tile:5d} | "
                  f"Moves: {game.moves_count:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Eps: {agent_stats['epsilon']:.3f} | "
                  f"Avg Score (100): {stats.get('avg_score', 0):.2f}")
            
            # Save model periodically
            if episode % save_freq == 0:
                model_path = os.path.join("saved_models", f"{model_name}_ep{episode}.pth")
                agent.save(model_path)
                
                # Save logs and plots
                logger.save_training_log(f"training_log_ep{episode}.json")
                logger.save_statistics(f"statistics_ep{episode}.json")
                
                # Create plots
                if len(logger.episode_scores) > 10:
                    plotter.plot_episode_scores(logger.episode_scores, 
                                               filename=f"scores_ep{episode}.png")
                    plotter.plot_episode_max_tiles(logger.episode_max_tiles,
                                                  filename=f"max_tiles_ep{episode}.png")
                    plotter.plot_combined_metrics(
                        logger.episode_scores,
                        logger.episode_max_tiles,
                        logger.episode_moves,
                        filename=f"combined_ep{episode}.png"
                    )
                    if agent.losses:
                        plotter.plot_loss_curve(agent.losses,
                                              filename=f"loss_ep{episode}.png")
                
                print(f"\n  ✓ Checkpoint saved at episode {episode}")
                print(f"    Best Score: {best_score} | Best Max Tile: {best_max_tile}\n")
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED!")
        print("=" * 70)
        print(f"\nFinal Statistics:")
        print(f"  Episodes trained: {episodes}")
        print(f"  Best score: {best_score}")
        print(f"  Best max tile: {best_max_tile}")
        print(f"  Average score (last 100): {stats.get('avg_score', 0):.2f}")
        print(f"  Average max tile (last 100): {stats.get('avg_max_tile', 0):.2f}")
        print()
        
        # Save final model and results
        final_model_path = os.path.join("saved_models", f"{model_name}_final.pth")
        agent.save(final_model_path)
        
        logger.save_training_log("training_log_final.json")
        logger.save_statistics("statistics_final.json")
        
        # Create final plots
        if len(logger.episode_scores) > 10:
            plotter.create_training_summary(
                logger.episode_scores,
                logger.episode_max_tiles,
                logger.episode_moves,
                agent.losses,
                filename="training_summary_final.png"
            )
        
        print(f"Final model saved to {final_model_path}")
        print("All logs and plots have been saved.")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print(f"Saving model at episode {episode}...")
        
        # Save current state
        interrupt_path = os.path.join("saved_models", f"{model_name}_interrupted_ep{episode}.pth")
        agent.save(interrupt_path)
        logger.save_training_log(f"training_log_interrupted_ep{episode}.json")
        logger.save_statistics(f"statistics_interrupted_ep{episode}.json")
        
        print(f"Model saved to {interrupt_path}")
        
    finally:
        # Clean up
        if ui is not None:
            ui.close()


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train DQN agent on 2048 game')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum steps per episode (default: 10000)')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Save model every N episodes (default: 100)')
    
    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                       help='Initial exploration rate (default: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                       help='Final exploration rate (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Exploration decay rate (default: 0.995)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--buffer-capacity', type=int, default=100000,
                       help='Replay buffer capacity (default: 100000)')
    parser.add_argument('--target-update', type=int, default=10,
                       help='Target network update frequency (default: 10)')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 256, 128],
                       help='Hidden layer sizes (default: 256 256 128)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization during training')
    parser.add_argument('--viz-freq', type=int, default=10,
                       help='Visualize every N episodes (default: 10)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for visualization (default: 10)')
    
    # Model
    parser.add_argument('--model-name', type=str, default='dqn_2048',
                       help='Name for saved model (default: dqn_2048)')
    
    args = parser.parse_args()
    
    # Run training
    train_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update,
        hidden_sizes=args.hidden_sizes,
        save_freq=args.save_freq,
        visualize=args.visualize,
        viz_freq=args.viz_freq,
        fps=args.fps,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
