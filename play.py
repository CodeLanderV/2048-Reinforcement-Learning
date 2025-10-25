"""
Play script for testing trained DQN agent on 2048 game.
Allows testing the model and manual play modes.
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game.game_2048 import Game2048
from src.agent.dqn_agent import DQNAgent
from src.ui.pygame_ui import GameUI, ManualGameUI
from src.utils.logger import GameLogger, save_game_state
from src.utils.plotter import TrainingPlotter


def play_agent(
    model_path: str,
    episodes: int = 10,
    max_steps: int = 10000,
    fps: int = 5,
    save_logs: bool = True
):
    """
    Play 2048 using trained DQN agent.
    
    Args:
        model_path: Path to trained model
        episodes: Number of episodes to play
        max_steps: Maximum steps per episode
        fps: Frames per second for visualization
        save_logs: Whether to save play logs
    """
    print("=" * 70)
    print("2048 DQN AGENT PLAY MODE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Episodes: {episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  FPS: {fps}")
    print("=" * 70)
    print()
    
    # Initialize components
    game = Game2048()
    agent = DQNAgent(
        state_size=16,
        action_size=4,
        epsilon_start=0.0,  # No exploration during testing
        epsilon_end=0.0
    )
    
    # Load trained model
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"✓ Model loaded successfully\n")
    else:
        print(f"✗ Model not found at {model_path}")
        print("Please train a model first using train.py")
        return
    
    # Initialize logger and UI
    logger = GameLogger()
    ui = GameUI()
    
    # Statistics
    all_scores = []
    all_max_tiles = []
    all_moves = []
    
    try:
        for episode in range(1, episodes + 1):
            print(f"\nEpisode {episode}/{episodes}")
            print("-" * 50)
            
            # Reset game
            state = game.reset()
            logger.start_episode(episode, mode='playing')
            
            done = False
            step = 0
            
            # Episode loop
            while not done and step < max_steps:
                # Get valid actions
                valid_actions = game.get_valid_moves()
                
                if not valid_actions:
                    done = True
                    break
                
                # Select action (greedy, no exploration)
                action = agent.select_action(state, valid_actions, training=False)
                
                # Take step
                next_state, reward, done, info = game.step(action)
                
                # Log step
                logger.log_step(action, reward, state, game.score, game.max_tile)
                
                # Update state
                state = next_state
                step += 1
                
                # Visualize
                should_continue = ui.update(
                    game.board,
                    game.score,
                    game.max_tile,
                    game.moves_count,
                    episode,
                    mode="Playing",
                    fps=fps
                )
                
                if not should_continue:
                    print("\nVisualization closed.")
                    return
            
            # End episode
            logger.end_episode(game.score, game.max_tile)
            
            # Store statistics
            all_scores.append(game.score)
            all_max_tiles.append(game.max_tile)
            all_moves.append(game.moves_count)
            
            # Show game over
            ui.show_game_over(game.score, game.max_tile)
            ui.wait_for_key()
            
            # Print episode results
            print(f"  Score: {game.score}")
            print(f"  Max Tile: {game.max_tile}")
            print(f"  Moves: {game.moves_count}")
        
        # Final statistics
        print("\n" + "=" * 70)
        print("PLAY SESSION COMPLETED")
        print("=" * 70)
        print(f"\nStatistics over {episodes} episodes:")
        print(f"  Average Score: {np.mean(all_scores):.2f}")
        print(f"  Best Score: {np.max(all_scores)}")
        print(f"  Average Max Tile: {np.mean(all_max_tiles):.2f}")
        print(f"  Best Max Tile: {int(np.max(all_max_tiles))}")
        print(f"  Average Moves: {np.mean(all_moves):.2f}")
        print("=" * 70)
        
        # Save logs
        if save_logs:
            logger.save_playing_log()
            print(f"\n✓ Play logs saved")
        
    except KeyboardInterrupt:
        print("\n\nPlay session interrupted by user!")
        
    finally:
        ui.close()


def play_manual():
    """
    Play 2048 manually with keyboard controls.
    """
    print("=" * 70)
    print("2048 MANUAL PLAY MODE")
    print("=" * 70)
    print("\nControls:")
    print("  Arrow Keys / WASD: Move tiles")
    print("  ESC / Q: Quit game")
    print("=" * 70)
    print()
    
    # Initialize game and UI
    game = Game2048()
    ui = ManualGameUI()
    logger = GameLogger()
    
    episode = 1
    logger.start_episode(episode, mode='playing')
    
    try:
        # Game loop
        while True:
            # Display current state
            ui.update(
                game.board,
                game.score,
                game.max_tile,
                game.moves_count,
                mode="Manual Play",
                fps=30
            )
            
            # Check if game is over
            if game.is_game_over():
                ui.show_game_over(game.score, game.max_tile)
                print(f"\nGame Over!")
                print(f"  Final Score: {game.score}")
                print(f"  Max Tile: {game.max_tile}")
                print(f"  Moves: {game.moves_count}")
                
                # Save game state
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_game_state(
                    {
                        'board': game.board,
                        'score': game.score,
                        'max_tile': game.max_tile,
                        'moves': game.moves_count
                    },
                    f"manual_game_{timestamp}.json"
                )
                
                ui.wait_for_key()
                break
            
            # Get action from user
            action = ui.get_action_from_key()
            
            if action is None:
                # User wants to quit
                print("\nGame quit by user.")
                break
            
            # Take action
            state = game.get_state()
            next_state, reward, done, info = game.step(action)
            
            # Log if move was valid
            if info['valid_move']:
                logger.log_step(action, reward, state, game.score, game.max_tile)
        
        # End episode and save
        logger.end_episode(game.score, game.max_tile)
        logger.save_playing_log()
        
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user!")
        
    finally:
        ui.close()


def watch_random_agent(episodes: int = 5, fps: int = 5):
    """
    Watch a random agent play (useful for testing).
    
    Args:
        episodes: Number of episodes
        fps: Frames per second
    """
    print("=" * 70)
    print("2048 RANDOM AGENT (for testing)")
    print("=" * 70)
    print()
    
    game = Game2048()
    ui = GameUI()
    
    try:
        for episode in range(1, episodes + 1):
            print(f"Episode {episode}/{episodes}")
            
            game.reset()
            done = False
            
            while not done:
                valid_actions = game.get_valid_moves()
                
                if not valid_actions:
                    break
                
                action = np.random.choice(valid_actions)
                _, _, done, _ = game.step(action)
                
                should_continue = ui.update(
                    game.board,
                    game.score,
                    game.max_tile,
                    game.moves_count,
                    episode,
                    mode="Random Agent",
                    fps=fps
                )
                
                if not should_continue:
                    return
            
            ui.show_game_over(game.score, game.max_tile)
            print(f"  Score: {game.score}, Max Tile: {game.max_tile}")
            ui.wait_for_key()
    
    finally:
        ui.close()


def main():
    """Main entry point for play script."""
    parser = argparse.ArgumentParser(description='Play 2048 with trained DQN agent or manually')
    
    parser.add_argument('--mode', type=str, default='agent',
                       choices=['agent', 'manual', 'random'],
                       help='Play mode: agent (trained DQN), manual (keyboard), or random')
    parser.add_argument('--model', type=str, default='saved_models/dqn_2048_final.pth',
                       help='Path to trained model (for agent mode)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to play (for agent/random mode)')
    parser.add_argument('--fps', type=int, default=5,
                       help='Frames per second for visualization')
    parser.add_argument('--max-steps', type=int, default=10000,
                       help='Maximum steps per episode')
    parser.add_argument('--no-save-logs', action='store_true',
                       help='Do not save play logs')
    
    args = parser.parse_args()
    
    if args.mode == 'agent':
        play_agent(
            model_path=args.model,
            episodes=args.episodes,
            max_steps=args.max_steps,
            fps=args.fps,
            save_logs=not args.no_save_logs
        )
    elif args.mode == 'manual':
        play_manual()
    elif args.mode == 'random':
        watch_random_agent(episodes=args.episodes, fps=args.fps)


if __name__ == "__main__":
    main()
