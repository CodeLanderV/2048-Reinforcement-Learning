"""
═══════════════════════════════════════════════════════════════════════════════
2048 Reinforcement Learning - Central Control Panel
═══════════════════════════════════════════════════════════════════════════════

This file provides a simple interface to train and evaluate different RL
algorithms for playing 2048. Just modify CONFIG and run!

QUICK START:
    python 2048RL.py train --algorithm dqn --episodes 2000
    python 2048RL.py play --model models/DQN/dqn_final.pth

ALGORITHMS AVAILABLE:
    - DQN:         Deep Q-Network (value-based, off-policy)
    - Double-DQN:  Reduces Q-value overestimation
    - MCTS:        Monte Carlo Tree Search (planning, no learning)
    - Policy-Grad: Direct policy optimization (not yet implemented)
"""

import sys
import warnings
from pathlib import Path

# Setup Python path and suppress numpy warnings
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Modify these to tune training behavior
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ─────────────────────────────────────────────────────────────────────
    # General Training Settings
    # ─────────────────────────────────────────────────────────────────────
    "algorithm": "dqn",         # Which algorithm: "dqn", "double-dqn", "mcts"
    "episodes": 2000,           # How many games to train on
    "enable_ui": True,          # Show pygame window? (slower but fun to watch)
    "enable_plots": True,       # Show live training graphs?
    
    # ─────────────────────────────────────────────────────────────────────
    # DQN Hyperparameters (Standard Deep Q-Network)
    # ─────────────────────────────────────────────────────────────────────
    # These settings are research-proven and optimized for 2048
    "dqn": {
        # Neural network training
        "learning_rate": 1e-4,          # How fast model learns (Adam optimizer)
        "gamma": 0.99,                  # Discount factor for future rewards
        "batch_size": 128,              # Samples per training step
        "gradient_clip": 5.0,           # Prevents gradient explosion
        "hidden_dims": (256, 256),      # Neural network architecture
        
        # Exploration schedule (ε-greedy)
        "epsilon_start": 1.0,           # Start: 100% random actions (explore)
        "epsilon_end": 0.1,             # End: 10% random actions (exploit learned policy)
        "epsilon_decay": 100000,        # Steps to decay from start→end
        
        # Experience replay & stability
        "replay_buffer_size": 100_000,  # How many past experiences to remember
        "target_update_interval": 1000, # Update target network every N steps
    },
    
    # ─────────────────────────────────────────────────────────────────────
    # Double DQN Hyperparameters (Reduces Q-value overestimation bias)
    # ─────────────────────────────────────────────────────────────────────
    # More exploration since Double DQN is inherently more stable
    "double_dqn": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "gradient_clip": 5.0,
        "hidden_dims": (256, 256),
        
        # INCREASED exploration vs standard DQN
        "epsilon_start": 1.0,
        "epsilon_end": 0.15,            # Keep 15% randomness (vs DQN's 10%)
        "epsilon_decay": 120000,        # Slower decay (vs DQN's 100k)
        
        "replay_buffer_size": 100_000,
        "target_update_interval": 1000,
    },
    
    # ─────────────────────────────────────────────────────────────────────
    # MCTS Hyperparameters (Monte Carlo Tree Search - Pure Planning)
    # ─────────────────────────────────────────────────────────────────────
    "mcts": {
        "simulations": 100,             # Tree search simulations per move
        "exploration_constant": 1.41,   # UCB exploration (√2 is theoretically optimal)
    },
    
    # ─────────────────────────────────────────────────────────────────────
    # Environment & Saving
    # ─────────────────────────────────────────────────────────────────────
    "invalid_move_penalty": -10.0,      # Punishment for invalid moves (prevents getting stuck)
    "save_dir": "models",               # Model checkpoint directory
    "checkpoint_interval": 100,         # Save model every N episodes
    "eval_episodes": 5,                 # Games to play during evaluation
}


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DQN TRAINING (DQN & Double DQN share 95% of code)
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn_variant(algorithm="dqn"):
    """
    Train DQN or Double DQN agent with configured settings.
    
    Both algorithms share the same training loop - only difference is:
    - DQN:        Q(s,a) = r + γ * max Q_target(s', a')
    - Double DQN: Q(s,a) = r + γ * Q_target(s', argmax Q_policy(s', a'))
    
    Args:
        algorithm: "dqn" or "double-dqn"
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
    from src.utils import TrainingTimer, EvaluationLogger
    
    # ─────────────────────────────────────────────────────────────────────
    # Setup: Import appropriate agent and configuration
    # ─────────────────────────────────────────────────────────────────────
    if algorithm == "dqn":
        from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
        algo_name = "DQN"
        config_key = "dqn"
        save_subdir = "DQN"
        save_prefix = "dqn"
        AgentClass = DQNAgent
        ModelConfigClass = DQNModelConfig
        AgentConfigClass = AgentConfig
    elif algorithm == "double-dqn":
        from src.agents.double_dqn import DoubleDQNAgent, DoubleDQNModelConfig, DoubleDQNAgentConfig
        algo_name = "DOUBLE DQN"
        config_key = "double_dqn"
        save_subdir = "DoubleDQN"
        save_prefix = "double_dqn"
        AgentClass = DoubleDQNAgent
        ModelConfigClass = DoubleDQNModelConfig
        AgentConfigClass = DoubleDQNAgentConfig
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print("=" * 80)
    print(f"🎮 TRAINING {algo_name} AGENT")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────
    # Initialize: Agent, Environment, Tracking
    # ─────────────────────────────────────────────────────────────────────
    timer = TrainingTimer().start()
    
    # Build agent with algorithm-specific config
    cfg = CONFIG[config_key]
    model_config = ModelConfigClass(
        output_dim=len(ACTIONS),
        hidden_dims=cfg["hidden_dims"]
    )
    agent_config = AgentConfigClass(
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay=cfg["epsilon_decay"],
        target_update_interval=cfg["target_update_interval"],
        replay_buffer_size=cfg["replay_buffer_size"],
        gradient_clip=cfg["gradient_clip"],
    )
    agent = AgentClass(
        model_config=model_config,
        agent_config=agent_config,
        action_space=ACTIONS
    )
    
    # Setup environment with configured penalty
    env_config = EnvironmentConfig(
        enable_ui=CONFIG["enable_ui"],
        invalid_move_penalty=CONFIG["invalid_move_penalty"]
    )
    env = GameEnvironment(env_config)
    
    # Tracking lists for metrics
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    
    # Setup live plotting (optional)
    if CONFIG["enable_plots"]:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    save_dir = Path(CONFIG["save_dir"]) / save_subdir
    episodes = CONFIG["episodes"]
    
    print(f"\n📊 Training for {episodes} episodes")
    print(f"💾 Models will be saved to: {save_dir}")
    print(f"🎯 Close plot window to stop early\n")
    
    best_score = 0
    best_tile = 0
    
    # ─────────────────────────────────────────────────────────────────────
    # Main Training Loop
    # ─────────────────────────────────────────────────────────────────────
    try:
        for episode in range(1, episodes + 1):
            # Reset for new episode
            state = env.reset()
            done = False
            episode_reward = 0
            
            # Play one full game
            while not done:
                # Agent selects action (ε-greedy)
                action = agent.select_action(state)
                
                # Execute action in environment
                result = env.step(action)
                
                # Store experience in replay buffer
                agent.store_transition(
                    state, action, result.reward, result.state, result.done
                )
                
                # Train agent if enough experiences collected
                if agent.can_optimize():
                    agent.optimize_model()
                
                # Update state and accumulate reward
                state = result.state
                episode_reward += result.reward
                done = result.done
            
            # ─────────────────────────────────────────────────────────────
            # Episode Complete: Track Metrics
            # ─────────────────────────────────────────────────────────────
            info = env.get_state()
            episode_rewards.append(episode_reward)
            episode_scores.append(info['score'])
            episode_max_tiles.append(info['max_tile'])
            
            best_score = max(best_score, info['score'])
            best_tile = max(best_tile, info['max_tile'])
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-50:])  # Last 50 episodes
                avg_score = np.mean(episode_scores[-50:])
                elapsed = timer.elapsed_str()
                print(
                    f"Ep {episode:4d} | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Score: {avg_score:6.0f} | "
                    f"Tile: {episode_max_tiles[-1]:4d} | "
                    f"ε: {agent.epsilon:.3f} | "
                    f"Time: {elapsed}"
                )
            
            # Save checkpoint periodically
            if episode % CONFIG["checkpoint_interval"] == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"{save_prefix}_ep{episode}.pth"
                agent.save(checkpoint_path, episode)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Update live plot
            if CONFIG["enable_plots"] and episode % 5 == 0:
                _update_training_plot(
                    ax1, ax2, episode_rewards, episode_scores, 
                    episode_max_tiles, algo_name
                )
                plt.pause(0.01)
                
                # Check if user closed plot window (early stop)
                if not plt.fignum_exists(fig.number):
                    print("\n⚠️  Plot closed - stopping early")
                    break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    # ─────────────────────────────────────────────────────────────────────
    # Training Complete: Save and Log Results
    # ─────────────────────────────────────────────────────────────────────
    finally:
        timer.stop()
        
        # Save final model
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / f"{save_prefix}_final.pth"
        agent.save(final_path, episode)
        print(f"\n💾 Final model saved: {final_path}")
        
        # Log evaluation to file
        logger = EvaluationLogger()
        final_avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
        
        logger.log_training(
            algorithm=algo_name,
            episodes=episode,
            final_avg_reward=final_avg_reward,
            max_tile=best_tile,
            final_score=best_score,
            training_time=timer.elapsed_str(),
            model_path=str(final_path),
            notes=f"LR={cfg['learning_rate']}, ε_end={cfg['epsilon_end']}, ε_decay={cfg['epsilon_decay']}"
        )
        
        # Cleanup
        env.close()
        if CONFIG["enable_plots"]:
            plt.ioff()
            plt.close('all')
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"⏱️  Total Time: {timer.elapsed_str()}")
        print(f"🏆 Best Score: {best_score}")
        print(f"🎯 Best Tile: {best_tile}")
        print(f"{'='*80}\n")


def _update_training_plot(ax1, ax2, rewards, scores, tiles, algo_name):
    """Helper: Update matplotlib training plots."""
    import numpy as np
    
    ax1.clear()
    ax2.clear()
    
    # Plot 1: Rewards with moving average
    ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
    if len(rewards) >= 50:
        moving_avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax1.plot(range(49, len(rewards)), moving_avg, color='blue', linewidth=2, label='MA-50')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'{algo_name} Training Progress - Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scores and Max Tiles
    ax2.plot(scores, alpha=0.3, color='green', label='Score')
    ax2.plot(tiles, alpha=0.3, color='red', label='Max Tile')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{algo_name} Training Progress - Game Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# MCTS TRAINING (Planning-only, no learning)
# ═══════════════════════════════════════════════════════════════════════════

def train_mcts():
    """
    Run MCTS simulations to evaluate performance.
    
    NOTE: MCTS is a planning algorithm - it doesn't "learn" from experience.
    Each move it builds a search tree and picks the best action. No model is saved.
    
    This function just runs games to evaluate MCTS performance.
    """
    import numpy as np
    from src.agents.mcts import MCTSAgent, MCTSConfig
    from src.environment import GameEnvironment, EnvironmentConfig
    from src.utils import TrainingTimer, EvaluationLogger
    
    print("=" * 80)
    print("🎮 RUNNING MCTS SIMULATIONS")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────
    # Setup: Agent and Environment
    # ─────────────────────────────────────────────────────────────────────
    timer = TrainingTimer().start()
    
    cfg = CONFIG["mcts"]
    agent = MCTSAgent(config=MCTSConfig(
        simulations=cfg["simulations"],
        exploration_constant=cfg["exploration_constant"]
    ))
    
    env_config = EnvironmentConfig(
        enable_ui=CONFIG["enable_ui"],
        invalid_move_penalty=CONFIG["invalid_move_penalty"]
    )
    env = GameEnvironment(env_config)
    
    # Tracking
    episode_scores = []
    episode_max_tiles = []
    episode_steps = []
    
    episodes = CONFIG["episodes"]
    print(f"\n📊 Running {episodes} MCTS simulations")
    print(f"🌲 {cfg['simulations']} tree searches per move\n")
    
    best_score = 0
    best_tile = 0
    
    # ─────────────────────────────────────────────────────────────────────
    # Simulation Loop
    # ─────────────────────────────────────────────────────────────────────
    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            board = env.get_board()
            done = False
            steps = 0
            
            # Play one game using MCTS tree search
            while not done:
                action = agent.select_action(state, board)
                result = env.step(action)
                
                state = result.state
                board = env.get_board()
                done = result.done
                steps += 1
            
            # Track metrics
            info = env.get_state()
            episode_scores.append(info['score'])
            episode_max_tiles.append(info['max_tile'])
            episode_steps.append(steps)
            
            best_score = max(best_score, info['score'])
            best_tile = max(best_tile, info['max_tile'])
            
            # Print progress every 5 games (MCTS is slower)
            if episode % 5 == 0:
                avg_score = np.mean(episode_scores[-10:])  # Last 10 games
                avg_tile = np.mean(episode_max_tiles[-10:])
                elapsed = timer.elapsed_str()
                print(
                    f"Game {episode:4d} | "
                    f"Score: {info['score']:6.0f} | "
                    f"Tile: {info['max_tile']:4d} | "
                    f"Steps: {steps:4d} | "
                    f"Avg Score: {avg_score:6.0f} | "
                    f"Time: {elapsed}"
                )
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted")
    
    # ─────────────────────────────────────────────────────────────────────
    # Complete: Log Results (no model to save)
    # ─────────────────────────────────────────────────────────────────────
    finally:
        timer.stop()
        
        # Log evaluation
        logger = EvaluationLogger()
        final_avg_score = float(np.mean(episode_scores[-100:])) if episode_scores else 0.0
        
        logger.log_training(
            algorithm="MCTS",
            episodes=episode,
            final_avg_reward=final_avg_score,  # MCTS doesn't have explicit rewards
            max_tile=best_tile,
            final_score=best_score,
            training_time=timer.elapsed_str(),
            model_path="N/A (MCTS doesn't save models)",
            notes=f"Simulations={cfg['simulations']}, C={cfg['exploration_constant']}"
        )
        
        env.close()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ MCTS Simulation Complete!")
        print(f"⏱️  Total Time: {timer.elapsed_str()}")
        print(f"🏆 Best Score: {best_score}")
        print(f"🎯 Best Tile: {best_tile}")
        print(f"📊 Avg Score: {np.mean(episode_scores):.1f}")
        print(f"📊 Avg Tile: {np.mean(episode_max_tiles):.1f}")
        print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════
# POLICY GRADIENT TRAINING (TODO)
# ═══════════════════════════════════════════════════════════════════════════

def train_policy_gradient():
    """Train Policy Gradient agent (not yet implemented)."""
    print("⚠️  Policy Gradient training not yet implemented!")
    print("This requires implementing the PolicyGradientAgent in src/agents/policy_gradient/")


# ═══════════════════════════════════════════════════════════════════════════
# PLAY MODE - Watch a trained model play
# ═══════════════════════════════════════════════════════════════════════════

def play_model(model_path=None, episodes=1, use_ui=True):
    """
    Load a trained model and watch it play.
    
    Args:
        model_path: Path to .pth model file (auto-detects algorithm)
        episodes: Number of games to play
        use_ui: Show pygame visualization
    """
    from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
    
    # Auto-detect model path if not provided
    if model_path is None:
        model_path = Path(CONFIG["save_dir"]) / "DQN" / "dqn_final.pth"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"💡 Train a model first: python 2048RL.py train --algorithm dqn")
        return
    
    # Detect algorithm from path
    if "double" in str(model_path).lower():
        from src.agents.double_dqn import DoubleDQNAgent, DoubleDQNModelConfig, DoubleDQNAgentConfig
        AgentClass = DoubleDQNAgent
        ModelConfigClass = DoubleDQNModelConfig
        AgentConfigClass = DoubleDQNAgentConfig
        config_key = "double_dqn"
        algo_name = "Double DQN"
    else:
        from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
        AgentClass = DQNAgent
        ModelConfigClass = DQNModelConfig
        AgentConfigClass = AgentConfig
        config_key = "dqn"
        algo_name = "DQN"
    
    print("=" * 80)
    print(f"🎮 PLAYING WITH {algo_name} MODEL")
    print("=" * 80)
    print(f"📂 Model: {model_path}\n")
    
    # Load agent
    cfg = CONFIG[config_key]
    model_config = ModelConfigClass(
        output_dim=len(ACTIONS),
        hidden_dims=cfg["hidden_dims"]
    )
    agent_config = AgentConfigClass(
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        epsilon_start=0.0,  # No exploration when playing
        epsilon_end=0.0,
        epsilon_decay=1,
        target_update_interval=cfg["target_update_interval"],
        replay_buffer_size=cfg["replay_buffer_size"],
        gradient_clip=cfg["gradient_clip"],
    )
    agent = AgentClass(
        model_config=model_config,
        agent_config=agent_config,
        action_space=ACTIONS
    )
    agent.load(model_path)
    
    # Setup environment
    env_config = EnvironmentConfig(
        enable_ui=use_ui,
        invalid_move_penalty=CONFIG["invalid_move_penalty"]
    )
    env = GameEnvironment(env_config)
    
    # Play episodes
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action = agent.act_greedy(state)
            result = env.step(action)
            
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1
        
        info = env.get_state()
        print(f"\nGame {ep}/{episodes}:")
        print(f"  Score: {info['score']}")
        print(f"  Max Tile: {info['max_tile']}")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
    
    env.close()


# ═══════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Parse command line arguments and run appropriate function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL agents for 2048",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train DQN for 2000 episodes
  python 2048RL.py train --algorithm dqn --episodes 2000
  
  # Train Double DQN with custom settings
  python 2048RL.py train --algorithm double-dqn --episodes 1000
  
  # Run MCTS simulations
  python 2048RL.py train --algorithm mcts --episodes 50
  
  # Watch trained model play
  python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train an agent')
    train_parser.add_argument(
        '--algorithm', '-a',
        choices=['dqn', 'double-dqn', 'mcts', 'policy-gradient'],
        default=CONFIG['algorithm'],
        help='Algorithm to train'
    )
    train_parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=CONFIG['episodes'],
        help='Number of episodes to train'
    )
    train_parser.add_argument(
        '--no-ui',
        action='store_true',
        help='Disable pygame UI (faster training)'
    )
    train_parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable live training plots'
    )
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Watch trained model play')
    play_parser.add_argument(
        '--model', '-m',
        help='Path to model file (.pth)'
    )
    play_parser.add_argument(
        '--episodes', '-e',
        type=int,
        default=1,
        help='Number of games to play'
    )
    play_parser.add_argument(
        '--no-ui',
        action='store_true',
        help='Disable pygame UI'
    )
    
    args = parser.parse_args()
    
    # Update CONFIG from command line
    if hasattr(args, 'episodes'):
        CONFIG['episodes'] = args.episodes
    if hasattr(args, 'no_ui') and args.no_ui:
        CONFIG['enable_ui'] = False
    if hasattr(args, 'no_plots') and args.no_plots:
        CONFIG['enable_plots'] = False
    
    # Execute command
    if args.command == 'train':
        if args.algorithm in ['dqn', 'double-dqn']:
            train_dqn_variant(args.algorithm)
        elif args.algorithm == 'mcts':
            train_mcts()
        elif args.algorithm == 'policy-gradient':
            train_policy_gradient()
    
    elif args.command == 'play':
        play_model(
            model_path=args.model,
            episodes=args.episodes,
            use_ui=not args.no_ui
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
