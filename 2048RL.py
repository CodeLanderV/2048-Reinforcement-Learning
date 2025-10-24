"""
═══════════════════════════════════════════════════════════════════════════════
2048 Reinforcement Learning - Central Control Panel
═══════════════════════════════════════════════════════════════════════════════

This file provides a simple interface to train and evaluate different RL
algorithms for playing 2048. Just modify CONFIG and run!

QUICK START:
    python 2048RL.py train --algorithm dqn --episodes 3000
    python 2048RL.py play --model models/DQN/dqn_final.pth

ALGORITHMS AVAILABLE:
    - DQN:         Deep Q-Network (value-based, off-policy)
    - Double-DQN:  Reduces Q-value overestimation

ARCHIVED ALGORITHMS (commented out):
    - MCTS:        Monte Carlo Tree Search (planning, no learning)
    - REINFORCE:   Monte Carlo Policy Gradient (on-policy learning)
"""

import sys
import warnings
import logging
import json
from pathlib import Path
from datetime import datetime

# Setup Python path and suppress numpy warnings
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Setup logging to capture all output
def setup_logging():
    """Setup logging to both console and file."""
    log_dir = Path(__file__).parent / "evaluations"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "logs.txt"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logging
logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Modify these to tune training behavior
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    # ─────────────────────────────────────────────────────────────────────
    # General Training Settings
    # ─────────────────────────────────────────────────────────────────────
    "algorithm": "dqn",         # Which algorithm: "dqn", "double-dqn", "mcts", "reinforce"
    "episodes": 10000,          # How many games to train on (increased for better convergence)
    "enable_ui": False,         # Show pygame window? (disabled for faster training)
    "enable_plots": True,       # Show live training graphs?
    
    # ─────────────────────────────────────────────────────────────────────
    # DQN Hyperparameters (Research-Proven Defaults from DeepMind)
    # ─────────────────────────────────────────────────────────────────────
    # These settings are optimized based on the original DQN paper and 
    # empirical testing for the 2048 game
    "dqn": {
        # Neural network training
        "learning_rate": 1e-4,          # Standard Adam learning rate (proven optimal)
        "gamma": 0.99,                  # High discount for long-term planning
        "batch_size": 128,              # Samples per training step
        "gradient_clip": 5.0,           # Prevents gradient explosion
        "hidden_dims": (256, 256),      # Adequate network capacity for 2048
        
        # Exploration schedule (ε-greedy) - CRITICAL FOR 2048
        "epsilon_start": 1.0,           # Start: 100% random actions (explore)
        "epsilon_end": 0.1,             # End: 10% random actions (prevents getting stuck)
        "epsilon_decay": 100000,        # Decay over 100k steps (balanced exploration)
        
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
    # REINFORCE Hyperparameters (Monte Carlo Policy Gradient)
    # ─────────────────────────────────────────────────────────────────────
    "reinforce": {
        "learning_rate": 0.001,         # Policy network learning rate
        "gamma": 0.99,                  # Discount factor for returns
        "hidden_dims": [256, 256],      # Policy network architecture
        "use_baseline": True,           # Subtract baseline to reduce variance
        "entropy_coef": 0.01,           # Entropy regularization (encourages exploration)
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
    print(f"TRAINING {algo_name} AGENT")
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
    moving_averages = []  # Track 100-episode moving average
    
    # Convergence detection parameters
    convergence_window = 100  # Calculate moving average over 100 episodes
    convergence_patience = 5000  # Stop if no improvement for 5000 episodes
    best_moving_avg = 0
    episodes_since_improvement = 0
    converged = False
    
    # Setup live plotting (optional)
    if CONFIG["enable_plots"]:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    save_dir = Path(CONFIG["save_dir"]) / save_subdir
    episodes = CONFIG["episodes"]
    
    print(f"\nTraining for maximum {episodes} episodes")
    print(f"Early stopping: Will stop if moving average doesn't improve for {convergence_patience} episodes")
    print(f"Models will be saved to: {save_dir}")
    print(f"Close plot window to stop early\n")
    
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
            
            # Calculate 100-episode moving average
            if len(episode_scores) >= convergence_window:
                moving_avg = sum(episode_scores[-convergence_window:]) / convergence_window
                moving_averages.append(moving_avg)
                
                # Check for convergence (improvement in moving average)
                if moving_avg > best_moving_avg * 1.01:  # 1% improvement threshold
                    best_moving_avg = moving_avg
                    episodes_since_improvement = 0
                else:
                    episodes_since_improvement += 1
                
                # Check if converged
                if episodes_since_improvement >= convergence_patience:
                    converged = True
                    print(f"\n[CONVERGENCE] Agent converged! No improvement for {convergence_patience} episodes")
                    print(f"[CONVERGENCE] Best moving average: {best_moving_avg:.2f}")
                    print(f"[CONVERGENCE] Stopping training early at episode {episode}\n")
                    break
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-50:])  # Last 50 episodes
                avg_score = np.mean(episode_scores[-50:])
                elapsed = timer.elapsed_str()
                
                # Add moving average info if available
                ma_info = f" | MA-100: {moving_averages[-1]:6.0f}" if moving_averages else ""
                convergence_info = f" | No-Imp: {episodes_since_improvement}" if len(episode_scores) >= convergence_window else ""
                
                print(
                    f"Ep {episode:4d} | "
                    f"Reward: {avg_reward:7.2f} | "
                    f"Score: {avg_score:6.0f}{ma_info} | "
                    f"Tile: {episode_max_tiles[-1]:4d} | "
                    f"ε: {agent.epsilon:.3f}{convergence_info} | "
                    f"Time: {elapsed}"
                )
            
            # Save checkpoint periodically
            if episode % CONFIG["checkpoint_interval"] == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"{save_prefix}_ep{episode}.pth"
                agent.save(checkpoint_path, episode)
                print(f"[CHECKPOINT] Saved: {checkpoint_path}")
            
            # Update live plot
            if CONFIG["enable_plots"] and episode % 5 == 0:
                _update_training_plot(
                    ax1, ax2, episode_rewards, episode_scores, 
                    episode_max_tiles, moving_averages, algo_name
                )
                plt.pause(0.01)
                
                # Check if user closed plot window (early stop)
                if not plt.fignum_exists(fig.number):
                    print("\n[WARNING] Plot closed - stopping early")
                    break
    
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
    
    # ─────────────────────────────────────────────────────────────────────
    # Training Complete: Save and Log Results
    # ─────────────────────────────────────────────────────────────────────
    finally:
        timer.stop()
        
        # Save final model
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / f"{save_prefix}_final.pth"
        agent.save(final_path, episode)
        print(f"\n[SAVE] Final model saved: {final_path}")
        
        # Save training plots
        if CONFIG["enable_plots"] and plt.fignum_exists(fig.number):
            plot_path = Path("evaluations") / f"{algo_name.replace(' ', '_')}_training_plot.png"
            plot_path.parent.mkdir(exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"[SAVE] Training plot saved: {plot_path}")
        
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
            notes=f"LR={cfg['learning_rate']}, epsilon_end={cfg['epsilon_end']}, epsilon_decay={cfg['epsilon_decay']}"
        )
        
        # Cleanup
        env.close()
        if CONFIG["enable_plots"]:
            plt.ioff()
            plt.close('all')
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        if converged:
            print(f"Reason: Converged (no improvement for {convergence_patience} episodes)")
            print(f"Best Moving Average (100ep): {best_moving_avg:.2f}")
        print(f"Total Episodes: {episode}")
        print(f"Total Time: {timer.elapsed_str()}")
        print(f"Best Score: {best_score}")
        print(f"Best Tile: {best_tile}")
        print(f"{'='*80}\n")


def _update_training_plot(ax1, ax2, rewards, scores, tiles, moving_averages, algo_name):
    """Helper: Update matplotlib training plots."""
    import numpy as np
    
    ax1.clear()
    ax2.clear()
    
    # Plot 1: Scores with moving average (100-episode window)
    ax1.scatter(range(len(scores)), scores, alpha=0.3, s=10, color='blue', label='Raw Score')
    if moving_averages:
        # Moving average starts at episode 100
        ma_start = 100
        ax1.plot(range(ma_start - 1, len(scores)), moving_averages, 
                color='red', linewidth=2, label='MA-100 (Convergence Metric)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{algo_name} Training Progress - Scores (Raw + 100-Episode Moving Average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max Tiles
    ax2.plot(scores, alpha=0.3, color='green', label='Score')
    ax2.plot(tiles, alpha=0.3, color='red', label='Max Tile')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{algo_name} Training Progress - Game Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# ARCHIVED: MCTS & REINFORCE ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════
# 
# These algorithms have been archived to keep the codebase focused on DQN/Double-DQN.
# The code is preserved below for future reference but commented out.
# To re-enable, uncomment the functions and update the main() algorithm routing.
#
# ═══════════════════════════════════════════════════════════════════════════

# # ═══════════════════════════════════════════════════════════════════════════
# # MCTS TRAINING (Planning-only, no learning)
# # ═══════════════════════════════════════════════════════════════════════════
# 
# def train_mcts():
#     """
#     Run MCTS simulations to evaluate performance.
#     
#     NOTE: MCTS is a planning algorithm - it doesn't "learn" from experience.
#     Each move it builds a search tree and picks the best action. No model is saved.
#     
#     This function just runs games to evaluate MCTS performance.
#     """
#     import numpy as np
#     from src.agents.mcts import MCTSAgent, MCTSConfig
#     from src.environment import GameEnvironment, EnvironmentConfig
#     from src.utils import TrainingTimer, EvaluationLogger
#     
#     print("=" * 80)
#     print("RUNNING MCTS SIMULATIONS")
#     print("=" * 80)
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Setup: Agent and Environment
#     # ─────────────────────────────────────────────────────────────────────
#     timer = TrainingTimer().start()
#     
#     cfg = CONFIG["mcts"]
#     agent = MCTSAgent(config=MCTSConfig(
#         simulations=cfg["simulations"],
#         exploration_constant=cfg["exploration_constant"]
#     ))
#     
#     env_config = EnvironmentConfig(
#         enable_ui=CONFIG["enable_ui"],
#         invalid_move_penalty=CONFIG["invalid_move_penalty"]
#     )
#     env = GameEnvironment(env_config)
#     
#     # Tracking
#     episode_scores = []
#     episode_max_tiles = []
#     episode_steps = []
#     
#     episodes = CONFIG["episodes"]
#     print(f"\nRunning {episodes} MCTS simulations")
#     print(f"🌲 {cfg['simulations']} tree searches per move\n")
#     
#     best_score = 0
#     best_tile = 0
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Simulation Loop
#     # ─────────────────────────────────────────────────────────────────────
#     try:
#         for episode in range(1, episodes + 1):
#             state = env.reset()
#             board = env.get_board()
#             done = False
#             steps = 0
#             
#             # Play one game using MCTS tree search
#             while not done:
#                 action = agent.select_action(state, board)
#                 result = env.step(action)
#                 
#                 state = result.state
#                 board = env.get_board()
#                 done = result.done
#                 steps += 1
#             
#             # Track metrics
#             info = env.get_state()
#             episode_scores.append(info['score'])
#             episode_max_tiles.append(info['max_tile'])
#             episode_steps.append(steps)
#             
#             best_score = max(best_score, info['score'])
#             best_tile = max(best_tile, info['max_tile'])
#             
#             # Print progress every 5 games (MCTS is slower)
#             if episode % 5 == 0:
#                 avg_score = np.mean(episode_scores[-10:])  # Last 10 games
#                 avg_tile = np.mean(episode_max_tiles[-10:])
#                 elapsed = timer.elapsed_str()
#                 print(
#                     f"Game {episode:4d} | "
#                     f"Score: {info['score']:6.0f} | "
#                     f"Tile: {info['max_tile']:4d} | "
#                     f"Steps: {steps:4d} | "
#                     f"Avg Score: {avg_score:6.0f} | "
#                     f"Time: {elapsed}"
#                 )
#     
#     except KeyboardInterrupt:
#         print("\n\n[WARNING] Simulation interrupted")
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Complete: Log Results (no model to save)
#     # ─────────────────────────────────────────────────────────────────────
#     finally:
#         timer.stop()
#         
#         # Log evaluation
#         logger = EvaluationLogger()
#         final_avg_score = float(np.mean(episode_scores[-100:])) if episode_scores else 0.0
#         
#         logger.log_training(
#             algorithm="MCTS",
#             episodes=episode,
#             final_avg_reward=final_avg_score,  # MCTS doesn't have explicit rewards
#             max_tile=best_tile,
#             final_score=best_score,
#             training_time=timer.elapsed_str(),
#             model_path="N/A (MCTS doesn't save models)",
#             notes=f"Simulations={cfg['simulations']}, C={cfg['exploration_constant']}"
#         )
#         
#         env.close()
#         
#         # Print summary
#         print(f"\n{'='*80}")
#         print(f"MCTS Simulation Complete!")
#         print(f"Total Time: {timer.elapsed_str()}")
#         print(f"Best Score: {best_score}")
#         print(f"Best Tile: {best_tile}")
#         print(f"Avg Score: {np.mean(episode_scores):.1f}")
#         print(f"Avg Tile: {np.mean(episode_max_tiles):.1f}")
#         print(f"{'='*80}\n")
# 
# 
# # ═══════════════════════════════════════════════════════════════════════════
# # REINFORCE TRAINING (Monte Carlo Policy Gradient)
# # ═══════════════════════════════════════════════════════════════════════════
# 
# def train_reinforce():
#     """
#     Train REINFORCE (Policy Gradient) agent.
#     
#     Key difference from DQN:
#     - Learns a stochastic policy π(a|s) that outputs action probabilities
#     - Updates after full episodes (Monte Carlo)
#     - On-policy: learns from its own experience only
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
#     from src.agents.reinforce import REINFORCEAgent, REINFORCEConfig
#     from src.utils import TrainingTimer, EvaluationLogger
#     
#     print("=" * 80)
#     print("TRAINING: REINFORCE (Monte Carlo Policy Gradient)")
#     print("=" * 80)
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Setup: Agent and Environment
#     # ─────────────────────────────────────────────────────────────────────
#     timer = TrainingTimer().start()
#     
#     cfg = CONFIG["reinforce"]
#     agent = REINFORCEAgent(
#         state_dim=16,
#         action_dim=4,
#         config=REINFORCEConfig(
#             learning_rate=cfg["learning_rate"],
#             gamma=cfg["gamma"],
#             hidden_dims=cfg["hidden_dims"],
#             use_baseline=cfg["use_baseline"],
#             entropy_coef=cfg["entropy_coef"]
#         )
#     )
#     
#     env_config = EnvironmentConfig(
#         enable_ui=CONFIG["enable_ui"],
#         invalid_move_penalty=CONFIG["invalid_move_penalty"]
#     )
#     env = GameEnvironment(env_config)
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Setup: Save directory and plotting
#     # ─────────────────────────────────────────────────────────────────────
#     save_dir = Path(CONFIG["save_dir"]) / "REINFORCE"
#     save_dir.mkdir(parents=True, exist_ok=True)
#     
#     if CONFIG["enable_plots"]:
#         plt.ion()
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#         fig.suptitle('REINFORCE Training Progress')
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Tracking variables
#     # ─────────────────────────────────────────────────────────────────────
#     episode_rewards = []
#     episode_scores = []
#     episode_max_tiles = []
#     moving_averages = []  # Track 100-episode moving average
#     best_score = 0
#     best_tile = 0
#     
#     # Convergence detection parameters
#     convergence_window = 100
#     convergence_patience = 5000
#     best_moving_avg = 0
#     episodes_since_improvement = 0
#     converged = False
#     
#     episodes = CONFIG["episodes"]
#     checkpoint_interval = CONFIG["checkpoint_interval"]
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Training Loop
#     # ─────────────────────────────────────────────────────────────────────
#     print(f"\nTraining for maximum {episodes} episodes...")
#     print(f"Early stopping: Will stop if moving average doesn't improve for {convergence_patience} episodes")
#     print(f"Checkpoints saved to: {save_dir}")
#     print(f"Updates: After each episode (Monte Carlo)")
#     print(f"Policy: Stochastic (samples from policy distribution)\n")
#     
#     for episode in range(1, episodes + 1):
#         state, _ = env.reset()
#         episode_reward = 0
#         done = False
#         
#         # ─────────────────────────────────────────────────────────────────
#         # Play full episode
#         # ─────────────────────────────────────────────────────────────────
#         while not done:
#             action = agent.select_action(state, env.board)
#             next_state, reward, done, _, info = env.step(ACTIONS[action])
#             
#             # Store transition
#             agent.store_transition(state, action, reward, next_state, done)
#             
#             state = next_state
#             episode_reward += reward
#         
#         # ─────────────────────────────────────────────────────────────────
#         # Update policy after episode completes (REINFORCE requirement)
#         # ─────────────────────────────────────────────────────────────────
#         agent.finish_episode()
#         
#         # ─────────────────────────────────────────────────────────────────
#         # Track metrics
#         # ─────────────────────────────────────────────────────────────────
#         episode_rewards.append(episode_reward)
#         episode_scores.append(env.board.score)
#         episode_max_tiles.append(env.board.max_tile())
#         
#         if env.board.score > best_score:
#             best_score = env.board.score
#         if env.board.max_tile() > best_tile:
#             best_tile = env.board.max_tile()
#         
#         # Calculate 100-episode moving average
#         if len(episode_scores) >= convergence_window:
#             moving_avg = sum(episode_scores[-convergence_window:]) / convergence_window
#             moving_averages.append(moving_avg)
#             
#             # Check for convergence
#             if moving_avg > best_moving_avg * 1.01:
#                 best_moving_avg = moving_avg
#                 episodes_since_improvement = 0
#             else:
#                 episodes_since_improvement += 1
#             
#             # Check if converged
#             if episodes_since_improvement >= convergence_patience:
#                 converged = True
#                 print(f"\n[CONVERGENCE] Agent converged! No improvement for {convergence_patience} episodes")
#                 print(f"[CONVERGENCE] Best moving average: {best_moving_avg:.2f}")
#                 break
#         
#         # ─────────────────────────────────────────────────────────────────
#         # Logging and visualization
#         # ─────────────────────────────────────────────────────────────────
#         if episode % 10 == 0:
#             avg_reward_last_10 = np.mean(episode_rewards[-10:])
#             avg_score_last_10 = np.mean(episode_scores[-10:])
#             avg_tile_last_10 = np.mean(episode_max_tiles[-10:])
#             
#             ma_info = f" | MA-100: {moving_averages[-1]:6.0f}" if moving_averages else ""
#             convergence_info = f" | No-Imp: {episodes_since_improvement}" if len(episode_scores) >= convergence_window else ""
#             
#             print(f"Episode {episode:4d} | "
#                   f"Reward: {episode_reward:7.1f} | "
#                   f"Score: {env.board.score:6.0f}{ma_info} | "
#                   f"MaxTile: {env.board.max_tile():4d} | "
#                   f"Avg(10): R={avg_reward_last_10:6.1f} S={avg_score_last_10:6.0f} T={avg_tile_last_10:4.0f}{convergence_info}")
#         
#         # Update plots
#         if CONFIG["enable_plots"] and episode % 20 == 0:
#             _update_training_plot(ax1, ax2, episode_rewards, episode_scores, 
#                                 episode_max_tiles, moving_averages, "REINFORCE")
#             plt.pause(0.01)
#         
#         # ─────────────────────────────────────────────────────────────────
#         # Save checkpoints
#         # ─────────────────────────────────────────────────────────────────
#         if episode % checkpoint_interval == 0:
#             agent.save(save_dir, episode)
#     
#     # ─────────────────────────────────────────────────────────────────────
#     # Final save and evaluation
#     # ─────────────────────────────────────────────────────────────────────
#     timer.stop()
#     
#     final_path = save_dir / "reinforce_final.pth"
#     agent.save(save_dir, "final")
#     
#     # Save training plots
#     if CONFIG["enable_plots"] and plt.fignum_exists(fig.number):
#         plot_path = Path("evaluations") / "REINFORCE_training_plot.png"
#         plot_path.parent.mkdir(exist_ok=True)
#         fig.savefig(plot_path, dpi=150, bbox_inches='tight')
#         print(f"\n[SAVE] Training plot saved: {plot_path}")
#     
#     # Log training results
#     logger = EvaluationLogger()
#     final_avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
#     
#     logger.log_training(
#         algorithm="REINFORCE",
#         episodes=episodes,
#         final_avg_reward=final_avg_reward,
#         max_tile=best_tile,
#         final_score=best_score,
#         training_time=timer.elapsed_str(),
#         model_path=str(final_path),
#         notes=f"LR={cfg['learning_rate']}, gamma={cfg['gamma']}, entropy={cfg['entropy_coef']}"
#     )
#     
#     # Cleanup
#     env.close()
#     if CONFIG["enable_plots"]:
#         plt.ioff()
#         plt.close('all')
#     
#     # Print summary
#     print(f"\n{'='*80}")
#     print(f"Training Complete!")
#     if converged:
#         print(f"Reason: Converged (no improvement for {convergence_patience} episodes)")
#         print(f"Best Moving Average (100ep): {best_moving_avg:.2f}")
#     print(f"Total Episodes: {episode}")
#     print(f"Total Time: {timer.elapsed_str()}")
#     print(f"Best Score: {best_score}")
#     print(f"Best Tile: {best_tile}")
#     print(f"{'='*80}\n")


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
        print(f"[ERROR] Model not found: {model_path}")
        print(f"[INFO] Train a model first: python 2048RL.py train --algorithm dqn")
        return
    
    # Detect algorithm from path
    if "reinforce" in str(model_path).lower():
        """
        ok this is to load the REINFORCE model into play.
        """
        from src.agents.reinforce import REINFORCEAgent, REINFORCEConfig
        
        print("=" * 80)
        print(f"PLAYING WITH REINFORCE MODEL")
        print("=" * 80)
        print(f"Model: {model_path}\n")
        
        cfg = CONFIG["reinforce"]
        agent = REINFORCEAgent(
            state_dim=16,
            action_dim=4,
            config=REINFORCEConfig(
                learning_rate=cfg["learning_rate"],
                gamma=cfg["gamma"],
                hidden_dims=cfg["hidden_dims"]
            )
        )
        agent.load(model_path)
        algo_name = "REINFORCE"
        
    elif "double" in str(model_path).lower():
        from src.agents.double_dqn import DoubleDQNAgent, DoubleDQNModelConfig, DoubleDQNAgentConfig
        AgentClass = DoubleDQNAgent
        ModelConfigClass = DoubleDQNModelConfig
        AgentConfigClass = DoubleDQNAgentConfig
        config_key = "double_dqn"
        algo_name = "Double DQN"
        
        print("=" * 80)
        print(f"PLAYING WITH {algo_name} MODEL")
        print("=" * 80)
        print(f"Model: {model_path}\n")
        
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
        
    else:
        from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
        AgentClass = DQNAgent
        ModelConfigClass = DQNModelConfig
        AgentConfigClass = AgentConfig
        config_key = "dqn"
        algo_name = "DQN"
        
        print("=" * 80)
        print(f"PLAYING WITH {algo_name} MODEL")
        print("=" * 80)
        print(f"Model: {model_path}\n")
        
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
    import time
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nGame {ep}/{episodes} starting...")
        
        while not done:
            # Select action
            action = agent.act_greedy(state)
            
            # Execute action
            result = env.step(action)
            
            # Add small delay so UI updates are visible
            if use_ui:
                time.sleep(0.1)  # 100ms delay between moves (adjust as needed)
            
            # Update state
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1
            
            # Handle pygame events to prevent freezing
            if use_ui and env.ui:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\n[INFO] Window closed by user")
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        print("\n[INFO] ESC pressed - stopping playback")
                        env.close()
                        return
        
        info = env.get_state()
        print(f"\nGame {ep}/{episodes} completed:")
        print(f"  Score: {info['score']}")
        print(f"  Max Tile: {info['max_tile']}")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
    
    print(f"\n[INFO] Finished playing {episodes} game(s)")
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
        choices=['dqn', 'double-dqn'],  # MCTS and REINFORCE archived
        default=CONFIG['algorithm'],
        help='Algorithm to train (DQN or Double-DQN)'
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
        # Training with proven hyperparameters (Optuna removed)
        print("\n[INFO] Starting training with research-proven hyperparameters")
        print("=" * 70)
        print("DQN Configuration:")
        print(f"  Learning Rate:    {CONFIG['dqn']['learning_rate']}")
        print(f"  Gamma:            {CONFIG['dqn']['gamma']}")
        print(f"  Epsilon End:      {CONFIG['dqn']['epsilon_end']}")
        print(f"  Epsilon Decay:    {CONFIG['dqn']['epsilon_decay']}")
        print(f"  Network Size:     {CONFIG['dqn']['hidden_dims']}")
        print(f"  Batch Size:       {CONFIG['dqn']['batch_size']}")
        print(f"  Episodes:         {CONFIG['episodes']}")
        print("=" * 70)
        print()
        
        # Run training
        if args.algorithm in ['dqn', 'double-dqn']:
            train_dqn_variant(args.algorithm)
        elif args.algorithm == 'mcts':
            print("[ERROR] MCTS algorithm has been archived and is no longer available.")
            print("[INFO] Please use 'dqn' or 'double-dqn' instead.")
            print("[INFO] To re-enable MCTS, uncomment the train_mcts() function in 2048RL.py")
        elif args.algorithm == 'reinforce':
            print("[ERROR] REINFORCE algorithm has been archived and is no longer available.")
            print("[INFO] Please use 'dqn' or 'double-dqn' instead.")
            print("[INFO] To re-enable REINFORCE, uncomment the train_reinforce() function in 2048RL.py")
    
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
