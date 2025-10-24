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
import re

# Setup Python path and suppress numpy warnings
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Try importing Optuna (optional for hyperparameter tuning)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

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
    "episodes": 10000,          # How many games to train on (increased for better results)
    "enable_ui": False,         # Show pygame window? (slower but fun to watch)
    "enable_plots": True,       # Show live training graphs?
    "hyperparameter_tuning": False,  # Enable hyperparameter search (set to False for faster training)
    
    # ─────────────────────────────────────────────────────────────────────
    # DQN Hyperparameters (Standard Deep Q-Network)
    # ─────────────────────────────────────────────────────────────────────
    # These settings are research-proven and optimized for 2048
    "dqn": {
        # Neural network training
        "learning_rate": 1e-4,          # Reduced from 3e-4 to prevent overfitting
        "gamma": 0.99,                  # Discount factor for future rewards
        "batch_size": 512,              # Samples per training step (increased for stability)
        "gradient_clip": 5.0,           # Prevents gradient explosion
        "hidden_dims": (512, 512, 256), # Neural network architecture (deeper and wider)
        
        # Exploration schedule (ε-greedy)
        "epsilon_start": 1.0,           # Start: 100% random actions (explore)
        "epsilon_end": 0.05,            # Increased from 0.01 to maintain more exploration
        "epsilon_decay": 250000,        # Increased from 200000 for slower decay
        
        # Experience replay & stability
        "replay_buffer_size": 500000,  # How many past experiences to remember (increased)
        "target_update_interval": 1000, # Update target network every N steps
    },
    
    # ─────────────────────────────────────────────────────────────────────
    # Double DQN Hyperparameters (Reduces Q-value overestimation bias)
    # ─────────────────────────────────────────────────────────────────────
    # More exploration since Double DQN is inherently more stable
    "double_dqn": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "batch_size": 256,
        "gradient_clip": 5.0,
        "hidden_dims": (512, 512, 256), # Same improved architecture
        
        # INCREASED exploration vs standard DQN
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,            # Keep 1% randomness
        "epsilon_decay": 200000,        # Longer decay
        
        "replay_buffer_size": 200_000,
        "target_update_interval": 1000,
    },
    
   
    
    # ─────────────────────────────────────────────────────────────────────
    # Environment & Saving
    # ─────────────────────────────────────────────────────────────────────
    "invalid_move_penalty": -10.0,      # Punishment for invalid moves (prevents getting stuck)
    "save_dir": "models",               # Model checkpoint directory
    "checkpoint_interval": 500,         # Save model every N episodes (less frequent)
    "eval_episodes": 5,                 # Games to play during evaluation
}


# ═══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING WITH OPTUNA
# ═══════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(algorithm: str, n_trials: int = 30, tune_episodes: int = 200):
    """
    Run Optuna hyperparameter optimization with short training runs.
    
    Uses quick 200-episode training sessions to evaluate different hyperparameter
    combinations, then returns the best configuration to use for full training.
    
    Args:
        algorithm: 'dqn', 'double-dqn', or 'reinforce'
        n_trials: Number of Optuna trials to run
        tune_episodes: Episodes per trial (kept short for speed)
    
    Returns:
        dict: Best hyperparameters found
    """
    if not OPTUNA_AVAILABLE:
        print("[ERROR] Optuna not installed. Install with: pip install optuna")
        print("[INFO] Skipping hyperparameter tuning, using default CONFIG values")
        return None
    
    import numpy as np
    from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
    
    print("=" * 80)
    print(f"HYPERPARAMETER TUNING: {algorithm.upper()}")
    print("=" * 80)
    print(f"Method: Optuna TPE Sampler")
    print(f"Trials: {n_trials}")
    print(f"Episodes per trial: {tune_episodes} (short runs for speed)")
    print("=" * 80)
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function - trains agent and returns score."""
        
        if algorithm in ['dqn', 'double-dqn']:
            # Sample DQN hyperparameters
            hidden_choice = trial.suggest_categorical("hidden_architecture", [0, 1, 2])
            hidden_options = [(128, 128), (256, 256), (512, 256)]
        
            config = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "epsilon_end": trial.suggest_float("epsilon_end", 0.01, 0.15),
                "epsilon_decay": trial.suggest_int("epsilon_decay", 50000, 150000),
                "replay_buffer_size": trial.suggest_categorical("replay_buffer_size", [50000, 100000]),
                "hidden_dims": hidden_options[hidden_choice],
            }
            
            # Quick DQN training
            from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
            from src.agents.double_dqn import DoubleDQNAgent, DoubleDQNModelConfig, DoubleDQNAgentConfig
            
            if algorithm == "double-dqn":
                model_config = DoubleDQNModelConfig(output_dim=len(ACTIONS), hidden_dims=config["hidden_dims"])
                agent_config = DoubleDQNAgentConfig(
                    gamma=config["gamma"], batch_size=config["batch_size"],
                    learning_rate=config["learning_rate"], epsilon_start=1.0,
                    epsilon_end=config["epsilon_end"], epsilon_decay=config["epsilon_decay"],
                    target_update_interval=1000, replay_buffer_size=config["replay_buffer_size"],
                    gradient_clip=5.0
                )
                agent = DoubleDQNAgent(model_config, agent_config, ACTIONS)
            else:
                model_config = DQNModelConfig(output_dim=len(ACTIONS), hidden_dims=config["hidden_dims"])
                agent_config = AgentConfig(
                    gamma=config["gamma"], batch_size=config["batch_size"],
                    learning_rate=config["learning_rate"], epsilon_start=1.0,
                    epsilon_end=config["epsilon_end"], epsilon_decay=config["epsilon_decay"],
                    target_update_interval=1000, replay_buffer_size=config["replay_buffer_size"],
                    gradient_clip=5.0
                )
                agent = DQNAgent(model_config, agent_config, ACTIONS)
            
            env_config = EnvironmentConfig(enable_ui=False, invalid_move_penalty=-100)
            env = GameEnvironment(env_config)
            
            scores = []
            for ep in range(tune_episodes):
                state = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state)
                    result = env.step(action)
                    agent.store_transition(state, action, result.reward, result.state, result.done)
                    if agent.can_optimize():
                        agent.optimize_model()
                    state = result.state
                    done = result.done
                scores.append(env.get_state()['score'])
            
            env.close()
            avg_score = float(np.mean(scores[-50:]))
            
        else:  # reinforce
            # Sample REINFORCE hyperparameters
            hidden_choice = trial.suggest_categorical("hidden_architecture", [0, 1, 2])
            hidden_options = [[128, 128], [256, 256], [512, 256]]
            
            config = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "hidden_dims": hidden_options[hidden_choice],
                "entropy_coef": trial.suggest_float("entropy_coef", 1e-4, 1e-1, log=True),
            }
            
            from src.agents.reinforce import REINFORCEAgent, REINFORCEConfig
            
            agent = REINFORCEAgent(
                state_dim=16, action_dim=4,
                config=REINFORCEConfig(
                    learning_rate=config["learning_rate"], gamma=config["gamma"],
                    hidden_dims=config["hidden_dims"], use_baseline=True,
                    entropy_coef=config["entropy_coef"]
                )
            )
            
            env_config = EnvironmentConfig(enable_ui=False, invalid_move_penalty=-100)
            env = GameEnvironment(env_config)
            
            scores = []
            for ep in range(tune_episodes):
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.select_action(state, env.board)
                    next_state, reward, done, _, _ = env.step(ACTIONS[action])
                    agent.store_transition(state, action, reward, next_state, done)
                    state = next_state
                agent.finish_episode()
                scores.append(env.board.score)
            
            env.close()
            avg_score = float(np.mean(scores[-50:]))
        
        print(f"[TRIAL {trial.number + 1}/{n_trials}] Score: {avg_score:.2f}")
        return avg_score
    
    # Run Optuna optimization
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Display and save results
    print(f"\n{'='*80}")
    print(f"BEST HYPERPARAMETERS FOUND")
    print(f"{'='*80}")
    print(f"Best Score: {study.best_value:.2f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("evaluations") / f"optuna_{algorithm}_{timestamp}.json"
    results_file.parent.mkdir(exist_ok=True)
    with open(results_file, "w") as f:
        json.dump({
            "algorithm": algorithm,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "n_trials": n_trials,
            "timestamp": timestamp
        }, f, indent=2)
    print(f"[SAVE] Tuning results saved to: {results_file}\n")
    
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED DQN TRAINING (DQN & Double DQN share 95% of code)
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn_variant(algorithm="dqn", resume_mode: str = None, resume_path: str = None):
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

    # Determine resume behavior (none/latest/second-last or explicit path)
    start_episode = 1
    if resume_path:
        resume_candidate = Path(resume_path)
        if resume_candidate.exists():
            try:
                # First, peek at checkpoint to get its architecture
                import torch as _torch
                ckpt_peek = _torch.load(resume_candidate, map_location='cpu', weights_only=False)
                ckpt_model_cfg = ckpt_peek.get('model_config', {})
                ckpt_hidden = ckpt_model_cfg.get('hidden_dims')
                
                # If architectures don't match, rebuild agent with checkpoint's architecture
                if ckpt_hidden and ckpt_hidden != cfg["hidden_dims"]:
                    print(f"[RESUME] Checkpoint has different architecture: {ckpt_hidden} vs current {cfg['hidden_dims']}")
                    print(f"[RESUME] Rebuilding agent to match checkpoint architecture...")
                    
                    # Rebuild with checkpoint's architecture
                    model_config = ModelConfigClass(
                        output_dim=len(ACTIONS),
                        hidden_dims=ckpt_hidden
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
                
                # Load checkpoint into agent
                agent.load(resume_candidate)
                # Attempt to extract episode number from filename
                m = re.search(r"_ep(\d+)\.pth", resume_candidate.name)
                start_episode = int(m.group(1)) + 1 if m else 1
                print(f"[RESUME] Resuming from provided checkpoint: {resume_candidate} (next ep {start_episode})")
                print(f"[RESUME] Loaded state - Steps: {agent.steps_done}, Epsilon: {agent.epsilon:.4f}")
            except Exception as e:
                print(f"[ERROR] Failed to load resume checkpoint {resume_candidate}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[WARNING] Resume path not found: {resume_candidate} - starting from scratch")
    elif resume_mode in ("latest", "second-last"):
        # Find numbered checkpoint files
        chk_files = list(save_dir.glob(f"{save_prefix}_ep*.pth"))
        def _ep_num(p):
            m = re.search(r"_ep(\d+)\.pth", p.name)
            return int(m.group(1)) if m else -1

        chk_files = sorted([p for p in chk_files if _ep_num(p) >= 0], key=_ep_num)
        if chk_files:
            idx = -1 if resume_mode == "latest" else -2
            if len(chk_files) >= abs(idx):
                candidate = chk_files[idx]
                try:
                    # Peek at checkpoint architecture
                    import torch as _torch
                    ckpt_peek = _torch.load(candidate, map_location='cpu', weights_only=False)
                    ckpt_model_cfg = ckpt_peek.get('model_config', {})
                    ckpt_hidden = ckpt_model_cfg.get('hidden_dims')
                    
                    # If architectures don't match, rebuild agent
                    if ckpt_hidden and ckpt_hidden != cfg["hidden_dims"]:
                        print(f"[RESUME] Checkpoint has different architecture: {ckpt_hidden} vs current {cfg['hidden_dims']}")
                        print(f"[RESUME] Rebuilding agent to match checkpoint architecture...")
                        
                        model_config = ModelConfigClass(
                            output_dim=len(ACTIONS),
                            hidden_dims=ckpt_hidden
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
                    
                    agent.load(candidate)
                    start_episode = _ep_num(candidate) + 1
                    print(f"[RESUME] Resuming from {resume_mode} checkpoint: {candidate} (next ep {start_episode})")
                    print(f"[RESUME] Loaded state - Steps: {agent.steps_done}, Epsilon: {agent.epsilon:.4f}")
                except Exception as e:
                    print(f"[ERROR] Failed to load checkpoint {candidate}: {e} - starting from scratch")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[WARNING] Not enough checkpoints to resume '{resume_mode}' - starting from scratch")
        else:
            print(f"[WARNING] No checkpoints found in {save_dir} - starting from scratch")
    
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
        for episode in range(start_episode, episodes + 1):
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

            # Report when we hit a new absolute best score (helps debug plateaus)
            if info['score'] > best_score:
                best_score = info['score']
                print(f"[NEW BEST] Ep {episode:4d} | New best score: {best_score} | Tile: {info['max_tile']}")
            else:
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
                
                # Add moving average info if available. Also show best MA and delta for clarity.
                if moving_averages:
                    cur_ma = moving_averages[-1]
                    ma_delta = cur_ma - best_moving_avg
                    ma_info = f" | MA-100: {cur_ma:6.0f} (best {best_moving_avg:6.0f}, Δ={ma_delta:+6.1f})"
                else:
                    ma_info = ""

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
    
    # Setup logging
    import logging
    logger = logging.getLogger()
    
    # Log play session start
    logger.info("=" * 80)
    logger.info(f"PLAY SESSION STARTED - Model: {model_path}")
    logger.info(f"Algorithm: {algo_name}")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"UI Enabled: {use_ui}")
    logger.info("=" * 80)
    
    # Play episodes
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        msg = f"GAME {ep}/{episodes} - Starting"
        print(f"\n{'='*80}")
        print(msg)
        print(f"{'='*80}")
        logger.info(msg)
        
        while not done:
            # Get current board state info
            info = env.get_state()
            
            # Get valid actions (test each action to see if board changes)
            valid_actions = []
            for action_idx in range(len(ACTIONS)):
                test_board = env.board.clone()
                test_result = test_board.step(ACTIONS[action_idx])
                if test_result.moved:
                    valid_actions.append(action_idx)
            
            # If no valid actions, game is over
            if not valid_actions:
                game_over_msg = f"Step {steps + 1}: No valid moves available - Game Over!"
                print(f"\n{game_over_msg}")
                logger.info(game_over_msg)
                done = True
                break
            
            # Select best action from valid actions only
            import torch
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                q_values = agent.policy_net(state_tensor).squeeze(0)
                
                # Mask invalid actions by setting their Q-values to -infinity
                masked_q_values = torch.full_like(q_values, float('-inf'))
                for valid_idx in valid_actions:
                    masked_q_values[valid_idx] = q_values[valid_idx]
                
                action = int(torch.argmax(masked_q_values).item())
            
            action_name = ACTIONS[action]
            
            # Execute action
            result = env.step(action)
            
            # Print and log step information
            step_msg = f"Step {steps + 1}: Action={action_name.upper()}, Reward={result.reward:+.1f}, Score={info['score']}->{result.info['score']}, MaxTile={result.info['max_tile']}, Empty={result.info['empty_cells']}, Valid={len(valid_actions)}"
            logger.info(step_msg)
            
            print(f"\nStep {steps + 1}:")
            print(f"  Action: {action_name.upper()} (Valid: {len(valid_actions)} options)")
            print(f"  Reward: {result.reward:+.1f}")
            print(f"  Score: {info['score']} -> {result.info['score']}")
            print(f"  Max Tile: {result.info['max_tile']}")
            print(f"  Empty Cells: {result.info['empty_cells']}")
            print(f"  Moved: {result.info.get('moved', 'N/A')}")
            print(f"  Done: {result.done}")
            
            # Print board state (original tile values)
            board_grid = env.board.grid
            print(f"  Board:")
            for row in board_grid:
                row_str = ' '.join(f'{int(val):5d}' for val in row)
                print(f"    {row_str}")
                logger.info(f"    {row_str}")
            
            # Update state
            state = result.state
            total_reward += result.reward
            done = result.done
            steps += 1
            
            # Safety check: prevent infinite loops
            if steps > 10000:
                safety_msg = f"[WARNING] Stopping after {steps} steps (safety limit)"
                print(f"\n{safety_msg}")
                logger.warning(safety_msg)
                break
            
            # Handle pygame events to prevent freezing
            if use_ui and env.ui:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_msg = "[INFO] Window closed by user"
                        print(f"\n{quit_msg}")
                        logger.info(quit_msg)
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        esc_msg = "[INFO] ESC pressed - stopping playback"
                        print(f"\n{esc_msg}")
                        logger.info(esc_msg)
                        env.close()
                        return
        
        # Final game summary
        final_info = env.get_state()
        avg_reward = total_reward/steps if steps > 0 else 0
        
        summary_lines = [
            f"GAME {ep}/{episodes} - COMPLETED",
            f"Final Score: {final_info['score']}",
            f"Max Tile: {final_info['max_tile']}",
            f"Total Steps: {steps}",
            f"Total Reward: {total_reward:.2f}",
            f"Average Reward/Step: {avg_reward:.2f}"
        ]
        
        print(f"\n{'='*80}")
        for line in summary_lines:
            print(line)
            logger.info(line)
        print(f"{'='*80}")
        logger.info("=" * 80)
    
    # Log session end
    final_msg = f"PLAY SESSION COMPLETED - {episodes} game(s) played"
    print(f"\n[INFO] {final_msg}")
    logger.info(final_msg)
    logger.info("=" * 80)
    
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
    train_parser.add_argument(
        '--tune-trials',
        type=int,
        default=0,  # Default to 0 = no tuning
        help='Number of Optuna trials for hyperparameter tuning before training (default: 0 = skip tuning)'
    )
    train_parser.add_argument(
        '--tune-episodes',
        type=int,
        default=200,
        help='Episodes per tuning trial (default: 200, kept short for speed)'
    )
    train_parser.add_argument(
        '--resume',
        choices=['none', 'latest', 'second-last'],
        default='none',
        help='Resume training from checkpoint: latest or second-last (default: none)'
    )
    train_parser.add_argument(
        '--resume-path',
        type=str,
        default=None,
        help='Explicit path to checkpoint (.pth) to resume from (overrides --resume)'
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
        # Optional hyperparameter tuning (only if --tune flag is provided)
        if hasattr(args, 'tune_trials') and args.tune_trials > 0:
            print("\n[INFO] Running hyperparameter tuning before training...\n")
            best_params = tune_hyperparameters(
                args.algorithm, 
                n_trials=args.tune_trials,
                tune_episodes=args.tune_episodes
            )
            
            if best_params:
                # Update CONFIG with best hyperparameters
                print(f"\n[INFO] Applying best hyperparameters for full training\n")
                print("=" * 70)
                for key, value in best_params.items():
                    if key in CONFIG[args.algorithm]:
                        old_value = CONFIG[args.algorithm][key]
                        CONFIG[args.algorithm][key] = value
                        print(f"  {key}: {old_value} -> {value}")
                print("=" * 70)
                print()
        else:
            print("\n[INFO] Skipping hyperparameter tuning, using default/configured parameters\n")
        
        # Run full training with optimized hyperparameters
        if args.algorithm in ['dqn', 'double-dqn']:
            # Pass resume options through to training function
            resume_mode = getattr(args, 'resume', None)
            resume_path = getattr(args, 'resume_path', None)
            if resume_mode == 'none' and not resume_path:
                resume_mode = None
            train_dqn_variant(args.algorithm, resume_mode=resume_mode, resume_path=resume_path)
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
