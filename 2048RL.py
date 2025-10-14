"""
2048 Reinforcement Learning - Central Control Panel
====================================================

Easy configuration and training for all algorithms.
Just edit the CONFIG section below and run!
"""

import sys
import warnings
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# CONFIGURATION - Edit these to tune your training!
# ============================================================================

CONFIG = {
    # Which algorithm to train?
    "algorithm": "dqn",  # Options: "dqn", "double-dqn", "policy-gradient", "mcts"
    
    # Training Settings
    "episodes": 2000,           # Number of training episodes
    "enable_ui": True,          # Show pygame window during training
    "enable_plots": True,       # Show live training plots
    
    # DQN / Double DQN Settings
    "dqn": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 50000,
        "replay_buffer_size": 100_000,
        "target_update_interval": 1000,
        "gradient_clip": 5.0,
        "hidden_dims": (256, 256),
    },
    
    # Policy Gradient Settings
    "policy_gradient": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "gradient_clip": 5.0,
        "hidden_dims": (256, 256),
    },
    
    # MCTS Settings
    "mcts": {
        "simulations": 100,
        "exploration_constant": 1.41,
    },
    
    # Environment Settings
    "invalid_move_penalty": -5.0,
    
    # Saving Settings
    "save_dir": "models",       # Where to save models
    "checkpoint_interval": 100, # Save every N episodes
    
    # Evaluation Settings
    "eval_episodes": 5,         # Episodes for final evaluation
}

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_dqn():
    """Train DQN agent with configured settings."""
    import matplotlib.pyplot as plt
    import numpy as np
    from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
    from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
    from src.utils import TrainingTimer, EvaluationLogger
    
    print("=" * 80)
    print("🎮 TRAINING DQN AGENT")
    print("=" * 80)
    
    # Setup timer
    timer = TrainingTimer().start()
    
    # Build agent with custom config
    cfg = CONFIG["dqn"]
    model_config = DQNModelConfig(
        output_dim=len(ACTIONS),
        hidden_dims=cfg["hidden_dims"]
    )
    agent_config = AgentConfig(
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
    agent = DQNAgent(model_config=model_config, agent_config=agent_config, action_space=ACTIONS)
    
    # Setup environment
    env_config = EnvironmentConfig(
        enable_ui=CONFIG["enable_ui"],
        invalid_move_penalty=CONFIG["invalid_move_penalty"]
    )
    env = GameEnvironment(env_config)
    
    # Training tracking
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    
    # Setup plotting
    if CONFIG["enable_plots"]:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    save_dir = Path(CONFIG["save_dir"]) / "DQN"
    episodes = CONFIG["episodes"]
    
    print(f"\n📊 Training for {episodes} episodes")
    print(f"💾 Models will be saved to: {save_dir}")
    print(f"🎯 Close plot window to stop early\n")
    
    best_score = 0
    best_tile = 0
    
    try:
        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(state)
                result = env.step(action)
                
                agent.store_transition(state, action, result.reward, result.state, result.done)
                
                if agent.can_optimize():
                    agent.optimize_model()
                
                state = result.state
                episode_reward += result.reward
                done = result.done
            
            # Track metrics
            info = env.get_state()
            episode_rewards.append(episode_reward)
            episode_scores.append(info['score'])
            episode_max_tiles.append(info['max_tile'])
            
            best_score = max(best_score, info['score'])
            best_tile = max(best_tile, info['max_tile'])
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_score = np.mean(episode_scores[-50:])
                elapsed = timer.elapsed_str()
                print(f"Ep {episode:4d} | Reward: {avg_reward:7.2f} | Score: {avg_score:6.0f} | "
                      f"Tile: {episode_max_tiles[-1]:4d} | ε: {agent.epsilon:.3f} | Time: {elapsed}")
            
            # Save checkpoint
            if episode % CONFIG["checkpoint_interval"] == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"dqn_ep{episode}.pth"
                agent.save(checkpoint_path, episode)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Update plot
            if CONFIG["enable_plots"] and episode % 5 == 0:
                ax1.clear()
                ax2.clear()
                
                ax1.plot(episode_rewards, alpha=0.3, color='blue')
                if len(episode_rewards) >= 50:
                    moving_avg = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
                    ax1.plot(range(49, len(episode_rewards)), moving_avg, color='blue', linewidth=2)
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.set_title('DQN Training Progress')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(episode_scores, alpha=0.3, color='green', label='Score')
                ax2.plot(episode_max_tiles, alpha=0.3, color='red', label='Max Tile')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.pause(0.01)
                
                if not plt.fignum_exists(fig.number):
                    print("\n⚠️  Plot closed - stopping early")
                    break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    
    finally:
        timer.stop()
        
        # Save final model
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "dqn_final.pth"
        agent.save(final_path, episode)
        print(f"\n💾 Final model saved: {final_path}")
        
        # Log evaluation
        logger = EvaluationLogger()
        final_avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
        
        logger.log_training(
            algorithm="DQN",
            episodes=episode,
            final_avg_reward=final_avg_reward,
            max_tile=best_tile,
            final_score=best_score,
            training_time=timer.elapsed_str(),
            model_path=str(final_path),
            notes=f"LR={cfg['learning_rate']}, ε={agent.epsilon:.4f}"
        )
        
        env.close()
        if CONFIG["enable_plots"]:
            plt.ioff()
            plt.close('all')
        
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"⏱️  Total Time: {timer.elapsed_str()}")
        print(f"🏆 Best Score: {best_score}")
        print(f"🎯 Best Tile: {best_tile}")
        print(f"{'='*80}\n")


def train_double_dqn():
    """Train Double DQN agent."""
    print("Double DQN training - Copy DQN implementation and adjust imports to double_dqn")
    print("Use same config structure from CONFIG['dqn']")


def train_policy_gradient():
    """Train Policy Gradient agent."""
    print("Policy Gradient training - Similar to DQN but use CONFIG['policy_gradient']")


def train_mcts():
    """Run MCTS simulations."""
    print("MCTS - No training needed, just run simulations with CONFIG['mcts']")


# ============================================================================
# PLAY FUNCTION
# ============================================================================

def play_model(model_path: str = None, episodes: int = None):
    """Watch a trained agent play."""
    import time
    import torch
    from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
    from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
    
    if model_path is None:
        model_path = f"models/{CONFIG['algorithm'].upper()}/{CONFIG['algorithm']}_final.pth"
    
    if episodes is None:
        episodes = CONFIG["eval_episodes"]
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"🎮 Loading model: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Print model info
    print(f"📊 Model Info:")
    print(f"   Episode trained: {checkpoint.get('episode', 'Unknown')}")
    print(f"   Steps trained: {checkpoint.get('steps', 'Unknown')}")
    print(f"   Final epsilon: {checkpoint.get('epsilon', 'Unknown')}")
    print(f"   Device: {device}")
    
    model_cfg = checkpoint.get("model_config", {})
    agent_cfg = checkpoint.get("agent_config", {})
    
    model_config = DQNModelConfig(**model_cfg) if model_cfg else DQNModelConfig(output_dim=len(ACTIONS))
    agent_config = AgentConfig(**agent_cfg) if agent_cfg else AgentConfig()
    
    agent = DQNAgent(model_config=model_config, agent_config=agent_config, action_space=ACTIONS, device=device)
    agent.policy_net.load_state_dict(checkpoint["model_state"])
    agent.target_net.load_state_dict(checkpoint["model_state"])
    agent.epsilon = 0.0
    
    print(f"✅ Model loaded successfully!\n")
    
    env_config = EnvironmentConfig(enable_ui=True)
    env = GameEnvironment(env_config)
    
    print(f"🎯 Playing {episodes} game(s)...")
    print(f"💡 Press ESC to quit, 'r' to restart\n")
    print("=" * 70)
    
    total_scores = []
    total_max_tiles = []
    total_steps = []
    
    try:
        for episode in range(1, episodes + 1):
            print(f"\n🎮 Game {episode}/{episodes}")
            print("-" * 70)
            
            state = env.reset()
            done = False
            steps = 0
            episode_reward = 0
            invalid_moves = 0
            valid_moves = 0
            
            while not done:
                # Agent selects action
                action = agent.act_greedy(state)
                action_name = ACTIONS[action]
                
                # Take step (environment handles UI events internally)
                result = env.step(action)
                
                # Track statistics
                episode_reward += result.reward
                if result.reward == -5.0:
                    invalid_moves += 1
                elif result.reward > 0:
                    valid_moves += 1
                
                # Print every 10 steps
                if steps % 10 == 0 and steps > 0:
                    info = env.get_state()
                    print(f"   Step {steps:3d}: Score={info['score']:5d} | MaxTile={info['max_tile']:4d} | "
                          f"Valid={valid_moves:3d} | Invalid={invalid_moves:3d}")
                
                # Check if user quit
                if result.info.get("terminated_by_user"):
                    print("\n👋 Quit by user")
                    env.close()
                    return
                
                state = result.state
                done = result.done
                steps += 1
                time.sleep(0.05)  # Faster playback
            
            # Final game statistics
            info = env.get_state()
            total_scores.append(info['score'])
            total_max_tiles.append(info['max_tile'])
            total_steps.append(steps)
            
            print("-" * 70)
            print(f"📊 Game {episode} Results:")
            print(f"   Final Score:      {info['score']:>6}")
            print(f"   Max Tile:         {info['max_tile']:>6}")
            print(f"   Total Steps:      {steps:>6}")
            print(f"   Valid Moves:      {valid_moves:>6}")
            print(f"   Invalid Moves:    {invalid_moves:>6}")
            print(f"   Total Reward:     {episode_reward:>6.1f}")
            print(f"   Empty Cells Left: {info['empty_cells']:>6}")
            print("-" * 70)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        env.close()
        
        # Print summary if multiple games
        if len(total_scores) > 0:
            print("\n" + "=" * 70)
            print("📈 SUMMARY STATISTICS")
            print("=" * 70)
            print(f"Games Played:       {len(total_scores)}")
            print(f"Average Score:      {sum(total_scores)/len(total_scores):.1f}")
            print(f"Best Score:         {max(total_scores)}")
            print(f"Worst Score:        {min(total_scores)}")
            print(f"Average Max Tile:   {sum(total_max_tiles)/len(total_max_tiles):.0f}")
            print(f"Best Max Tile:      {max(total_max_tiles)}")
            print(f"Average Steps:      {sum(total_steps)/len(total_steps):.1f}")
            print("=" * 70)
        
        print("\n✅ Done!")


# ============================================================================
# MAIN MENU
# ============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*80)
    print("CURRENT CONFIGURATION")
    print("="*80)
    print(f"Algorithm:        {CONFIG['algorithm']}")
    print(f"Episodes:         {CONFIG['episodes']}")
    print(f"UI Enabled:       {CONFIG['enable_ui']}")
    print(f"Plots Enabled:    {CONFIG['enable_plots']}")
    print(f"Save Directory:   {CONFIG['save_dir']}")
    
    if CONFIG['algorithm'] in ['dqn', 'double-dqn']:
        cfg = CONFIG['dqn']
        print(f"\nDQN Settings:")
        print(f"  Learning Rate:  {cfg['learning_rate']}")
        print(f"  Gamma:          {cfg['gamma']}")
        print(f"  Batch Size:     {cfg['batch_size']}")
        print(f"  Epsilon Decay:  {cfg['epsilon_decay']}")
    
    print("="*80 + "\n")


def main():
    """Main menu."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="2048 RL - Easy Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 2048RL.py train              # Train with current CONFIG
  python 2048RL.py train --episodes 5000  # Override episodes
  python 2048RL.py play               # Play trained model
  python 2048RL.py config             # Show current config
        """
    )
    
    parser.add_argument(
        "action",
        choices=["train", "play", "config"],
        help="Action to perform"
    )
    parser.add_argument("--episodes", type=int, help="Override number of episodes")
    parser.add_argument("--algorithm", type=str, help="Override algorithm")
    parser.add_argument("--model", type=str, help="Path to model for playing")
    
    args = parser.parse_args()
    
    # Override config if specified
    if args.episodes:
        CONFIG["episodes"] = args.episodes
    if args.algorithm:
        CONFIG["algorithm"] = args.algorithm
    
    if args.action == "config":
        print_config()
    
    elif args.action == "train":
        print_config()
        
        if CONFIG["algorithm"] == "dqn":
            train_dqn()
        elif CONFIG["algorithm"] == "double-dqn":
            train_double_dqn()
        elif CONFIG["algorithm"] == "policy-gradient":
            train_policy_gradient()
        elif CONFIG["algorithm"] == "mcts":
            train_mcts()
        else:
            print(f"❌ Unknown algorithm: {CONFIG['algorithm']}")
    
    elif args.action == "play":
        play_model(model_path=args.model, episodes=args.episodes)


if __name__ == "__main__":
    main()
