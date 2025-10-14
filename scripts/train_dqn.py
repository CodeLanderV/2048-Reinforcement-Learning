"""Train DQN agent on 2048."""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS
from src.utils import TrainingTimer, EvaluationLogger


def train_dqn(episodes: int = 2000, save_dir: Path = Path("models/DQN")):
    """Train DQN agent with timer and evaluation logging."""
    
    print("=" * 80)
    print("🎮 TRAINING DQN AGENT")
    print("=" * 80)
    
    # Setup timer
    timer = TrainingTimer().start()
    
    # Setup agent and environment
    model_config = DQNModelConfig(output_dim=len(ACTIONS))
    agent_config = AgentConfig()
    agent = DQNAgent(model_config=model_config, agent_config=agent_config, action_space=ACTIONS)
    
    env_config = EnvironmentConfig(enable_ui=True)
    env = GameEnvironment(env_config)
    
    # Training tracking
    episode_rewards = []
    episode_scores = []
    episode_max_tiles = []
    losses = []
    
    # Setup matplotlib for live plotting
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
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
            episode_loss = []
            
            while not done:
                action = agent.select_action(state)
                result = env.step(action)
                
                agent.store_transition(state, action, result.reward, result.state, result.done)
                
                if agent.can_optimize():
                    loss = agent.optimize_model()
                    if loss is not None:
                        episode_loss.append(loss)
                
                state = result.state
                episode_reward += result.reward
                done = result.done
            
            # Track metrics
            info = env.get_state()
            episode_rewards.append(episode_reward)
            episode_scores.append(info['score'])
            episode_max_tiles.append(info['max_tile'])
            
            if episode_loss:
                losses.append(np.mean(episode_loss))
            
            # Update best
            if info['score'] > best_score:
                best_score = info['score']
            if info['max_tile'] > best_tile:
                best_tile = info['max_tile']
            
            # Print progress every 10 episodes
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_score = np.mean(episode_scores[-50:])
                elapsed = timer.elapsed_str()
                print(f"Ep {episode:4d} | Reward: {avg_reward:7.2f} | Score: {avg_score:6.0f} | "
                      f"Tile: {episode_max_tiles[-1]:4d} | ε: {agent.epsilon:.3f} | "
                      f"Time: {elapsed}")
            
            # Save checkpoint every 100 episodes
            if episode % 100 == 0:
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"dqn_2048_ep{episode}.pth"
                agent.save(checkpoint_path, episode)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Update plot
            if episode % 5 == 0:
                ax1.clear()
                ax2.clear()
                
                # Plot rewards
                ax1.plot(episode_rewards, alpha=0.3, color='blue')
                if len(episode_rewards) >= 50:
                    moving_avg = np.convolve(episode_rewards, np.ones(50)/50, mode='valid')
                    ax1.plot(range(49, len(episode_rewards)), moving_avg, color='blue', linewidth=2)
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.set_title('Training Progress')
                ax1.grid(True, alpha=0.3)
                
                # Plot scores and max tiles
                ax2.plot(episode_scores, alpha=0.3, color='green', label='Score')
                ax2.plot(episode_max_tiles, alpha=0.3, color='red', label='Max Tile')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Value')
                ax2.set_title('Performance Metrics')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.pause(0.01)
                
                if not plt.fignum_exists(fig.number):
                    print("\n⚠️  Plot closed - stopping training early")
                    break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    finally:
        # Stop timer
        timer.stop()
        
        # Save final model
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / "dqn_2048_final.pth"
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
            notes=f"Epsilon final: {agent.epsilon:.4f}"
        )
        
        # Cleanup
        env.close()
        plt.ioff()
        plt.close('all')
        
        print(f"\n{'='*80}")
        print(f"✅ Training Complete!")
        print(f"⏱️  Total Time: {timer.elapsed_str()}")
        print(f"🏆 Best Score: {best_score}")
        print(f"🎯 Best Tile: {best_tile}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000, help="Number of episodes")
    args = parser.parse_args()
    
    train_dqn(episodes=args.episodes)
