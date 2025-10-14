"""Watch a trained agent play 2048."""

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

# Suppress numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS


def load_agent(checkpoint_path: Path) -> DQNAgent:
    """Load trained DQN agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_cfg_dict = checkpoint.get("model_config", {})
    agent_cfg_dict = checkpoint.get("agent_config", {})
    
    model_config = DQNModelConfig(**model_cfg_dict) if model_cfg_dict else DQNModelConfig(output_dim=len(ACTIONS))
    agent_config = AgentConfig(**agent_cfg_dict) if agent_cfg_dict else AgentConfig()
    
    agent = DQNAgent(
        model_config=model_config,
        agent_config=agent_config,
        action_space=ACTIONS,
        device=device
    )
    
    agent.policy_net.load_state_dict(checkpoint["model_state"])
    agent.target_net.load_state_dict(checkpoint["model_state"])
    agent.epsilon = 0.0  # No exploration during play
    
    return agent


def main():
    parser = argparse.ArgumentParser(description="Watch trained 2048 agent")
    parser.add_argument("model", type=Path, help="Path to model checkpoint (.pth)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of games")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between moves")
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return
    
    print(f"Loading model: {args.model}")
    agent = load_agent(args.model)
    print(f"Model loaded!\n")
    
    env_config = EnvironmentConfig(enable_ui=True)
    env = GameEnvironment(env_config)
    
    print(f"Playing {args.episodes} game(s)...")
    print("Press ESC to quit\n")
    
    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            steps = 0
            
            while not done:
                # Handle UI events
                if env.ui:
                    event = env.ui.handle_events()
                    if event == "quit":
                        print("\nQuit by user")
                        return
                
                # Agent selects action
                action = agent.act_greedy(state)
                
                # Take step
                result = env.step(action)
                state = result.state
                done = result.done
                steps += 1
                
                # Small delay to watch the game
                if args.delay > 0:
                    time.sleep(args.delay)
            
            # Print results
            info = env.get_state()
            print(f"Game {episode}: Score={info['score']:>6} | MaxTile={info['max_tile']:>4} | Steps={steps:>4}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        env.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
