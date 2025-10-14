
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
from src.environment import ACTIONS, EnvironmentConfig, GameEnvironment


def build_agent(checkpoint_path: Path, device: torch.device) -> DQNAgent:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg_dict = checkpoint.get("model_config", {})
    agent_cfg_dict = checkpoint.get("agent_config", {})

    model_config = DQNModelConfig(**model_cfg_dict) if model_cfg_dict else DQNModelConfig(output_dim=len(ACTIONS))
    agent_config = AgentConfig(**agent_cfg_dict) if agent_cfg_dict else AgentConfig()
    agent = DQNAgent(model_config=model_config, agent_config=agent_config, action_space=ACTIONS, device=device)
    agent.policy_net.load_state_dict(checkpoint["model_state"])
    agent.target_net.load_state_dict(checkpoint["model_state"])
    agent.epsilon = 0.0
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play 2048 with a trained DQN agent")
    parser.add_argument("model_path", type=Path, help="Path to the saved model checkpoint (.pth)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", help="Disable pygame UI and run headless")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between agent moves in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🎮 Loading model from: {args.model_path}")
    print(f"💻 Device: {device}")
    
    try:
        agent = build_agent(args.model_path, device)
        print(f"✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

    env_config = EnvironmentConfig(seed=args.seed, enable_ui=not args.headless)
    env = GameEnvironment(env_config)
    
    print(f"🎯 Starting {args.episodes} episode(s)...")
    print(f"{'🖥️  UI enabled' if not args.headless else '🚫 Headless mode'}")
    print()

    try:
        for episode in range(1, args.episodes + 1):
            state = env.reset()
            done = False
            steps = 0
            while not done:
                if env.ui:
                    event = env.ui.handle_events()
                    if event == "quit":
                        print("\n👋 Quit by user")
                        return
                    if event == "restart":
                        state = env.reset()
                        continue
                action = agent.act_greedy(state)
                result = env.step(action)
                state = result.state
                done = result.done
                steps += 1
                if env.ui and args.delay > 0:
                    time.sleep(args.delay)

            info = env.get_state()
            print(
                f"📊 Episode {episode}: Score={info['score']} MaxTile={info['max_tile']} Steps={steps} Empty={info['empty_cells']}"
            )
    except Exception as e:
        print(f"\n❌ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n✅ Done!")


if __name__ == "__main__":
    main()
