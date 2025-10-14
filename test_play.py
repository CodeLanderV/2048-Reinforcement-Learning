"""Simple test to debug play functionality."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.dqn import DQNAgent, DQNModelConfig, AgentConfig
from src.environment import GameEnvironment, EnvironmentConfig, ACTIONS

# Load model
model_path = Path("models/DQN/dqn_final.pth")
print(f"Loading model: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Episode: {checkpoint.get('episode', 'N/A')}")

model_cfg = checkpoint.get("model_config", {})
agent_cfg = checkpoint.get("agent_config", {})

print(f"Model config: {model_cfg}")
print(f"Agent config: {agent_cfg}")

model_config = DQNModelConfig(**model_cfg) if model_cfg else DQNModelConfig(output_dim=len(ACTIONS))
agent_config = AgentConfig(**agent_cfg) if agent_cfg else AgentConfig()

agent = DQNAgent(model_config=model_config, agent_config=agent_config, action_space=ACTIONS, device=device)
agent.policy_net.load_state_dict(checkpoint["model_state"])
agent.target_net.load_state_dict(checkpoint["model_state"])
agent.epsilon = 0.0

print(f"Model loaded successfully!")
print(f"Agent epsilon: {agent.epsilon}")

# Create environment with UI
env_config = EnvironmentConfig(enable_ui=True)
env = GameEnvironment(env_config)

print(f"\nStarting game...")
state = env.reset()
print(f"Initial state shape: {state.shape}")
print(f"UI enabled: {env.ui is not None}")

done = False
steps = 0
max_steps = 1000  # Safety limit

try:
    while not done and steps < max_steps:
        # Select action
        action = agent.act_greedy(state)
        print(f"Step {steps}: Action {action} ({ACTIONS[action]})", end="")
        
        # Take step
        result = env.step(action)
        
        print(f" -> Reward: {result.reward:.2f}, Done: {result.done}")
        
        # Check for user quit
        if result.info.get("terminated_by_user"):
            print("User quit!")
            break
        
        state = result.state
        done = result.done
        steps += 1
        
        time.sleep(0.2)  # Slower so we can see
    
    info = env.get_state()
    print(f"\nGame finished!")
    print(f"Steps: {steps}")
    print(f"Score: {info['score']}")
    print(f"Max Tile: {info['max_tile']}")
    print(f"Empty cells: {info['empty_cells']}")

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
finally:
    env.close()
    print("Done!")
