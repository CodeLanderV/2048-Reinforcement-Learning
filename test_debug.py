"""Simple test script to debug the issue."""
import sys
print("Python version:", sys.version)
print("Starting imports...")

try:
    import torch
    print("✅ PyTorch imported")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import pygame
    print("✅ Pygame imported")
except Exception as e:
    print(f"❌ Pygame import failed: {e}")
    sys.exit(1)

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.environment import GameEnvironment
    print("✅ GameEnvironment imported")
except Exception as e:
    print(f"❌ GameEnvironment import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from pathlib import Path
    checkpoint_path = Path("SavedModels/DQN/dqn_2048_final.pth")
    if checkpoint_path.exists():
        print(f"✅ Checkpoint exists: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"✅ Checkpoint loaded, keys: {list(checkpoint.keys())}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
except Exception as e:
    print(f"❌ Checkpoint loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All checks passed!")
