# 2048 Reinforcement Learning Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Reinforcement Learning project for training and comparing different RL algorithms on the 2048 game.

## 🎯 Objective

To design, build, and train Reinforcement Learning agents capable of playing the game 2048. The primary focus is on building a robust, modular architecture from the ground up to facilitate the comparison of multiple RL algorithms.

## 🚀 Features

- **Multiple RL Algorithms**: Implementations of DQN, PPO, A3C, Rainbow, and more
- **Modular Architecture**: Easy to extend and experiment with new algorithms
- **Comprehensive Evaluation**: Built-in metrics and comparison tools
- **Visual Interface**: Pygame-based rendering for watching agent gameplay
- **Experiment Tracking**: Integration with TensorBoard and Weights & Biases
- **Configurable**: YAML-based configuration system for easy experimentation

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **ML Framework**: PyTorch 2.0+
- **Numerical Operations**: NumPy
- **UI/Rendering**: Pygame
- **Tracking**: TensorBoard / Weights & Biases
- **Environment**: OpenAI Gym

## 📁 Project Structure

```
2048-Reinforcement-Learning/
├── src/
│   ├── game/              # 2048 game logic and Gym environment
│   ├── agents/            # RL agent implementations (DQN, etc.)
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops and replay buffer
│   └── utils/             # Helper functions
├── configs/               # YAML configuration files
├── logs/                  # Training logs (TensorBoard)
├── checkpoints/           # Saved model weights
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

**Note**: This is a simplified structure to get started quickly. You can add more directories (tests, notebooks, evaluation, ui) as your project grows.

See [STRUCTURE.md](STRUCTURE.md) for detailed directory descriptions.

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/CodeLanderV/2048-Reinforcement-Learning.git
   cd 2048-Reinforcement-Learning
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

## 🎮 Usage

### Training an Agent

```python
from src.training.trainer import Trainer
from src.utils.config_loader import load_config

# Load configuration
config = load_config('configs/dqn.yaml')

# Create trainer
trainer = Trainer(config)

# Train the agent
trainer.train()
```

Or use the command line:

```bash
python -m src.training.train --config configs/dqn.yaml
```

### Evaluating an Agent

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator(checkpoint_path='checkpoints/best_model.pth')
results = evaluator.evaluate(num_episodes=100)
print(f"Average Score: {results['mean_score']}")
```

### Visualizing Gameplay

```python
from src.ui.renderer import GameRenderer

renderer = GameRenderer(agent=trained_agent)
renderer.play()
```

## 📊 Algorithms

### Implemented
- [ ] **DQN** (Deep Q-Network)
- [ ] **Double DQN**
- [ ] **Dueling DQN**
- [ ] **PPO** (Proximal Policy Optimization)
- [ ] **A3C** (Asynchronous Advantage Actor-Critic)
- [ ] **Rainbow DQN**

### Planned
- [ ] **SAC** (Soft Actor-Critic)
- [ ] **TD3** (Twin Delayed DDPG)
- [ ] **AlphaZero-style** approach

## 📈 Results

Results and comparisons will be documented in the `results/` directory as experiments are completed.

## 🧪 Testing

Run tests using pytest:

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=src tests/
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [2048 Game](https://play2048.co/)
- [Deep Q-Network Paper](https://arxiv.org/abs/1312.5602)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Rainbow Paper](https://arxiv.org/abs/1710.02298)

## 👥 Authors

- **CodeLanderV** - [GitHub](https://github.com/CodeLanderV)

## 🙏 Acknowledgments

- OpenAI Gym for the environment framework
- PyTorch team for the deep learning framework
- The RL community for research and resources
