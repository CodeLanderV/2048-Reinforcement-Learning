# 2048 Reinforcement Learning Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Reinforcement Learning project for training and comparing different RL algorithms on the 2048 game.

## 🎯 Objective

To design, build, and train Reinforcement Learning agents capable of playing the game 2048. The primary focus is on building a robust, modular architecture from the ground up to facilitate the comparison of multiple RL algorithms.


## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **ML Framework**: PyTorch 2.0+
- **Numerical Operations**: NumPy
- **UI/Rendering**: Pygame
- **Tracking**: TensorBoard / Weights & Biases
- **Environment**: OpenAI Gym

## 📁 Project Structure


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


## 📊 Algorithms

### Implemented
- [ ] **DQN** (Deep Q-Network)
~_- [ ] **Double DQN**
- [ ] **Dueling DQN**
- [ ] **PPO** (Proximal Policy Optimization)
- [ ] **A3C** (Asynchronous Advantage Actor-Critic)
- [ ] **Rainbow DQN**_~

## 📈 Results

Results and comparisons will be documented in the `results/` directory as experiments are completed.


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## 👥 Authors

- **CodeLanderV** - [GitHub](https://github.com/CodeLanderV)

## 🙏 Acknowledgments

- OpenAI Gym for the environment framework
- PyTorch team for the deep learning framework
- The RL community for research and resources
