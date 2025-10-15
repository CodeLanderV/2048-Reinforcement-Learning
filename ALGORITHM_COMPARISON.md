# Algorithm Comparison: 2048 Reinforcement Learning

## Overview

This project implements **4 different algorithms** for playing 2048. Each has unique characteristics and trade-offs.

---

## 🧠 Algorithm Comparison Table

| Algorithm | Type | Learning | Policy | Memory | Network | Best For |
|-----------|------|----------|--------|--------|---------|----------|
| **DQN** | Value-Based | Off-Policy | ε-greedy | Replay Buffer | Q-Network | Stable, sample-efficient learning |
| **Double DQN** | Value-Based | Off-Policy | ε-greedy | Replay Buffer | 2 Q-Networks | Reducing Q-value overestimation |
| **MCTS** | Planning | None | UCB | None | None | No training data, strong baseline |
| **REINFORCE** | Policy Gradient | On-Policy | Stochastic π(a\|s) | Episode Buffer | Policy Network | Direct policy optimization |

---

## 📊 Detailed Algorithm Descriptions

### 1. DQN (Deep Q-Network)

**Type:** Value-based reinforcement learning  
**Learning:** Off-policy (learns from any experience)  
**Policy:** ε-greedy (explores with probability ε)

#### Key Features:
- **Q-Value Function:** Learns `Q(s,a)` = expected return for taking action `a` in state `s`
- **Bellman Update:** `Q(s,a) = r + γ * max Q_target(s', a')`
- **Experience Replay:** Stores transitions in buffer, samples randomly for training
- **Target Network:** Separate network for stable Q-value targets (updated every 1000 steps)
- **Epsilon Decay:** Exploration rate decreases exponentially over time

#### Advantages:
- ✅ Sample efficient (reuses experience)
- ✅ Stable learning with target networks
- ✅ Off-policy (can learn from demonstrations)

#### Disadvantages:
- ❌ Can overestimate Q-values (fixed by Double DQN)
- ❌ Requires careful hyperparameter tuning

#### When to Use:
- Standard baseline for value-based RL
- When you have limited data
- When you need stable, predictable training

---

### 2. Double DQN

**Type:** Value-based reinforcement learning (improved DQN)  
**Learning:** Off-policy  
**Policy:** ε-greedy

#### Key Features:
- **Decoupled Selection/Evaluation:** 
  - Policy network **selects** action: `a* = argmax Q_policy(s', a)`
  - Target network **evaluates** action: `Q_target(s', a*)`
- **Reduces Overestimation:** Addresses DQN's tendency to overestimate Q-values
- **Same Architecture as DQN:** Just changes the target calculation

#### Advantages:
- ✅ All DQN advantages
- ✅ More accurate Q-value estimates
- ✅ Often achieves better final performance

#### Disadvantages:
- ❌ Slightly more complex than DQN
- ❌ Still requires hyperparameter tuning

#### When to Use:
- **Always prefer over vanilla DQN** (strictly better)
- When Q-value accuracy matters
- When you want state-of-the-art value-based performance

---

### 3. MCTS (Monte Carlo Tree Search)

**Type:** Planning algorithm (no learning)  
**Learning:** None  
**Policy:** Implicit through tree search

#### Key Features:
- **UCB Tree Search:** Uses Upper Confidence Bound to balance exploration/exploitation
- **Four Phases:**
  1. **Selection:** Traverse tree using UCB until reaching unexpanded node
  2. **Expansion:** Add new child node for unexplored action
  3. **Simulation:** Random playout from new node until game over
  4. **Backpropagation:** Update visit counts and values up the tree
- **No Neural Network:** Pure algorithmic approach
- **No Memory:** Builds fresh tree for each move

#### Advantages:
- ✅ No training required (works immediately)
- ✅ No hyperparameters to tune (just simulations count)
- ✅ Guaranteed to improve with more compute
- ✅ Strong baseline performance

#### Disadvantages:
- ❌ Computationally expensive per move
- ❌ Doesn't learn from experience
- ❌ Scales poorly to large action spaces

#### When to Use:
- Quick evaluation without training
- Strong baseline to compare against
- When you have computation time per move but no training time
- Domains where perfect simulation is possible

---

### 4. REINFORCE (Monte Carlo Policy Gradient)

**Type:** Policy gradient reinforcement learning  
**Learning:** On-policy (learns only from its own experience)  
**Policy:** Stochastic π(a|s) - outputs probability distribution over actions

#### Key Features:
- **Direct Policy Optimization:** Learns policy π(a|s) directly (not via Q-values)
- **Policy Gradient Theorem:** `∇J(θ) = E[∇log π(a|s) * G_t]`
- **Monte Carlo Returns:** Uses full episode returns `G_t = Σ γ^k * r_{t+k}`
- **Stochastic Policy:** Naturally explores through probability distribution
- **Episode-Based Updates:** Updates after each complete episode

#### Advantages:
- ✅ Directly optimizes policy (no Q-value intermediary)
- ✅ Natural exploration through stochastic policy
- ✅ Can learn stochastic optimal policies
- ✅ Works well in continuous action spaces

#### Disadvantages:
- ❌ High variance in gradient estimates
- ❌ Sample inefficient (on-policy only)
- ❌ Requires full episodes (can't update mid-game)
- ❌ Slower convergence than value-based methods

#### When to Use:
- When you need stochastic policies
- Problems where optimal policy is stochastic
- As foundation for advanced policy gradient methods (A3C, PPO, TRPO)
- When you want to understand policy gradient theory

---

## 🔬 Mathematical Formulations

### DQN Update Rule
```
Q(s,a) ← Q(s,a) + α * [r + γ * max Q_target(s', a') - Q(s,a)]
```

### Double DQN Update Rule
```
a* = argmax_a Q_policy(s', a)
Q(s,a) ← Q(s,a) + α * [r + γ * Q_target(s', a*) - Q(s,a)]
```

### MCTS UCB Formula
```
UCB(node) = Q(node) + c * sqrt(log(N_parent) / N_node)
```
Where:
- `Q(node)` = average value (exploitation)
- `c * sqrt(...)` = exploration bonus
- `c = √2` theoretically optimal

### REINFORCE Policy Gradient
```
∇J(θ) = E_τ [Σ_t ∇log π_θ(a_t|s_t) * G_t]
```
Where:
- `G_t = Σ_{k=0}^{T-t} γ^k * r_{t+k}` (discounted return)
- `∇log π_θ(a_t|s_t)` = score function
- Increases probability of actions with high returns

---

## 🎯 Performance Characteristics (2048 Game)

### Expected Performance (After Training)

| Algorithm | Max Tile (Avg) | Score (Avg) | Training Time | Inference Speed |
|-----------|----------------|-------------|---------------|-----------------|
| **DQN** | 1024-2048 | 15,000-25,000 | ~2 hours | Fast |
| **Double DQN** | 2048-4096 | 20,000-35,000 | ~2 hours | Fast |
| **MCTS** | 512-1024 | 8,000-15,000 | None | Slow (tree search) |
| **REINFORCE** | 512-1024 | 10,000-18,000 | ~3-4 hours | Fast |

### Training Characteristics

| Algorithm | Sample Efficiency | Stability | Hyperparameter Sensitivity |
|-----------|-------------------|-----------|----------------------------|
| **DQN** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ (Moderate) |
| **Double DQN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ (Moderate) |
| **MCTS** | N/A | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (Minimal) |
| **REINFORCE** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ (High) |

---

## 🚀 Usage Guide

### Training Each Algorithm

```bash
# DQN
python 2048RL.py train --algorithm dqn --episodes 2000

# Double DQN
python 2048RL.py train --algorithm double-dqn --episodes 2000

# MCTS (no training, just evaluation)
python 2048RL.py train --algorithm mcts --episodes 50

# REINFORCE
python 2048RL.py train --algorithm reinforce --episodes 3000
```

### Playing with Trained Models

```bash
# DQN
python 2048RL.py play --model models/DQN/dqn_final.pth

# Double DQN
python 2048RL.py play --model models/DoubleDQN/double_dqn_final.pth

# REINFORCE
python 2048RL.py play --model models/REINFORCE/reinforce_final.pth

# MCTS (no model needed)
python 2048RL.py train --algorithm mcts --episodes 5
```

---

## 🔧 Hyperparameter Tuning

### Supported Algorithms
- ✅ DQN
- ✅ Double DQN
- ❌ MCTS (no hyperparameters to tune)
- ⚠️ REINFORCE (coming soon)

### Tuning Methods

```bash
# Grid Search (exhaustive)
python hyperparam_tuning.py --algorithm dqn --method grid --trials 20

# Random Search (faster)
python hyperparam_tuning.py --algorithm double-dqn --method random --trials 30

# Optuna (Bayesian optimization - most efficient)
python hyperparam_tuning.py --algorithm dqn --method optuna --trials 50
```

---

## 📚 References

### DQN
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." Nature.

### Double DQN
- van Hasselt et al. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI.

### MCTS
- Browne et al. (2012). "A Survey of Monte Carlo Tree Search Methods." IEEE TCIAIG.
- Kocsis & Szepesvári (2006). "Bandit based Monte-Carlo Planning." ECML.

### REINFORCE
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine Learning, 8, 229-256.

---

## 🎓 Learning Path

### For Beginners:
1. **Start with MCTS** - Understand tree search without neural networks
2. **Move to DQN** - Learn value-based RL with neural networks
3. **Try Double DQN** - See how small algorithmic changes improve performance
4. **Explore REINFORCE** - Understand policy gradient methods

### For Researchers:
- Compare all algorithms on same task
- Analyze learning curves and convergence
- Study hyperparameter sensitivity
- Implement advanced variants (A3C, PPO, SAC)

---

## 🤔 Which Algorithm Should I Use?

### Quick Decision Tree:

**Need best performance?**  
→ **Double DQN** (most reliable)

**No time for training?**  
→ **MCTS** (works immediately)

**Learning about RL theory?**  
→ **REINFORCE** (foundational algorithm)

**Building on value-based methods?**  
→ **DQN** (standard baseline)

**Need stochastic policy?**  
→ **REINFORCE** (only one with explicit stochastic policy)

---

## 📈 Expected Training Curves

### DQN / Double DQN
- **Episodes 0-500:** Rapid improvement (random → basic strategy)
- **Episodes 500-1500:** Steady improvement (learning merge patterns)
- **Episodes 1500-2000:** Fine-tuning (approaching optimal play)

### REINFORCE
- **Episodes 0-1000:** High variance, slow improvement
- **Episodes 1000-2500:** Gradual learning of better policies
- **Episodes 2500-3000:** Stabilization (may need more episodes)

### MCTS
- **Immediate performance** (no training curve)
- Performance scales linearly with simulation count

---

*Last Updated: October 15, 2025*
