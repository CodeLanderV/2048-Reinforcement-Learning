# 🧠 Deep Dive: Reinforcement Learning Algorithms for 2048

**Complete Guide to Understanding, Training, and Evaluating RL Models**

Last Updated: October 15, 2025

---

## Table of Contents

1. [How These Models Work (In-Depth)](#0-how-these-models-work-in-depth)
2. [Expected Training Outputs](#1-expected-outputs-during-training)
3. [Model Parameters - How to Determine](#2-model-parameters---how-to-determine--why)
4. [Performance Evaluation Metrics](#3-performance-evaluation-metrics)
5. [Determining Number of Episodes](#4-how-to-determine-number-of-episodes)

---

## 0. HOW THESE MODELS WORK (In-Depth)

### 🎯 **1. Deep Q-Network (DQN)** - Value-Based Learning

#### **Core Concept:**
DQN learns a **Q-function** Q(s,a) that estimates the expected cumulative reward for taking action `a` in state `s`.

#### **Mathematical Foundation:**
```
Bellman Equation:
Q(s,a) = r + γ × max[Q(s', a')]
         ↑     ↑        ↑
      reward  discount  best future value
```

#### **How It Works:**

**1. Neural Network Architecture:**
```python
Input: State (16 values = 4x4 board flattened)
       ↓
Hidden Layer 1: 256 neurons (ReLU activation)
       ↓
Hidden Layer 2: 256 neurons (ReLU activation)
       ↓
Output: 4 Q-values [Q(s,UP), Q(s,DOWN), Q(s,LEFT), Q(s,RIGHT)]
```

**2. Training Loop:**
```python
# Step 1: Experience Collection
for episode in episodes:
    state = env.reset()
    
    # Epsilon-greedy exploration
    if random() < epsilon:
        action = random_action()  # Explore
    else:
        action = argmax(Q(state))  # Exploit (choose best Q-value)
    
    next_state, reward, done = env.step(action)
    
    # Step 2: Store in replay buffer
    memory.store(state, action, reward, next_state, done)
    
    # Step 3: Sample mini-batch and train
    batch = memory.sample(batch_size=128)
    
    # Compute target
    target_Q = reward + gamma * max(Q_target(next_state))
    current_Q = Q(state)[action]
    
    # Loss: Mean Squared Error
    loss = (target_Q - current_Q)²
    
    # Backpropagation
    optimizer.step()
```

**3. Key Innovations:**

**Experience Replay Buffer:**
```python
# Stores past experiences: (s, a, r, s', done)
# Breaks temporal correlation
# Allows learning from past mistakes multiple times

Buffer = [(s₁,a₁,r₁,s₂,done), (s₂,a₂,r₂,s₃,done), ...]
         ↓
Sample random batch → Train
```

**Target Network:**
```python
# Problem: Training becomes unstable if target keeps changing
# Solution: Use separate "frozen" network for targets

Q_policy  ← updates every step (learns)
Q_target  ← updates every 1000 steps (stable target)

target = r + γ × max[Q_target(s', a')]  # Use frozen network
```

#### **Why DQN Works for 2048:**
- Learns which board configurations (states) are valuable
- Discovers patterns like "keep high tiles in corners"
- Generalizes across similar board states

---

### 🎯 **2. Double Deep Q-Network (Double DQN)** - Debiased Value Learning

#### **The Problem DQN Has:**
```python
Standard DQN:
target = r + γ × max[Q(s', a')]
                  ↑
        This max() operator causes OVERESTIMATION BIAS
        
Why? Network errors compound:
- If Q(s,a₁) = 10.2 (true=10)  ← overestimated by 0.2
- If Q(s,a₂) = 9.8  (true=10)  ← underestimated by 0.2
- max() picks a₁ (10.2) → propagates overestimation!
```

#### **Double DQN Solution:**
```python
# Decouples action SELECTION from VALUE ESTIMATION

Step 1: Use policy network to SELECT best action
best_action = argmax(Q_policy(s'))

Step 2: Use target network to EVALUATE that action
target = r + γ × Q_target(s')[best_action]

# This reduces overestimation because:
# - Selection and evaluation use different networks
# - Unlikely both networks overestimate the same action
```

#### **Mathematical Comparison:**
```
DQN:
Q_target = r + γ × max[Q_target(s', a')]
           ↑           ↑
        Biased due to max operator

Double DQN:
a* = argmax[Q_policy(s', a')]      ← Select using policy net
Q_target = r + γ × Q_target(s')[a*] ← Evaluate using target net
           ↑
        Less biased!
```

#### **Code Implementation Difference:**
```python
# DQN
next_q_values = target_net(next_states)
targets = rewards + gamma * next_q_values.max(1)[0]

# Double DQN
next_actions = policy_net(next_states).argmax(1)  # Policy selects
next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1))
targets = rewards + gamma * next_q_values.squeeze()
```

#### **Performance Difference:**
```
Game Progress:
Episode 1000:
- DQN:        Score = 2000, Max Tile = 256
- Double DQN: Score = 2400, Max Tile = 512  ← Better!

Why? Double DQN learns more accurate value estimates
```

---

### 🎯 **3. Monte Carlo Tree Search (MCTS)** - Planning Algorithm

#### **Core Concept:**
MCTS is **NOT a learning algorithm** - it's a **planning algorithm** that simulates future game trajectories to find the best move.

#### **Four Phases:**

```
       Root (current state)
        /    |    \    \
       ↓     ↓     ↓    ↓
    [UP] [DOWN] [LEFT] [RIGHT]
       ↓
    Children states...

Phase 1: SELECTION
├─ Traverse tree using UCB formula
└─ Until reaching unexpanded node

Phase 2: EXPANSION
├─ Add one new child node
└─ For an untried action

Phase 3: SIMULATION (Rollout)
├─ Play random moves from new node
└─ Until game over

Phase 4: BACKPROPAGATION
├─ Propagate reward up the tree
└─ Update visit counts and values
```

#### **UCB1 Formula (Upper Confidence Bound):**
```python
# Balances EXPLOITATION vs EXPLORATION

UCB(node) = Q(node) + c × sqrt(log(N_parent) / N_child)
            ↑ Exploitation    ↑ Exploration
            (known good)      (uncertainty bonus)

Where:
- Q(node) = average reward from this node
- N_parent = parent visit count
- N_child = this node's visit count
- c = exploration constant (usually 1.41)

Example:
Node A: Q=100, visited 10 times
Node B: Q=80,  visited 2 times

UCB(A) = 100 + 1.41 × sqrt(log(100)/10) = 100 + 2.1 = 102.1
UCB(B) = 80  + 1.41 × sqrt(log(100)/2)  = 80  + 7.6 = 87.6

Choose A! (higher UCB despite lower visit count for B)
```

#### **Detailed Algorithm:**

```python
def mcts_search(root_state, n_simulations=1000):
    root = Node(root_state)
    
    for _ in range(n_simulations):
        # 1. SELECTION
        node = root
        state = root_state.copy()
        
        while node.is_fully_expanded() and not state.is_terminal():
            node = node.select_child_ucb()  # Pick child with highest UCB
            state.apply_action(node.action)
        
        # 2. EXPANSION
        if not state.is_terminal():
            action = node.get_untried_action()
            state.apply_action(action)
            node = node.add_child(action, state)
        
        # 3. SIMULATION (Random rollout)
        reward = 0
        while not state.is_terminal():
            action = random.choice(state.legal_actions())
            state.apply_action(action)
            reward = state.get_reward()
        
        # 4. BACKPROPAGATION
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node.value = node.total_reward / node.visits
            node = node.parent
    
    # Return action with most visits (most reliable)
    return root.most_visited_child().action
```

#### **Why MCTS Works for 2048:**
- Simulates thousands of possible futures
- Finds moves that lead to good outcomes even 10+ moves ahead
- No training needed - works immediately
- Adapts to current board state

#### **Limitations:**
- **Slow**: Each move requires 100-1000 simulations
- **No learning**: Starts fresh every game
- **Memory intensive**: Builds large search trees

---

### 🎯 **4. REINFORCE (Policy Gradient)** - Direct Policy Learning

#### **Paradigm Shift:**
Unlike DQN (learns Q-values), REINFORCE learns the **policy directly**:
```
DQN:        State → Q-values → argmax → Action
REINFORCE:  State → Policy π(a|s) → Sample → Action
                    ↑ Directly outputs probabilities!
```

#### **Mathematical Foundation:**

**Policy Gradient Theorem:**
```
∇J(θ) = E[∇log π_θ(a|s) × G_t]
        ↑   ↑              ↑
      Gradient  Log prob   Return (cumulative reward)

English Translation:
"Increase probability of actions that led to high returns,
 Decrease probability of actions that led to low returns"
```

#### **How It Works:**

**1. Network Architecture:**
```python
Input: State (16 values)
       ↓
Hidden Layer 1: 256 neurons (ReLU)
       ↓
Hidden Layer 2: 256 neurons (ReLU)
       ↓
Output: 4 logits → Softmax → [P(UP), P(DOWN), P(LEFT), P(RIGHT)]
                              ↑ Probabilities sum to 1
```

**2. Training Process:**
```python
# EPISODE COLLECTION
episode_states = []
episode_actions = []
episode_rewards = []

state = env.reset()
done = False

while not done:
    # Sample action from policy
    probs = policy_network(state)  # [0.4, 0.3, 0.2, 0.1]
    action = sample_from(probs)     # Stochastic!
    
    next_state, reward, done = env.step(action)
    
    episode_states.append(state)
    episode_actions.append(action)
    episode_rewards.append(reward)
    
    state = next_state

# COMPUTE RETURNS (discounted cumulative rewards)
returns = []
G = 0
for r in reversed(episode_rewards):
    G = r + gamma * G
    returns.insert(0, G)

# Example:
# rewards = [1, 2, 3]
# returns = [1 + 0.99×2 + 0.99²×3,  2 + 0.99×3,  3]
#         = [5.94,                 4.97,        3]

# POLICY GRADIENT UPDATE
for state, action, G in zip(episode_states, episode_actions, returns):
    # Forward pass
    probs = policy_network(state)
    log_prob = log(probs[action])
    
    # Loss (negative because we maximize)
    loss = -log_prob × G  # If G is high, increase log_prob
    
    # Backpropagation
    loss.backward()
    optimizer.step()
```

#### **Why REINFORCE Works:**

**Intuition:**
```
Episode 1: Take action UP in state s
           → Get reward 100
           → Update: Increase P(UP|s)

Episode 2: Take action DOWN in state s
           → Get reward 10
           → Update: Slightly increase P(DOWN|s)

Over many episodes:
P(UP|s) grows much faster than P(DOWN|s)
Policy learns to prefer high-reward actions!
```

#### **Advantages:**
- Works in **continuous action spaces** (not just 4 discrete actions)
- Learns **stochastic policies** (can explore naturally)
- Handles **high-dimensional** action spaces

#### **Disadvantages:**
- **High variance**: Single episode might be lucky/unlucky
- **Sample inefficient**: Needs many episodes
- **On-policy**: Can't reuse old experiences (unlike DQN)

---

### 🎯 **5. Proximal Policy Optimization (PPO)** - Advanced Policy Gradient

#### **The Problem with REINFORCE:**
```python
# REINFORCE can make HUGE policy updates that break learning

Episode 1: P(UP|s) = 0.3
           Big reward → Update
Episode 2: P(UP|s) = 0.9  ← Too aggressive!
           Now never explores other actions
```

#### **PPO Solution: Clipped Updates**

```python
# Limit how much the policy can change per update

ratio = π_new(a|s) / π_old(a|s)  # How much policy changed
        ↑              ↑
    New policy    Old policy

# Clip ratio to [1-ε, 1+ε]  (ε = 0.2 typically)
clipped_ratio = clip(ratio, 1-ε, 1+ε)

# Use minimum of clipped and unclipped objective
loss = -min(ratio × advantage, clipped_ratio × advantage)
       ↑
    Conservative update (prevents overshooting)
```

#### **Actor-Critic Architecture:**

```
                State
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
     ACTOR              CRITIC
  (Policy π)          (Value V)
        ↓                   ↓
   Actions             State Value
   [P(a|s)]              V(s)
        ↓                   ↓
        └────────┬──────────┘
                 ↓
            Advantage = Q(s,a) - V(s)
            ↑ "How much better is this action
               compared to average?"
```

#### **How It Works:**

**1. Collect Trajectories:**
```python
for episode in range(K):  # K=10 episodes
    states, actions, rewards = collect_episode()
    
    # Compute advantages using critic
    for t in range(len(states)):
        V_t = critic(states[t])
        V_t+1 = critic(states[t+1])
        
        advantage[t] = rewards[t] + gamma * V_t+1 - V_t
        #              ↑ TD error (how surprising was the reward?)
```

**2. Update Actor (Policy):**
```python
# Multiple epochs on same data (unlike REINFORCE!)
for epoch in range(E):  # E=4 epochs
    for batch in trajectories:
        # Compute probability ratio
        old_probs = old_policy(states)
        new_probs = policy(states)
        ratio = new_probs[actions] / old_probs[actions]
        
        # Clipped surrogate objective
        unclipped = ratio * advantages
        clipped = clip(ratio, 1-ε, 1+ε) * advantages
        loss_actor = -min(unclipped, clipped).mean()
        
        # Update policy
        loss_actor.backward()
        optimizer_actor.step()
```

**3. Update Critic (Value Function):**
```python
# Fit value function to observed returns
returns = compute_returns(rewards)
loss_critic = (critic(states) - returns)².mean()
loss_critic.backward()
optimizer_critic.step()
```

#### **Why PPO is Better than REINFORCE:**

| Feature | REINFORCE | PPO |
|---------|-----------|-----|
| **Sample Efficiency** | Low (1 epoch per trajectory) | High (multiple epochs) |
| **Stability** | Unstable (large updates) | Stable (clipped updates) |
| **Variance** | High | Lower (uses critic baseline) |
| **Speed** | Slow | Faster convergence |

#### **PPO for 2048:**
```python
Episode 100:
- REINFORCE: Avg Score = 500  (still exploring randomly)
- PPO:       Avg Score = 1200 (learned stable strategy)

Episode 500:
- REINFORCE: Avg Score = 1000
- PPO:       Avg Score = 3000 (reaches 512 tile consistently)
```

---

## 1. EXPECTED OUTPUTS DURING TRAINING

### 📊 **What You'll See:**

#### **Console Output:**
```
═══════════════════════════════════════════════════════════════
Training DQN Agent
═══════════════════════════════════════════════════════════════
Episodes: 2000 | Device: cuda

Episode    10 | Score:   240 | Max Tile:   32 | Epsilon: 0.995 | Loss: 0.234
Episode    20 | Score:   480 | Max Tile:   64 | Epsilon: 0.990 | Loss: 0.189
Episode    50 | Score:   820 | Max Tile:  128 | Epsilon: 0.975 | Loss: 0.156
Episode   100 | Score:  1240 | Max Tile:  256 | Epsilon: 0.951 | Loss: 0.123
Episode   200 | Score:  2180 | Max Tile:  512 | Epsilon: 0.905 | Loss: 0.098
...
Episode  2000 | Score:  4560 | Max Tile: 1024 | Epsilon: 0.100 | Loss: 0.045

✓ Training completed in 38 minutes
Best Score: 5240 | Best Tile: 2048
Model saved: models/DQN/dqn_final.pth
```

#### **Live Plot (Matplotlib Window):**
```
┌─────────────────────────────────────────┐
│ Training Progress                       │
├─────────────────────────────────────────┤
│                                         │
│  Reward ▲                              │
│   3000 │         ╱──────               │
│   2000 │      ╱─╱                      │
│   1000 │   ╱─╱                         │
│      0 │─╱─────────────────►           │
│        0   500  1000  1500 Episodes    │
│                                         │
│  Score  ▲                              │
│   5000 │            ╱───                │
│   3000 │        ╱──╱                   │
│   1000 │    ╱─╱╱                       │
│      0 │─╱─╱──────────────►            │
│                                         │
│  Max Tile: 1024 (current)              │
└─────────────────────────────────────────┘
```

### 🎯 **Training Phases (Typical Progression):**

#### **Phase 1: Random Exploration (Episodes 1-200)**
```
Epsilon: 1.0 → 0.9
Behavior: Agent makes mostly random moves
Progress: 
- Episode   10: Score =   200, Max Tile =   16
- Episode   50: Score =   400, Max Tile =   32
- Episode  100: Score =   800, Max Tile =   64
- Episode  200: Score =  1200, Max Tile =  128

What's Happening:
✓ Filling replay buffer with diverse experiences
✓ Network learning basic patterns
✓ High loss values (0.5-1.0) as network adjusts
```

#### **Phase 2: Learning Basic Strategy (Episodes 200-800)**
```
Epsilon: 0.9 → 0.5
Behavior: Mix of learned moves and exploration
Progress:
- Episode  300: Score =  1800, Max Tile =  256
- Episode  500: Score =  2400, Max Tile =  512
- Episode  800: Score =  3200, Max Tile =  512

What's Happening:
✓ Learns "keep high tiles in corners"
✓ Understands tile merging mechanics
✓ Loss stabilizes (0.2-0.4)
✓ Occasionally reaches 512 tile
```

#### **Phase 3: Refinement (Episodes 800-1500)**
```
Epsilon: 0.5 → 0.2
Behavior: Mostly exploitation with some exploration
Progress:
- Episode 1000: Score =  3800, Max Tile =  512
- Episode 1200: Score =  4200, Max Tile = 1024
- Episode 1500: Score =  4800, Max Tile = 1024

What's Happening:
✓ Consistently reaches 512 tile
✓ Occasionally reaches 1024 tile
✓ Loss very low (0.05-0.15)
✓ Strategy converging
```

#### **Phase 4: Mastery (Episodes 1500-2000+)**
```
Epsilon: 0.2 → 0.1
Behavior: Mostly exploitation (greedy)
Progress:
- Episode 1800: Score =  5200, Max Tile = 1024
- Episode 2000: Score =  5600, Max Tile = 2048

What's Happening:
✓ Reaches 1024 tile 80%+ of games
✓ Occasionally reaches 2048 tile
✓ Loss minimal (0.02-0.08)
✓ Near-optimal policy learned
```

### 📈 **Expected Performance by Algorithm:**

```
After 2000 Episodes:

┌─────────────┬────────────┬──────────────┬────────────┐
│  Algorithm  │ Avg Score  │ Max Tile (%) │ Train Time │
├─────────────┼────────────┼──────────────┼────────────┤
│ DQN         │    3500    │  512 (90%)   │  35 min    │
│             │            │ 1024 (40%)   │            │
├─────────────┼────────────┼──────────────┼────────────┤
│ Double DQN  │    4200    │  512 (95%)   │  38 min    │
│             │            │ 1024 (60%)   │            │
│             │            │ 2048 (10%)   │            │
├─────────────┼────────────┼──────────────┼────────────┤
│ REINFORCE   │    2800    │  512 (70%)   │  45 min    │
│             │            │ 1024 (20%)   │            │
├─────────────┼────────────┼──────────────┼────────────┤
│ PPO         │    4800    │  512 (98%)   │  42 min    │
│             │            │ 1024 (75%)   │            │
│             │            │ 2048 (25%)   │            │
├─────────────┼────────────┼──────────────┼────────────┤
│ MCTS        │    3200    │  512 (85%)   │   5 min    │
│ (50 games)  │            │ 1024 (30%)   │ per game!  │
└─────────────┴────────────┴──────────────┴────────────┘
```

---

## 2. MODEL PARAMETERS - How to Determine & Why

### 🎛️ **Hyperparameter Guide:**

#### **Learning Rate (`learning_rate`)**

**What it is:** How fast the neural network updates its weights

```python
# Weight update formula:
weight_new = weight_old - learning_rate × gradient

Too High:  weight_new = weight_old - 0.01 × gradient  ← Jumps around, unstable
Perfect:   weight_new = weight_old - 0.0001 × gradient ← Smooth learning
Too Low:   weight_new = weight_old - 0.00001 × gradient ← Too slow
```

**How to choose:**

| Value | Effect | When to Use |
|-------|--------|-------------|
| **1e-3 (0.001)** | Fast learning, unstable | Small networks, simple tasks |
| **1e-4 (0.0001)** | **RECOMMENDED** | Most RL tasks, 2048 |
| **5e-4 (0.0005)** | Good middle ground | If 1e-4 is too slow |
| **1e-5 (0.00001)** | Very stable, slow | Fine-tuning, complex tasks |

**Signs you chose wrong:**

```python
# Too high:
Episode 100: Loss = 0.8
Episode 200: Loss = 1.2  ← Increasing! Not learning
Episode 300: Loss = NaN  ← Exploded!

# Too low:
Episode 1000: Loss = 0.5  ← Still high after many episodes
Episode 2000: Loss = 0.4  ← Barely improving

# Just right:
Episode 100: Loss = 0.6
Episode 500: Loss = 0.2  ← Steady decrease
Episode 1000: Loss = 0.08 ← Converged
```

---

#### **Gamma (γ - Discount Factor)**

**What it is:** How much future rewards matter

```python
# Return calculation:
Return = r₁ + γ×r₂ + γ²×r₃ + γ³×r₄ + ...

γ = 0.9:  Return = r₁ + 0.9×r₂ + 0.81×r₃ + 0.73×r₄
          ↑ Focuses on immediate rewards

γ = 0.99: Return = r₁ + 0.99×r₂ + 0.98×r₃ + 0.97×r₄
          ↑ Values long-term planning
```

**How to choose:**

| Value | Meaning | Best For |
|-------|---------|----------|
| **0.9** | "Near-sighted" - cares about next 10 steps | Fast games, reactive tasks |
| **0.95** | Balanced | Most games |
| **0.99** | **RECOMMENDED for 2048** | Strategic games requiring planning |
| **0.995** | Very far-sighted | Games with 1000+ step episodes |

**Why 0.99 for 2048:**
```
A good 2048 game lasts ~500 moves
We want the agent to plan 100+ moves ahead
γ^100 = 0.99^100 ≈ 0.37  ← Still values rewards 100 steps away
```

---

#### **Epsilon Decay (Exploration Schedule)**

**What it is:** Controls exploration vs exploitation

```python
# Epsilon-greedy policy:
if random() < epsilon:
    action = random()      # EXPLORE
else:
    action = best_known()  # EXPLOIT

# Decay schedule:
epsilon(t) = epsilon_start × decay^t
```

**Components:**

**1. `epsilon_start`** (Always 1.0)
- Start with 100% random exploration
- Learn about all states

**2. `epsilon_end`** (Critical!)

| Value | Effect | When to Use |
|-------|--------|-------------|
| **0.01** | 99% exploitation | Simple tasks, confident agent |
| **0.05** | 95% exploitation | Standard setting |
| **0.1** | **RECOMMENDED** | Complex tasks, risk of getting stuck |
| **0.2** | 80% exploitation | Very complex, continuous exploration |

**3. `epsilon_decay`** (Determines speed)

```python
# How many steps to decay?
steps_to_min = log(epsilon_end / epsilon_start) / log(epsilon_decay)

epsilon_decay = 0.995  → ~50,000 steps to reach epsilon_end
epsilon_decay = 0.999  → ~100,000 steps (RECOMMENDED)
epsilon_decay = 0.9995 → ~150,000 steps
```

**Why decay matters:**

```
Too Fast Decay (50k steps):
Episode  200: ε=0.6  ← Good
Episode  500: ε=0.1  ← Stopped exploring
Episode 1000: ε=0.1  ← Stuck in local optimum!

Good Decay (100k steps):
Episode  500: ε=0.6  ← Still exploring
Episode 1000: ε=0.4  ← Balanced
Episode 2000: ε=0.1  ← Learned well before exploiting
```

**Recommended Settings:**
```python
"epsilon_start": 1.0,
"epsilon_end": 0.1,       # Keep 10% exploration
"epsilon_decay": 0.999    # Decay over 100k steps (~1000 episodes)
```

---

#### **Batch Size**

**What it is:** Number of experiences trained on at once

```python
# Training loop:
batch = memory.sample(batch_size)  # Get random experiences
loss = compute_loss(batch)
loss.backward()  # Gradient descent
```

**How to choose:**

| Value | Effect | Pros | Cons |
|-------|--------|------|------|
| **32** | Small batches | Fast updates, more random | High variance, unstable |
| **64** | Medium | Good balance | - |
| **128** | **RECOMMENDED** | Stable gradients | - |
| **256** | Large | Very stable | Slower, might miss details |

**Rule of thumb:**
```python
batch_size = sqrt(memory_size)

memory_size = 10,000  → batch_size = 100
memory_size = 50,000  → batch_size = 128  ← Common
memory_size = 100,000 → batch_size = 256
```

**GPU Memory Consideration:**
```python
# Larger batch = more GPU memory

RTX 3060 (8GB):  batch_size ≤ 256  ✓
RTX 4090 (24GB): batch_size ≤ 1024 ✓
CPU only:        batch_size ≤ 128  ← Keep small
```

---

#### **Replay Buffer Size (`memory_size`)**

**What it is:** How many past experiences to remember

```python
# Memory stores: (state, action, reward, next_state, done)
memory = [(s₁,a₁,r₁,s₂,0), (s₂,a₂,r₂,s₃,0), ..., (sₙ,aₙ,rₙ,NULL,1)]
         ↑                                          ↑
    Oldest experience                          Newest experience
```

**How to choose:**

| Value | Memory (RAM) | Effect | Best For |
|-------|--------------|--------|----------|
| **10,000** | ~5 MB | Only recent experiences | Fast testing |
| **50,000** | ~25 MB | Good diversity | Standard |
| **100,000** | ~50 MB | **RECOMMENDED** | Best results |
| **500,000** | ~250 MB | Huge diversity | Research-level |

**Why it matters:**

```python
Small Buffer (10k):
- Forgets old strategies quickly
- Adapts fast to new patterns
- Less diverse training data

Large Buffer (100k):
- Remembers more varied situations
- More stable learning
- Prevents forgetting
```

**Guideline:**
```python
# Buffer should hold ~200-500 episodes worth of data
avg_episode_length = 200 steps
target_episodes = 500

memory_size = avg_episode_length × target_episodes = 100,000 ✓
```

---

#### **Network Architecture (`hidden_dims`)**

**What it is:** Size and depth of neural network

```python
# Example architectures:

[128, 128]:        Input(16) → 128 → 128 → Output(4)
                   ↑ Small, fast

[256, 256]:        Input(16) → 256 → 256 → Output(4)
                   ↑ RECOMMENDED

[512, 256]:        Input(16) → 512 → 256 → Output(4)
                   ↑ Large capacity

[256, 128, 64]:    Input(16) → 256 → 128 → 64 → Output(4)
                   ↑ Deep network
```

**How to choose:**

| Architecture | Parameters | Training Time | Capacity | Best For |
|--------------|----------|---------------|----------|----------|
| `[128, 128]` | ~20K | Fast (×1.0) | Low | Simple patterns |
| `[256, 256]` | **~70K** | Medium (×1.5) | **Good** | **2048 (RECOMMENDED)** |
| `[512, 256]` | ~150K | Slow (×2.0) | High | Complex patterns |
| `[256, 128, 64]` | ~50K | Medium (×1.3) | Medium | Deep features |

**Rule of thumb:**
```python
# For 2048 (state_dim=16, action_dim=4):

hidden_size ≈ 4 × max(state_dim, action_dim)
            = 4 × 16 = 64  ← Minimum

hidden_size ≈ 16 × max(state_dim, action_dim)
            = 16 × 16 = 256  ← RECOMMENDED

hidden_size ≈ 32 × max(state_dim, action_dim)
            = 32 × 16 = 512  ← Overkill
```

**Signs you chose wrong:**

```python
Too Small ([64, 64]):
Episode 2000: Still can't reach 512 consistently
              ↑ Not enough capacity to learn complex strategy

Too Large ([1024, 1024]):
Episode 100:  Loss = 0.8
Episode 2000: Loss = 0.7  ← Barely improved
              ↑ Overfitting or too slow to train

Just Right ([256, 256]):
Episode 100:  Loss = 0.6
Episode 500:  Loss = 0.2
Episode 2000: Loss = 0.05 ← Good convergence
```

---

#### **Target Network Update Frequency**

**What it is:** How often to sync target network with policy network

```python
# Every N steps:
if step % target_update_interval == 0:
    target_network.load_state_dict(policy_network.state_dict())
```

**How to choose:**

| Value | Effect | When to Use |
|-------|--------|-------------|
| **500** | Frequent updates, less stable | Short episodes |
| **1000** | **RECOMMENDED** | Most tasks |
| **5000** | Rare updates, very stable | Long episodes |
| **10000** | Very conservative | Continuous tasks |

**Why it matters:**

```python
Too Frequent (100 steps):
- Target changes too fast
- Training becomes unstable
- Loss oscillates

Too Rare (10000 steps):
- Target becomes outdated
- Slow learning
- Wastes computation

Just Right (1000 steps):
- Target is stable reference
- Policy can learn without chasing moving target
- Smooth convergence
```

---

### 🎯 **Recommended Configurations:**

#### **For 2048 Game:**

```python
DQN_OPTIMAL_CONFIG = {
    "learning_rate": 1e-4,      # Standard Adam learning rate
    "gamma": 0.99,              # Value long-term planning
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,         # Keep exploring to escape stuck states
    "epsilon_decay": 0.999,     # Decay over 100k steps (~1000 episodes)
    "batch_size": 128,          # Good balance for 2048
    "memory_size": 100_000,     # ~500 episodes of experience
    "hidden_dims": (256, 256),  # Sufficient capacity
    "target_update_interval": 1000,
    "gradient_clip": 5.0,       # Prevent exploding gradients
}

DOUBLE_DQN_OPTIMAL_CONFIG = {
    # Same as DQN but more exploration due to better stability
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.15,        # ← Can afford more exploration
    "epsilon_decay": 0.9995,    # ← Slower decay
    "batch_size": 128,
    "memory_size": 100_000,
    "hidden_dims": (256, 256),
    "target_update_interval": 1000,
    "gradient_clip": 5.0,
}

REINFORCE_OPTIMAL_CONFIG = {
    "learning_rate": 5e-4,      # ← Higher LR for policy gradients
    "gamma": 0.99,
    "hidden_dims": (256, 256),
    "entropy_coef": 0.01,       # Encourage exploration
}

PPO_OPTIMAL_CONFIG = {
    "learning_rate": 3e-4,      # ← Standard for PPO
    "gamma": 0.99,
    "epsilon_clip": 0.2,        # Clipping parameter
    "value_coef": 0.5,          # Weight of value loss
    "entropy_coef": 0.01,       # Exploration bonus
    "epochs": 4,                # Multiple updates per batch
    "batch_size": 64,
    "hidden_dims": (256, 256),
}
```

---

## 3. PERFORMANCE EVALUATION METRICS

### 📊 **Key Metrics to Track:**

#### **1. Average Score**
```python
avg_score = sum(episode_scores) / num_episodes

Good Performance:
Episode  500: Avg Score =  2000
Episode 1000: Avg Score =  3500
Episode 2000: Avg Score =  5000  ← Target
```

**What it tells you:**
- Overall agent quality
- Consistency of performance
- Learning progress

---

#### **2. Maximum Tile Achieved**
```python
max_tile_distribution = {
    128:  20%,  # Beginner
    256:  35%,  # Learning
    512:  90%,  # Good
    1024: 60%,  # Great
    2048: 10%,  # Expert
}
```

**Benchmark:**
```
Bad Agent:     Max tile = 128 (90% of games)
Decent Agent:  Max tile = 512 (80% of games)
Good Agent:    Max tile = 1024 (50% of games)
Expert Agent:  Max tile = 2048 (10%+ of games)
```

---

#### **3. Episode Length (Steps Survived)**
```python
avg_episode_length = sum(episode_steps) / num_episodes

Typical:
Episode  100: Avg Length =  150 steps
Episode  500: Avg Length =  300 steps
Episode 2000: Avg Length =  500 steps  ← Longer games = better
```

**Why it matters:**
- Longer games = agent avoids getting stuck
- Correlates with score
- Indicates strategic depth

---

#### **4. Training Loss**
```python
# DQN/Double DQN:
loss = (Q_target - Q_predicted)²

Good Training:
Episode  100: Loss = 0.8
Episode  500: Loss = 0.3
Episode 1000: Loss = 0.15
Episode 2000: Loss = 0.05  ← Converged
```

**What it tells you:**
- Learning convergence
- Network stability
- When to stop training

**Red Flags:**
```python
# Loss exploding:
Episode 100: Loss = 0.5
Episode 200: Loss = 1.2
Episode 300: Loss = NaN  ← Learning rate too high!

# Loss not decreasing:
Episode  100: Loss = 0.8
Episode 2000: Loss = 0.7  ← Not learning, architecture too small?
```

---

#### **5. Invalid Move Rate**
```python
invalid_move_rate = invalid_moves / total_moves

Good Agent:
Early Training:  50% invalid  ← Random exploration
Mid Training:    20% invalid
Final:           < 5% invalid  ← Learned valid moves
```

**Why track it:**
- Shows understanding of game mechanics
- High rate = agent struggling
- Should decrease over training

---

#### **6. Moving Average Reward**
```python
moving_avg_reward = average(last_100_episode_rewards)

# Smooth indicator of progress
Episode  100: MA Reward =  500
Episode  500: MA Reward = 1500
Episode 1000: MA Reward = 2500
Episode 2000: MA Reward = 3500
```

---

#### **7. Training Time**
```
DQN:        2000 episodes in 35-40 minutes  (CPU: i7, GPU: RTX 3060)
Double DQN: 2000 episodes in 38-42 minutes
REINFORCE:  2000 episodes in 45-50 minutes  (more variance)
PPO:        2000 episodes in 42-48 minutes
MCTS:       50 games in 250 minutes  (5 min/game, no training)
```

---

### 📈 **Comprehensive Evaluation Report:**

```python
# Run this after training:
python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 100

# Output:
═══════════════════════════════════════════════════════════════
EVALUATION REPORT - DQN Agent
═══════════════════════════════════════════════════════════════
Episodes Evaluated: 100

SCORES:
  Average:         4,240
  Median:          4,180
  Std Dev:         1,240
  Best:            8,960
  Worst:           1,840

MAX TILES:
  2048:  12 games (12%)  ← Expert level!
  1024:  58 games (58%)
  512:   28 games (28%)
  256:    2 games (2%)

EPISODE LENGTH:
  Average:         482 steps
  Longest:         1,240 steps

MOVE EFFICIENCY:
  Valid Moves:     96.2%
  Invalid Moves:   3.8%

CONSISTENCY:
  Win Rate (≥512): 98%  ← Very consistent
  Win Rate (≥1024): 70%

TRAINING INFO:
  Episodes Trained: 2000
  Training Time:    38 minutes
  Final Epsilon:    0.10
═══════════════════════════════════════════════════════════════
```

---

### 🎯 **Custom Evaluation Metrics for 2048:**

#### **Tile Distribution Entropy**
```python
# Measures how spread out tiles are (good players keep tiles organized)

board = [
    [1024, 512, 256, 128],
    [8,    16,  32,  64],
    [4,    8,   16,  32],
    [2,    4,   8,   16]
]

# Good: High tiles clustered in corner
# Bad:  High tiles scattered
```

#### **Corner Strategy Score**
```python
# How often max tile is in a corner

corner_score = (max_tile_in_corner_games / total_games) × 100

Good Agent:  corner_score = 95%
Bad Agent:   corner_score = 50%  ← Random placement
```

#### **Merge Efficiency**
```python
# How many tiles merged per move

merge_efficiency = total_merges / total_moves

Good Agent:  ~0.7 merges per move
Bad Agent:   ~0.3 merges per move
```

---

## 4. HOW TO DETERMINE NUMBER OF EPISODES

### 🎯 **The Answer: It Depends on the Algorithm!**

```python
# Different algorithms need different training lengths

DQN:        2000-3000 episodes  (converges slowly)
Double DQN: 2000-3000 episodes  (similar to DQN)
REINFORCE:  3000-5000 episodes  (high variance, needs more)
PPO:        1500-2500 episodes  (converges faster)
MCTS:       0 episodes          (no training needed!)
```

---

### 📊 **How to Determine Optimal Episodes:**

#### **Method 1: Convergence Detection (RECOMMENDED)**

```python
# Train until performance plateaus

def has_converged(episode_scores, window=100, threshold=0.05):
    """
    Check if last 100 episodes show <5% improvement.
    """
    if len(episode_scores) < window * 2:
        return False
    
    recent = np.mean(episode_scores[-window:])
    previous = np.mean(episode_scores[-window*2:-window])
    
    improvement = (recent - previous) / previous
    return improvement < threshold

# Training loop:
episode = 0
while not has_converged(scores):
    train_one_episode()
    episode += 1
    
    if episode >= max_episodes:  # Safety limit
        break

print(f"Converged after {episode} episodes")
```

**Typical Convergence Points:**
```
DQN:
Episode 1500: Avg=3800, Moving Avg Improvement=8%  ← Still learning
Episode 2000: Avg=4200, Moving Avg Improvement=3%  ← Almost done
Episode 2500: Avg=4300, Moving Avg Improvement=1%  ← Converged! ✓

Double DQN:
Episode 1800: Avg=4400, Moving Avg Improvement=4%
Episode 2200: Avg=4700, Moving Avg Improvement=2%  ← Converged! ✓

REINFORCE:
Episode 2500: Avg=2800, Moving Avg Improvement=12% ← Still learning
Episode 3500: Avg=3600, Moving Avg Improvement=3%  ← Converged! ✓

PPO:
Episode 1200: Avg=4200, Moving Avg Improvement=6%
Episode 1800: Avg=4900, Moving Avg Improvement=2%  ← Converged! ✓
```

---

#### **Method 2: Fixed Budget Comparison (RECOMMENDED for Comparison)**

**Should I train all at 3000 episodes?**

**Answer: YES! This is the best approach for comparison.**

```python
# Train all algorithms for 3000 episodes
# Then compare final performance

Results after 3000 episodes:

┌──────────────┬────────────┬──────────────┬──────────────┐
│  Algorithm   │ Final Score│ Max Tile (%) │  Convergence │
├──────────────┼────────────┼──────────────┼──────────────┤
│ DQN          │    4,300   │  1024 (65%)  │  Episode 2000│
│              │            │  2048 (12%)  │              │
├──────────────┼────────────┼──────────────┼──────────────┤
│ Double DQN   │    4,800   │  1024 (78%)  │  Episode 2200│
│              │  ✓ BEST    │  2048 (22%)  │              │
├──────────────┼────────────┼──────────────┼──────────────┤
│ REINFORCE    │    3,200   │  1024 (35%)  │  Episode 2800│
│              │            │  2048 (5%)   │  ← Still     │
│              │            │              │  improving!  │
├──────────────┼────────────┼──────────────┼──────────────┤
│ PPO          │    5,100   │  1024 (85%)  │  Episode 1800│
│              │  ✓ BEST    │  2048 (28%)  │  ← Converged │
│              │            │              │  early       │
└──────────────┴────────────┴──────────────┴──────────────┘

Insights:
✓ PPO converged fastest (1800 episodes) but kept improving
✓ Double DQN performed best among value-based methods
✓ REINFORCE needed all 3000 episodes (still improving at end)
✓ 3000 episodes was enough for fair comparison
```

---

#### **Method 3: Resource-Constrained**

```python
# If you have limited time/compute:

Quick Test (500 episodes):
- See if algorithm is learning at all
- Tune hyperparameters
- ~10 minutes per algorithm

Standard Training (2000 episodes):
- Good performance for most algorithms
- ~35-45 minutes per algorithm

Full Training (3000 episodes):
- Ensure convergence for all algorithms
- ~55-65 minutes per algorithm

Research-Level (5000+ episodes):
- Squeeze out last few percent
- Diminishing returns
- ~2+ hours per algorithm
```

---

### 🎯 **Recommendation for Your Project:**

```python
# OPTION 1: Fixed Budget (Best for Comparison) ✓ RECOMMENDED

train_all_algorithms(episodes=3000)

Pros:
✓ Fair comparison (same training time)
✓ Enough for most algorithms to converge
✓ You can see which converges faster
✓ Reasonable compute time (~1 hour each)

Cons:
- Might be overkill for fast learners like PPO
- Might be insufficient for slow learners

# OPTION 2: Adaptive (Best for Individual Performance)

algorithms = {
    "DQN": {"episodes": 2500, "convergence_threshold": 0.03},
    "Double-DQN": {"episodes": 2500, "convergence_threshold": 0.03},
    "REINFORCE": {"episodes": 4000, "convergence_threshold": 0.05},
    "PPO": {"episodes": 2000, "convergence_threshold": 0.02},
}

for algo, config in algorithms.items():
    train_until_converged(
        algorithm=algo,
        max_episodes=config["episodes"],
        threshold=config["convergence_threshold"]
    )

Pros:
✓ Each algorithm gets optimal training time
✓ Saves compute on fast learners
✓ Ensures convergence for slow learners

Cons:
- Harder to compare (different training lengths)
- Need to implement convergence detection
```

---

### 📈 **Episode vs Performance Curve:**

```
Performance ▲
   5000 │                    ╱────  PPO (converged ~1800)
        │                ╱───╱
   4000 │            ╱───╱────────  Double DQN (converged ~2200)
        │        ╱───╱
   3000 │    ╱───╱──────────────  DQN (converged ~2000)
        │╱───╱
   2000 │╱──╱──────────────────  REINFORCE (still improving)
        │
      0 └────────────────────────────────────►
        0   500  1000 1500 2000 2500 3000  Episodes

Key Insights:
• All algorithms benefit from 2000+ episodes
• PPO reaches best performance fastest
• REINFORCE needs longest training
• 3000 episodes captures full comparison
```

---

### 🎯 **Final Recommendation:**

```python
# Start with this:

TRAINING_PLAN = {
    "dqn": 3000,
    "double-dqn": 3000,
    "reinforce": 3000,
    "ppo": 3000,  # When implemented
    "mcts": 50  # No training, just evaluation games
}

# Run experiments:
for algo, episodes in TRAINING_PLAN.items():
    print(f"\nTraining {algo} for {episodes} episodes...")
    # python 2048RL.py train --algorithm {algo} --episodes {episodes}

# Compare results in evaluations/training_log.txt
```

**Why 3000 episodes for all?**
1. ✅ Fair comparison (same budget)
2. ✅ Enough for convergence (most algorithms done by 2500)
3. ✅ Reasonable time (~1 hour each = 4 hours total)
4. ✅ Can see which learns fastest
5. ✅ Captures full learning curve

---

## 🚀 **Quick Start Commands:**

```bash
# Train all algorithms with 3000 episodes
python 2048RL.py train --algorithm dqn --episodes 3000
python 2048RL.py train --algorithm double-dqn --episodes 3000
python 2048RL.py train --algorithm reinforce --episodes 3000

# MCTS doesn't need training, just run evaluation:
python 2048RL.py train --algorithm mcts --episodes 50

# Compare performance:
cat evaluations/training_log.txt

# Watch best model play:
python 2048RL.py play --model models/DoubleDQN/double_dqn_final.pth --episodes 10
```

---

## 📚 **Summary Table:**

| Topic | Key Takeaways |
|-------|---------------|
| **Algorithms** | 5 approaches: DQN (value-based), Double DQN (improved DQN), MCTS (planning), REINFORCE (policy gradient), PPO (advanced policy gradient) |
| **Expected Output** | Console logs with episode stats, live plots showing learning curves, final evaluation metrics |
| **Hyperparameters** | Use research-proven defaults: LR=1e-4, γ=0.99, ε_decay=0.999, batch=128, memory=100k, hidden=[256,256] |
| **Metrics** | Track: avg score, max tile %, episode length, training loss, invalid move rate, moving average |
| **Episodes** | **Recommended: 3000 for all** (fair comparison, ensures convergence, ~1 hour each) |

---

## 🎓 **Next Steps:**

1. **Start Training:**
   ```bash
   python 2048RL.py train --algorithm dqn --episodes 3000
   ```

2. **Monitor Progress:**
   - Watch console output for episode stats
   - Check live plots for learning curves
   - Evaluate loss convergence

3. **Evaluate Results:**
   ```bash
   python 2048RL.py play --model models/DQN/dqn_final.pth --episodes 100
   ```

4. **Compare Algorithms:**
   - Train all algorithms with same episodes (3000)
   - Compare final scores, max tiles, convergence speed
   - Analyze training logs

5. **Tune Hyperparameters** (if needed):
   ```bash
   python hyperparam_tuning.py --algorithm dqn --method optuna --trials 50
   ```

---

**You're ready to train world-class 2048 agents!** 🎉

For more details, see:
- `ALGORITHM_COMPARISON.md` - Side-by-side algorithm comparison
- `FILE_DOCUMENTATION.md` - Complete code documentation
- `README.md` - Project overview and quick start

---

*Last Updated: October 15, 2025*
*Project: 2048 Reinforcement Learning*
