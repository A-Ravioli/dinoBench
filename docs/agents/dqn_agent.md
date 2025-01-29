# DQN Agent

A Deep Q-Network implementation for the Chrome dinosaur game.

## Overview

This agent uses a convolutional neural network to process game screenshots and learn optimal actions through Q-learning. It implements experience replay and target networks for stable training.

## Architecture

### Neural Network
```
Input: 84x84x1 grayscale image
│
├─ Conv2D(32, 8x8, stride=4)
│   └─ ReLU
│
├─ Conv2D(64, 4x4, stride=2)
│   └─ ReLU
│
├─ Conv2D(64, 3x3, stride=1)
│   └─ ReLU
│
├─ Flatten
│
├─ Linear(3136 -> 512)
│   └─ ReLU
│
└─ Linear(512 -> 3)
```

### Training Components
- Experience Replay Buffer (100,000 transitions)
- Target Network (updated every 10 episodes)
- Adam Optimizer (lr=1e-4)
- Huber Loss

## Hyperparameters

```python
BATCH_SIZE = 32
GAMMA = 0.99          # Discount factor
EPS_START = 1.0       # Starting exploration rate
EPS_END = 0.01       # Final exploration rate
EPS_DECAY = 100000   # Exploration decay steps
TARGET_UPDATE = 10    # Episodes between target updates
REPLAY_BUFFER_SIZE = 100000
LEARNING_RATE = 1e-4
```

## Performance

Baseline performance metrics:
- Average Score: ~50
- Best Score: ~200
- Training Time: ~2 hours
- Hardware: CPU or single GPU

## Training Process

1. **Initialization**
   - Initialize policy and target networks
   - Create empty replay buffer
   - Set exploration rate to 1.0

2. **Episode Loop**
   - Reset environment
   - Process each frame:
     1. Select action (ε-greedy)
     2. Execute action
     3. Store transition
     4. Train on random batch
   - Update target network if needed
   - Decay exploration rate

3. **Training Step**
   - Sample random batch
   - Compute Q-values
   - Calculate target values
   - Update policy network
   - Clip gradients

## Usage Example

```python
from benchmark import DinoBenchmark
from benchmark import DQNAgent

# Create and benchmark agent
benchmark = DinoBenchmark()
agent = DQNAgent("path/to/model.pth")
results = benchmark.run_benchmark(agent, n_episodes=100)
```

## Implementation Details

### State Processing
- Convert game screen to grayscale
- Resize to 84x84
- Normalize pixel values

### Action Selection
- During training: ε-greedy policy
- During evaluation: greedy policy
- Actions:
  - 0: Do nothing
  - 1: Jump
  - 2: Duck

### Reward Structure
Uses environment rewards:
- +0.1 for surviving
- +1.0 for passing obstacle
- -10.0 for collision

## Training Tips

1. **Exploration**
   - Start with high exploration
   - Decay slowly for better learning
   - Consider curriculum learning

2. **Stability**
   - Use target network
   - Clip gradients
   - Monitor loss values

3. **Performance**
   - Increase buffer size for better stability
   - Adjust batch size based on hardware
   - Fine-tune learning rate

## Results Reproduction

To reproduce the baseline results:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the agent:
   ```bash
   python train.py
   ```

3. Run benchmark:
   ```bash
   python benchmark.py
   ```

Expected training progression:
- Episode 100: ~10 average score
- Episode 500: ~30 average score
- Episode 1000: ~50 average score

## Known Limitations

1. **Training Time**
   - Requires many episodes for good performance
   - CPU training is slow

2. **Performance Cap**
   - Struggles with very fast game speeds
   - Limited by frame-based input

3. **Memory Usage**
   - Large replay buffer needed
   - GPU memory for training

## Future Improvements

1. **Architecture**
   - Try different CNN architectures
   - Add recurrent layers
   - Implement attention mechanisms

2. **Training**
   - Prioritized experience replay
   - Dueling DQN
   - Rainbow DQN improvements

3. **Features**
   - Frame stacking
   - Reward shaping
   - State preprocessing

## References

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
2. [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
3. [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) 