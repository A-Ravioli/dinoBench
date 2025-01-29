# DinoBench 🦖

A standardized benchmark for testing and comparing reinforcement learning algorithms on the Chrome dinosaur game.

## Overview

DinoBench is a platform for:
- Benchmarking RL algorithms against a faithful recreation of the Chrome dinosaur game
- Comparing different approaches with standardized metrics
- Maintaining a global leaderboard of best performances
- Ensuring reproducibility through saved seeds

## Features

- **Custom Game Environment**: Accurate recreation of Chrome's dinosaur game using Pygame
- **Standardized Interface**: OpenAI Gym-compatible environment
- **Benchmark Framework**: Tools for fair comparison of different agents
- **Reproducible Results**: Seed tracking for replicating high scores
- **Visualization**: Watch your agents play in real-time

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dinoBench.git
cd dinoBench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Play the Game

Try the game yourself:
```bash
python play_game.py
```

Controls:
- SPACE/UP: Jump
- DOWN: Duck
- Q: Quit

### Run the Benchmark

Test the example agents:
```bash
python benchmark.py
```

### Train a DQN Agent

Train the example DQN implementation:
```bash
python train.py
```

## Creating Your Own Agent

1. Create a new Python file (e.g., `my_agent.py`)
2. Implement the `DinoAgent` interface:

```python
from benchmark import DinoAgent

class MyAgent(DinoAgent):
    def __init__(self):
        # Initialize your agent
        pass
    
    def act(self, state):
        """
        Args:
            state: 84x84x1 grayscale image of game state
        Returns:
            action: 0 (do nothing), 1 (jump), or 2 (duck)
        """
        # Your agent's decision logic here
        return action
    
    def name(self):
        return "My Awesome Agent"
```

3. Run the benchmark:
```python
from benchmark import DinoBenchmark

benchmark = DinoBenchmark()
my_agent = MyAgent()
benchmark.run_benchmark(my_agent, n_episodes=100)
```

## Environment Details

### Observation Space
- Type: Box(84, 84, 1)
- 84x84 grayscale image of the game state

### Action Space
- Type: Discrete(3)
- Actions:
  - 0: Do nothing
  - 1: Jump
  - 2: Duck

### Reward Structure
- +0.1: Surviving each timestep
- +1.0: Successfully passing an obstacle
- -10.0: Collision with obstacle

### Game Mechanics
- Dinosaur can jump over or duck under obstacles
- Obstacles include:
  - Small cacti
  - Large cacti
  - Cactus groups
  - Birds at different heights
- Game speed increases with score
- Precise collision detection

## Benchmark System

### Metrics Tracked
- High Score: Best single-episode performance
- Average Score: Mean score over multiple episodes
- Standard Deviation: Consistency measure
- Median Score: Typical performance indicator
- Success Rate: Percentage of episodes above threshold

### Leaderboard Categories
1. All-Time High Scores
2. Most Consistent Performance
3. Best Average Score
4. Recent Benchmark Runs

### Reproducibility
- All high scores include seeds
- Benchmark runs can be replicated exactly
- Performance statistics over multiple episodes

## Example Agents

1. **Random Agent** (Baseline)
   - Makes random actions
   - Serves as minimum performance baseline

2. **DQN Agent** (Example Implementation)
   - Deep Q-Network with CNN architecture
   - Processes raw game images
   - Learns through experience replay

## Contributing

1. Fork the repository
2. Create your feature branch
3. Implement and test your agent
4. Submit a pull request with:
   - Your agent implementation
   - Benchmark results
   - Brief description of your approach

## Tips for High Scores

1. **State Processing**
   - Consider extracting relevant features
   - Distance to obstacles
   - Obstacle types and patterns

2. **Action Selection**
   - Timing is crucial for jumps
   - Duck only for birds
   - Avoid unnecessary actions

3. **Training Strategies**
   - Start with simpler scenarios
   - Gradually increase difficulty
   - Use curriculum learning

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Chrome's dinosaur game
- Built with PyTorch and Pygame
- Uses OpenAI Gym interface
