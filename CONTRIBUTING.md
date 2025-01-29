# Contributing to DinoBench

Thank you for your interest in contributing to DinoBench! This document provides guidelines and instructions for contributing.

## Types of Contributions

### 1. New Agents

The primary way to contribute is by implementing new agents. Your agent should:
- Inherit from the `DinoAgent` base class
- Implement the required methods
- Include documentation of your approach
- Provide benchmark results

### 2. Environment Improvements

You can also contribute to the game environment:
- Bug fixes
- Performance optimizations
- New features
- Better visualization

### 3. Documentation

Help improve our documentation:
- Fix typos or unclear instructions
- Add examples and tutorials
- Improve code comments
- Write guides for specific approaches

## Submitting an Agent

1. **Implementation**
   ```python
   # my_agent.py
   from benchmark import DinoAgent
   
   class MyAgent(DinoAgent):
       def __init__(self):
           # Document your initialization
           pass
       
       def act(self, state):
           # Document your action selection logic
           pass
       
       def name(self):
           return "My Agent Name"
   ```

2. **Documentation**
   - Create a markdown file (e.g., `docs/agents/my_agent.md`)
   - Describe your approach
   - Include any relevant papers or resources
   - Document hyperparameters and architecture

3. **Benchmark Results**
   - Run the standard benchmark
   - Include results in your documentation
   - Save any trained models
   - Document reproduction steps

4. **Pull Request**
   - Create a new branch
   - Add your implementation
   - Add documentation
   - Submit PR with benchmark results

## Code Style

Follow these guidelines:
- Use Python type hints
- Follow PEP 8
- Document all public methods
- Include docstring examples
- Write clear commit messages

Example:
```python
from typing import ndarray
import numpy as np

class MyAgent(DinoAgent):
    """
    My awesome agent implementation.
    
    Uses approach X to achieve Y...
    
    Args:
        param1: Description of param1
        param2: Description of param2
    """
    
    def act(self, state: ndarray) -> int:
        """
        Select action based on current state.
        
        Args:
            state: 84x84x1 grayscale image
            
        Returns:
            action: 0 (nothing), 1 (jump), or 2 (duck)
            
        Example:
            >>> agent = MyAgent()
            >>> state = env.reset()
            >>> action = agent.act(state)
        """
        pass
```

## Running Tests

Before submitting:
1. Run the benchmark suite
   ```bash
   python benchmark.py --agent MyAgent --episodes 100
   ```

2. Run style checks
   ```bash
   flake8 my_agent.py
   mypy my_agent.py
   ```

3. Test documentation
   ```bash
   python -m doctest my_agent.py
   ```

## Benchmark Requirements

Your agent should:
1. Run without errors for 100 episodes
2. Complete each episode in reasonable time
3. Handle all game scenarios
4. Include reproducibility information

## Review Process

1. **Initial Check**
   - Code style compliance
   - Documentation completeness
   - Benchmark results included

2. **Technical Review**
   - Implementation correctness
   - Code efficiency
   - Best practices followed

3. **Benchmark Verification**
   - Results reproduction
   - Performance validation
   - Resource usage check

4. **Final Steps**
   - Address review comments
   - Update documentation
   - Merge approval

## Getting Help

- Open an issue for questions
- Join discussions
- Read existing documentation
- Check closed issues

## Code of Conduct

- Be respectful and inclusive
- Give constructive feedback
- Help others learn
- Share knowledge

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 