import torch
import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from dino_env import DinoEnv

class DinoAgent(ABC):
    """Base class for all dinosaur game agents"""
    @abstractmethod
    def act(self, state):
        """Return action given state"""
        pass
    
    @abstractmethod
    def name(self):
        """Return name of the agent"""
        pass

class DQNAgent(DinoAgent):
    """Example DQN Agent implementation"""
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from train import DQN  # Import here to avoid circular import
        self.model = DQN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state)
            return q_values.max(1)[1].item()
    
    def name(self):
        return "DQN Agent"

class RandomAgent(DinoAgent):
    """Example Random Agent implementation"""
    def act(self, state):
        return np.random.randint(0, 3)
    
    def name(self):
        return "Random Agent"

class DinoBenchmark:
    def __init__(self, save_file="benchmark_results.json"):
        self.save_file = save_file
        self.results = self._load_results()
        
    def _load_results(self):
        """Load existing benchmark results"""
        try:
            with open(self.save_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "high_scores": [],
                "recent_runs": []
            }
    
    def _save_results(self):
        """Save benchmark results"""
        with open(self.save_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def run_benchmark(self, agent, n_episodes=100, render_best=True):
        """Run benchmark for given agent"""
        env = DinoEnv()
        scores = []
        best_score = 0
        best_seed = None
        
        print(f"\nBenchmarking {agent.name()}")
        print(f"Running {n_episodes} episodes...")
        
        start_time = time.time()
        
        for episode in range(n_episodes):
            seed = np.random.randint(0, 1000000)
            state, _ = env.reset(seed=seed)
            episode_score = 0
            done = False
            
            while not done:
                action = agent.act(state)
                state, _, done, _, info = env.step(action)
                episode_score = info['score']
            
            scores.append(episode_score)
            
            if episode_score > best_score:
                best_score = episode_score
                best_seed = seed
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, Current Score: {episode_score}, Best Score: {best_score}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate statistics
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        
        # Record results
        run_result = {
            "agent_name": agent.name(),
            "timestamp": datetime.now().isoformat(),
            "episodes": n_episodes,
            "best_score": best_score,
            "best_seed": best_seed,
            "average_score": float(avg_score),
            "std_score": float(std_score),
            "median_score": float(median_score),
            "duration": duration
        }
        
        # Update high scores
        self.results["high_scores"].append({
            "score": best_score,
            "agent_name": agent.name(),
            "timestamp": datetime.now().isoformat(),
            "seed": best_seed
        })
        self.results["high_scores"].sort(key=lambda x: x["score"], reverse=True)
        self.results["high_scores"] = self.results["high_scores"][:10]  # Keep top 10
        
        # Update recent runs
        self.results["recent_runs"].append(run_result)
        self.results["recent_runs"] = self.results["recent_runs"][-20:]  # Keep last 20 runs
        
        self._save_results()
        
        # Print results
        print("\nBenchmark Results:")
        print(f"Best Score: {best_score}")
        print(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
        print(f"Median Score: {median_score}")
        print(f"Time taken: {duration:.2f} seconds")
        
        # Demonstrate best run if requested
        if render_best and best_seed is not None:
            print("\nDemonstrating best run...")
            state, _ = env.reset(seed=best_seed)
            done = False
            while not done:
                action = agent.act(state)
                state, _, done, _, info = env.step(action)
                time.sleep(0.01)  # Slow down for visualization
            print(f"Demonstration complete. Score: {info['score']}")
        
        env.close()
        return run_result
    
    def print_leaderboard(self):
        """Print the current leaderboard"""
        print("\n=== DINO GAME LEADERBOARD ===")
        print("Top 10 All-Time High Scores:")
        print("Rank  Score  Agent  Date")
        print("-" * 40)
        
        for i, entry in enumerate(self.results["high_scores"], 1):
            date = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d")
            print(f"{i:2d}.   {entry['score']:4d}   {entry['agent_name']:<20} {date}")
    
    def print_recent_runs(self):
        """Print recent benchmark runs"""
        print("\n=== RECENT BENCHMARK RUNS ===")
        print("Agent  Avg Score  Best Score  Date")
        print("-" * 50)
        
        for run in reversed(self.results["recent_runs"]):
            date = datetime.fromisoformat(run["timestamp"]).strftime("%Y-%m-%d %H:%M")
            print(f"{run['agent_name']:<20} {run['average_score']:9.2f} {run['best_score']:10d} {date}")

def main():
    # Create benchmark instance
    benchmark = DinoBenchmark()
    
    # Example usage with random agent
    random_agent = RandomAgent()
    benchmark.run_benchmark(random_agent, n_episodes=5)
    
    # If a trained model exists, test it too
    model_path = Path("best_dino_model.pth")
    if model_path.exists():
        dqn_agent = DQNAgent(model_path)
        benchmark.run_benchmark(dqn_agent, n_episodes=5)
    
    # Print results
    benchmark.print_leaderboard()
    benchmark.print_recent_runs()

if __name__ == "__main__":
    main() 