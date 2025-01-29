import torch
from train import DQN
from dino_env import DinoEnv
import time

def demonstrate_agent():
    # Initialize environment and model
    env = DinoEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = DQN().to(device)
    model.load_state_dict(torch.load('best_dino_model.pth'))
    model.eval()
    
    print("Demonstrating trained agent...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Get action from model
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(state_tensor).max(1)[1].view(1, 1)
                
                # Take action in environment
                state, reward, done, _, info = env.step(action.item())
                total_reward += reward
                
                # Add a small delay to make it easier to watch
                time.sleep(0.01)
            
            print(f"Game Over! Score: {info['score']}")
            time.sleep(1)  # Pause between games
            
    except KeyboardInterrupt:
        print("\nDemonstration ended by user")
    
    env.close()

if __name__ == "__main__":
    demonstrate_agent() 