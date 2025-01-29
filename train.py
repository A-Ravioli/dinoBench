import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import time
from dino_env import DinoEnv

# DQN Network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 3)  # 3 actions: nothing, jump, duck

    def forward(self, x):
        # Ensure input is in the correct format (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        if x.dim() == 4 and x.shape[1] == 84:
            x = x.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        
        x = x.contiguous()  # Make tensor contiguous in memory
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Ensure states are float32 numpy arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 100000
    TARGET_UPDATE = 10
    REPLAY_BUFFER_SIZE = 100000
    LEARNING_RATE = 1e-4
    EVALUATION_INTERVAL = 100  # Episodes between evaluations
    EVALUATION_EPISODES = 5    # Number of episodes for each evaluation
    TARGET_SCORE = 1000       # Target score to achieve
    CONSISTENT_TARGET = 3     # Number of consecutive evaluations above target
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = DinoEnv()
    
    # Initialize networks
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    # Training metrics
    steps_done = 0
    episode_scores = []
    best_eval_score = 0
    above_target_count = 0
    
    def select_action(state, eps_threshold):
        if random.random() > eps_threshold:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                q_values = policy_net(state)
                return q_values.max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(3)]], device=device)
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
        action_batch = torch.LongTensor(batch[1]).to(device)
        reward_batch = torch.FloatTensor(batch[2]).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
        done_batch = torch.FloatTensor(batch[4]).to(device)
        
        state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
        
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
    def evaluate_model(n_episodes=5):
        eval_scores = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
                state, reward, done, _, info = env.step(action.item())
                total_reward += reward
            
            eval_scores.append(info['score'])
        
        return np.mean(eval_scores)
    
    print("Starting training...")
    print(f"Target score: {TARGET_SCORE}")
    print(f"Training until {CONSISTENT_TARGET} consecutive evaluations above target")
    print(f"Using device: {device}")
    
    episode = 0
    try:
        while True:
            state, _ = env.reset()
            total_reward = 0
            
            # One episode
            while True:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
                action = select_action(state, eps_threshold)
                next_state, reward, done, _, info = env.step(action.item())
                
                memory.push(state, action.item(), reward, next_state, done)
                state = next_state
                total_reward += reward
                
                optimize_model()
                steps_done += 1
                
                if done:
                    episode_scores.append(info['score'])
                    if episode % 10 == 0:  # Print progress every 10 episodes
                        print(f"Episode {episode}, Score: {info['score']}, Epsilon: {eps_threshold:.3f}")
                    break
            
            # Update target network
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            # Evaluate model
            if episode % EVALUATION_INTERVAL == 0:
                eval_score = evaluate_model(EVALUATION_EPISODES)
                print(f"\nEvaluation at episode {episode}")
                print(f"Average Score: {eval_score:.1f}")
                print(f"Best Score: {best_eval_score:.1f}")
                print(f"Consecutive Above Target: {above_target_count}")
                
                if eval_score > TARGET_SCORE:
                    above_target_count += 1
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        torch.save(policy_net.state_dict(), 'best_dino_model.pth')
                        print(f"New best model saved with score: {eval_score:.1f}")
                else:
                    above_target_count = 0
                
                if above_target_count >= CONSISTENT_TARGET:
                    print(f"\nTraining complete! Achieved target score consistently over {CONSISTENT_TARGET} evaluations")
                    break
            
            episode += 1
            
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    
    env.close()
    return policy_net

if __name__ == "__main__":
    model = train() 