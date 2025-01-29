import pygame
from dino_env import DinoEnv

def play_game():
    env = DinoEnv()
    done = False
    total_reward = 0
    
    # Initial reset
    env.reset()
    
    print("Controls:")
    print("SPACE / UP ARROW - Jump")
    print("DOWN ARROW - Duck")
    print("Q - Quit")
    
    while not done:
        # Process events
        action = 0  # Default action (do nothing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
        
        # Get pressed keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or keys[pygame.K_UP]:
            action = 1  # Jump
        elif keys[pygame.K_DOWN]:
            action = 2  # Duck
        
        # Step the environment
        _, reward, done, _, info = env.step(action)
        total_reward += reward
        
        # Display score
        pygame.display.set_caption(f"Dino Game - Score: {info['score']}")
    
    print(f"\nGame Over! Final Score: {info['score']}")
    env.close()

if __name__ == "__main__":
    play_game() 