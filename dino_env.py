import numpy as np
import pygame
import random
import gymnasium as gym
from gymnasium import spaces

class DinoEnv(gym.Env):
    def __init__(self):
        super(DinoEnv, self).__init__()
        
        # Initialize Pygame
        pygame.init()
        
        # Constants
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 300
        self.GROUND_Y = 250
        self.GRAVITY = 0.8
        self.JUMP_SPEED = -16
        self.GAME_SPEED = 7
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (83, 83, 83)
        self.BACKGROUND = (247, 247, 247)
        
        # Set up display
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Chrome Dino Game")
        
        # Game objects
        self.dino_width = 44
        self.dino_height = 47
        self.dino_x = 80
        self.dino_y = self.GROUND_Y - self.dino_height
        self.dino_vel_y = 0
        self.is_jumping = False
        self.is_ducking = False
        
        # Animation variables
        self.step_index = 0
        self.animation_timer = 0
        self.leg_up = True
        
        # Obstacles
        self.obstacles = []
        self.obstacle_types = [
            {"type": "CACTUS_SMALL", "width": 20, "height": 40, "y": self.GROUND_Y - 40},
            {"type": "CACTUS_BIG", "width": 25, "height": 50, "y": self.GROUND_Y - 50},
            {"type": "CACTUS_GROUP", "width": 50, "height": 40, "y": self.GROUND_Y - 40},
            {"type": "BIRD_LOW", "width": 40, "height": 30, "y": self.GROUND_Y - 40},
            {"type": "BIRD_HIGH", "width": 40, "height": 30, "y": self.GROUND_Y - 80}
        ]
        
        # Ground
        self.ground_x = 0
        self.GROUND_LINE_Y = self.GROUND_Y + 5
        
        # Score and game speed
        self.score = 0
        self.clock = pygame.time.Clock()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.dino_y = self.GROUND_Y - self.dino_height
        self.dino_vel_y = 0
        self.is_jumping = False
        self.is_ducking = False
        self.step_index = 0
        self.animation_timer = 0
        self.leg_up = True
        self.ground_x = 0
        self.obstacles = []
        self.score = 0
        self._spawn_obstacle()
        return self._get_observation(), {}
    
    def step(self, action):
        reward = 0.1  # Small reward for surviving
        done = False
        
        # Handle actions
        if action == 1 and not self.is_jumping:  # Jump
            self.dino_vel_y = self.JUMP_SPEED
            self.is_jumping = True
            self.is_ducking = False
        elif action == 2 and not self.is_jumping:  # Duck
            self.is_ducking = True
        else:
            self.is_ducking = False
        
        # Update dinosaur position
        if self.is_jumping:
            self.dino_y += self.dino_vel_y
            self.dino_vel_y += self.GRAVITY
            
            if self.dino_y >= self.GROUND_Y - self.dino_height:
                self.dino_y = self.GROUND_Y - self.dino_height
                self.dino_vel_y = 0
                self.is_jumping = False
        
        # Update ground position
        self.ground_x -= self.GAME_SPEED
        if self.ground_x <= -self.SCREEN_WIDTH:
            self.ground_x = 0
        
        # Update animation
        self.animation_timer += 1
        if self.animation_timer >= 5:  # Change animation every 5 frames
            self.leg_up = not self.leg_up
            self.animation_timer = 0
        
        # Update obstacles
        for obstacle in self.obstacles[:]:
            obstacle["x"] -= self.GAME_SPEED
            if obstacle["x"] < -obstacle["width"]:
                self.obstacles.remove(obstacle)
                self.score += 1
                reward = 1.0  # Reward for passing obstacle
        
        # Spawn new obstacles
        if len(self.obstacles) < 3 and random.random() < 0.02:
            self._spawn_obstacle()
        
        # Check collisions
        dino_rect = pygame.Rect(
            self.dino_x,
            self.dino_y,
            self.dino_width * (1.3 if self.is_ducking else 1),
            self.dino_height * (0.6 if self.is_ducking else 1)
        )
        
        for obstacle in self.obstacles:
            obstacle_rect = pygame.Rect(
                obstacle["x"],
                obstacle["y"],
                obstacle["width"],
                obstacle["height"]
            )
            if dino_rect.colliderect(obstacle_rect):
                done = True
                reward = -10.0  # Penalty for collision
        
        # Draw game state
        self._draw()
        
        return self._get_observation(), reward, done, False, {"score": self.score}
    
    def _spawn_obstacle(self):
        obstacle_type = random.choice(self.obstacle_types)
        min_distance = 300  # Minimum distance between obstacles
        
        if not self.obstacles or self.obstacles[-1]["x"] < self.SCREEN_WIDTH - min_distance:
            self.obstacles.append({
                "x": self.SCREEN_WIDTH,
                "y": obstacle_type["y"],
                "width": obstacle_type["width"],
                "height": obstacle_type["height"],
                "type": obstacle_type["type"]
            })
    
    def _draw_dino(self):
        if self.is_ducking:
            # Draw ducking dinosaur
            body_height = self.dino_height * 0.6
            pygame.draw.rect(self.screen, self.BLACK,
                           (self.dino_x, self.dino_y + self.dino_height - body_height,
                            self.dino_width * 1.3, body_height))
            
            # Draw leg
            leg_height = 10 if self.leg_up else 5
            pygame.draw.rect(self.screen, self.BLACK,
                           (self.dino_x + self.dino_width * 0.7,
                            self.dino_y + self.dino_height - leg_height,
                            5, leg_height))
        else:
            # Draw body
            pygame.draw.rect(self.screen, self.BLACK,
                           (self.dino_x, self.dino_y,
                            self.dino_width, self.dino_height * 0.8))
            
            # Draw head
            pygame.draw.rect(self.screen, self.BLACK,
                           (self.dino_x + self.dino_width * 0.6,
                            self.dino_y,
                            self.dino_width * 0.4, self.dino_height * 0.4))
            
            # Draw eye
            pygame.draw.circle(self.screen, self.WHITE,
                             (self.dino_x + self.dino_width * 0.8,
                              self.dino_y + self.dino_height * 0.2), 2)
            
            # Draw leg
            if not self.is_jumping:
                leg_height = 15 if self.leg_up else 8
                pygame.draw.rect(self.screen, self.BLACK,
                               (self.dino_x + self.dino_width * 0.3,
                                self.dino_y + self.dino_height * 0.8,
                                5, leg_height))
    
    def _draw_obstacle(self, obstacle):
        if "BIRD" in obstacle["type"]:
            # Draw bird body
            pygame.draw.ellipse(self.screen, self.BLACK,
                              (obstacle["x"], obstacle["y"],
                               obstacle["width"], obstacle["height"]))
            
            # Draw wings
            wing_y = obstacle["y"] + (5 if self.leg_up else -5)
            pygame.draw.line(self.screen, self.BLACK,
                           (obstacle["x"] + obstacle["width"] * 0.3, obstacle["y"] + obstacle["height"] * 0.5),
                           (obstacle["x"] + obstacle["width"] * 0.3, wing_y), 2)
            pygame.draw.line(self.screen, self.BLACK,
                           (obstacle["x"] + obstacle["width"] * 0.7, obstacle["y"] + obstacle["height"] * 0.5),
                           (obstacle["x"] + obstacle["width"] * 0.7, wing_y), 2)
        else:
            # Draw cactus
            if "SMALL" in obstacle["type"]:
                pygame.draw.rect(self.screen, self.BLACK,
                               (obstacle["x"], obstacle["y"],
                                obstacle["width"], obstacle["height"]))
            elif "BIG" in obstacle["type"]:
                pygame.draw.rect(self.screen, self.BLACK,
                               (obstacle["x"], obstacle["y"],
                                obstacle["width"], obstacle["height"]))
                # Add spikes
                pygame.draw.line(self.screen, self.BLACK,
                               (obstacle["x"] + obstacle["width"] * 0.5, obstacle["y"] + obstacle["height"] * 0.2),
                               (obstacle["x"] + obstacle["width"] * 0.8, obstacle["y"] + obstacle["height"] * 0.3), 2)
            else:  # CACTUS_GROUP
                for i in range(3):
                    x_offset = i * (obstacle["width"] // 3)
                    height_var = random.randint(-5, 5)
                    pygame.draw.rect(self.screen, self.BLACK,
                                   (obstacle["x"] + x_offset,
                                    obstacle["y"] + height_var,
                                    obstacle["width"] // 4,
                                    obstacle["height"]))
    
    def _draw(self):
        # Clear screen
        self.screen.fill(self.BACKGROUND)
        
        # Draw ground
        pygame.draw.line(self.screen, self.BLACK, 
                        (0, self.GROUND_LINE_Y), 
                        (self.SCREEN_WIDTH, self.GROUND_LINE_Y))
        
        # Draw dinosaur
        self._draw_dino()
        
        # Draw obstacles
        for obstacle in self.obstacles:
            self._draw_obstacle(obstacle)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (20, 20))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def _get_observation(self):
        # Convert pygame surface to grayscale numpy array
        screen_array = pygame.surfarray.array3d(self.screen)
        screen_array = np.mean(screen_array, axis=2).astype(np.uint8)
        # Resize to 84x84
        observation = pygame.transform.scale(
            pygame.surfarray.make_surface(screen_array),
            (84, 84)
        )
        observation_array = pygame.surfarray.array2d(observation).astype(np.uint8)
        return observation_array.reshape(84, 84, 1)
    
    def close(self):
        pygame.quit()
        
    def render(self):
        # Already rendering in step function
        pass 