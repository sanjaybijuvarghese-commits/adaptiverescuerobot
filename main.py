import pygame

from config import *
from environment import Environment
from visualization import Visualization

pygame.init()

# Create Window
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Adaptive Disaster Rescue Robot")

clock = pygame.time.Clock()

# Create Environment
environment = Environment()

# Create Visualization
visualization = Visualization()

running = True

while running:

    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw Everything
    visualization.draw(screen, environment)

    pygame.display.flip()

pygame.quit()