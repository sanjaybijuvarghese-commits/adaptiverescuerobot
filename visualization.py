import pygame
from config import *

class Visualization:

    def __init__(self):

        # Load Images
        self.robot_img = pygame.image.load(
            "assets/robot.png"
        ).convert_alpha()

        self.robot_img = pygame.transform.scale(
            self.robot_img,
            (CELL_SIZE - 10, CELL_SIZE - 10)
        )

        self.survivor_img = pygame.image.load(
            "assets/survivor symbol.png"
        ).convert_alpha()

        self.survivor_img = pygame.transform.scale(
            self.survivor_img,
            (CELL_SIZE - 14, CELL_SIZE - 14)
        )

        self.hazard_img = pygame.image.load(
            "assets/hazard.png"
        ).convert_alpha()

        self.hazard_img = pygame.transform.scale(
            self.hazard_img,
            (CELL_SIZE - 8, CELL_SIZE - 8)
        )

    # -------------------------------------------------

    def draw(self, screen, environment):

        screen.fill(WHITE)

        # -----------------------------
        # Draw Grid
        # -----------------------------

        for row in range(ROWS + 1):

            pygame.draw.line(
                screen,
                GRAY,
                (0, row * CELL_SIZE),
                (WINDOW_WIDTH, row * CELL_SIZE),
                1
            )

        for col in range(COLS + 1):

            pygame.draw.line(
                screen,
                GRAY,
                (col * CELL_SIZE, 0),
                (col * CELL_SIZE, WINDOW_HEIGHT),
                1
            )

        # -----------------------------
        # Draw Obstacles
        # -----------------------------

        for row, col in environment.obstacles:

            obstacle = pygame.Rect(

                col * CELL_SIZE + 3,
                row * CELL_SIZE + 3,

                CELL_SIZE - 6,
                CELL_SIZE - 6
            )

            pygame.draw.rect(
                screen,
                OBSTACLE,
                obstacle,
                border_radius=6
            )

        # -----------------------------
        # Draw Hazards
        # -----------------------------

        for row, col in environment.hazards:

            screen.blit(

                self.hazard_img,

                (
                    col * CELL_SIZE + 4,
                    row * CELL_SIZE + 4
                )
            )

        # -----------------------------
        # Draw Survivors
        # -----------------------------

        for row, col in environment.survivors:

            screen.blit(

                self.survivor_img,

                (
                    col * CELL_SIZE + 7,
                    row * CELL_SIZE + 7
                )
            )

        # -----------------------------
        # Draw Robot
        # -----------------------------

        row, col = environment.robot

        screen.blit(

            self.robot_img,

            (
                col * CELL_SIZE + 5,
                row * CELL_SIZE + 5
            )
        )