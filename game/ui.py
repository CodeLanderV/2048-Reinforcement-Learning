"""Simple pygame UI for the 2048 board."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import pygame

from .board import GameBoard


class GameUI:
    TILE_COLORS: Dict[int, Tuple[int, int, int]] = {
        0: (204, 192, 179),
        2: (238, 228, 218),
        4: (237, 224, 200),
        8: (242, 177, 121),
        16: (245, 149, 99),
        32: (246, 124, 95),
        64: (246, 94, 59),
        128: (237, 207, 114),
        256: (237, 204, 97),
        512: (237, 200, 80),
        1024: (237, 197, 63),
        2048: (237, 194, 46),
    }

    BACKGROUND_COLOR = (187, 173, 160)
    BOARD_PADDING = 15
    TILE_PADDING = 10
    WINDOW_SIZE = 500

    def __init__(self, board: GameBoard, caption: str = "2048") -> None:
        pygame.init()
        pygame.display.set_caption(caption)
        self.board = board
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        self.font = pygame.font.SysFont("arial", 48, bold=True)
        self.small_font = pygame.font.SysFont("arial", 24)

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def handle_events(self) -> Optional[str]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key in (pygame.K_r, pygame.K_R):
                    return "restart"
                mapping = {
                    pygame.K_UP: "up",
                    pygame.K_DOWN: "down",
                    pygame.K_LEFT: "left",
                    pygame.K_RIGHT: "right",
                }
                if event.key in mapping:
                    return mapping[event.key]
        return None

    def draw(self) -> None:
        self.screen.fill(self.BACKGROUND_COLOR)
        grid = self.board.grid
        tile_size = (self.WINDOW_SIZE - 2 * self.BOARD_PADDING) // self.board.GRID_SIZE
        for row in range(self.board.GRID_SIZE):
            for col in range(self.board.GRID_SIZE):
                x = self.BOARD_PADDING + col * tile_size
                y = self.BOARD_PADDING + row * tile_size
                value = int(grid[row, col])
                color = self.TILE_COLORS.get(value, (60, 58, 50))
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(
                        x + self.TILE_PADDING,
                        y + self.TILE_PADDING,
                        tile_size - 2 * self.TILE_PADDING,
                        tile_size - 2 * self.TILE_PADDING,
                    ),
                    border_radius=8,
                )
                if value:
                    text_surface = self.font.render(str(value), True, (119, 110, 101))
                    text_rect = text_surface.get_rect(center=(x + tile_size / 2, y + tile_size / 2))
                    self.screen.blit(text_surface, text_rect)
        score_surface = self.small_font.render(f"Score: {self.board.score}", True, (119, 110, 101))
        self.screen.blit(score_surface, (self.BOARD_PADDING, self.WINDOW_SIZE - 40))
        pygame.display.update()
