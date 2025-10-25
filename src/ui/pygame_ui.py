"""
Pygame UI for 2048 Game
Provides interactive visualization of the game.
"""

import pygame
import numpy as np
from typing import Optional, Tuple
import sys


class GameUI:
    """
    Pygame-based UI for 2048 game.
    
    Features:
    - Visual representation of the game board
    - Real-time display of score, max tile, and moves
    - Color-coded tiles based on value
    - Smooth animations (optional)
    """
    
    # Color scheme for tiles
    COLORS = {
        0: (205, 193, 180),
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
        4096: (60, 58, 50),
        8192: (60, 58, 50),
    }
    
    FONT_COLORS = {
        0: (205, 193, 180),
        2: (119, 110, 101),
        4: (119, 110, 101),
    }
    
    def __init__(self, size: int = 4, cell_size: int = 100, margin: int = 10):
        """
        Initialize Pygame UI.
        
        Args:
            size: Board size (4x4)
            cell_size: Size of each cell in pixels
            margin: Margin between cells
        """
        pygame.init()
        
        self.size = size
        self.cell_size = cell_size
        self.margin = margin
        
        # Calculate window size
        self.board_width = size * cell_size + (size + 1) * margin
        self.info_height = 100
        self.window_width = self.board_width
        self.window_height = self.board_width + self.info_height
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("2048 - DQN Agent")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.background_color = (187, 173, 160)
        self.text_color = (119, 110, 101)
        self.light_text_color = (249, 246, 242)
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        
        # Game state display
        self.score = 0
        self.max_tile = 0
        self.moves = 0
        self.episode = 0
    
    def get_tile_color(self, value: int) -> Tuple[int, int, int]:
        """
        Get color for tile value.
        
        Args:
            value: Tile value
            
        Returns:
            RGB color tuple
        """
        if value in self.COLORS:
            return self.COLORS[value]
        # For values > 8192
        return (60, 58, 50)
    
    def get_font_color(self, value: int) -> Tuple[int, int, int]:
        """
        Get font color for tile value.
        
        Args:
            value: Tile value
            
        Returns:
            RGB color tuple
        """
        if value <= 4:
            return self.text_color
        return self.light_text_color
    
    def draw_board(self, board: np.ndarray):
        """
        Draw the game board.
        
        Args:
            board: 4x4 numpy array representing the game state
        """
        # Fill background
        self.screen.fill(self.background_color)
        
        # Draw board background
        pygame.draw.rect(
            self.screen,
            (187, 173, 160),
            (0, self.info_height, self.board_width, self.board_width)
        )
        
        # Draw tiles
        for i in range(self.size):
            for j in range(self.size):
                value = int(board[i, j])
                
                # Calculate position
                x = j * (self.cell_size + self.margin) + self.margin
                y = i * (self.cell_size + self.margin) + self.margin + self.info_height
                
                # Draw tile background
                tile_color = self.get_tile_color(value)
                pygame.draw.rect(
                    self.screen,
                    tile_color,
                    (x, y, self.cell_size, self.cell_size),
                    border_radius=5
                )
                
                # Draw tile value
                if value != 0:
                    font_color = self.get_font_color(value)
                    
                    # Choose font size based on number of digits
                    if value < 100:
                        font = self.font_large
                    elif value < 1000:
                        font = self.font_medium
                    else:
                        font = self.font_small
                    
                    text = font.render(str(value), True, font_color)
                    text_rect = text.get_rect(
                        center=(x + self.cell_size // 2, y + self.cell_size // 2)
                    )
                    self.screen.blit(text, text_rect)
    
    def draw_info(self, score: int, max_tile: int, moves: int, episode: int = 0, 
                  mode: str = "Training"):
        """
        Draw game information panel.
        
        Args:
            score: Current score
            max_tile: Maximum tile value
            moves: Number of moves
            episode: Current episode (for training)
            mode: Display mode (Training/Playing)
        """
        # Update stored values
        self.score = score
        self.max_tile = max_tile
        self.moves = moves
        self.episode = episode
        
        # Draw info background
        pygame.draw.rect(
            self.screen,
            (250, 248, 239),
            (0, 0, self.window_width, self.info_height)
        )
        
        # Draw mode
        mode_text = self.font_medium.render(f"{mode}", True, self.text_color)
        self.screen.blit(mode_text, (10, 10))
        
        # Draw episode (if training)
        if mode == "Training":
            episode_text = self.font_small.render(f"Episode: {episode}", True, self.text_color)
            self.screen.blit(episode_text, (10, 45))
        
        # Draw score
        score_text = self.font_small.render(f"Score: {score}", True, self.text_color)
        self.screen.blit(score_text, (self.window_width // 3, 20))
        
        # Draw max tile
        max_tile_text = self.font_small.render(f"Max: {max_tile}", True, self.text_color)
        self.screen.blit(max_tile_text, (self.window_width // 3, 50))
        
        # Draw moves
        moves_text = self.font_small.render(f"Moves: {moves}", True, self.text_color)
        self.screen.blit(moves_text, (2 * self.window_width // 3, 20))
    
    def update(self, board: np.ndarray, score: int, max_tile: int, moves: int,
               episode: int = 0, mode: str = "Training", fps: int = 10):
        """
        Update the display.
        
        Args:
            board: Game board state
            score: Current score
            max_tile: Maximum tile
            moves: Number of moves
            episode: Current episode
            mode: Display mode
            fps: Frames per second
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            
            # Allow ESC to close
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return False
        
        # Draw everything
        self.draw_info(score, max_tile, moves, episode, mode)
        self.draw_board(board)
        
        # Update display
        pygame.display.flip()
        
        # Control FPS
        self.clock.tick(fps)
        
        return True
    
    def close(self):
        """Close the Pygame window."""
        pygame.quit()
    
    def wait_for_key(self, keys: Optional[list] = None) -> Optional[int]:
        """
        Wait for a key press.
        
        Args:
            keys: List of allowed keys (None for any key)
            
        Returns:
            Key code or None if window closed
        """
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
                
                if event.type == pygame.KEYDOWN:
                    if keys is None or event.key in keys:
                        return event.key
                    
                    if event.key == pygame.K_ESCAPE:
                        self.close()
                        return None
            
            self.clock.tick(30)
        
        return None
    
    def show_game_over(self, score: int, max_tile: int):
        """
        Display game over screen.
        
        Args:
            score: Final score
            max_tile: Maximum tile achieved
        """
        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Game Over text
        game_over_text = self.font_large.render("Game Over!", True, (255, 255, 255))
        game_over_rect = game_over_text.get_rect(
            center=(self.window_width // 2, self.window_height // 2 - 60)
        )
        self.screen.blit(game_over_text, game_over_rect)
        
        # Score text
        score_text = self.font_medium.render(f"Score: {score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(
            center=(self.window_width // 2, self.window_height // 2)
        )
        self.screen.blit(score_text, score_rect)
        
        # Max tile text
        max_text = self.font_medium.render(f"Max Tile: {max_tile}", True, (255, 255, 255))
        max_rect = max_text.get_rect(
            center=(self.window_width // 2, self.window_height // 2 + 40)
        )
        self.screen.blit(max_text, max_rect)
        
        # Update display
        pygame.display.flip()


class ManualGameUI(GameUI):
    """
    Extended UI for manual gameplay with keyboard controls.
    """
    
    def get_action_from_key(self) -> Optional[int]:
        """
        Get action from keyboard input.
        
        Returns:
            Action (0=left, 1=right, 2=up, 3=down) or None
        """
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        return 0  # Left
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        return 1  # Right
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        return 2  # Up
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        return 3  # Down
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        return None
            
            self.clock.tick(30)
        
        return None
