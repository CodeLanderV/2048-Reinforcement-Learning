"""
Pygame UI for 2048 game
"""

import pygame
import sys
from game_logic import Game2048
from constants import *


class Game2048UI:
    """Pygame interface for 2048 game"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("2048")
        self.clock = pygame.time.Clock()
        self.game = Game2048()
        
        # Load fonts
        self.font_small = pygame.font.Font(None, FONT_SIZE_SMALL)
        self.font_medium = pygame.font.Font(None, FONT_SIZE_MEDIUM)
        self.font_large = pygame.font.Font(None, FONT_SIZE_LARGE)
        
        self.running = True
    
    def handle_events(self):
        """Handle keyboard and window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_r:
                    # Reset game
                    self.game.reset()
                
                elif not self.game.is_game_over():
                    # Handle arrow keys
                    if event.key == pygame.K_UP:
                        self.game.move(0)
                    elif event.key == pygame.K_RIGHT:
                        self.game.move(1)
                    elif event.key == pygame.K_DOWN:
                        self.game.move(2)
                    elif event.key == pygame.K_LEFT:
                        self.game.move(3)
    
    def draw(self):
        """Draw the game state"""
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw score
        self.draw_score()
        
        # Draw grid
        self.draw_grid()
        
        # Draw game over message
        if self.game.is_game_over():
            self.draw_game_over()
        
        # Draw win message
        if self.game.has_won() and not self.game.is_game_over():
            self.draw_win()
        
        pygame.display.flip()
    
    def draw_score(self):
        """Draw the score at the top"""
        score_text = f"Score: {self.game.get_score()}"
        text_surface = self.font_medium.render(score_text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, 40))
        self.screen.blit(text_surface, text_rect)
        
        # Draw instructions
        instructions = "Arrow keys to play | R to restart | ESC to quit"
        inst_surface = self.font_small.render(instructions, True, TEXT_COLOR)
        inst_rect = inst_surface.get_rect(center=(WINDOW_WIDTH // 2, 70))
        self.screen.blit(inst_surface, inst_rect)
    
    def draw_grid(self):
        """Draw the game grid and tiles"""
        board = self.game.get_board()
        
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = board[i, j]
                
                # Calculate tile position
                x = GRID_PADDING + j * (TILE_SIZE + GRID_PADDING)
                y = 100 + GRID_PADDING + i * (TILE_SIZE + GRID_PADDING)
                
                # Draw tile background
                color = TILE_COLORS.get(value, TILE_COLORS[0])
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x, y, TILE_SIZE, TILE_SIZE),
                    border_radius=5
                )
                
                # Draw tile number
                if value != 0:
                    self.draw_tile_number(value, x, y)
    
    def draw_tile_number(self, value, x, y):
        """Draw the number on a tile"""
        font_size = FONT_SIZE_TILE.get(value, 30)
        font = pygame.font.Font(None, font_size)
        
        # Choose text color based on tile value
        text_color = TEXT_COLOR if value <= 4 else TEXT_COLOR_LIGHT
        
        text_surface = font.render(str(value), True, text_color)
        text_rect = text_surface.get_rect(
            center=(x + TILE_SIZE // 2, y + TILE_SIZE // 2)
        )
        self.screen.blit(text_surface, text_rect)
    
    def draw_game_over(self):
        """Draw game over overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(200)
        overlay.fill((238, 228, 218))
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        game_over_text = "Game Over!"
        text_surface = self.font_large.render(game_over_text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40))
        self.screen.blit(text_surface, text_rect)
        
        # Final score
        score_text = f"Final Score: {self.game.get_score()}"
        score_surface = self.font_medium.render(score_text, True, TEXT_COLOR)
        score_rect = score_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
        self.screen.blit(score_surface, score_rect)
        
        # Restart instruction
        restart_text = "Press R to restart"
        restart_surface = self.font_small.render(restart_text, True, TEXT_COLOR)
        restart_rect = restart_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(restart_surface, restart_rect)
    
    def draw_win(self):
        """Draw win message overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(150)
        overlay.fill((237, 194, 46))
        self.screen.blit(overlay, (0, 0))
        
        # Win text
        win_text = "You Win!"
        text_surface = self.font_large.render(win_text, True, TEXT_COLOR_LIGHT)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
        self.screen.blit(text_surface, text_rect)
        
        # Continue instruction
        continue_text = "Keep playing or press R to restart"
        continue_surface = self.font_small.render(continue_text, True, TEXT_COLOR_LIGHT)
        continue_rect = continue_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20))
        self.screen.blit(continue_surface, continue_rect)
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game2048UI()
    game.run()
