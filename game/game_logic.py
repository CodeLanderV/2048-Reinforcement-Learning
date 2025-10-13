"""
Core 2048 game logic (no UI dependencies)
"""

import random
import numpy as np
from constants import GRID_SIZE


class Game2048:
    """2048 game logic implementation"""
    
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.board = None
        self.score = 0
        self.game_over = False
        self.won = False
        self.reset()
    
    def reset(self):
        """Reset the game to initial state"""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.game_over = False
        self.won = False
        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()
    
    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            # 90% chance of 2, 10% chance of 4
            self.board[row, col] = 2 if random.random() < 0.9 else 4
            return True
        return False
    
    def move(self, direction):
        """
        Execute a move in the given direction
        
        Args:
            direction: 0=up, 1=right, 2=down, 3=left
        
        Returns:
            bool: True if move was valid (board changed), False otherwise
        """
        original_board = self.board.copy()
        original_score = self.score
        
        if direction == 0:  # Up
            self._move_up()
        elif direction == 1:  # Right
            self._move_right()
        elif direction == 2:  # Down
            self._move_down()
        elif direction == 3:  # Left
            self._move_left()
        else:
            return False
        
        # Check if board changed
        moved = not np.array_equal(original_board, self.board)
        
        if moved:
            self.add_random_tile()
            if self._check_game_over():
                self.game_over = True
            if self._check_win():
                self.won = True
        
        return moved
    
    def _move_left(self):
        """Move all tiles left"""
        for i in range(self.grid_size):
            self.board[i] = self._merge_line(self.board[i])
    
    def _move_right(self):
        """Move all tiles right"""
        for i in range(self.grid_size):
            self.board[i] = self._merge_line(self.board[i][::-1])[::-1]
    
    def _move_up(self):
        """Move all tiles up"""
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T
    
    def _move_down(self):
        """Move all tiles down"""
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T
    
    def _merge_line(self, line):
        """
        Merge a single line (row or column) to the left
        
        Example: [2, 0, 2, 4] -> [4, 4, 0, 0]
        """
        # Remove zeros
        non_zero = line[line != 0]
        
        # Merge adjacent equal tiles
        merged = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                # Merge tiles
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                skip = True
            else:
                merged.append(non_zero[i])
        
        # Pad with zeros
        merged = merged + [0] * (self.grid_size - len(merged))
        return np.array(merged)
    
    def _check_game_over(self):
        """Check if no more moves are possible"""
        # Check for empty cells
        if np.any(self.board == 0):
            return False
        
        # Check for possible merges horizontally
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
        
        # Check for possible merges vertically
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False
        
        return True
    
    def _check_win(self):
        """Check if player reached 2048"""
        return np.any(self.board == 2048)
    
    def get_board(self):
        """Return a copy of the current board"""
        return self.board.copy()
    
    def get_score(self):
        """Return current score"""
        return self.score
    
    def is_game_over(self):
        """Return True if game is over"""
        return self.game_over
    
    def has_won(self):
        """Return True if player won"""
        return self.won
    
    def get_available_moves(self):
        """Return list of available moves (directions that would change the board)"""
        available = []
        for direction in range(4):
            original_board = self.board.copy()
            temp_game = Game2048(self.grid_size)
            temp_game.board = original_board.copy()
            temp_game.score = self.score
            if temp_game.move(direction):
                available.append(direction)
        return available
