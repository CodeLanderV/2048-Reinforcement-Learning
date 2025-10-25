"""
2048 Game Logic Implementation
Handles the game state, moves, and rules for the 2048 game.
"""

import numpy as np
import copy
from typing import Tuple, List, Optional


class Game2048:
    """
    Represents the 2048 game environment.
    
    The game board is a 4x4 grid where tiles with powers of 2 appear.
    Players can move tiles in 4 directions (up, down, left, right).
    When two tiles with the same number touch, they merge into one.
    Goal: Create a tile with the number 2048.
    """
    
    def __init__(self, size: int = 4):
        """
        Initialize the game board.
        
        Args:
            size: Size of the board (default: 4x4)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.moves_count = 0
        self.game_over = False
        
        # Add two initial tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            The initial board state
        """
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.max_tile = 0
        self.moves_count = 0
        self.game_over = False
        
        self.add_random_tile()
        self.add_random_tile()
        
        return self.get_state()
    
    def add_random_tile(self) -> bool:
        """
        Add a random tile (2 or 4) to an empty position on the board.
        90% chance of 2, 10% chance of 4.
        
        Returns:
            True if a tile was added, False if board is full
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        
        if not empty_cells:
            return False
        
        # Choose random empty cell
        row, col = empty_cells[np.random.randint(len(empty_cells))]
        
        # 90% chance of 2, 10% chance of 4
        self.board[row, col] = 2 if np.random.random() < 0.9 else 4
        
        return True
    
    def get_state(self) -> np.ndarray:
        """
        Get the current board state.
        
        Returns:
            Copy of the current board
        """
        return self.board.copy()
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Get list of empty cell coordinates.
        
        Returns:
            List of (row, col) tuples for empty cells
        """
        return list(zip(*np.where(self.board == 0)))
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over (no valid moves available).
        
        Returns:
            True if game is over, False otherwise
        """
        # Check if there are empty cells
        if len(self.get_empty_cells()) > 0:
            return False
        
        # Check if any adjacent cells can be merged
        for i in range(self.size):
            for j in range(self.size):
                current = self.board[i, j]
                
                # Check right neighbor
                if j < self.size - 1 and self.board[i, j + 1] == current:
                    return False
                
                # Check down neighbor
                if i < self.size - 1 and self.board[i + 1, j] == current:
                    return False
        
        return True
    
    def _merge_line(self, line: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Merge a single line (row or column) to the left.
        
        Args:
            line: 1D array representing a line of tiles
            
        Returns:
            Tuple of (merged line, score gained)
        """
        # Remove zeros
        non_zero = line[line != 0]
        
        merged = []
        score_gained = 0
        skip = False
        
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            
            if i < len(non_zero) - 1 and non_zero[i] == non_zero[i + 1]:
                # Merge tiles
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                score_gained += merged_value
                skip = True
            else:
                merged.append(non_zero[i])
        
        # Pad with zeros
        merged.extend([0] * (len(line) - len(merged)))
        
        return np.array(merged, dtype=np.int32), score_gained
    
    def move_left(self) -> Tuple[bool, int]:
        """
        Move all tiles to the left.
        
        Returns:
            Tuple of (whether board changed, score gained)
        """
        old_board = self.board.copy()
        total_score = 0
        
        for i in range(self.size):
            self.board[i, :], score = self._merge_line(self.board[i, :])
            total_score += score
        
        board_changed = not np.array_equal(old_board, self.board)
        
        return board_changed, total_score
    
    def move_right(self) -> Tuple[bool, int]:
        """
        Move all tiles to the right.
        
        Returns:
            Tuple of (whether board changed, score gained)
        """
        # Flip horizontally, move left, flip back
        self.board = np.fliplr(self.board)
        changed, score = self.move_left()
        self.board = np.fliplr(self.board)
        
        return changed, score
    
    def move_up(self) -> Tuple[bool, int]:
        """
        Move all tiles up.
        
        Returns:
            Tuple of (whether board changed, score gained)
        """
        # Transpose, move left, transpose back
        self.board = self.board.T
        changed, score = self.move_left()
        self.board = self.board.T
        
        return changed, score
    
    def move_down(self) -> Tuple[bool, int]:
        """
        Move all tiles down.
        
        Returns:
            Tuple of (whether board changed, score gained)
        """
        # Transpose, move right, transpose back
        self.board = self.board.T
        changed, score = self.move_right()
        self.board = self.board.T
        
        return changed, score
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one game step with the given action.
        
        Args:
            action: 0=left, 1=right, 2=up, 3=down
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        if self.game_over:
            return self.get_state(), 0, True, {'game_over': True}
        
        # Map action to move function
        move_functions = {
            0: self.move_left,
            1: self.move_right,
            2: self.move_up,
            3: self.move_down
        }
        
        # Execute move
        changed, score_gained = move_functions[action]()
        
        reward = 0
        
        # Only process if the move changed the board
        if changed:
            self.score += score_gained
            self.moves_count += 1
            
            # Add new random tile
            self.add_random_tile()
            
            # Update max tile
            self.max_tile = int(np.max(self.board))
            
            # Calculate reward
            reward = self.calculate_reward(score_gained)
            
            # Check game over
            self.game_over = self.is_game_over()
        else:
            # Invalid move (board didn't change)
            reward = -1  # Small penalty for invalid moves
        
        info = {
            'score': self.score,
            'max_tile': self.max_tile,
            'moves': self.moves_count,
            'game_over': self.game_over,
            'valid_move': changed
        }
        
        return self.get_state(), reward, self.game_over, info
    
    def calculate_reward(self, score_gained: int) -> float:
        """
        Calculate reward based on game state.
        Uses log2(max_tile) * 2 as base reward.
        Adds bonuses for keeping highest tile in corner and snake pattern.
        
        Args:
            score_gained: Score gained from the last move
            
        Returns:
            Calculated reward value
        """
        reward = 0
        
        # Base reward: log2(max_tile) * 2
        if self.max_tile > 0:
            reward += np.log2(self.max_tile) * 2
        
        # Bonus for score gained
        reward += score_gained * 0.1
        
        # Bonus for keeping highest tile in corner
        corner_bonus = self.calculate_corner_bonus()
        reward += corner_bonus
        
        # Bonus for snake pattern
        snake_bonus = self.calculate_snake_bonus()
        reward += snake_bonus
        
        # Bonus for empty cells (encourages keeping board open)
        empty_cells = len(self.get_empty_cells())
        reward += empty_cells * 0.5
        
        return reward
    
    def calculate_corner_bonus(self) -> float:
        """
        Calculate bonus for keeping highest tile in a corner.
        
        Returns:
            Corner bonus value
        """
        max_val = np.max(self.board)
        corners = [
            self.board[0, 0],
            self.board[0, -1],
            self.board[-1, 0],
            self.board[-1, -1]
        ]
        
        if max_val in corners:
            return 5.0  # Good bonus for corner placement
        
        return 0
    
    def calculate_snake_bonus(self) -> float:
        """
        Calculate bonus for maintaining a snake pattern (descending values).
        Snake pattern: highest values arranged in a snake from one corner.
        
        Returns:
            Snake pattern bonus value
        """
        bonus = 0
        
        # Check if tiles are generally arranged in descending order
        # Flatten board and get positions of tiles in descending order
        flat_board = self.board.flatten()
        sorted_indices = np.argsort(flat_board)[::-1]  # Descending order
        
        # Check monotonicity along rows
        for i in range(self.size):
            row = self.board[i, :]
            if i % 2 == 0:  # Even rows: left to right
                if all(row[j] >= row[j+1] for j in range(self.size-1) if row[j] != 0):
                    bonus += 1
            else:  # Odd rows: right to left
                if all(row[j] <= row[j+1] for j in range(self.size-1) if row[j] != 0):
                    bonus += 1
        
        return bonus
    
    def get_valid_moves(self) -> List[int]:
        """
        Get list of valid moves (moves that change the board).
        
        Returns:
            List of valid action indices
        """
        valid_moves = []
        
        for action in range(4):
            # Create temporary copy
            temp_game = copy.deepcopy(self)
            
            # Try the move
            if action == 0:
                changed, _ = temp_game.move_left()
            elif action == 1:
                changed, _ = temp_game.move_right()
            elif action == 2:
                changed, _ = temp_game.move_up()
            else:
                changed, _ = temp_game.move_down()
            
            if changed:
                valid_moves.append(action)
        
        return valid_moves
    
    def clone(self) -> 'Game2048':
        """
        Create a deep copy of the game state.
        
        Returns:
            New Game2048 instance with copied state
        """
        return copy.deepcopy(self)
    
    def __str__(self) -> str:
        """
        String representation of the board.
        
        Returns:
            Formatted board string
        """
        result = f"Score: {self.score}, Max Tile: {self.max_tile}, Moves: {self.moves_count}\n"
        result += "-" * (self.size * 6 + 1) + "\n"
        
        for i in range(self.size):
            result += "|"
            for j in range(self.size):
                value = self.board[i, j]
                if value == 0:
                    result += "     |"
                else:
                    result += f"{value:5d}|"
            result += "\n"
            result += "-" * (self.size * 6 + 1) + "\n"
        
        return result
