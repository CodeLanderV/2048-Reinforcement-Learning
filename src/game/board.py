"""Core 2048 game board implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class MoveResult:
    """Container with details about the outcome of a move."""
    moved: bool
    score_gain: int
    board: np.ndarray


class GameBoard:
    """Representation of the 2048 board and rules."""

    GRID_SIZE: int = 4
    START_TILES: int = 2
    FOUR_PROBABILITY: float = 0.1

    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = np.random.default_rng(seed)
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        self.score: int = 0
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the board and spawn initial tiles."""
        self.grid.fill(0)
        self.score = 0
        for _ in range(self.START_TILES):
            self._spawn_tile()
        return self.grid.copy()

    def step(self, direction: str) -> MoveResult:
        """Perform a move in the given direction."""
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError(f"Invalid move direction: {direction}")

        rotated = self._rotate_board_for_direction(direction)
        moved, score_gain = self._collapse_left(rotated)
        if moved:
            self._spawn_tile(board=rotated)
            self.score += score_gain
        self._undo_rotation(rotated, direction)
        return MoveResult(moved=moved, score_gain=score_gain, board=self.grid.copy())

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of empty cell coordinates."""
        return list(zip(*np.where(self.grid == 0)))

    def max_tile(self) -> int:
        """Get maximum tile value on board."""
        return int(self.grid.max())

    def is_full(self) -> bool:
        """Check if board has no empty cells."""
        return not np.any(self.grid == 0)

    def can_move(self) -> bool:
        """Check if any valid move exists."""
        if not self.is_full():
            return True
        grid = self.grid
        # Check horizontal adjacents
        if np.any(grid[:, :-1] == grid[:, 1:]):
            return True
        # Check vertical adjacents
        if np.any(grid[:-1, :] == grid[1:, :]):
            return True
        return False

    def is_game_over(self) -> bool:
        """Check if game is over (no valid moves)."""
        return not self.can_move()

    def clone(self) -> "GameBoard":
        """Create a deep copy of the board."""
        clone = GameBoard()
        clone.grid = self.grid.copy()
        clone.score = self.score
        return clone

    def to_array(self) -> np.ndarray:
        """Get board as numpy array."""
        return self.grid.copy()

    def to_normalized_state(self) -> np.ndarray:
        """Return log2 normalized board suitable for neural networks."""
        with np.errstate(divide="ignore"):
            log_board = np.where(self.grid > 0, np.log2(self.grid), 0.0)
        # Normalize by 15.0 (log2(32768)) to handle tiles beyond 2048
        # Values range from 0 (empty) to ~1 (tiles up to 32768)
        return (log_board / 15.0).astype(np.float32)  # Normalize to [0, ~1] range

    # Private helper methods
    def _spawn_tile(self, board: Optional[np.ndarray] = None) -> None:
        """Spawn a new tile (2 or 4) in random empty cell."""
        board = board if board is not None else self.grid
        empties = list(zip(*np.where(board == 0)))
        if not empties:
            return
        row, col = empties[self.random.integers(len(empties))]
        value = 4 if self.random.random() < self.FOUR_PROBABILITY else 2
        board[row, col] = value

    def _rotate_board_for_direction(self, direction: str) -> np.ndarray:
        """Rotate board so direction becomes left."""
        if direction == "left":
            return self.grid
        if direction == "right":
            return np.rot90(self.grid, k=2)
        if direction == "up":
            return np.rot90(self.grid, k=1)
        if direction == "down":
            return np.rot90(self.grid, k=-1)
        raise ValueError(direction)

    def _undo_rotation(self, board: np.ndarray, direction: str) -> None:
        """Restore original orientation after collapse."""
        if direction == "left":
            self.grid = board
        elif direction == "right":
            self.grid = np.rot90(board, k=2)
        elif direction == "up":
            self.grid = np.rot90(board, k=-1)
        elif direction == "down":
            self.grid = np.rot90(board, k=1)
        else:
            raise ValueError(direction)

    def _collapse_left(self, board: np.ndarray) -> Tuple[bool, int]:
        """Collapse and merge tiles to the left."""
        moved = False
        score_gain = 0
        for row in range(self.GRID_SIZE):
            line = board[row, :]
            original = line.copy()
            cleaned = line[line != 0]
            merged: List[int] = []
            skip = False
            for idx in range(len(cleaned)):
                if skip:
                    skip = False
                    continue
                tile = cleaned[idx]
                if idx + 1 < len(cleaned) and cleaned[idx + 1] == tile:
                    merged_value = tile * 2
                    merged.append(merged_value)
                    score_gain += merged_value
                    skip = True
                else:
                    merged.append(int(tile))
            merged_array = np.array(merged, dtype=np.int32)
            new_line = np.pad(merged_array, (0, self.GRID_SIZE - len(merged_array)))
            if not np.array_equal(original, new_line):
                moved = True
            board[row, :] = new_line
        return moved, score_gain
