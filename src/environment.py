"""
═══════════════════════════════════════════════════════════════════════════════
2048 Game Environment - Gym-style RL Interface
═══════════════════════════════════════════════════════════════════════════════

This module wraps the 2048 game logic in a standard Reinforcement Learning
interface compatible with OpenAI Gym conventions.

STATE REPRESENTATION:
    - Neural network input: Flattened 16-dimensional vector
    - Each cell value is log2-normalized: log2(tile_value) if tile > 0, else 0
    - Example: [0, 1, 2, 3, ...] represents [empty, 2, 4, 8, ...]
    - Range: 0 (empty) to ~11 (2048 tile)

ACTION SPACE:
    - 4 discrete actions: [0, 1, 2, 3] = ["up", "down", "left", "right"]
    - Can use either integer index or string direction

REWARD STRUCTURE:
    - Primary: Score gained from merging tiles (e.g., merge two 4's → reward +8)
    - Penalty: Invalid moves (walls/no merge possible) → reward -10
    - Episode ends when no valid moves remain (game over)

USAGE:
    env = GameEnvironment(EnvironmentConfig(enable_ui=True))
    state = env.reset()
    result = env.step(action=0)  # Move up
    print(f"Reward: {result.reward}, Done: {result.done}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .game import GameBoard, GameUI


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

ACTIONS: List[str] = ["up", "down", "left", "right"]  # Action space (4 directions)


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StepResult:
    """
    Result returned from environment.step().
    
    Attributes:
        state: Log-normalized board state (16D flattened vector)
        reward: Immediate reward (score gain - penalty for invalid moves)
        done: True if game is over (no valid moves)
        info: Dict with {"moved", "score", "max_tile", "empty_cells"}
    """
    state: np.ndarray
    reward: float
    done: bool
    info: Dict


@dataclass
class EnvironmentConfig:
    """
    Configuration for 2048 game environment.
    
    Attributes:
        seed: Random seed for reproducibility (None = random)
        invalid_move_penalty: Negative reward for invalid moves (-10 recommended)
        enable_ui: Show pygame window for visualization
    """
    seed: Optional[int] = None
    invalid_move_penalty: float = -10.0
    enable_ui: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Main Environment Class
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GameEnvironment:
    """
    Gym-style Reinforcement Learning environment for 2048.
    
    Provides standard RL interface:
        - reset() → initial state
        - step(action) → StepResult(state, reward, done, info)
        - close() → cleanup
    
    Example:
        env = GameEnvironment()
        state = env.reset()
        
        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            state, reward, done = result.state, result.reward, result.done
        
        env.close()
    """
    
    config: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self) -> None:
        """Initialize game board and optional UI."""
        self.board = GameBoard(seed=self.config.seed)
        self.ui: Optional[GameUI] = None
        if self.config.enable_ui:
            self.ui = GameUI(self.board)
        self._last_score = 0

    def reset(self) -> np.ndarray:
        """
        Reset game to initial state (two random tiles).
        
        Returns:
            state: Log-normalized board state (16D vector)
        """
        board = self.board.reset()
        self._last_score = 0
        if self.ui:
            self.ui.draw()
        return self._build_state(board)

    def step(self, action: int | str) -> StepResult:
        """
        Execute one action and return results.
        
        Args:
            action: Integer (0-3) or string ("up", "down", "left", "right")
        
        Returns:
            StepResult with:
                - state: New board state after action
                - reward: Score gained (+ penalty if invalid move + bonuses)
                - done: True if no more valid moves
                - info: Dict with game statistics
        """
        direction = self._action_to_direction(action)
        
        # ─────────────────────────────────────────────────────────────────
        # Handle UI events (quit/restart) if pygame window is open
        # ─────────────────────────────────────────────────────────────────
        if self.ui:
            user_event = self.ui.handle_events()
            
            # User closed window → terminate episode
            if user_event == "quit":
                info = self.get_state()
                info.update({"moved": False, "terminated_by_user": True})
                return StepResult(
                    state=self._build_state(self.board.grid),
                    reward=0.0,
                    done=True,
                    info=info,
                )
            
            # User pressed R → restart game
            if user_event == "restart":
                board = self.board.reset()
                self._last_score = 0
                self.ui.draw()
                info = self.get_state()
                info.update({"moved": False, "restarted_by_user": True})
                return StepResult(
                    state=self._build_state(board),
                    reward=0.0,
                    done=False,
                    info=info,
                )
        
        # ─────────────────────────────────────────────────────────────────
        # Execute move and calculate reward
        # ─────────────────────────────────────────────────────────────────
        result = self.board.step(direction)
        moved = result.moved  # Did board change?
        
        # Base reward = score gained from merges (logarithmic scaling)
        # This emphasizes larger merges more
        base_reward = float(result.score_gain)
        if base_reward > 0:
            # Use log scaling to reward larger merges more
            base_reward = np.log2(base_reward + 1) * 10  # Scale up for significance
        
        reward = base_reward
        
        # Penalty for invalid moves (hits wall, no tiles merge)
        if not moved:
            reward += self.config.invalid_move_penalty
        else:
            # Bonus for valid moves to encourage action
            reward += 1.0
            
            # Bonus for maintaining empty cells (helps avoid getting stuck)
            empty_cells = len(self.board.get_empty_cells())
            reward += empty_cells * 0.5  # Small bonus per empty cell
            
            # Progressive tile milestone rewards (encourages reaching higher tiles)
            max_tile = self.board.max_tile()
            if max_tile >= 2048:
                reward += 1000  # Huge bonus for reaching 2048
            elif max_tile >= 1024:
                reward += 500
            elif max_tile >= 512:
                reward += 250
            elif max_tile >= 256:
                reward += 100
            elif max_tile >= 128:
                reward += 50
        
        # Update tracking and build new state
        self._last_score = self.board.score
        state = self._build_state(result.board)
        done = self.board.is_game_over()  # No more valid moves?
        
        # Package info for logging/debugging
        info = {
            "moved": moved,
            "score": self.board.score,
            "max_tile": self.board.max_tile(),
            "empty_cells": len(self.board.get_empty_cells()),
        }
        
        # Update pygame display
        if self.ui:
            self.ui.draw()
        
        return StepResult(state=state, reward=reward, done=done, info=info)

    def close(self) -> None:
        """Clean up pygame resources (call at end of training/playing)."""
        if self.ui:
            self.ui.close()

    def render(self) -> None:
        """Force redraw pygame window (if UI enabled)."""
        if self.ui:
            self.ui.draw()

    def get_state(self) -> Dict:
        """
        Get detailed game state information.
        
        Returns:
            Dict with:
                - board: Raw 4x4 grid (tile values)
                - log_board: Log2-normalized board
                - score: Current game score
                - max_tile: Highest tile value on board
                - empty_cells: Number of empty spaces
        """
        flat_board = self.board.to_array().astype(np.int32)
        log_board = self.board.to_normalized_state()
        return {
            "board": flat_board,
            "log_board": log_board,
            "score": self.board.score,
            "max_tile": self.board.max_tile(),
            "empty_cells": len(self.board.get_empty_cells()),
        }
    
    def get_board(self) -> GameBoard:
        """Get underlying GameBoard object (needed for MCTS tree search)."""
        return self.board

    def sample_action(self) -> int:
        """Sample random action from action space [0, 1, 2, 3]."""
        return int(np.random.randint(0, len(ACTIONS)))

    @staticmethod
    def action_space() -> List[str]:
        """Get list of valid action strings: ["up", "down", "left", "right"]."""
        return list(ACTIONS)

    # ───────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ───────────────────────────────────────────────────────────────────────

    def _build_state(self, board: np.ndarray) -> np.ndarray:
        """
        Convert raw board to neural network input format.
        
        Transformation:
            - Raw board: [[2, 4], [8, 0], ...] (tile values)
            - Log2:      [[1, 2], [3, 0], ...] (log2 of values)
            - Normalize: [[0.09, 0.18], [0.27, 0], ...] (divide by 11.0 for 2048 tile)
            - Flatten:   [0.09, 0.18, 0.27, 0, ...] (16D vector)
        
        This normalization helps neural networks learn better by keeping
        values in a normalized [0, 1] range where 1.0 represents a 2048 tile.
        
        Args:
            board: 4x4 numpy array of tile values
        
        Returns:
            Flattened log2-normalized state (16D float32 vector, range [0, 1])
        """
        with np.errstate(divide="ignore"):  # Ignore log2(0) warnings
            log_board = np.where(board > 0, np.log2(board), 0.0)
        # Normalize by 11.0 (log2(2048)) to keep values in [0, 1] range
        normalized = (log_board / 11.0).astype(np.float32)
        return normalized.flatten()

    @staticmethod
    def _action_to_direction(action: int | str) -> str:
        """
        Convert action to direction string.
        
        Args:
            action: Either integer index (0-3) or string ("up", "down", ...)
        
        Returns:
            Direction string ("up", "down", "left", "right")
        
        Raises:
            ValueError: If action is invalid
        """
        if isinstance(action, str):
            if action not in ACTIONS:
                raise ValueError(f"Unknown action: {action}. Use one of {ACTIONS}")
            return action
        
        idx = int(action)
        if not 0 <= idx < len(ACTIONS):
            raise ValueError(f"Action index {idx} out of range [0-{len(ACTIONS)-1}]")
        return ACTIONS[idx]
