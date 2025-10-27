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
        self._prev_max_tile = 0  # Track max tile for milestone bonuses

    def reset(self) -> np.ndarray:
        """
        Reset game to initial state (two random tiles).
        
        Returns:
            state: Log-normalized board state (16D vector)
        """
        board = self.board.reset()
        self._last_score = 0
        self._prev_max_tile = 0  # Reset max tile tracking
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
        # Execute move and calculate SMART reward function
        # ─────────────────────────────────────────────────────────────────
        result = self.board.step(direction)
        moved = result.moved  # Did board change?
        
        # Invalid move penalty (strong deterrent)
        if not moved:
            reward = self.config.invalid_move_penalty
        else:
            grid = self.board.grid
            max_tile = self.board.max_tile()
            empty_cells = len(self.board.get_empty_cells())
            
            # ═════════════════════════════════════════════════════════════
            # COMPONENT 1: Score-based reward (weighted by max tile)
            # ═════════════════════════════════════════════════════════════
            # Key insight: Same score means MORE with higher max tile
            # Score of 100 with max=512 is better than score=100 with max=64
            
            score_gain = float(result.score_gain)
            if max_tile >= 64:
                # Amplify score rewards as max tile increases
                tile_multiplier = np.log2(max_tile) / 10.0  # 64→0.6, 512→0.9, 1024→1.0
                score_reward = score_gain * tile_multiplier
            else:
                # Early game: smaller multiplier (prevent optimizing for small merges)
                score_reward = score_gain * 0.2
            
            reward = score_reward
            
            # ═════════════════════════════════════════════════════════════
            # COMPONENT 2: Max tile progression bonuses (EXPONENTIAL)
            # ═════════════════════════════════════════════════════════════
            # Massive one-time bonuses for reaching new tiles
            
            if max_tile > self._prev_max_tile:
                # Exponential milestone rewards: higher tiles = exponentially better!
                milestone_bonus = max_tile * 3.0
                reward += milestone_bonus
                self._prev_max_tile = max_tile
                # Examples: 128→+384, 256→+768, 512→+1536, 1024→+3072
            
            # ═════════════════════════════════════════════════════════════
            # COMPONENT 3: Corner strategy (POSITIVE REINFORCEMENT ONLY)
            # ═════════════════════════════════════════════════════════════
            # Only reward good behavior, don't punish exploration
            
            corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
            corner_values = [grid[r, c] for r, c in corners]
            max_in_corner = max_tile in corner_values
            
            if max_tile >= 32 and max_in_corner:
                # Reward scales with tile value (keeping 512 in corner > keeping 64)
                corner_bonus = np.log2(max_tile) ** 1.2 * 2.0
                reward += corner_bonus
                # Examples: 64→+10, 128→+14, 256→+19, 512→+25, 1024→+33
            
            # ═════════════════════════════════════════════════════════════
            # COMPONENT 4: Board quality heuristics
            # ═════════════════════════════════════════════════════════════
            
            # 4a. Empty cells bonus (breathing room = flexibility)
            if empty_cells >= 4:
                reward += empty_cells * 1.0
            elif empty_cells <= 2:
                # Mild penalty for cramped board (dangerous territory)
                reward -= (3 - empty_cells) * 1.0
            
            # 4b. Merge potential (adjacent equal tiles = future opportunities)
            merge_potential = self._count_merge_potential(grid)
            if merge_potential >= 3:
                reward += merge_potential * 0.8
            
            # 4c. Monotonicity bonus (tiles in decreasing order = snake pattern)
            if max_tile >= 128:
                # Check if board has good monotonic structure
                monotonicity = self._calculate_monotonicity(grid)
                reward += monotonicity * 2.0
            
            # 4d. Smoothness penalty (large differences between adjacent tiles = bad)
            if max_tile >= 64:
                smoothness = self._calculate_smoothness(grid)
                reward += smoothness * 0.5  # Negative values penalize rough boards
        
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
                - max_tile: HISTORICAL maximum tile achieved during episode (not current board max)
                - empty_cells: Number of empty spaces
        """
        flat_board = self.board.to_array().astype(np.int32)
        log_board = self.board.to_normalized_state()
        return {
            "board": flat_board,
            "log_board": log_board,
            "score": self.board.score,
            "max_tile": self._prev_max_tile,  # Historical max, not current board max
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

    def _calculate_snake_bonus(self, grid: np.ndarray, max_tile: int) -> float:
        """
        Reward snake pattern: monotonically decreasing values from corner.
        
        Checks all four corners and rewards the best snake pattern found.
        Snake pattern means tiles decrease in value as you move away from corner.
        
        Args:
            grid: 4x4 board grid
            max_tile: Highest tile value
            
        Returns:
            Bonus reward for good snake patterns
        """
        best_score = 0.0
        
        # Check all 4 corners for snake patterns
        patterns = [
            # Top-left corner: scan right then snake down
            [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0)],
            # Top-right corner: scan left then snake down  
            [(0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (1, 1), (1, 2), (1, 3)],
            # Bottom-left corner: scan right then snake up
            [(3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (2, 2), (2, 1), (2, 0)],
            # Bottom-right corner: scan left then snake up
            [(3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (2, 1), (2, 2), (2, 3)],
        ]
        
        for pattern in patterns:
            score = 0.0
            prev_val = float('inf')
            
            for r, c in pattern:
                val = grid[r, c]
                if val == 0:
                    continue
                    
                # Reward if tiles decrease monotonically
                if val <= prev_val:
                    # Bigger tiles = bigger reward for proper ordering
                    score += np.log2(val) if val > 0 else 0
                else:
                    # Penalty for breaking the snake (tile is bigger than previous)
                    score -= 2.0
                    
                prev_val = val
            
            best_score = max(best_score, score)
        
        return best_score * 0.5  # Scale down to reasonable range
    
    def _calculate_edge_bonus(self, grid: np.ndarray) -> float:
        """
        Reward keeping high-value tiles on edges (not in center).
        
        Center tiles are harder to merge and block movement.
        Edge tiles maintain flexibility.
        
        Args:
            grid: 4x4 board grid
            
        Returns:
            Bonus for good edge alignment
        """
        bonus = 0.0
        
        # Get all edge positions
        edges = (
            [(0, c) for c in range(4)] +  # Top row
            [(3, c) for c in range(4)] +  # Bottom row
            [(r, 0) for r in range(1, 3)] +  # Left column (excluding corners)
            [(r, 3) for r in range(1, 3)]    # Right column (excluding corners)
        )
        
        # Get center positions
        center = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        # Calculate average tile value on edges vs center
        edge_vals = [grid[r, c] for r, c in edges if grid[r, c] > 0]
        center_vals = [grid[r, c] for r, c in center if grid[r, c] > 0]
        
        if edge_vals and center_vals:
            avg_edge = np.mean([np.log2(v) for v in edge_vals])
            avg_center = np.mean([np.log2(v) for v in center_vals])
            
            # Reward if edges have higher average value than center
            if avg_edge > avg_center:
                bonus = (avg_edge - avg_center) * 2.0
        
        return bonus
    
    def _calculate_order_bonus(self, grid: np.ndarray, max_tile: int) -> float:
        """
        Reward having tiles near max tile in descending order.
        
        Checks the row/column containing max tile for monotonicity.
        
        Args:
            grid: 4x4 board grid
            max_tile: Highest tile value
            
        Returns:
            Bonus for proper tile ordering
        """
        # Find max tile position
        max_pos = np.argwhere(grid == max_tile)
        if len(max_pos) == 0:
            return 0.0
        
        r, c = max_pos[0]
        bonus = 0.0
        
        # Check row containing max tile for monotonicity
        row = grid[r, :]
        row_bonus = self._check_monotonicity(row)
        
        # Check column containing max tile for monotonicity
        col = grid[:, c]
        col_bonus = self._check_monotonicity(col)
        
        bonus = max(row_bonus, col_bonus)
        return bonus * 2.0
    
    def _check_monotonicity(self, line: np.ndarray) -> float:
        """
        Check if a line of tiles is monotonically decreasing.
        
        Args:
            line: 1D array of tile values
            
        Returns:
            Score for monotonicity (higher = more monotonic)
        """
        score = 0.0
        non_zero = line[line > 0]
        
        if len(non_zero) <= 1:
            return 0.0
        
        # Check both directions (left-to-right and right-to-left)
        decreasing_score = 0.0
        increasing_score = 0.0
        
        for i in range(len(non_zero) - 1):
            if non_zero[i] >= non_zero[i + 1]:
                decreasing_score += 1.0
            if non_zero[i] <= non_zero[i + 1]:
                increasing_score += 1.0
        
        # Return best monotonicity direction
        return max(decreasing_score, increasing_score)
    
    def _calculate_monotonicity(self, grid: np.ndarray) -> float:
        """
        Calculate monotonicity score for entire grid.
        
        Checks all rows and columns for monotonic ordering (either increasing 
        or decreasing). Higher score = more organized board structure.
        
        Args:
            grid: 4x4 board grid
            
        Returns:
            Total monotonicity score across all rows and columns
        """
        total_monotonicity = 0.0
        
        # Check all rows
        for r in range(4):
            total_monotonicity += self._check_monotonicity(grid[r, :])
        
        # Check all columns
        for c in range(4):
            total_monotonicity += self._check_monotonicity(grid[:, c])
        
        return total_monotonicity
    
    def _count_merge_potential(self, grid: np.ndarray) -> int:
        """
        Count pairs of adjacent equal tiles (potential merges).
        
        More merge potential = more flexibility for next moves.
        
        Args:
            grid: 4x4 board grid
            
        Returns:
            Number of adjacent tile pairs that can merge
        """
        count = 0
        
        # Check horizontal adjacents
        for r in range(4):
            for c in range(3):
                if grid[r, c] > 0 and grid[r, c] == grid[r, c + 1]:
                    count += 1
        
        # Check vertical adjacents
        for r in range(3):
            for c in range(4):
                if grid[r, c] > 0 and grid[r, c] == grid[r + 1, c]:
                    count += 1
        
        return count
    
    def _calculate_smoothness(self, grid: np.ndarray) -> float:
        """
        Calculate smoothness of the grid (how similar adjacent tiles are).
        
        Smoothness measures the difference between adjacent tiles.
        Lower differences = smoother board = easier to merge tiles.
        
        Args:
            grid: 4x4 board grid
            
        Returns:
            Smoothness score (0 = perfect smoothness, higher = more rough)
        """
        smoothness = 0.0
        
        # Check horizontal adjacents
        for r in range(4):
            for c in range(3):
                if grid[r, c] > 0 and grid[r, c + 1] > 0:
                    diff = abs(np.log2(grid[r, c]) - np.log2(grid[r, c + 1]))
                    smoothness += diff
        
        # Check vertical adjacents
        for r in range(3):
            for c in range(4):
                if grid[r, c] > 0 and grid[r + 1, c] > 0:
                    diff = abs(np.log2(grid[r, c]) - np.log2(grid[r + 1, c]))
                    smoothness += diff
        
        # Return negative (we want to minimize smoothness, so reward low values)
        # Normalize by dividing by typical smoothness (~40-60)
        return -smoothness / 50.0

    def _build_state(self, board: np.ndarray) -> np.ndarray:
        """
        Convert raw board to neural network input format.
        
        Transformation:
            - Raw board: [[2, 4], [8, 0], ...] (tile values)
            - Log2:      [[1, 2], [3, 0], ...] (log2 of values)
            - Normalize: [[0.06, 0.13], [0.19, 0], ...] (divide by 15.0 for 32768 tile)
            - Flatten:   [0.06, 0.13, 0.19, 0, ...] (16D vector)
        
        This normalization helps neural networks learn better by keeping
        values in a normalized [0, ~1] range. Using 15.0 (log2(32768)) allows
        tiles beyond 2048 without exceeding 1.0.
        
        Args:
            board: 4x4 numpy array of tile values
        
        Returns:
            Flattened log2-normalized state (16D float32 vector, range [0, ~1])
        """
        with np.errstate(divide="ignore"):  # Ignore log2(0) warnings
            log_board = np.where(board > 0, np.log2(board), 0.0)
        # Normalize by 15.0 (log2(32768)) to handle tiles beyond 2048
        # 2048 tile = 11/15 = 0.73, 4096 = 12/15 = 0.8, etc.
        normalized = (log_board / 15.0).astype(np.float32)
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
