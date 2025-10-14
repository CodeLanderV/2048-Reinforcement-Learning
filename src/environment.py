"""Environment wrapper for 2048 game - Gym-style interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .game import GameBoard, GameUI


# Action space
ACTIONS: List[str] = ["up", "down", "left", "right"]


@dataclass
class StepResult:
    """Result from environment step."""
    state: np.ndarray
    reward: float
    done: bool
    info: Dict


@dataclass
class EnvironmentConfig:
    """Configuration for game environment."""
    seed: Optional[int] = None
    invalid_move_penalty: float = -5.0
    enable_ui: bool = False


@dataclass
class GameEnvironment:
    """Gym-style environment for 2048 game."""
    
    config: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    def __post_init__(self) -> None:
        self.board = GameBoard(seed=self.config.seed)
        self.ui: Optional[GameUI] = None
        if self.config.enable_ui:
            self.ui = GameUI(self.board)
        self._last_score = 0

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        board = self.board.reset()
        self._last_score = 0
        if self.ui:
            self.ui.draw()
        return self._build_state(board)

    def step(self, action: int | str) -> StepResult:
        """Execute action and return result."""
        direction = self._action_to_direction(action)
        
        # Handle UI events if enabled
        if self.ui:
            user_event = self.ui.handle_events()
            if user_event == "quit":
                info = self.get_state()
                info.update({"moved": False, "terminated_by_user": True})
                return StepResult(
                    state=self._build_state(self.board.grid),
                    reward=0.0,
                    done=True,
                    info=info,
                )
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
        
        # Execute move
        result = self.board.step(direction)
        moved = result.moved
        reward = float(result.score_gain)
        if not moved:
            reward += self.config.invalid_move_penalty
        
        self._last_score = self.board.score
        state = self._build_state(result.board)
        done = self.board.is_game_over()
        
        info = {
            "moved": moved,
            "score": self.board.score,
            "max_tile": self.board.max_tile(),
            "empty_cells": len(self.board.get_empty_cells()),
        }
        
        if self.ui:
            self.ui.draw()
        
        return StepResult(state=state, reward=reward, done=done, info=info)

    def close(self) -> None:
        """Clean up resources."""
        if self.ui:
            self.ui.close()

    def render(self) -> None:
        """Render current state (if UI enabled)."""
        if self.ui:
            self.ui.draw()

    def get_state(self) -> Dict:
        """Get current state information."""
        flat_board = self.board.to_array().astype(np.int32)
        log_board = self.board.to_normalized_state()
        return {
            "board": flat_board,
            "log_board": log_board,
            "score": self.board.score,
            "max_tile": self.board.max_tile(),
            "empty_cells": len(self.board.get_empty_cells()),
        }

    def sample_action(self) -> int:
        """Sample random action."""
        return int(np.random.randint(0, len(ACTIONS)))

    @staticmethod
    def action_space() -> List[str]:
        """Get available actions."""
        return list(ACTIONS)

    def _build_state(self, board: np.ndarray) -> np.ndarray:
        """Convert board to neural network input (flattened log-normalized)."""
        with np.errstate(divide="ignore"):
            log_board = np.where(board > 0, np.log2(board), 0.0)
        return log_board.flatten().astype(np.float32)

    @staticmethod
    def _action_to_direction(action: int | str) -> str:
        """Convert action index or string to direction."""
        if isinstance(action, str):
            if action not in ACTIONS:
                raise ValueError(f"Unknown action: {action}")
            return action
        idx = int(action)
        if not 0 <= idx < len(ACTIONS):
            raise ValueError(f"Action index out of range: {idx}")
        return ACTIONS[idx]
