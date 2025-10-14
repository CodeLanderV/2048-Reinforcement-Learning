"""Monte Carlo Tree Search Agent for 2048."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ...game import GameBoard


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    simulations: int = 100
    exploration_constant: float = 1.41


class MCTSNode:
    """Node in MCTS tree."""
    
    def __init__(self, board: GameBoard, parent: Optional[MCTSNode] = None, action: Optional[int] = None):
        self.board = board.clone()
        self.parent = parent
        self.action = action
        self.children: List[Optional[MCTSNode]] = [None] * 4
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self) -> bool:
        """Check if all children have been created."""
        return all(child is not None for child in self.children)

    def best_child(self, exploration_constant: float) -> MCTSNode:
        """Select child with highest UCB score."""
        best_score = float("-inf")
        best_node = None
        
        for child in self.children:
            if child is None:
                continue
            if child.visits == 0:
                score = float("inf")
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_node = child
        
        return best_node  # type: ignore[return-value]

    def expand(self, action: int) -> MCTSNode:
        """Expand tree by creating child node."""
        child_board = self.board.clone()
        directions = ["up", "down", "left", "right"]
        child_board.step(directions[action])
        child_node = MCTSNode(child_board, parent=self, action=action)
        self.children[action] = child_node
        return child_node


class MCTSAgent:
    """Monte Carlo Tree Search agent."""
    
    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        action_space: Optional[List[str]] = None,
    ):
        self.config = config if config is not None else MCTSConfig()
        self.action_space = action_space if action_space is not None else ["up", "down", "left", "right"]

    def select_action(self, state: np.ndarray, board: GameBoard) -> int:
        """Select action using MCTS."""
        root = MCTSNode(board)
        
        for _ in range(self.config.simulations):
            node = root
            
            # Selection
            while node.is_fully_expanded() and not node.board.is_game_over():
                node = node.best_child(self.config.exploration_constant)
            
            # Expansion
            if not node.board.is_game_over():
                available_actions = [i for i in range(4) if node.children[i] is None]
                if available_actions:
                    action = random.choice(available_actions)
                    node = node.expand(action)
            
            # Simulation
            reward = self._simulate(node.board)
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        # Return most visited child
        best_action = max(range(4), key=lambda a: root.children[a].visits if root.children[a] else 0)
        return best_action

    def act_greedy(self, state: np.ndarray, board: GameBoard) -> int:
        """Same as select_action for MCTS."""
        return self.select_action(state, board)

    def _simulate(self, board: GameBoard) -> float:
        """Simulate random playthrough."""
        sim_board = board.clone()
        total_reward = 0.0
        
        while not sim_board.is_game_over():
            direction = random.choice(["up", "down", "left", "right"])
            result = sim_board.step(direction)
            total_reward += result.score_gain
        
        return total_reward

    def save(self, path: Path, episode: int) -> None:
        """MCTS doesn't need saving (no learned parameters)."""
        pass

    def load(self, path: Path) -> None:
        """MCTS doesn't need loading (no learned parameters)."""
        pass
