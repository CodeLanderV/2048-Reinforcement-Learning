"""
Monte Carlo Tree Search (MCTS) Agent for 2048

This agent uses random simulations to evaluate moves.
No training required - works immediately!
"""

import sys
import os
import numpy as np
import copy
from typing import List, Tuple

# Add parent directory to path to import game modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'game'))
from game_logic import Game2048


class MCTSAgent:
    """
    Monte Carlo Tree Search agent for 2048.
    
    Strategy:
    1. For each possible action, simulate N random games
    2. Track the average score achieved
    3. Choose the action with best average score
    """
    
    def __init__(self, num_simulations=50, max_depth=10):
        """
        Initialize MCTS agent.
        
        Args:
            num_simulations (int): Number of random playouts per action
            max_depth (int): Maximum depth for each simulation
        """
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    
    def choose_action(self, game: Game2048) -> int:
        """
        Choose the best action using MCTS.
        
        Args:
            game (Game2048): Current game state
        
        Returns:
            int: Best action (0=up, 1=right, 2=down, 3=left)
        """
        # Get available moves
        available_actions = game.get_available_moves()
        
        if not available_actions:
            return 0  # Game over, doesn't matter
        
        if len(available_actions) == 1:
            return available_actions[0]  # Only one option
        
        # Evaluate each action
        action_scores = {}
        
        for action in available_actions:
            # Run multiple simulations for this action
            total_score = 0
            
            for _ in range(self.num_simulations):
                score = self._simulate(game, action)
                total_score += score
            
            # Average score for this action
            avg_score = total_score / self.num_simulations
            action_scores[action] = avg_score
        
        # Choose action with best average score
        best_action = max(action_scores, key=action_scores.get)
        
        return best_action
    
    def _simulate(self, game: Game2048, first_action: int) -> float:
        """
        Simulate a random game starting with a specific first action.
        
        Args:
            game (Game2048): Current game state
            first_action (int): First action to take
        
        Returns:
            float: Final score achieved in simulation
        """
        # Create a copy of the game
        sim_game = self._copy_game(game)
        
        # Take the first action
        sim_game.move(first_action)
        
        # Play randomly until game over or max depth
        depth = 0
        while not sim_game.is_game_over() and depth < self.max_depth:
            # Get available moves
            available = sim_game.get_available_moves()
            if not available:
                break
            
            # Choose random action
            action = np.random.choice(available)
            sim_game.move(action)
            depth += 1
        
        # Return final score
        return sim_game.get_score()
    
    def _copy_game(self, game: Game2048) -> Game2048:
        """
        Create a deep copy of the game state.
        
        Args:
            game (Game2048): Game to copy
        
        Returns:
            Game2048: Copy of the game
        """
        new_game = Game2048()
        new_game.board = game.board.copy()
        new_game.score = game.score
        new_game.game_over = game.game_over
        new_game.won = game.won
        return new_game
    
    def play_game(self, verbose=True) -> Tuple[int, int]:
        """
        Play a complete game using MCTS.
        
        Args:
            verbose (bool): Print game progress
        
        Returns:
            tuple: (final_score, highest_tile)
        """
        game = Game2048()
        moves = 0
        
        if verbose:
            print("Starting MCTS game...")
            print(f"Simulations per move: {self.num_simulations}")
            print(f"Max depth per simulation: {self.max_depth}")
            print("="*50)
        
        while not game.is_game_over():
            # Choose action
            action = self.choose_action(game)
            
            # Execute action
            moved = game.move(action)
            
            if moved:
                moves += 1
                if verbose and moves % 10 == 0:
                    print(f"Move {moves}: Score={game.get_score()}, "
                          f"Max Tile={np.max(game.get_board())}")
        
        final_score = game.get_score()
        highest_tile = int(np.max(game.get_board()))
        
        if verbose:
            print("="*50)
            print(f"Game Over!")
            print(f"Final Score: {final_score}")
            print(f"Highest Tile: {highest_tile}")
            print(f"Total Moves: {moves}")
            print("\nFinal Board:")
            print(game.get_board())
        
        return final_score, highest_tile


# Test the agent
if __name__ == "__main__":
    print("Testing MCTS Agent for 2048")
    print("="*50)
    
    # Create agent with moderate settings
    agent = MCTSAgent(num_simulations=30, max_depth=8)
    
    # Play one game
    print("\nPlaying one game...\n")
    score, tile = agent.play_game(verbose=True)
    
    print("\n" + "="*50)
    print("Test complete!")
