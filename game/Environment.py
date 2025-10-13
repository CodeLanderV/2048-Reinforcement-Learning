"""
2048 RL Environment (No OpenAI Gym Dependency)

A simple, lightweight environment wrapper for the 2048 game.
Designed for RL agent training.
"""

import numpy as np
from game_logic import Game2048


class Game2048Environment:
    """
    Environment wrapper for 2048 game.
    
    No OpenAI Gym dependency - just a simple, clean interface for RL agents.
    """
    
    def __init__(self):
        """Initialize the environment"""
        self.game = Game2048()
        self.action_space_n = 4  # up, right, down, left
        self.observation_shape = (4, 4)  # 4x4 board
        
    def reset(self):
        """
        Reset the environment to start a new episode.
        
        Returns:
            state (np.ndarray): Initial board state (4x4 array of raw values)
        """
        self.game.reset()
        return self.get_state()
    
    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action (int): Action to take (0=up, 1=right, 2=down, 3=left)
        
        Returns:
            tuple: (state, reward, done, info)
                - state (np.ndarray): New board state
                - reward (float): Reward from this action (score increase)
                - done (bool): True if episode is over
                - info (dict): Additional information
        """
        # Track score before move
        old_score = self.game.get_score()
        
        # Execute the move
        moved = self.game.move(action)
        
        # Calculate reward (score increase)
        new_score = self.game.get_score()
        reward = new_score - old_score
        
        # Check if game is over
        done = self.game.is_game_over()
        
        # Get new state
        state = self.get_state()
        
        # Additional info
        info = {
            'score': new_score,
            'highest_tile': np.max(state),
            'moved': moved,  # Was the move valid?
        }
        
        return state, reward, done, info
    
    def get_state(self):
        """
        Get the current state of the environment.
        
        Returns:
            np.ndarray: Board state (4x4 array of raw tile values)
        """
        return self.game.get_board()
    
    def get_available_actions(self):
        """
        Get list of actions that would change the board.
        
        Returns:
            list: Valid action indices
        """
        return self.game.get_available_moves()
    
    def render(self, mode='human'):
        """
        Render the current state.
        
        Args:
            mode (str): Rendering mode ('human' for text, 'rgb_array' for image)
        """
        if mode == 'human':
            board = self.get_state()
            print("\n" + "="*30)
            print(f"Score: {self.game.get_score()}")
            print("="*30)
            for row in board:
                print("|", end="")
                for val in row:
                    if val == 0:
                        print("    |", end="")
                    else:
                        print(f"{val:4d}|", end="")
                print()
            print("="*30)
    
    def close(self):
        """Clean up resources (if any)"""
        pass


# Example usage
if __name__ == "__main__":
    print("Testing 2048 Environment")
    print("="*50)
    
    env = Game2048Environment()
    
    # Reset environment
    state = env.reset()
    print("\nInitial State:")
    env.render()
    
    # Take some random actions
    print("\nTaking 5 random actions...")
    for i in range(5):
        action = np.random.choice([0, 1, 2, 3])
        action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        state, reward, done, info = env.step(action)
        
        print(f"\nAction: {action_names[action]}")
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")
        env.render()
        
        if done:
            print("\nGame Over!")
            break
    
    print("\n" + "="*50)
    print("Test complete!")
