"""
Logging utilities for training and gameplay.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import numpy as np


class GameLogger:
    """
    Logger for game events, training progress, and statistics.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Training logs
        self.training_log = []
        self.episode_scores = []
        self.episode_max_tiles = []
        self.episode_moves = []
        self.episode_rewards = []
        
        # Playing logs
        self.playing_log = []
        
        # Current episode data
        self.current_episode_data = {}
    
    def start_episode(self, episode: int, mode: str = "training"):
        """
        Start logging a new episode.
        
        Args:
            episode: Episode number
            mode: 'training' or 'playing'
        """
        self.current_episode_data = {
            'episode': episode,
            'mode': mode,
            'start_time': datetime.now().isoformat(),
            'moves': [],
            'rewards': [],
            'score': 0,
            'max_tile': 0,
            'moves_count': 0
        }
    
    def log_step(self, action: int, reward: float, state: np.ndarray, 
                 score: int, max_tile: int):
        """
        Log a single step in the episode.
        
        Args:
            action: Action taken
            reward: Reward received
            state: Game state
            score: Current score
            max_tile: Current max tile
        """
        if self.current_episode_data:
            self.current_episode_data['moves'].append({
                'action': int(action),
                'reward': float(reward)
            })
            self.current_episode_data['rewards'].append(float(reward))
            self.current_episode_data['score'] = int(score)
            self.current_episode_data['max_tile'] = int(max_tile)
            self.current_episode_data['moves_count'] += 1
    
    def end_episode(self, final_score: int, final_max_tile: int, 
                   additional_info: Dict = None):
        """
        End the current episode and save logs.
        
        Args:
            final_score: Final score
            final_max_tile: Final max tile
            additional_info: Additional information to log
        """
        if self.current_episode_data:
            self.current_episode_data['end_time'] = datetime.now().isoformat()
            self.current_episode_data['final_score'] = int(final_score)
            self.current_episode_data['final_max_tile'] = int(final_max_tile)
            self.current_episode_data['total_reward'] = sum(self.current_episode_data['rewards'])
            
            if additional_info:
                self.current_episode_data.update(additional_info)
            
            # Add to appropriate log
            if self.current_episode_data['mode'] == 'training':
                self.training_log.append(self.current_episode_data.copy())
                self.episode_scores.append(final_score)
                self.episode_max_tiles.append(final_max_tile)
                self.episode_moves.append(self.current_episode_data['moves_count'])
                self.episode_rewards.append(self.current_episode_data['total_reward'])
            else:
                self.playing_log.append(self.current_episode_data.copy())
            
            self.current_episode_data = {}
    
    def save_training_log(self, filename: str = None):
        """
        Save training log to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"Training log saved to {filepath}")
        return filepath
    
    def save_playing_log(self, filename: str = None):
        """
        Save playing log to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"playing_log_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.playing_log, f, indent=2)
        
        print(f"Playing log saved to {filepath}")
        return filepath
    
    def save_statistics(self, filename: str = None):
        """
        Save training statistics to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"statistics_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        stats = {
            'total_episodes': len(self.episode_scores),
            'scores': {
                'mean': float(np.mean(self.episode_scores)) if self.episode_scores else 0,
                'std': float(np.std(self.episode_scores)) if self.episode_scores else 0,
                'min': int(np.min(self.episode_scores)) if self.episode_scores else 0,
                'max': int(np.max(self.episode_scores)) if self.episode_scores else 0,
                'all': [int(s) for s in self.episode_scores]
            },
            'max_tiles': {
                'mean': float(np.mean(self.episode_max_tiles)) if self.episode_max_tiles else 0,
                'std': float(np.std(self.episode_max_tiles)) if self.episode_max_tiles else 0,
                'min': int(np.min(self.episode_max_tiles)) if self.episode_max_tiles else 0,
                'max': int(np.max(self.episode_max_tiles)) if self.episode_max_tiles else 0,
                'all': [int(t) for t in self.episode_max_tiles]
            },
            'moves': {
                'mean': float(np.mean(self.episode_moves)) if self.episode_moves else 0,
                'std': float(np.std(self.episode_moves)) if self.episode_moves else 0,
                'min': int(np.min(self.episode_moves)) if self.episode_moves else 0,
                'max': int(np.max(self.episode_moves)) if self.episode_moves else 0,
                'all': [int(m) for m in self.episode_moves]
            },
            'rewards': {
                'mean': float(np.mean(self.episode_rewards)) if self.episode_rewards else 0,
                'std': float(np.std(self.episode_rewards)) if self.episode_rewards else 0,
                'min': float(np.min(self.episode_rewards)) if self.episode_rewards else 0,
                'max': float(np.max(self.episode_rewards)) if self.episode_rewards else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {filepath}")
        return filepath
    
    def get_statistics(self) -> Dict:
        """
        Get current training statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.episode_scores:
            return {}
        
        return {
            'episodes': len(self.episode_scores),
            'avg_score': np.mean(self.episode_scores[-100:]),
            'avg_max_tile': np.mean(self.episode_max_tiles[-100:]),
            'avg_moves': np.mean(self.episode_moves[-100:]),
            'best_score': np.max(self.episode_scores),
            'best_max_tile': np.max(self.episode_max_tiles)
        }


def save_game_state(game_state: Dict[str, Any], filename: str, state_dir: str = "game_states"):
    """
    Save game state to file.
    
    Args:
        game_state: Dictionary containing game state
        filename: Output filename
        state_dir: Directory to save state
    """
    os.makedirs(state_dir, exist_ok=True)
    filepath = os.path.join(state_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_state = {}
    for key, value in game_state.items():
        if isinstance(value, np.ndarray):
            serializable_state[key] = value.tolist()
        else:
            serializable_state[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_state, f, indent=2)
    
    print(f"Game state saved to {filepath}")
    return filepath


def load_game_state(filename: str, state_dir: str = "game_states") -> Dict[str, Any]:
    """
    Load game state from file.
    
    Args:
        filename: Input filename
        state_dir: Directory containing state
        
    Returns:
        Dictionary containing game state
    """
    filepath = os.path.join(state_dir, filename)
    
    with open(filepath, 'r') as f:
        state = json.load(f)
    
    # Convert lists back to numpy arrays
    if 'board' in state and isinstance(state['board'], list):
        state['board'] = np.array(state['board'], dtype=np.int32)
    
    print(f"Game state loaded from {filepath}")
    return state
