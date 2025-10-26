"""
═══════════════════════════════════════════════════════════════════════════════
Training Utilities - Timers, Logging, and Helper Functions
═══════════════════════════════════════════════════════════════════════════════

This module provides utilities for tracking training progress and logging results.

KEY CLASSES:
    - TrainingTimer:     Track training duration with human-readable output
    - EvaluationLogger:  Log training results to file for comparison

USAGE:
    # Time your training
    timer = TrainingTimer().start()
    # ... train model ...
    timer.stop()
    print(f"Training took: {timer.elapsed_str()}")  # "2:15:30"
    
    # Log results
    logger = EvaluationLogger()
    logger.log_training(
        algorithm="DQN",
        episodes=2000,
        final_avg_reward=145.3,
        max_tile=256,
        final_score=2048,
        training_time=timer.elapsed_str(),
        model_path="models/DQN/dqn_final.pth",
        notes="LR=1e-4, ε_end=0.1"
    )
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Training Timer
# ═══════════════════════════════════════════════════════════════════════════

class TrainingTimer:
    """
    Simple timer for tracking training duration.
    
    Provides human-readable elapsed time strings (e.g., "2:15:30" for 2h 15m 30s)
    
    Example:
        timer = TrainingTimer().start()
        # ... train model ...
        print(f"Took: {timer.elapsed_str()}")  # "0:45:23"
        timer.stop()
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start the timer (call at beginning of training)."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer (call at end of training)."""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        If timer is still running, returns time since start().
        If timer is stopped, returns total duration.
        
        Returns:
            Elapsed seconds as float
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def elapsed_str(self) -> str:
        """
        Get elapsed time as human-readable string.
        
        Format: "H:MM:SS" (e.g., "2:15:30" = 2 hours, 15 minutes, 30 seconds)
        
        Returns:
            Formatted time string
        """
        seconds = int(self.elapsed())
        return str(timedelta(seconds=seconds))


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation Logger
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationLogger:
    """
    Logger for tracking and comparing model training sessions.
    
    Appends training results to 'evaluations/training_log.txt' with:
        - Algorithm name and hyperparameters
        - Training duration and episodes
        - Final performance metrics (score, max tile, reward)
        - Model save location
    
    This helps compare different training runs and hyperparameter settings.
    
    Example:
        logger = EvaluationLogger()
        logger.log_training(
            algorithm="DQN",
            episodes=2000,
            final_avg_reward=145.3,
            max_tile=256,
            final_score=2048,
            training_time="2:15:30",
            model_path="models/DQN/dqn_final.pth",
            notes="LR=1e-4, ε_end=0.1"
        )
        
        # Later, view all training history
        print(logger.get_summary())
    """
    
    def __init__(self, log_dir: Path = Path("evaluations")):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to store log file (default: "evaluations/")
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "training_log.txt"
    
    def log_training(
        self,
        algorithm: str,
        episodes: int,
        final_avg_reward: float,
        max_tile: int,
        final_score: int,
        training_time: str,
        model_path: str,
        notes: str = ""
    ):
        """
        Log training session results to file.
        
        Appends formatted entry to training_log.txt with timestamp and metrics.
        
        Args:
            algorithm: Algorithm name ("DQN", "Double DQN", "MCTS")
            episodes: Number of training episodes completed
            final_avg_reward: Average reward over last 100 episodes
            max_tile: Highest tile achieved during training
            final_score: Highest score achieved during training
            training_time: Formatted duration string (e.g., "2:15:30")
            model_path: Where model was saved
            notes: Additional info (hyperparameters, observations)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
{'='*80}
Training Session: {timestamp}
{'='*80}
Algorithm:          {algorithm}
Episodes:           {episodes}
Training Time:      {training_time}
Final Avg Reward:   {final_avg_reward:.2f}
Best Max Tile:      {max_tile}
Best Score:         {final_score}
Model Saved:        {model_path}
Notes:              {notes}
{'='*80}

"""
        
        # Append to log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        # Also print to console
        print(log_entry)
        print(f"[LOG] Evaluation logged to: {self.log_file}")
    
    def log_evaluation(
        self,
        model_name: str,
        num_games: int,
        avg_score: float,
        avg_max_tile: float,
        best_score: int,
        best_tile: int,
    ):
        """
        Log evaluation of a trained model (playing N games).
        
        Use this to test a trained model's performance.
        
        Args:
            model_name: Model filename or description
            num_games: Number of evaluation games played
            avg_score: Average score across all games
            avg_max_tile: Average highest tile across all games
            best_score: Best score in any single game
            best_tile: Best tile in any single game
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
{'-'*80}
Evaluation: {timestamp}
{'-'*80}
Model:              {model_name}
Games Played:       {num_games}
Average Score:      {avg_score:.2f}
Average Max Tile:   {avg_max_tile:.0f}
Best Score:         {best_score}
Best Tile:          {best_tile}
{'-'*80}

"""
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(log_entry)
    
    def get_summary(self) -> str:
        """
        Get all logged training sessions as a single string.
        
        Returns:
            Full contents of training_log.txt
        """
        if not self.log_file.exists():
            return "No training sessions logged yet."
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            return f.read()


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable duration string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2:15:30" for 2h 15m 30s)
    """
    return str(timedelta(seconds=int(seconds)))


def save_config(config_dict: Dict, save_path: Path):
    """
    Save configuration dictionary to JSON file.
    
    Useful for saving hyperparameters alongside trained models.
    
    Args:
        config_dict: Dictionary of configuration values
        save_path: Where to save JSON file
    
    Example:
        save_config(CONFIG["dqn"], Path("models/DQN/config.json"))
    """
    import json
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)
