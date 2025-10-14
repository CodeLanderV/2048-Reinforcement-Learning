"""Utility functions for training and evaluation."""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


class TrainingTimer:
    """Timer to track training duration."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def elapsed_str(self) -> str:
        """Get elapsed time as formatted string."""
        seconds = int(self.elapsed())
        return str(timedelta(seconds=seconds))


class EvaluationLogger:
    """Logger for tracking and comparing model evaluations."""
    
    def __init__(self, log_dir: Path = Path("evaluations")):
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
        """Log training results."""
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
        print(f"✅ Evaluation logged to: {self.log_file}")
    
    def log_evaluation(
        self,
        model_name: str,
        num_games: int,
        avg_score: float,
        avg_max_tile: float,
        best_score: int,
        best_tile: int,
    ):
        """Log model evaluation results."""
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
        """Get summary of all logged training sessions."""
        if not self.log_file.exists():
            return "No training sessions logged yet."
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            return f.read()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    return str(timedelta(seconds=int(seconds)))


def save_config(config_dict: Dict, save_path: Path):
    """Save configuration to JSON file."""
    import json
    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)
