"""
═══════════════════════════════════════════════════════════════════════════════
Training Metrics Logger
═══════════════════════════════════════════════════════════════════════════════

Saves training metrics to JSON for post-training analysis and plotting.

USAGE:
    # During training
    metrics = MetricsLogger(algorithm="DQN", save_dir="evaluations")
    
    for episode in range(episodes):
        # ... train ...
        metrics.log_episode(
            episode=episode,
            score=score,
            max_tile=max_tile,
            reward=reward,
            steps=steps,
            epsilon=epsilon,
            loss=loss
        )
    
    metrics.save()
    
    # After training - generate plots
    from src.plot_from_logs import plot_training_metrics
    plot_training_metrics("evaluations/DQN_metrics.json")
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any


class MetricsLogger:
    """
    Logs training metrics to JSON file for post-training visualization.
    """
    
    def __init__(
        self,
        algorithm: str,
        save_dir: str = "evaluations",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metrics logger.
        
        Args:
            algorithm: Algorithm name (e.g., "DQN", "Double-DQN")
            save_dir: Directory to save metrics file
            config: Training configuration dictionary
        """
        self.algorithm = algorithm
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training session metadata
        self.metadata = {
            "algorithm": algorithm,
            "start_time": datetime.now().isoformat(),
            "config": config or {}
        }
        
        # Episode-by-episode metrics
        self.episodes: List[int] = []
        self.scores: List[int] = []
        self.max_tiles: List[int] = []
        self.rewards: List[float] = []
        self.steps: List[int] = []
        self.epsilons: List[float] = []
        self.losses: List[Optional[float]] = []
        
        # Session tracking
        self.best_score = 0
        self.best_tile = 0
    
    def log_episode(
        self,
        episode: int,
        score: int,
        max_tile: int,
        reward: float,
        steps: int,
        epsilon: float,
        loss: Optional[float] = None
    ) -> None:
        """
        Log metrics for a single episode.
        
        Args:
            episode: Episode number
            score: Game score
            max_tile: Highest tile achieved
            reward: Total episode reward
            steps: Number of steps/moves
            epsilon: Current exploration rate
            loss: Average training loss (if available)
        """
        self.episodes.append(int(episode))
        self.scores.append(int(score))
        self.max_tiles.append(int(max_tile))
        self.rewards.append(float(reward))
        self.steps.append(int(steps))
        self.epsilons.append(float(epsilon))
        self.losses.append(float(loss) if loss is not None else None)
        
        # Track bests
        if score > self.best_score:
            self.best_score = int(score)
        if max_tile > self.best_tile:
            self.best_tile = int(max_tile)
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save metrics to JSON file.
        
        Args:
            filename: Output filename (default: auto-generated)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.algorithm.replace(' ', '_')}_metrics_{timestamp}.json"
        
        # Finalize metadata
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["total_episodes"] = len(self.episodes)
        self.metadata["best_score"] = self.best_score
        self.metadata["best_tile"] = self.best_tile
        
        # Compile full data
        data = {
            "metadata": self.metadata,
            "metrics": {
                "episodes": self.episodes,
                "scores": self.scores,
                "max_tiles": self.max_tiles,
                "rewards": self.rewards,
                "steps": self.steps,
                "epsilons": self.epsilons,
                "losses": self.losses
            }
        }
        
        # Save to file
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(save_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.episodes:
            return {}
        
        import numpy as np
        
        return {
            "total_episodes": len(self.episodes),
            "best_score": self.best_score,
            "best_tile": self.best_tile,
            "avg_score": float(np.mean(self.scores)),
            "avg_max_tile": float(np.mean(self.max_tiles)),
            "avg_reward": float(np.mean(self.rewards)),
            "avg_steps": float(np.mean(self.steps)),
            "final_epsilon": self.epsilons[-1] if self.epsilons else None
        }
