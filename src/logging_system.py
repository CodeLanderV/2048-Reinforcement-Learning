"""
═══════════════════════════════════════════════════════════════════════════════
Unified Logging System - Main, Training, and Testing Logs
═══════════════════════════════════════════════════════════════════════════════

This module provides a comprehensive logging system that writes to three files:
    - mainlog.txt:     Everything (training + testing + general info)
    - training_log.txt: Training episodes, metrics, checkpoints
    - testing_log.txt:  Evaluation games, performance metrics

USAGE:
    from src.logging_system import setup_logging, log_training, log_testing, log_main
    
    # Setup at start of program
    setup_logging()
    
    # Log training events
    log_training(f"Episode {ep} | Score: {score} | Reward: {reward}")
    
    # Log testing events
    log_testing(f"Game {game} | Score: {score} | Max Tile: {tile}")
    
    # Log general info (goes to all logs)
    log_main("Starting new training session...")
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global loggers
_main_logger: Optional[logging.Logger] = None
_training_logger: Optional[logging.Logger] = None
_testing_logger: Optional[logging.Logger] = None


def setup_logging(log_dir: Path = None) -> None:
    """
    Initialize the three-tier logging system.
    
    Creates:
        - evaluations/mainlog.txt: All events
        - evaluations/training_log.txt: Training only
        - evaluations/testing_log.txt: Testing only
    
    Args:
        log_dir: Directory for log files (default: evaluations/)
    """
    global _main_logger, _training_logger, _testing_logger
    
    if log_dir is None:
        log_dir = Path("evaluations")
    log_dir.mkdir(exist_ok=True)
    
    # Clear any existing handlers
    for logger_name in ['main', 'training', 'testing']:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
    
    # ─────────────────────────────────────────────────────────────────────
    # Main Logger: Everything goes here
    # ─────────────────────────────────────────────────────────────────────
    _main_logger = logging.getLogger('main')
    main_file = log_dir / "mainlog.txt"
    
    main_handler = logging.FileHandler(main_file, mode='a', encoding='utf-8')
    main_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    _main_logger.addHandler(main_handler)
    _main_logger.addHandler(console_handler)
    
    # ─────────────────────────────────────────────────────────────────────
    # Training Logger: Training episodes only
    # ─────────────────────────────────────────────────────────────────────
    _training_logger = logging.getLogger('training')
    training_file = log_dir / "training_log.txt"
    
    training_handler = logging.FileHandler(training_file, mode='a', encoding='utf-8')
    training_handler.setLevel(logging.INFO)
    training_handler.setFormatter(formatter)
    
    _training_logger.addHandler(training_handler)
    
    # ─────────────────────────────────────────────────────────────────────
    # Testing Logger: Evaluation games only
    # ─────────────────────────────────────────────────────────────────────
    _testing_logger = logging.getLogger('testing')
    testing_file = log_dir / "testing_log.txt"
    
    testing_handler = logging.FileHandler(testing_file, mode='a', encoding='utf-8')
    testing_handler.setLevel(logging.INFO)
    testing_handler.setFormatter(formatter)
    
    _testing_logger.addHandler(testing_handler)
    
    # Log initialization
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _main_logger.info("="*80)
    _main_logger.info(f"LOGGING SYSTEM INITIALIZED - {timestamp}")
    _main_logger.info("="*80)


def log_main(message: str) -> None:
    """
    Log to main log (and console).
    Use for general program events, initialization, errors.
    
    Args:
        message: Message to log
    """
    if _main_logger is None:
        setup_logging()
    _main_logger.info(message)


def log_training(message: str) -> None:
    """
    Log training event to training_log.txt and mainlog.txt.
    Use for episode updates, loss, convergence info.
    
    Args:
        message: Training message to log
    """
    if _training_logger is None:
        setup_logging()
    _training_logger.info(message)
    _main_logger.info(f"[TRAIN] {message}")


def log_testing(message: str) -> None:
    """
    Log testing event to testing_log.txt and mainlog.txt.
    Use for evaluation games, performance metrics.
    
    Args:
        message: Testing message to log
    """
    if _testing_logger is None:
        setup_logging()
    _testing_logger.info(message)
    _main_logger.info(f"[TEST] {message}")


def log_training_session_start(algorithm: str, episodes: int, config: dict) -> None:
    """
    Log the start of a training session with configuration.
    
    Args:
        algorithm: Algorithm name (DQN, Double DQN)
        episodes: Total episodes to train
        config: Hyperparameter configuration dict
    """
    log_main("="*80)
    log_main(f"TRAINING SESSION STARTED: {algorithm}")
    log_main("="*80)
    log_training(f"Algorithm: {algorithm}")
    log_training(f"Target Episodes: {episodes}")
    log_training("Configuration:")
    for key, value in config.items():
        log_training(f"  {key}: {value}")
    log_training("="*80)


def log_training_session_end(
    algorithm: str,
    episodes_completed: int,
    best_score: int,
    best_tile: int,
    training_time: str,
    converged: bool = False
) -> None:
    """
    Log the end of a training session with final metrics.
    
    Args:
        algorithm: Algorithm name
        episodes_completed: Number of episodes trained
        best_score: Best score achieved
        best_tile: Highest tile achieved
        training_time: Formatted training duration
        converged: Whether training converged early
    """
    log_training("="*80)
    log_training("TRAINING SESSION COMPLETED")
    log_training("="*80)
    log_training(f"Algorithm: {algorithm}")
    log_training(f"Episodes Completed: {episodes_completed}")
    log_training(f"Best Score: {best_score}")
    log_training(f"Best Tile: {best_tile}")
    log_training(f"Training Time: {training_time}")
    if converged:
        log_training("Status: CONVERGED (early stopping)")
    log_training("="*80)
    log_main(f"Training completed: {episodes_completed} episodes in {training_time}")


def log_checkpoint(episode: int, path: str) -> None:
    """
    Log model checkpoint save.
    
    Args:
        episode: Episode number
        path: Path where model was saved
    """
    log_training(f"[CHECKPOINT] Episode {episode} saved to: {path}")


def log_evaluation_start(model_path: str, num_games: int) -> None:
    """
    Log start of model evaluation.
    
    Args:
        model_path: Path to model being evaluated
        num_games: Number of games to play
    """
    log_main("="*80)
    log_main("EVALUATION SESSION STARTED")
    log_main("="*80)
    log_testing(f"Model: {model_path}")
    log_testing(f"Games: {num_games}")
    log_testing("-"*80)


def log_evaluation_game(
    game_num: int,
    score: int,
    max_tile: int,
    steps: int,
    reached_2048: bool = False
) -> None:
    """
    Log results of a single evaluation game.
    
    Args:
        game_num: Game number
        score: Final score
        max_tile: Highest tile reached
        steps: Number of moves
        reached_2048: Whether 2048 tile was reached
    """
    win_marker = "🏆 WIN" if reached_2048 else ""
    log_testing(
        f"Game {game_num:4d} | Score: {score:6d} | "
        f"Max Tile: {max_tile:5d} | Steps: {steps:4d} {win_marker}"
    )


def log_evaluation_summary(
    num_games: int,
    avg_score: float,
    max_score: int,
    avg_tile: float,
    max_tile: int,
    win_rate: float,
    tile_distribution: dict
) -> None:
    """
    Log evaluation session summary statistics.
    
    Args:
        num_games: Total games played
        avg_score: Average score
        max_score: Best score
        avg_tile: Average max tile
        max_tile: Highest tile achieved
        win_rate: Percentage of games reaching 2048
        tile_distribution: Dict of {tile_value: count}
    """
    log_testing("="*80)
    log_testing("EVALUATION SUMMARY")
    log_testing("="*80)
    log_testing(f"Total Games:     {num_games}")
    log_testing(f"Average Score:   {avg_score:.2f}")
    log_testing(f"Max Score:       {max_score}")
    log_testing(f"Average Tile:    {avg_tile:.0f}")
    log_testing(f"Max Tile:        {max_tile}")
    log_testing(f"Win Rate:        {win_rate:.1f}% (2048 reached)")
    log_testing("")
    log_testing("Tile Distribution:")
    for tile in sorted(tile_distribution.keys(), reverse=True):
        count = tile_distribution[tile]
        percentage = (count / num_games) * 100
        log_testing(f"  {tile:5d}: {count:3d} games ({percentage:5.1f}%)")
    log_testing("="*80)


def cleanup_logging() -> None:
    """Close all log handlers and cleanup."""
    global _main_logger, _training_logger, _testing_logger
    
    for logger in [_main_logger, _training_logger, _testing_logger]:
        if logger:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    _main_logger = None
    _training_logger = None
    _testing_logger = None
