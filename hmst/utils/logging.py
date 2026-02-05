"""
Logging utilities for HMST training and inference.
"""

import logging
import sys
from typing import Optional
import wandb


def setup_logger(
    name: str = 'hmst',
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    Logger for training and inference metrics.

    Supports both local logging and W&B integration.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        project_name: str = 'hmst',
        run_name: Optional[str] = None
    ):
        self.use_wandb = use_wandb

        if use_wandb:
            wandb.init(project=project_name, name=run_name)

        self.metrics = {}

    def log(self, metrics: dict, step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number
        """
        self.metrics.update(metrics)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """Log histogram of values."""
        if self.use_wandb:
            wandb.log({name: wandb.Histogram(values)}, step=step)

    def finish(self):
        """Finish logging session."""
        if self.use_wandb:
            wandb.finish()

    def get_metrics(self) -> dict:
        """Get current metrics."""
        return self.metrics.copy()
