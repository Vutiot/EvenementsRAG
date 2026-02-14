"""
Logging configuration for EvenementsRAG.

Provides centralized logging setup using loguru with:
- Console output with colors
- File output with rotation
- Different log levels for different components
- Structured logging support

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Starting process")
    logger.debug("Debug information", extra={"key": "value"})
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from config.settings import settings


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure the global logger.

    Args:
        log_file: Path to log file (defaults to settings.LOG_FILE)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log files (e.g., "10 MB", "1 day")
        retention: How long to keep old logs
    """
    # Remove default logger
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file or settings.LOG_FILE:
        log_path = Path(log_file or settings.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            ),
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logger initialized at level {level}")


def get_logger(name: str) -> logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    # Ensure logger is set up
    if not logger._core.handlers:
        setup_logger(level=settings.LOG_LEVEL)

    # Return logger with context
    return logger.bind(name=name)


# Initialize logger on import
setup_logger(level=settings.LOG_LEVEL)


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger(__name__)

    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")

    # Test structured logging
    test_logger.info(
        "Structured log example",
        extra={
            "user": "test_user",
            "action": "fetch_data",
            "count": 42,
        }
    )
