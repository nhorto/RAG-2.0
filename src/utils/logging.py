"""Logging configuration for RAG system."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_file: str = None,
    level: str = "INFO",
    format_string: str = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_file: Path to log file (if None, uses default)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string

    Returns:
        Logger instance
    """
    # Create logs directory
    if log_file is None:
        project_root = Path(__file__).parent.parent.parent
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / f"rag_system_{datetime.now().strftime('%Y%m%d')}.log"

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger("rag_system")
    logger.setLevel(getattr(logging, level.upper()))

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance.

    Args:
        name: Logger name (if None, uses 'rag_system')

    Returns:
        Logger instance
    """
    if name is None:
        name = "rag_system"

    logger = logging.getLogger(name)

    # Set up logging if not already configured
    if not logger.handlers:
        setup_logging()

    return logger
