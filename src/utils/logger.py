import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from src.core.config import settings

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "mediflow.log"


def setup_logging() -> None:
    """
    Configures structured logging for the application using structlog.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        handlers=[stdout_handler, file_handler],
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True) if settings.environment == "dev" else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Retrieves a bound logger with the specified name.
    """
    return structlog.get_logger(name)
