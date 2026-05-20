import logging
import sys
from importlib.util import find_spec
from typing import Optional

from configs.settings import settings

LOGURU_AVAILABLE = find_spec("loguru") is not None


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to a log file.
        json_format: Whether to use JSON-structured log format.

    Returns:
        Configured standard library Logger.
    """
    log_dir = settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt_text = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d — %(message)s"
    )
    fmt_json = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"module":"%(name)s","func":"%(funcName)s","line":%(lineno)d,"msg":"%(message)s"}'
    )

    formatter = logging.Formatter(fmt_json if json_format else fmt_text)

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    if log_file:
        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()), handlers=handlers, force=True
    )

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("finsight")


def get_logger(name: str) -> logging.Logger:
    """Return a named child logger under the 'finsight' namespace."""
    return logging.getLogger(f"finsight.{name}")


# Module-level default logger
logger = setup_logging(
    level="DEBUG" if settings.DEBUG else "INFO",
    log_file="finsight.log",
    json_format=settings.ENVIRONMENT == "production",
)
