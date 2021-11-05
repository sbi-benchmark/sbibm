import logging
from typing import Optional


def get_logger(
    name: str, level: Optional[int] = logging.INFO, console_logging: bool = True
) -> logging.Logger:
    """Gets logger with given name, while setting level and optionally adding handler

    Note: Logging to `sys.stdout` for Jupyter as done in this Gist
    https://gist.github.com/joshbode/58fac7ababc700f51e2a9ecdebe563ad

    Args:
        name: Name of logger
        level: Log level
        console_logging: Whether or not to log to console

    Returns:
        Logger
    """
    log = logging.getLogger(name)

    if level is not None:
        log.setLevel(level)

    has_stream_handler = False
    for h in log.handlers:
        if type(h) == logging.StreamHandler:
            has_stream_handler = True
    if console_logging and not has_stream_handler:
        console_handler = logging.StreamHandler()
        log.addHandler(console_handler)

    return log
