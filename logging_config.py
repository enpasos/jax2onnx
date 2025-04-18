import logging
import os
import atexit
import threading

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore


# Create a special null handler that can safely absorb log messages during shutdown
class NullHandler(logging.Handler):
    def emit(self, record):
        pass


file_handlers = []
_lock = threading.Lock()


def shutdown_logging():
    """No-op shutdown: no file handlers to flush or close."""
    logging.raiseExceptions = False


def configure_logging():

    # Remove all FileHandlers from all loggers (root and children)
    for logger_name in list(logging.root.manager.loggerDict):
        logger = logging.getLogger(logger_name)
        logger.handlers = [
            h for h in logger.handlers if not isinstance(h, logging.FileHandler)
        ]
    logging.getLogger().handlers = [
        h
        for h in logging.getLogger().handlers
        if not isinstance(h, logging.FileHandler)
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(base_dir, "pyproject.toml")
    level = "DEBUG"
    fmt = "%(levelname)s:%(name)s:%(message)s"
    if os.path.exists(pyproject_path):
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        log_cfg = config.get("tool", {}).get("jax2onnx", {}).get("logging", {})
        level = log_cfg.get("level", level)
        fmt = log_cfg.get("format", fmt)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    atexit.register(shutdown_logging)
