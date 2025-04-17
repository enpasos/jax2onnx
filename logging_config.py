import logging
import os

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore


def configure_logging():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(base_dir, "pyproject.toml")

    level = "DEBUG"
    fmt = "%(levelname)s:%(name)s:%(message)s"
    log_file = None

    if os.path.exists(pyproject_path):
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        log_cfg = config.get("tool", {}).get("jax2onnx", {}).get("logging", {})
        level = log_cfg.get("level", level)
        fmt = log_cfg.get("format", fmt)
        log_file = log_cfg.get("file", log_file)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(os.path.join(base_dir, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
