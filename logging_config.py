import logging
import os
import atexit

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore


def configure_logging():
    """Configures logging based on pyproject.toml [tool.jax2onnx.logging]."""
    # Load config from pyproject.toml
    config = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(base_dir, "pyproject.toml")
    if os.path.exists(pyproject_path):
        read_mode = "rb" if getattr(tomllib, "__name__", "") == "tomllib" else "r"
        with open(pyproject_path, read_mode) as f:
            config = (
                tomllib.load(f).get("tool", {}).get("jax2onnx", {}).get("logging", {})
            )

    default_level = config.get("default_level", "INFO").upper()
    log_format = config.get("format", "%(levelname)s:%(name)s:%(message)s")
    specific_levels = config.get("levels", {})

    # Reset root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, default_level, logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # Apply per-logger levels
    for logger_name, lvl in specific_levels.items():
        log_obj = logging.getLogger(logger_name)
        log_obj.setLevel(getattr(logging, lvl.upper(), logging.INFO))
        print(f"[CONFIGURE] Set level {lvl.upper()} on logger: {logger_name}")

    # Ensure clean shutdown
    atexit.register(lambda: setattr(logging, "raiseExceptions", False))
