import logging
import os

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore


def configure_logging():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject_path = os.path.join(base_dir, "pyproject.toml")
    level = "WARNING"
    fmt = "%(levelname)s:%(name)s:%(message)s"
    if os.path.exists(pyproject_path):
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        log_cfg = config.get("tool", {}).get("jax2onnx", {}).get("logging", {})
        level = log_cfg.get("level", level)
        fmt = log_cfg.get("format", fmt)
    logging.basicConfig(level=getattr(logging, level, logging.WARNING), format=fmt)


configure_logging()
