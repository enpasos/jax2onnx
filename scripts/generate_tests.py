# scripts/generate_tests.py

from __future__ import annotations

import os
from pathlib import Path
import sys


def _ensure_repository_root_on_import_path() -> None:
    """Make the repository-local ``tests`` package importable when run directly."""

    repository_root = str(Path(__file__).resolve().parent.parent)
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)


def _configure_jax_environment() -> None:
    """Force CPU execution with x64 enabled to avoid JAX runtime warnings."""

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> None:
    _configure_jax_environment()
    _ensure_repository_root_on_import_path()

    from tests.t_generator import generate_all_tests  # delayed import; sets env first

    print("Generating tests for all plugins...")
    generate_all_tests()


if __name__ == "__main__":
    main()
