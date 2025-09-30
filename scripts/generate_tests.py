# file: scripts/generate_tests.py

from __future__ import annotations

import os


def _configure_jax_environment() -> None:
    """Force CPU execution with x64 enabled to avoid JAX runtime warnings."""

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main() -> None:
    _configure_jax_environment()

    from tests.t_generator import generate_all_tests  # delayed import; sets env first

    print("Generating tests for all plugins...")
    generate_all_tests()


if __name__ == "__main__":
    main()
