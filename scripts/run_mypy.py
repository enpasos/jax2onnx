#!/usr/bin/env python3
# scripts/run_mypy.py

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
import os


def main(argv: list[str]) -> int:
    poetry = shutil.which("poetry")
    if poetry is not None:
        return subprocess.call([poetry, "run", "mypy", *argv])

    repo_root = Path(__file__).resolve().parent.parent
    local_runner = repo_root / "python-venv"
    if (
        local_runner.exists()
        and local_runner.is_file()
        and os.access(local_runner, os.X_OK)
    ):
        return subprocess.call([str(local_runner), "-m", "mypy", *argv])

    mypy_exe = shutil.which("mypy")
    if mypy_exe is None:
        sys.stderr.write(
            "mypy hook: neither 'poetry' nor 'mypy' is available on PATH\n"
        )
        return 1
    return subprocess.call([mypy_exe, *argv])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
