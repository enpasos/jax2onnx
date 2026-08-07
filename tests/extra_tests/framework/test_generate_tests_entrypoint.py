# tests/extra_tests/framework/test_generate_tests_entrypoint.py

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_generate_tests_bootstraps_repository_import_path(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "scripts" / "generate_tests.py"
    generator = repository_root / "tests" / "t_generator.py"
    probe = f"""
import importlib.util
from pathlib import Path
import sys

script = Path({str(script)!r})
repository_root = Path({str(repository_root)!r})
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or '.').resolve() != repository_root
]
spec = importlib.util.spec_from_file_location('generate_tests_probe', script)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._ensure_repository_root_on_import_path()
from tests import t_generator
assert Path(t_generator.__file__).resolve() == Path({str(generator)!r})
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
