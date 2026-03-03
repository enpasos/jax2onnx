# tests/extra_tests/framework/test_ad_registration_helpers_only.py

from __future__ import annotations

from pathlib import Path
import re


def test_ad_jvp_registry_writes_are_centralized():
    repo_root = Path(__file__).resolve().parents[3]
    plugins_root = repo_root / "jax2onnx" / "plugins"
    helper_file = plugins_root / "jax" / "_autodiff_utils.py"

    pattern = re.compile(r"\bad\.primitive_jvps\s*\[")
    offenders: list[str] = []

    for py_file in sorted(plugins_root.rglob("*.py")):
        if py_file == helper_file:
            continue
        if pattern.search(py_file.read_text(encoding="utf-8")):
            offenders.append(str(py_file.relative_to(repo_root)))

    assert not offenders, (
        "Direct writes to ad.primitive_jvps are only allowed in "
        "jax2onnx/plugins/jax/_autodiff_utils.py.\n"
        "Migrate these files to helper registration:\n"
        + "\n".join(f"- {entry}" for entry in offenders)
    )
