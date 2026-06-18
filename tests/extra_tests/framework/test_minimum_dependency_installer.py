# tests/extra_tests/framework/test_minimum_dependency_installer.py

from __future__ import annotations

from pathlib import Path

from scripts.install_minimum_dependencies import (
    minimum_requirement,
    minimum_requirements,
)


def test_minimum_requirement_extracts_lower_bound_with_upper_bound() -> None:
    assert (
        minimum_requirement("orbax-checkpoint>=0.11.6,<0.11.37")
        == "orbax-checkpoint==0.11.6"
    )


def test_minimum_requirement_preserves_extras() -> None:
    assert minimum_requirement("jax[cuda12_pip]>=0.8.1") == "jax[cuda12_pip]==0.8.1"


def test_project_runtime_dependencies_all_have_minimum_versions() -> None:
    requirements = minimum_requirements(Path("pyproject.toml"))

    assert "jax==0.8.1" in requirements
    assert "flax==0.12.1" in requirements
    assert all("==" in requirement for requirement in requirements)
