# tests/extra_tests/framework/test_coverage_generators.py

from __future__ import annotations

from scripts import generate_jnp_operator_coverage as jnp_coverage


def _jnp_status(
    op: str,
    *,
    doc_usage: dict[str, set[str]] | None = None,
    prim_usage: dict[str, set[str]] | None = None,
    component_usage: dict[str, set[str]] | None = None,
) -> str:
    status, _, _ = jnp_coverage._status_for_op(
        op,
        doc_usage=doc_usage or {},
        prim_usage=prim_usage or {},
        component_usage=component_usage or {},
    )
    return status


def test_jnp_coverage_uses_numpy_component_metadata_as_direct_signal() -> None:
    status = _jnp_status(
        "pow",
        component_usage={"pow": {"jax/numpy/pow"}},
    )

    assert status == "covered"


def test_jnp_coverage_marks_docs_aliases_as_indirect_coverage() -> None:
    status = _jnp_status(
        "arccos",
        doc_usage={"acos": {"jax/numpy/acos"}},
        prim_usage={"acos": {"jax/numpy/acos"}},
    )

    assert status == "covered_indirect"


def test_jnp_coverage_marks_lower_level_primitive_reuse_as_indirect() -> None:
    status = _jnp_status(
        "multiply",
        prim_usage={"mul": {"jax/lax/mul"}},
    )

    assert status == "covered_indirect"


def test_jnp_coverage_marks_static_numpy_entries_as_non_functional() -> None:
    assert _jnp_status("float32") == "non_functional"
    assert _jnp_status("einsum_path") == "non_functional"


def test_jnp_coverage_marks_helper_apis_as_composite() -> None:
    assert _jnp_status("allclose") == "composite"
    assert _jnp_status("positive") == "composite"
