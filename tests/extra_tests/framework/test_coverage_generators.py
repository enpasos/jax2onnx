# tests/extra_tests/framework/test_coverage_generators.py

from __future__ import annotations

from pathlib import Path

import pytest

from scripts._coverage_generation import write_or_check_generated
from scripts import generate_jnp_operator_coverage as jnp_coverage
from scripts import generate_lax_operator_coverage as lax_coverage
from scripts import generate_onnx_operator_coverage as onnx_coverage


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


def _lax_status(
    op: str,
    *,
    doc_usage: dict[str, set[str]] | None = None,
    prim_usage: dict[str, set[str]] | None = None,
) -> str:
    status, _, _ = lax_coverage._status_for_op(
        op,
        doc_usage=doc_usage or {},
        prim_usage=prim_usage or {},
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


@pytest.mark.parametrize(
    ("op", "primitive"),
    [
        ("angle", "atan2"),
        ("argsort", "sort"),
        ("around", "round"),
        ("bitwise_count", "population_count"),
        ("cbrt", "cbrt"),
        ("deg2rad", "mul"),
        ("degrees", "mul"),
        ("empty", "broadcast_in_dim"),
        ("fft2", "fft"),
        ("fftn", "fft"),
        ("float_power", "pow"),
        ("hfft", "irfft"),
        ("hypot", "sqrt"),
        ("ifft2", "ifft"),
        ("ifftn", "ifft"),
        ("ihfft", "rfft"),
        ("inner", "dot_general"),
        ("irfft2", "irfft"),
        ("irfftn", "irfft"),
        ("iscomplex", "imag"),
        ("isreal", "imag"),
        ("kron", "mul"),
        ("log10", "log"),
        ("log2", "log"),
        ("log1p", "log1p"),
        ("nextafter", "nextafter"),
        ("ptp", "reduce_max"),
        ("rad2deg", "mul"),
        ("radians", "mul"),
        ("rfft2", "rfft"),
        ("rfftn", "rfft"),
        ("signbit", "shift_right_arithmetic"),
        ("sort_complex", "sort"),
    ],
)
def test_jnp_coverage_marks_verified_lax_reuse_as_indirect(
    op: str, primitive: str
) -> None:
    status = _jnp_status(
        op,
        prim_usage={primitive: {f"jax/lax/{primitive}"}},
    )

    assert status == "covered_indirect"


def test_jnp_coverage_marks_identity_as_iota_reuse() -> None:
    assert (
        _jnp_status("identity", prim_usage={"iota": {"jax/lax/iota"}})
        == "covered_indirect"
    )


def test_jnp_coverage_marks_static_numpy_entries_as_non_functional() -> None:
    assert _jnp_status("float32") == "non_functional"
    assert _jnp_status("einsum_path") == "non_functional"


def test_jnp_coverage_marks_helper_apis_as_composite() -> None:
    assert _jnp_status("allclose") == "composite"
    assert _jnp_status("positive") == "composite"


@pytest.mark.parametrize(
    "op",
    [
        "argwhere",
        "argpartition",
        "average",
        "bincount",
        "convolve",
        "correlate",
        "diff",
        "divmod",
        "extract",
        "fmax",
        "fmin",
        "heaviside",
        "intersect1d",
        "isin",
        "lexsort",
        "linalg.slogdet",
        "logaddexp",
        "logaddexp2",
        "mask_indices",
        "modf",
        "nan_to_num",
        "nonzero",
        "packbits",
        "place",
        "poly",
        "polymul",
        "polyval",
        "tril_indices",
        "tril_indices_from",
        "triu_indices",
        "triu_indices_from",
        "unique_all",
        "unique_counts",
        "unique_inverse",
        "unique_values",
        "unpackbits",
    ],
)
def test_jnp_coverage_marks_verified_composite_math_apis_as_composite(
    op: str,
) -> None:
    assert _jnp_status(op) == "composite"


def test_lax_coverage_marks_broadcast_like_as_composite() -> None:
    assert _lax_status("broadcast_like") == "composite"


def test_lax_coverage_marks_trace_helpers_as_composite() -> None:
    assert _lax_status("stage") == "composite"


def test_lax_coverage_uses_ormqr_plugin_signal() -> None:
    assert (
        _lax_status("linalg.ormqr", prim_usage={"ormqr": {"jax/lax/ormqr"}})
        == "covered"
    )


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("QuantizeLinear", "Decide quantization scope"),
        ("SequenceConstruct", "add container plugins"),
        ("GridSample", "Vision-specific op"),
        ("SoftmaxCrossEntropyLoss", "demanded by target models"),
    ],
)
def test_onnx_coverage_recommends_next_action_for_uncovered_categories(
    op: str, expected: str
) -> None:
    assert expected in onnx_coverage._recommend_for_uncovered(op)


def test_onnx_coverage_open_action_bucket_summary_counts_only_uncovered_ops() -> None:
    summary = onnx_coverage.build_open_action_bucket_summary(
        official_ops=["Add", "GridSample", "QuantizeLinear"],
        metadata_usage={"Add": {"jax/lax/add"}},
        lowering_usage={},
    )

    assert "Vision-specific native ops: `1`" in summary
    assert "Quantization scope: `1`" in summary
    assert "General triage: `0`" in summary
    assert "Add" not in summary


def test_write_or_check_generated_accepts_current_file(tmp_path: Path) -> None:
    target = tmp_path / "coverage.md"
    target.write_text("# Coverage\n", encoding="utf-8")

    write_or_check_generated(
        target,
        "# Coverage",
        check=True,
        label="test coverage page",
    )


def test_write_or_check_generated_reports_stale_file(tmp_path: Path) -> None:
    target = tmp_path / "coverage.md"
    target.write_text("# Old\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        write_or_check_generated(
            target,
            "# New",
            check=True,
            label="test coverage page",
        )

    message = str(exc_info.value)
    assert "test coverage page is stale" in message
    assert "--- current" in message
    assert "+++ generated" in message


def test_write_or_check_generated_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "coverage.md"

    write_or_check_generated(
        target,
        "# Coverage",
        check=False,
        label="test coverage page",
    )

    assert target.read_text(encoding="utf-8") == "# Coverage\n"
