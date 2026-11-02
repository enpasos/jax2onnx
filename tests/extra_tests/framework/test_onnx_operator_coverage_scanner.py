# tests/extra_tests/framework/test_onnx_operator_coverage_scanner.py

"""Coverage scanning reads source without importing the converter or JAX."""

from pathlib import Path

import pytest

from scripts import generate_onnx_operator_coverage as coverage


def _scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    monkeypatch.setattr(coverage, "PLUGIN_ROOT", tmp_path)
    (tmp_path / "sample.py").write_text(source, encoding="utf-8")
    metadata, lowering = coverage.collect_plugin_operator_usage(
        plugin_root=tmp_path, include_examples=False
    )
    return dict(metadata), dict(lowering)


def test_scanner_reads_actual_builder_and_shared_lowering_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lowering = _scan(
        tmp_path,
        monkeypatch,
        """
def lower(ctx, eqn):
    ctx.builder.Neg(x)
    ctx.builder.op("MatMulInteger", [lhs, rhs])
    builder.op_multi_out("TopK", [x, k], num_outputs=2)
    lower_unary_elementwise(ctx, eqn, op_name="Elu")
    lower_unary_elementwise_with_optional_cast(ctx, eqn, op_name="Sqrt")
    lower_reduction(ctx, eqn, op_type="ReduceMax")
    builder_reduce_with_axes(ctx, x, op_type="ReduceLogSumExp")
""",
    )

    assert lowering == {
        name: {"sample"}
        for name in (
            "Neg",
            "MatMulInteger",
            "TopK",
            "Elu",
            "Sqrt",
            "ReduceMax",
            "ReduceLogSumExp",
        )
    }


def test_scanner_resolves_locally_selected_reduction_names_at_emitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lowering = _scan(
        tmp_path,
        monkeypatch,
        """
def lower(ctx, eqn):
    op_name = None
    if is_absolute:
        op_name = "ReduceL1"
    elif is_squared:
        op_name = "ReduceSumSquare"
    builder_reduce_with_axes(ctx, x, op_type=op_name)
""",
    )

    assert lowering == {
        "ReduceL1": {"sample"},
        "ReduceSumSquare": {"sample"},
    }


def test_scanner_keeps_metadata_and_lowering_evidence_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, lowering = _scan(
        tmp_path,
        monkeypatch,
        """
metadata = [{"doc": "https://onnx.ai/onnx/operators/onnx__Relu.html"}]
op_name = "Gelu"
example = 'ctx.builder.Softplus(x)'
# ctx.builder.LeakyRelu(x)
inspect_expected_graph(op_type="LogSoftmax")

def unrelated():
    op_type = "ReduceMin"

def lower(ctx, eqn, op_type):
    lower_reduction(ctx, eqn, op_type=op_type)
""",
    )

    assert metadata == {"Relu": {"sample"}}
    assert lowering == {}


def test_scanner_does_not_borrow_operator_names_from_nested_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, lowering = _scan(
        tmp_path,
        monkeypatch,
        """
def lower(ctx, eqn, op_type):
    def unused():
        op_type = "ReduceMin"
    lower_reduction(ctx, eqn, op_type=op_type)
""",
    )

    assert lowering == {}


def test_scanner_reads_literal_generated_metadata_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, lowering = _scan(
        tmp_path,
        monkeypatch,
        """
COMPONENTS = [
    {"component": op, "doc": f"https://onnx.ai/onnx/operators/onnx__{op}.html"}
    for op in ("LayerNormalization", "ReduceMean")
]
""",
    )

    assert metadata == {
        "LayerNormalization": {"sample"},
        "ReduceMean": {"sample"},
    }
    assert lowering == {}


@pytest.mark.parametrize(
    "iteration",
    [
        "for op in discover_ops()",
        'for op in ("Unused",) if enabled',
    ],
)
def test_scanner_does_not_guess_dynamic_or_filtered_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, iteration: str
) -> None:
    metadata, lowering = _scan(
        tmp_path,
        monkeypatch,
        '[{"doc": f"https://onnx.ai/onnx/operators/onnx__{op}.html"} '
        + iteration
        + "]\n",
    )

    assert metadata == {}
    assert lowering == {}
