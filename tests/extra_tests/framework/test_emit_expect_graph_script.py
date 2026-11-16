# tests/extra_tests/framework/test_emit_expect_graph_script.py

"""scripts/emit_expect_graph.py prints the structure that a generated test checks."""

from __future__ import annotations

from typing import Any

import pytest

from scripts import emit_expect_graph
from tests import t_generator as tgen


def test_snippet_for_an_authored_testcase(capsys: pytest.CaptureFixture[str]) -> None:
    assert emit_expect_graph.main(["neg"]) == 0

    assert capsys.readouterr().out == (
        "# neg\nexpect_graph(\n    ['Neg:3'],\n    no_unused_inputs=True,\n)\n"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # A module built by construct_and_call, exported with x64 enabled.
        ("linear_no_bias_f64", "['Gemm:3x64']"),
        # A symbolic batch dimension.
        ("linear_symbolic_batch_dynamic", "symbols={'B': None}"),
    ],
)
def test_generated_variants_are_selectable(
    name: str, expected: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert emit_expect_graph.main([name]) == 0

    snippet = capsys.readouterr().out
    assert snippet.startswith(f"# {name}\nexpect_graph(\n")
    assert expected in snippet


def test_unknown_testcase_is_reported() -> None:
    with pytest.raises(SystemExit, match="Unknown testcase.*no_such_testcase"):
        emit_expect_graph.main(["no_such_testcase"])


def test_export_uses_the_options_of_the_generated_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The snippet is only useful if it describes the graph that the generated
    # test checks, so the script must not keep export options of its own.
    class _Exported(Exception):
        pass

    captured: dict[str, Any] = {}

    def record(**kwargs: Any) -> str:
        captured.update(kwargs)
        raise _Exported

    monkeypatch.setattr(emit_expect_graph, "to_onnx", record)
    tp = emit_expect_graph._select_test_params(["neg"])["neg"]

    with pytest.raises(_Exported):
        emit_expect_graph._regenerate_spec(tp, None)

    options = tgen.to_onnx_options_for_test_param(tp)
    assert {name: captured[name] for name in options} == options
    assert captured["inputs"] == tgen.input_specs_for_test_param(tp, tp["callable"])
    assert set(captured) == {*options, "fn", "inputs", "return_mode", "output_path"}
