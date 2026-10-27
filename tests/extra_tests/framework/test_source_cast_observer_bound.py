# tests/extra_tests/framework/test_source_cast_observer_bound.py

"""The passive artifact return accepts bounded math and rejects excess storage."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import onnx
import onnx_ir as ir
import pytest

from tests.extra_tests.converter.test_source_cast_selected_emission import (
    assert_selected_model,
    export_selected,
)


def _capture(tmp_path: Path, proto: onnx.ModelProto) -> tuple[Any, Any]:
    path = Path(__file__).resolve().parents[3] / "scripts/_source_cast_observer.py"
    spec = spec_from_file_location("_cast_observer_bound_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    observer = module._Observation(tmp_path, module.NODE_ID, ModuleType("owner"))
    observer.model_type = onnx.ModelProto
    code = object()
    observer.codes = {"compare": object(), "runtime": object(), "artifact": code}
    frame = SimpleNamespace(f_code=code, f_locals={}, f_back=None)
    observer._profile(frame, "return", (None, proto, None, [([], [])]))
    return module, observer.report


@pytest.mark.parametrize("nodes", [128, 129])
def test_actual_artifact_return_has_a_finite_128_node_limit(
    nodes: int,
    tmp_path: Path,
) -> None:
    proto = onnx.ModelProto()
    for index in range(nodes):
        proto.graph.node.add(
            op_type="Identity", input=[f"v{index}"], output=[f"v{index + 1}"]
        )
    module, report = _capture(tmp_path, proto)
    if nodes == 128:
        assert report["observer_errors"] == []
        assert len(report["artifact_events"]) == 1
        assert len(report["artifact_events"][0]["graph_ops"]) == nodes
        assert (tmp_path / module.MODEL_NAME).read_bytes() == proto.SerializeToString()
    else:
        assert report["artifact_events"] == []
        assert "artifact node count exceeds bound" in str(report["observer_errors"])
        assert not (tmp_path / module.MODEL_NAME).exists()


@pytest.mark.parametrize("fault", ["bytes", "operator", "symbol"])
def test_larger_node_allowance_preserves_byte_and_per_node_bounds(
    fault: str,
    tmp_path: Path,
) -> None:
    proto = onnx.ModelProto()
    proto.graph.node.add(op_type="Identity", input=["x"], output=["y"])
    if fault == "bytes":
        proto.doc_string = "x" * (1024 * 1024)
        expected = "original model exceeds bound"
    elif fault == "operator":
        proto.graph.node[0].op_type = "x" * 129
        expected = "artifact node metadata exceeds bound"
    else:
        proto.graph.node[0].input[0] = "x" * 257
        expected = "artifact symbol metadata exceeds bound"
    module, report = _capture(tmp_path, proto)
    assert report["artifact_events"] == []
    assert expected in str(report["observer_errors"])
    assert not (tmp_path / module.MODEL_NAME).exists()


def test_genuine_selected_artifact_fits_node_and_one_mib_envelope(
    tmp_path: Path,
) -> None:
    with jax.enable_x64(True):
        model = export_selected()
    receipt = assert_selected_model(model)
    proto = ir.to_proto(model)
    compute = 63 if receipt.contract.algorithm == "DIRECT_RNE_53_TO_11" else 65
    assert len(proto.graph.node) <= compute + 59 + 1 <= 125
    assert len(proto.SerializeToString()) <= 1024 * 1024
    module, report = _capture(tmp_path, proto)
    assert report["observer_errors"] == [] and len(report["artifact_events"]) == 1
    assert (tmp_path / module.MODEL_NAME).read_bytes() == proto.SerializeToString()
