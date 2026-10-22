# tests/extra_tests/converter/test_source_native_nnx_log_softmax.py

"""Native NNX final-classifier spelling preserves its real named Function."""

from typing import Any

from flax import nnx
import jax
from jax import export as jax_export
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import validate_source_jaxpr
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_model_native_bindings import (
    _fresh,
    _contexts,
    _payload,
)


def _native_log_softmax(x: Any) -> Any:
    return nnx.log_softmax(x, axis=-1)


@onnx_function(name="NativeNNXClassifierLogSoftmax")
def _named_log_softmax(x: Any) -> Any:
    return nnx.log_softmax(x, axis=-1)


def _call_named(x: Any) -> Any:
    return _named_log_softmax(x)


@pytest.mark.parametrize("x64", (False, True))
def test_source_nnx_log_softmax_keeps_native_typed_trace_and_called_function(
    x64: bool,
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]
        specs = [jax.ShapeDtypeStruct((batch, 10), np.float32)]
        native = jax.make_jaxpr(_fresh(_native_log_softmax))(*specs)
        with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
            activated = jax.make_jaxpr(_fresh(_native_log_softmax))(*specs)
        assert typed_trace_manifest(activated) == typed_trace_manifest(native)
        assert _contexts(activated) == _contexts(native)
        assert [_payload(v) for v in activated.consts] == [
            _payload(v) for v in native.consts
        ]
        assert all(e.primitive.name != "nnx.log_softmax" for e in activated.eqns)
        validate_source_jaxpr(native, 23)
        model = ui._to_onnx(
            _call_named, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert isinstance(model, onnx.ModelProto)
        onnx.checker.check_model(model, full_check=True)
        functions = [
            f for f in model.functions if "NativeNNXClassifierLogSoftmax" in f.name
        ]
        assert len(functions) == 1
        function = functions[0]
        assert any(
            n.domain == function.domain and n.op_type == function.name
            for n in model.graph.node
        )
        assert any(n.op_type == "Exp" for n in function.node)
        assert any(n.op_type == "Log" for n in function.node)
        assert all(n.op_type != "LogSoftmax" for n in function.node)
        value = model.graph.output[0].type.tensor_type
        assert value.elem_type == onnx.TensorProto.FLOAT
        assert value.shape.dim[0].dim_param == "B"
        assert value.shape.dim[1].dim_value == 10
