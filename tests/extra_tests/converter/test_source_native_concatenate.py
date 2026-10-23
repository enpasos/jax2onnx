# tests/extra_tests/converter/test_source_native_concatenate.py

"""Native concatenate follows Tile through the unchanged CLS token geometry."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Primitive, Var
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.plugins.jax.lax.concatenate import ConcatenatePlugin
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_mlp_runtime_acceptance import (
    _runtime_configuration,
)
from tests.extra_tests.converter.test_source_model_native_bindings import (
    _contexts,
    _fresh,
    _payload,
)
from tests.extra_tests.converter.test_source_native_tile import (
    _sample,
    _check_one_artifact,
)


def _concat_token(x: Any, token: Any) -> Any:
    cls_tokens = jnp.tile(token, (x.shape[0], 1, 1))
    return jnp.concatenate([cls_tokens, x], axis=1)


@onnx_function(name="NativeConcatClsToken")
def _named_concat_token(x: Any, token: Any) -> Any:
    return _concat_token(x, token)


def _call_named_concat_token(x: Any, token: Any) -> Any:
    return _named_concat_token(x, token)


def _trace(fn: Callable[..., Any], specs: list[Any]) -> Any:
    native = jax.make_jaxpr(_fresh(fn))(*specs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
    assert typed_trace_manifest(activated) == typed_trace_manifest(native)
    assert _contexts(activated) == _contexts(native)
    assert [_payload(v) for v in activated.consts] == [
        _payload(v) for v in native.consts
    ]
    assert all(e.primitive.name != "jax.numpy.concatenate" for e in activated.eqns)
    assert any(e.primitive is jax.lax.concatenate_p for e in native.eqns)
    validate_source_jaxpr(native, 23)
    return native


@pytest.mark.parametrize("x64", (False, True))
@pytest.mark.parametrize("named", (False, True), ids=("root", "function"))
def test_native_token_tile_then_concatenate_is_bit_exact_for_b2_b5(
    x64: bool, named: bool
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]
        specs = [
            jax.ShapeDtypeStruct((batch, 49, 256), np.float32),
            jax.ShapeDtypeStruct((1, 1, 256), np.float32),
        ]
        graph = _trace(_concat_token, specs)
        concatenate = next(
            e for e in graph.eqns if e.primitive is jax.lax.concatenate_p
        )
        assert concatenate.params == {"dimension": 1}
        assert concatenate.outvars[0].aval.shape == (batch, 50, 256)
        fn = _call_named_concat_token if named else _concat_token
        samples = []
        for size in (2, 5):
            x = (np.arange(size * 49 * 256).reshape(size, 49, 256) / 8 - 2).astype(
                np.float32
            )
            token = (np.arange(256).reshape(1, 1, 256) / 16 - 8).astype(np.float32)
            token[0, 0, 0] = np.float32(-0.0)
            samples.append(_sample(fn, (x, token)))
        model = ui._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        if named:
            functions = [f for f in model.functions if "NativeConcatClsToken" in f.name]
            assert len(functions) == 1
            function = functions[0]
            assert any(
                n.domain == function.domain and n.op_type == function.name
                for n in model.graph.node
            )
            assert any(n.op_type == "Concat" for n in function.node)
        value = model.graph.output[0].type.tensor_type
        assert value.elem_type == onnx.TensorProto.FLOAT
        assert value.shape.dim[0].dim_param == "B"
        assert tuple(d.dim_value for d in value.shape.dim[1:]) == (50, 256)
        _check_one_artifact(model, samples)


@pytest.mark.parametrize(
    "dtype", ("bool", "uint32", "int32", "float16", "float32", "float64")
)
def test_native_concatenate_keeps_dtype_and_exact_input_bytes(dtype: str) -> None:
    with _runtime_configuration(True):
        batch = jax_export.symbolic_shape("B")[0]

        def original(x: Any, y: Any) -> Any:
            return jnp.concatenate((x, y), axis=-1)

        specs = [
            jax.ShapeDtypeStruct((batch, 2), np.dtype(dtype)),
            jax.ShapeDtypeStruct((batch, 3), np.dtype(dtype)),
        ]
        _trace(original, specs)
        samples = []
        for size in (2, 5):
            left = np.arange(size * 2).reshape(size, 2)
            right = np.arange(size * 3).reshape(size, 3) + 7
            if dtype == "bool":
                left, right = left % 2 == 0, right % 3 == 1
            elif dtype == "float64":
                left = left.astype(np.float64) + np.float64(2.0**-40)
                right = right.astype(np.float64) - np.float64(2.0**-40)
            elif dtype == "uint32":
                left = left.astype(np.uint32) + np.uint32(1 << 31)
                right = right.astype(np.uint32) + np.uint32(1 << 31)
            values = (left.astype(np.dtype(dtype)), right.astype(np.dtype(dtype)))
            samples.append(_sample(original, values))
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert model.graph.output[
            0
        ].type.tensor_type.elem_type == onnx.helper.np_dtype_to_tensor_dtype(
            np.dtype(dtype)
        )
        assert all(n.op_type != "Cast" for n in model.graph.node)
        _check_one_artifact(model, samples)


def test_native_explicit_dtype_concatenate_preserves_its_original_conversion_boundary() -> (
    None
):
    with _runtime_configuration(True):

        def original(x: Any, y: Any) -> Any:
            return jnp.concatenate([x, y], axis=1, dtype=np.float64)

        specs = [
            jax.ShapeDtypeStruct((2, 2), np.float32),
            jax.ShapeDtypeStruct((2, 3), np.int32),
        ]
        _trace(original, specs)
        values = (
            np.asarray([[0.0, -0.0], [1.0, -1.0]], np.float32),
            np.asarray([[16777217, 16777219, -16777217], [1, 2, 3]], np.int32),
        )
        sample = _sample(original, values)
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        assert (
            model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        )
        assert any(n.op_type == "Cast" for n in model.graph.node)
        _check_one_artifact(model, [sample])


def test_source_native_concatenate_rechecks_existing_owner_before_builder_access() -> (
    None
):
    with _runtime_configuration(False):
        batch = jax_export.symbolic_shape("B")[0]
        graph = jax.make_jaxpr(lambda x, y: jax.lax.concatenate((x, y), 1))(
            jax.ShapeDtypeStruct((batch, 2), np.float32),
            jax.ShapeDtypeStruct((batch, 3), np.float32),
        )
        equation = graph.eqns[0]
        context = SimpleNamespace(precision_policy=PrecisionPolicy.SOURCE, opset=23)
        changed = equation.replace(primitive=Primitive("concatenate"))
        with pytest.raises(SourcePrecisionError, match="canonical native primitive"):
            ConcatenatePlugin().lower(context, changed)
        changed = equation.replace(
            outvars=[Var(equation.outvars[0].aval.update(dtype=np.dtype(np.int32)))]
        )
        with pytest.raises(SourcePrecisionError, match="native data dtype"):
            ConcatenatePlugin().lower(context, changed)
        changed = equation.replace(
            outvars=[Var(equation.outvars[0].aval.update(weak_type=True))]
        )
        with pytest.raises(SourcePrecisionError, match="weakness"):
            ConcatenatePlugin().lower(context, changed)
