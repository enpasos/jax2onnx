# tests/extra_tests/converter/test_source_native_split.py

"""Native Split transport for the unchanged Transformer's rotary split leaf."""

from collections.abc import Callable
import hashlib
from types import SimpleNamespace
from typing import Any

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Primitive, Var, native_split_primitive
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_split import validate_native_split
from jax2onnx.plugins.jax.lax.split import SplitPlugin
from jax2onnx.precision import (
    CheckStatus,
    SampleCriteria,
    SourceSample,
    validate_source_samples,
)
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
    _same,
)


def _rotary_split(x: Any) -> Any:
    # This is the original rotary leaf spelling: two pieces on the last axis.
    return jnp.split(x, 2, axis=-1)


@onnx_function(name="NativeRotarySplit")
def _named_rotary_split(x: Any) -> Any:
    return _rotary_split(x)


def _call_named_rotary_split(x: Any) -> Any:
    return _named_rotary_split(x)


def _trace(fn: Callable[..., Any], specs: list[Any]) -> Any:
    native = jax.make_jaxpr(_fresh(fn))(*specs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
    assert typed_trace_manifest(activated) == typed_trace_manifest(native)
    assert _contexts(activated) == _contexts(native)
    assert [_payload(v) for v in activated.consts] == [
        _payload(v) for v in native.consts
    ]
    assert all(e.primitive.name != "jax.numpy.split" for e in activated.eqns)
    assert any(e.primitive is native_split_primitive() for e in native.eqns)
    validate_source_jaxpr(native, 23)
    return native


def _sample(fn: Callable[..., Any], values: tuple[Any, ...]) -> SourceSample:
    eager = fn(*values)
    compiled = jax.jit(_fresh(fn))
    _same(compiled(*values), eager)
    _same(compiled(*values), eager)
    return SourceSample(
        values,
        tuple(np.asarray(v) for v in jax.tree.leaves(eager)),
        SampleCriteria("bitwise"),
    )


def _check(model: Any, samples: list[SourceSample]) -> None:
    assert isinstance(model, onnx.ModelProto)
    onnx.checker.check_model(model, full_check=True)
    digest = hashlib.sha256(model.SerializeToString()).hexdigest()
    for optimization in ("disabled", "all"):
        result = validate_source_samples(
            model, samples=samples, optimization=optimization
        )
        assert result.status is CheckStatus.VALIDATED, result
        assert len(result.samples) == len(samples)
        result.require_sample_acceptance()
    assert hashlib.sha256(model.SerializeToString()).hexdigest() == digest


@pytest.mark.parametrize("x64", (False, True))
@pytest.mark.parametrize("named", (False, True), ids=("root", "function"))
def test_native_rotary_split_has_one_bit_exact_b2_b5_artifact(
    x64: bool, named: bool
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]
        specs = [jax.ShapeDtypeStruct((batch, 2, 4), np.float32)]
        native = _trace(_rotary_split, specs)
        equation = next(
            e for e in native.eqns if e.primitive is native_split_primitive()
        )
        assert equation.params == {"axis": 2, "sizes": (2, 2)}
        assert len(equation.outvars) == 2
        assert all(v.aval.shape == (batch, 2, 2) for v in equation.outvars)
        fn = _call_named_rotary_split if named else _rotary_split
        samples = []
        for size in (2, 5):
            data = (np.arange(size * 8).reshape(size, 2, 4) / 8 - 3).astype(np.float32)
            data[0, 0, 0] = np.float32(-0.0)
            samples.append(_sample(fn, (data,)))
        model = ui._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        if named:
            functions = [f for f in model.functions if "NativeRotarySplit" in f.name]
            assert len(functions) == 1
            function = functions[0]
            assert any(
                n.domain == function.domain and n.op_type == function.name
                for n in model.graph.node
            )
            assert any(n.op_type == "Split" for n in function.node)
        assert len(model.graph.output) == 2
        for output in model.graph.output:
            value = output.type.tensor_type
            assert value.elem_type == onnx.TensorProto.FLOAT
            assert value.shape.dim[0].dim_param == "B"
            assert tuple(d.dim_value for d in value.shape.dim[1:]) == (2, 2)
        nodes = [*model.graph.node, *(n for f in model.functions for n in f.node)]
        assert any(n.op_type == "Split" and len(n.output) == 2 for n in nodes)
        _check(model, samples)


@pytest.mark.parametrize(
    "dtype", ("bool", "int8", "uint32", "int64", "float16", "float32", "float64")
)
def test_native_indices_split_keeps_exact_dtype_bytes_and_empty_piece(
    dtype: str,
) -> None:
    with _runtime_configuration(True):
        batch = jax_export.symbolic_shape("B")[0]

        def original(x: Any) -> Any:
            return jnp.split(x, np.asarray([2, 2, 5]), axis=-1)

        specs = [jax.ShapeDtypeStruct((batch, 8), np.dtype(dtype))]
        native = _trace(original, specs)
        equation = next(
            e for e in native.eqns if e.primitive is native_split_primitive()
        )
        assert equation.params == {"axis": 1, "sizes": (2, 0, 3, 3)}
        samples = []
        for size in (2, 5):
            data = np.arange(size * 8).reshape(size, 8)
            if dtype == "bool":
                data = data % 3 == 1
            elif dtype == "float64":
                data = data.astype(np.float64) + np.float64(2.0**-40)
            elif dtype == "uint32":
                data = data.astype(np.uint32) + np.uint32(1 << 31)
            else:
                data = data - 3
            samples.append(_sample(original, (data.astype(np.dtype(dtype)),)))
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        dtype_code = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        assert model.graph.input[0].type.tensor_type.elem_type == dtype_code
        assert all(
            v.type.tensor_type.elem_type == dtype_code for v in model.graph.output
        )
        assert model.graph.output[1].type.tensor_type.shape.dim[1].dim_value == 0
        _check(model, samples)


def _equation() -> Any:
    batch = jax_export.symbolic_shape("B")[0]
    graph = jax.make_jaxpr(_rotary_split)(
        jax.ShapeDtypeStruct((batch, 2, 4), np.float32)
    )
    return graph, next(e for e in graph.eqns if e.primitive is native_split_primitive())


def _validate(equation: Any, formals: set[Any]) -> None:
    site = _Site("native Split contract test", 23, equation.primitive.name)
    validate_native_split(
        equation,
        [_tensor(v, site) for v in equation.invars],
        [_tensor(v, site) for v in equation.outvars],
        formals,
        site,
    )


def test_native_split_schema_retains_actual_identity_axis_sizes_scope_and_types() -> (
    None
):
    with _runtime_configuration(True):
        graph, equation = _equation()
        trace = str(graph)
        site = _Site("native Split contract test", 23, "split")
        source = _tensor(equation.invars[0], site)
        assert source.symbol is not None
        formals = {source.symbol}
        _validate(equation, formals)
        left, right = equation.outvars
        batch = equation.invars[0].aval.shape[0]
        other = jax_export.symbolic_shape("B")[0]
        corruptions = {
            "custom identity": equation.replace(primitive=Primitive("split")),
            "second input": equation.replace(
                invars=[*equation.invars, *equation.invars]
            ),
            "missing output": equation.replace(outvars=[left]),
            "effects": equation.replace(effects={object()}),
            "unknown parameter": equation.replace(
                params={**equation.params, "placement": None}
            ),
            "negative axis": equation.replace(params={**equation.params, "axis": -1}),
            "axis rank": equation.replace(params={**equation.params, "axis": 3}),
            "boolean axis": equation.replace(params={**equation.params, "axis": True}),
            "list sizes": equation.replace(params={**equation.params, "sizes": [2, 2]}),
            "negative size": equation.replace(
                params={**equation.params, "sizes": (-1, 5)}
            ),
            "boolean size": equation.replace(
                params={**equation.params, "sizes": (True, 3)}
            ),
            "sum": equation.replace(params={**equation.params, "sizes": (1, 2)}),
            "symbolic split axis": equation.replace(
                params={**equation.params, "axis": 0}
            ),
            "result dtype": equation.replace(
                outvars=[left, Var(right.aval.update(dtype=np.dtype(np.float64)))]
            ),
            "result weakness": equation.replace(
                outvars=[left, Var(right.aval.update(weak_type=True))]
            ),
            "result shape": equation.replace(
                outvars=[left, Var(right.aval.update(shape=(batch, 2, 3)))]
            ),
            "foreign scope": equation.replace(
                outvars=[left, Var(right.aval.update(shape=(other, 2, 2)))]
            ),
            "placement": equation.replace(
                invars=[
                    SimpleNamespace(
                        aval=SimpleNamespace(
                            shape=equation.invars[0].aval.shape,
                            dtype=np.dtype(np.float32),
                            weak_type=False,
                            sharding=object(),
                        )
                    )
                ]
            ),
            "int64 sizes": equation.replace(
                params={**equation.params, "sizes": (1 << 63, 0)}
            ),
        }
        for name, changed in corruptions.items():
            with pytest.raises(SourcePrecisionError) as error:
                _validate(changed, formals)
            assert error.value, name
        with pytest.raises(SourcePrecisionError, match="actual formals"):
            _validate(equation, set())
        assert str(graph) == trace


def test_native_split_rejects_missing_size_schema_before_builder_access() -> None:
    with _runtime_configuration(False):
        _, equation = _equation()
        changed = equation.replace(params={"axis": 2, "sizes": [2, 2]})
        context = SimpleNamespace(precision_policy=PrecisionPolicy.SOURCE, opset=23)
        with pytest.raises(SourcePrecisionError, match="invalid native split sizes"):
            SplitPlugin().lower(context, changed)
