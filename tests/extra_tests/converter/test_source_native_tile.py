# tests/extra_tests/converter/test_source_native_tile.py

"""Canonical native Tile transport and the observed original14 token geometry."""

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
from jax2onnx._compat.jax import Primitive, Var, native_tile_primitive
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    _Site,
    _tensor,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_tile import (
    require_native_tile_emission,
    validate_native_tile,
)
from jax2onnx.plugins.jax.numpy.tile import JnpTilePlugin
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


def _token_tile(x: Any, token: Any) -> Any:
    # Exact observed Tile input/reps: (1,1,256), (B,1,1). Keeping x as a
    # separate result also checks that its genuine B is retained by one artifact.
    return jnp.tile(token, (x.shape[0], 1, 1)), x


@onnx_function(name="NativeTileClsToken")
def _named_token_tile(x: Any, token: Any) -> Any:
    return _token_tile(x, token)


def _call_named_token_tile(x: Any, token: Any) -> Any:
    # Resolve the decorated global while plugin activation owns its binding.
    return _named_token_tile(x, token)


def _same_full_world_trace(
    fn: Callable[..., Any], specs: list[Any], *, require_tile: bool = True
) -> Any:
    native = jax.make_jaxpr(_fresh(fn))(*specs)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
    assert typed_trace_manifest(activated) == typed_trace_manifest(native)
    assert _contexts(activated) == _contexts(native)
    assert [_payload(v) for v in activated.consts] == [
        _payload(v) for v in native.consts
    ]
    validate_source_jaxpr(native, 23)
    canonical = native_tile_primitive()
    if canonical is None:
        # The support floor's actual native program owns its existing exact
        # reshape/broadcast/reshape semantics; no synthetic Tile is admitted.
        assert all(e.primitive.name != "tile" for e in native.eqns)
        assert {e.primitive.name for e in native.eqns} <= {
            "reshape",
            "broadcast_in_dim",
        }
    elif require_tile:
        equation = next(e for e in native.eqns if e.primitive is canonical)
        assert len(equation.invars) == len(equation.outvars) == 1
        assert type(equation.params["reps"]) is tuple
    return native


def _sample(fn: Callable[..., Any], values: tuple[Any, ...]) -> SourceSample:
    eager = fn(*values)
    compiled = jax.jit(_fresh(fn))
    _same(compiled(*values), eager)
    _same(compiled(*values), eager)
    expected = tuple(np.asarray(v) for v in jax.tree.leaves(eager))
    return SourceSample(values, expected, SampleCriteria("bitwise"))


def _check_one_artifact(model: Any, samples: list[SourceSample]) -> None:
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
def test_original14_singleton_token_tile_is_bit_exact_for_b2_b5(
    x64: bool, named: bool
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]
        specs = [
            jax.ShapeDtypeStruct((batch, 49, 256), np.float32),
            jax.ShapeDtypeStruct((1, 1, 256), np.float32),
        ]
        native = _same_full_world_trace(_token_tile, specs)
        if native_tile_primitive() is not None:
            equation = next(e for e in native.eqns if e.primitive.name == "tile")
            assert equation.invars[0].aval.shape == (1, 1, 256)
            assert equation.params == {"reps": (batch, 1, 1)}
            assert equation.outvars[0].aval.shape == (batch, 1, 256)
        fn = _call_named_token_tile if named else _token_tile
        samples = []
        for size in (2, 5):
            x = (np.arange(size * 49 * 256).reshape(size, 49, 256) / 8).astype(
                np.float32
            )
            token = (np.arange(256).reshape(1, 1, 256) / 16 - 8).astype(np.float32)
            token[0, 0, 0] = np.float32(-0.0)
            samples.append(_sample(fn, (x, token)))
        model = ui._to_onnx(
            fn, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        if named:
            functions = [f for f in model.functions if "NativeTileClsToken" in f.name]
            assert len(functions) == 1
            function = functions[0]
            assert any(
                n.domain == function.domain and n.op_type == function.name
                for n in model.graph.node
            )
            if native_tile_primitive() is not None:
                assert any(n.op_type == "Tile" for n in function.node)
        output = model.graph.output[0].type.tensor_type
        assert output.elem_type == onnx.TensorProto.FLOAT
        assert output.shape.dim[0].dim_param == "B"
        assert tuple(d.dim_value for d in output.shape.dim[1:]) == (1, 256)
        if native_tile_primitive() is not None:
            nodes = [*model.graph.node, *(n for f in model.functions for n in f.node)]
            assert any(n.op_type == "Tile" for n in nodes)
        _check_one_artifact(model, samples)


@pytest.mark.parametrize(
    "dtype,x64",
    [
        ("bool", False),
        ("int8", False),
        ("uint8", False),
        ("int32", False),
        ("uint32", False),
        ("int64", True),
        ("uint64", True),
        ("float16", False),
        ("float32", False),
        ("float64", True),
    ],
)
def test_existing_batch_tile_transports_exact_dtype_and_bytes(
    dtype: str, x64: bool
) -> None:
    with _runtime_configuration(x64):
        batch = jax_export.symbolic_shape("B")[0]

        def original(x: Any) -> Any:
            # The final physical dimension of this real/discrete tensor is two;
            # it must never be inferred to be a complex component pair.
            return jnp.tile(x, (1, 3))

        specs = [jax.ShapeDtypeStruct((batch, 2), np.dtype(dtype))]
        _same_full_world_trace(original, specs)
        samples = []
        for size in (2, 5):
            data = np.arange(size * 2).reshape(size, 2)
            if dtype == "bool":
                data = data % 2 == 0
            elif dtype == "float64":
                data = data.astype(np.float64) + np.float64(2.0**-40)
            elif dtype in {"uint32", "uint64"}:
                data = data.astype(np.dtype(dtype)) + np.asarray(
                    1 << 31, np.dtype(dtype)
                )
            else:
                data = data - 2
            values = (data.astype(np.dtype(dtype)),)
            samples.append(_sample(original, values))
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        expected_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
        assert model.graph.input[0].type.tensor_type.elem_type == expected_type
        assert model.graph.output[0].type.tensor_type.elem_type == expected_type
        _check_one_artifact(model, samples)


def _native_token_equation() -> tuple[Any, Any]:
    batch = jax_export.symbolic_shape("B")[0]
    graph = jax.make_jaxpr(_token_tile)(
        jax.ShapeDtypeStruct((batch, 49, 256), np.float32),
        jax.ShapeDtypeStruct((1, 1, 256), np.float32),
    )
    canonical = native_tile_primitive()
    equation = next((e for e in graph.eqns if e.primitive is canonical), None)
    return graph, equation


def _validate(equation: Any, formal_symbols: set[Any]) -> None:
    site = _Site("native Tile contract test", 23, equation.primitive.name)
    validate_native_tile(
        equation,
        [_tensor(v, site) for v in equation.invars],
        [_tensor(v, site) for v in equation.outvars],
        formal_symbols,
        site,
    )


def test_native_tile_schema_keeps_identity_scope_type_shape_and_int64_guards() -> None:
    with _runtime_configuration(True):
        graph, equation = _native_token_equation()
        trace = str(graph)
        if equation is None:
            # Actual floor absence is checked positively above. A same-named
            # custom owner still cannot claim this canonical primitive role.
            assert native_tile_primitive() is None
            assert {e.primitive.name for e in graph.eqns} <= {
                "reshape",
                "broadcast_in_dim",
            }
            stand_in = SimpleNamespace(
                primitive=Primitive("tile"), invars=graph.invars, outvars=graph.outvars
            )
            with pytest.raises(
                SourcePrecisionError, match="canonical native primitive"
            ):
                _validate(stand_in, set())
            return
        site = _Site("native Tile contract test", 23, "tile")
        target = _tensor(equation.outvars[0], site)
        assert target.symbol is not None
        formals = {target.symbol}
        _validate(equation, formals)
        operand, result = equation.invars[0], equation.outvars[0]
        batch = result.aval.shape[0]
        other_batch = jax_export.symbolic_shape("B")[0]
        corruptions = {
            "custom identity": equation.replace(primitive=Primitive("tile")),
            "second operand": equation.replace(invars=[operand, operand]),
            "effects": equation.replace(effects={object()}),
            "placement": equation.replace(
                invars=[
                    SimpleNamespace(
                        aval=SimpleNamespace(
                            shape=operand.aval.shape,
                            dtype=operand.aval.dtype,
                            weak_type=operand.aval.weak_type,
                            sharding=object(),
                        )
                    )
                ]
            ),
            "extra parameter": equation.replace(
                params={**equation.params, "mode": None}
            ),
            "list reps": equation.replace(params={"reps": [batch, 1, 1]}),
            "rank": equation.replace(params={"reps": (batch, 1)}),
            "foreign scope": equation.replace(params={"reps": (other_batch, 1, 1)}),
            "affine repeat": equation.replace(params={"reps": (batch + 1, 1, 1)}),
            "B squared": equation.replace(
                invars=[Var(operand.aval.update(shape=(batch, 1, 256)))]
            ),
            "symbolic multiple": equation.replace(
                invars=[Var(operand.aval.update(shape=(2, 1, 256)))]
            ),
            "dtype": equation.replace(
                outvars=[Var(result.aval.update(dtype=np.dtype(np.float64)))]
            ),
            "weakness": equation.replace(
                outvars=[Var(result.aval.update(weak_type=True))]
            ),
            "shape": equation.replace(
                outvars=[Var(result.aval.update(shape=(batch, 1, 257)))]
            ),
            "negative repeat": equation.replace(params={"reps": (batch, -1, 1)}),
            "boolean repeat": equation.replace(params={"reps": (batch, True, 1)}),
            "int64 repeat": equation.replace(params={"reps": (batch, 1 << 63, 0)}),
            "int64 size": equation.replace(
                params={"reps": (batch, 1 << 62, 4)},
                outvars=[Var(result.aval.update(shape=(batch, 1 << 62, 1024)))],
            ),
        }
        for name, changed in corruptions.items():
            with pytest.raises(SourcePrecisionError) as error:
                _validate(changed, formals)
            assert error.value, name
        with pytest.raises(SourcePrecisionError, match="actual formals"):
            _validate(equation, set())
        assert str(graph) == trace


def test_symbolic_tile_requires_actual_origin_before_accessing_the_builder() -> None:
    with _runtime_configuration(False):
        graph, equation = _native_token_equation()
        if equation is None:
            assert native_tile_primitive() is None
            validate_source_jaxpr(graph, 23)
            return
        origins = []

        def missing_origin(value: Any) -> None:
            origins.append(value)
            return None

        context = SimpleNamespace(
            precision_policy=PrecisionPolicy.SOURCE,
            opset=23,
            get_symbolic_dim_origin=missing_origin,
        )
        with pytest.raises(SourcePrecisionError, match="no actual input origin"):
            JnpTilePlugin().lower(context, equation)
        assert len(origins) == 1 and origins[0] is equation.params["reps"][0]
        with pytest.raises(SourcePrecisionError, match="no actual input origin"):
            require_native_tile_emission(equation, context)


@pytest.mark.parametrize(
    "shape,repeats", [((), ()), ((0, 2), (3, 1)), ((2, 0), (1, 2))]
)
def test_static_scalar_and_empty_tile_keep_native_rank_and_storage(
    shape: tuple[int, ...], repeats: tuple[int, ...]
) -> None:
    with _runtime_configuration(False):

        def original(x: Any) -> Any:
            return jnp.tile(x, repeats)

        specs = [jax.ShapeDtypeStruct(shape, np.float32)]
        _same_full_world_trace(original, specs, require_tile=False)
        values = (np.zeros(shape, np.float32),)
        sample = _sample(original, values)
        model = ui._to_onnx(
            original, specs, precision_policy=PrecisionPolicy.SOURCE, opset=23
        )
        _check_one_artifact(model, [sample])
