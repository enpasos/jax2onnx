# tests/extra_tests/converter/test_source_symbolic_contraction.py

"""One native B survives ordered contraction, exact transpose and scoped ports.

The original contraction recipes and their missing numerical criteria stay in
other files. These independent dyadic coordinate samples establish only their
own declared bitwise acceptance, for B=2/5 and the recorded CPU execution modes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import jax
from jax import export
from jax.extend.core import jaxpr_as_fun
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import onnx_function, user_interface as ui
from jax2onnx._compat.jax import Primitive, Var, atomic_symbolic_dimension
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    _Site,
    _tensor,
    _validate_dot,
    _validate_tensor_layout,
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import (
    normalized_shape_dimensions,
    representation_for_aval,
)
from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin
from jax2onnx.plugins.plugin_system import FunctionPlugin, ONNX_FUNCTION_PLUGIN_REGISTRY
from jax2onnx.plugins.jax.lax.transpose import TransposePlugin
from jax2onnx.precision import (
    CheckStatus,
    PRECISION_METADATA_KEY,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    precision_report_to_json,
    validate_source_samples,
)
from tests.extra_tests.converter.source_target_evidence import require_target_pair
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_generalized_contraction import (
    _oracle,
    _payload,
    _same,
)
from tests.extra_tests.converter.test_source_tensor_matrix import (
    _body,
    _port,
)

_BITWISE = SampleCriteria("bitwise")
_WIDTHS = [("float32", False), ("float32", True), ("float64", True)]
_MATRIX = (((1,), (0,)), ((), ()))
_ORIENTATION = (((1,), (1,)), ((), ()))


def _globals() -> dict[str, Any]:
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }


def _settings() -> tuple[Any, ...]:
    return tuple(getattr(jax.config, name) for name in _globals())


@pytest.fixture(autouse=True)
def _restore_settings() -> Iterator[None]:
    previous, effective = _globals(), _settings()
    before = jnp.matmul, jnp.transpose, jnp.add
    try:
        with (
            jax.enable_x64(False),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)
        assert (_globals(), _settings()) == (previous, effective)
        assert (jnp.matmul, jnp.transpose, jnp.add) == before
        assert active_precision_policy() is PrecisionPolicy.LEGACY


def _fresh(fn: Any) -> Any:
    def call(*args: Any) -> Any:
        return fn(*args)

    return call


def _specs(shapes: Any, dtype: Any) -> list[Any]:
    (batch,) = export.symbolic_shape("B")
    return [
        jax.ShapeDtypeStruct(
            tuple(batch if dim == "B" else dim for dim in shape), dtype
        )
        for shape in shapes
    ]


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (
            np.dtype(v.aval.dtype).name,
            normalized_shape_dimensions(v.aval.shape),
            bool(v.aval.weak_type),
        )
        for v in variables
    )


@contextmanager
def _observe_functions() -> Any:
    """Associate each outer trace before recursive child builds append theirs."""
    bodies, built = [], []
    original_trace = FunctionPlugin._trace_function_body
    original_build = FunctionPlugin._build_function

    def trace(plugin: Any, *args: Any, **kwargs: Any) -> Any:
        closed = original_trace(plugin, *args, **kwargs)
        bodies.append(closed)
        return closed

    def build(plugin: Any, *args: Any, **kwargs: Any) -> Any:
        index = len(bodies)
        definition = original_build(plugin, *args, **kwargs)
        assert len(bodies) > index
        built.append((definition, bodies[index]))
        return definition

    with (
        patch.object(FunctionPlugin, "_trace_function_body", trace),
        patch.object(FunctionPlugin, "_build_function", build),
    ):
        yield built


def _save(path: Path, arrays: Any) -> None:
    np.savez(path, **{f"out{i}": np.asarray(v) for i, v in enumerate(arrays)})


def _observe(
    fn: Any,
    specs: Any,
    feeds: Any,
    expected: Any,
    directory: Path,
    *,
    output_shapes: Any,
    opset: int = 23,
    named: bool = False,
) -> tuple[Any, Any, Any]:
    """Freeze native observations and artifact before checking either target role."""
    directory.mkdir(parents=True, exist_ok=True)
    before = _globals(), _settings()
    frozen = [tuple(np.asarray(v).copy() for v in values) for values in feeds]
    native = jax.make_jaxpr(_fresh(fn))(*specs)
    native_manifest = typed_trace_manifest(native)
    (directory / "native-trace.txt").write_text(str(native))
    (directory / "native-trace.json").write_text(json.dumps(native_manifest, indent=2))
    (directory / "criteria.json").write_text(json.dumps(asdict(_BITWISE)))
    references = []
    for index, (values, wanted) in enumerate(zip(feeds, expected, strict=True)):
        eager = tuple(np.asarray(v).copy() for v in jax.tree.leaves(fn(*values)))
        compiled = tuple(
            np.asarray(v).copy() for v in jax.tree.leaves(jax.jit(_fresh(fn))(*values))
        )
        _save(directory / f"inputs-{index}.npz", frozen[index])
        _save(directory / f"eager-{index}.npz", eager)
        _save(directory / f"fresh-jit-{index}.npz", compiled)
        _save(directory / f"coordinate-{index}.npz", wanted)
        references.append((eager, compiled))
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(_fresh(fn))(*specs)
    (directory / "activated-trace.txt").write_text(str(activated))
    (directory / "activated-trace.json").write_text(
        json.dumps(typed_trace_manifest(activated), indent=2)
    )
    assert _signature(_body(native).invars) == _signature(_body(activated).invars)
    assert _signature(_body(native).outvars) == _signature(_body(activated).outvars)
    if not named:
        assert typed_trace_manifest(activated) == native_manifest
    with _observe_functions() as built:
        artifact = ui._to_onnx(
            fn, specs, opset=opset, precision_policy=PrecisionPolicy.SOURCE
        )
    payload = artifact.SerializeToString()
    (directory / "source.onnx").write_bytes(payload)
    onnx.checker.check_model(artifact, full_check=True)
    report = get_precision_report(artifact)
    assert report is not None and report.profile == "source-static-v1"
    (directory / "source-report.json").write_text(precision_report_to_json(report))
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert tuple(v.shape for v in report.traced_outputs) == tuple(output_shapes)
    assert tuple(v.shape for v in report.exported_outputs) == tuple(output_shapes)
    for port, declaration in zip(artifact.graph.input, report.inputs, strict=True):
        _port(port, declaration.traced.dtype, declaration.traced.shape)
    for port, declaration in zip(
        artifact.graph.output, report.exported_outputs, strict=True
    ):
        _port(port, declaration.dtype, declaration.shape)
    observations, results = [], []
    for optimization in ("disabled", "all"):
        for role in (0, 1):
            samples = [
                SourceSample(values, outputs[role], _BITWISE)
                for values, outputs in zip(frozen, references, strict=True)
            ]
            result = validate_source_samples(
                artifact, samples=samples, optimization=optimization
            )
            results.append(result)
            observations.append(
                {
                    "optimization": optimization,
                    "native_role": role,
                    "artifact_sha256": result.artifact_sha256,
                    "target": None if result.target is None else asdict(result.target),
                    "checks": [asdict(c) for c in result.checks],
                    "samples": [asdict(s) for s in result.samples],
                }
            )
    (directory / "comparisons-before-guards.json").write_text(
        json.dumps(observations, indent=2)
    )
    for role in (0, 1):
        samples = [
            SourceSample(values, outputs[role], _BITWISE)
            for values, outputs in zip(frozen, references, strict=True)
        ]
        require_target_pair(
            artifact,
            samples,
            results[role],
            results[2 + role],
            evidence_path=directory / f"target-boundary-{role}.json",
        )
    for outputs, wanted in zip(references, expected, strict=True):
        for role in outputs:
            assert len(role) == len(wanted)
            for actual, target in zip(role, wanted, strict=True):
                _same(actual, target)
    assert artifact.SerializeToString() == payload
    assert typed_trace_manifest(native) == native_manifest
    for values, saved in zip(feeds, frozen, strict=True):
        for value, original in zip(values, saved, strict=True):
            _same(value, original)
    assert (_globals(), _settings()) == before
    return artifact, native, built


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
@pytest.mark.parametrize("opset", [23, 26])
def test_actual_native_weight_first_geometry_bias_and_aliases(
    dtype: str, x64: bool, opset: int, tmp_path: Path
) -> None:
    with jax.enable_x64(x64):
        weight = _payload(dtype, (64, 128), right=True)
        bias = _payload(dtype, (64,), right=True)
        original_parameters = weight.tobytes(), bias.tobytes()

        def fn(x: Any) -> Any:
            native = jax.lax.dot_general(weight, x, _ORIENTATION)
            restored = jax.lax.transpose(native, (1, 0))
            return restored, restored + bias, restored, x

        feeds = [(_payload(dtype, (batch, 128)),) for batch in (2, 5)]
        wanted = []
        for (value,) in feeds:
            raw = _oracle(weight, value, _ORIENTATION, dtype)
            result = np.transpose(raw, (1, 0)).copy()
            wanted.append((result, np.add(result, bias), result, value))
        model, trace, _ = _observe(
            fn,
            _specs([("B", 128)], dtype),
            feeds,
            wanted,
            tmp_path,
            output_shapes=[("B", 64), ("B", 64), ("B", 64), ("B", 128)],
            opset=opset,
        )
        body = _body(trace)
        dot = next(e for e in body.eqns if e.primitive is jax.lax.dot_general_p)
        transpose = next(e for e in body.eqns if e.primitive is jax.lax.transpose_p)
        assert _signature(dot.invars) == (
            (dtype, (64, 128), False),
            (dtype, ("B", 128), False),
        )
        assert dot.params["dimension_numbers"] == _ORIENTATION
        assert _signature(dot.outvars) == ((dtype, (64, "B"), False),)
        assert transpose.invars[0] is dot.outvars[0]
        assert transpose.params == {"permutation": (1, 0)}
        assert body.outvars[0] is body.outvars[2] is transpose.outvars[0]
        assert body.outvars[3] is body.invars[0]
        assert sum(n.op_type == "MatMul" for n in model.graph.node) == 1
        assert (weight.tobytes(), bias.tobytes()) == original_parameters


_GEOMETRIES = (
    # Leading public X becomes a nonleading free B on the native lhs.
    (
        ("B", 2, 3, 4),
        (4, 5, 3),
        (1, 3, 0, 2),
        (((3, 1), (2, 0)), ((), ())),
        (1, 0, 2),
        ("B", 2, 5),
        False,
    ),
    # The native free B occupies the last of four lhs axes.
    (
        ("B", 2, 3, 4),
        (4, 5, 3),
        (1, 2, 3, 0),
        (((2, 1), (0, 2)), ((), ())),
        (1, 0, 2),
        ("B", 2, 5),
        False,
    ),
    # Unequal ranks with free B on the rhs; contracting order is explicitly 4,3.
    (
        ("B", 4, 3),
        (3, 2, 4),
        (1, 0, 2),
        (((2, 0), (0, 2)), ((), ())),
        (1, 0),
        ("B", 2),
        True,
    ),
    # A supplied static batch axis precedes lhs free axes in native output.
    (
        ("B", 2, 3, 4),
        (5, 2, 3),
        (1, 3, 0, 2),
        (((3,), (2,)), ((0,), (1,))),
        (2, 0, 1, 3),
        ("B", 2, 4, 5),
        False,
    ),
    (("B", 3), (3,), (0, 1), (((1,), (0,)), ((), ())), (0,), ("B",), False),
    # Supplied batch order (native lhs axes 1,0) remains distinct from free B.
    (
        ("B", 2, 3, 4),
        (3, 2, 4, 5),
        (1, 2, 0, 3),
        (((3,), (2,)), ((1, 0), (0, 1))),
        (2, 0, 1, 3),
        ("B", 3, 2, 5),
        False,
    ),
)


@pytest.mark.parametrize(
    "geometry",
    _GEOMETRIES,
    ids=[
        "lhs-axis-two",
        "lhs-axis-three",
        "rhs-axis-one",
        "static-batch-before-free",
        "rank-one-result",
        "ordered-static-batch-axes",
    ],
)
@pytest.mark.parametrize("dtype,x64", _WIDTHS)
def test_native_unequal_ranks_ordered_contracts_and_free_axes(
    geometry: Any, dtype: str, x64: bool, tmp_path: Path
) -> None:
    shape, static_shape, input_order, dimensions, output_order, output_shape, right = (
        geometry
    )
    with jax.enable_x64(x64):
        static = _payload(dtype, static_shape, right=True)

        def fn(x: Any) -> Any:
            moved = jax.lax.transpose(x, input_order)
            operands = (static, moved) if right else (moved, static)
            return jax.lax.transpose(
                jax.lax.dot_general(*operands, dimensions), output_order
            )

        feeds, wanted = [], []
        for batch in (2, 5):
            value = _payload(dtype, tuple(batch if d == "B" else d for d in shape))
            moved = np.transpose(value, input_order)
            operands = (static, moved) if right else (moved, static)
            feeds.append((value,))
            wanted.append(
                (
                    np.transpose(
                        _oracle(*operands, dimensions, dtype), output_order
                    ).copy(),
                )
            )
        _observe(
            fn,
            _specs([shape], dtype),
            feeds,
            wanted,
            tmp_path,
            output_shapes=[output_shape],
        )


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
def test_paired_nonleading_native_B_axes_preserve_supplied_batch_order(
    dtype: str, x64: bool, tmp_path: Path
) -> None:
    dimensions = (((2,), (0,)), ((1,), (2,)))
    with jax.enable_x64(x64):

        def fn(left: Any, right: Any) -> Any:
            return jax.lax.dot_general(
                jax.lax.transpose(left, (1, 0, 2)),
                jax.lax.transpose(right, (1, 2, 0)),
                dimensions,
            )

        feeds = [
            (_payload(dtype, (b, 2, 3)), _payload(dtype, (b, 3, 4), right=True))
            for b in (2, 5)
        ]
        wanted = [
            (
                _oracle(
                    np.transpose(a, (1, 0, 2)),
                    np.transpose(b, (1, 2, 0)),
                    dimensions,
                    dtype,
                ),
            )
            for a, b in feeds
        ]
        _observe(
            fn,
            _specs([("B", 2, 3), ("B", 3, 4)], dtype),
            feeds,
            wanted,
            tmp_path,
            output_shapes=[("B", 2, 4)],
        )


@onnx_function(name="SymbolicNativeWeightFirst")
def _named_raw(x: Any, weight: Any) -> Any:
    return jax.lax.dot_general(weight, x, _ORIENTATION)


@onnx_function(name="SymbolicNativeRestoreAxes")
def _named_restore(raw: Any) -> Any:
    return jax.lax.transpose(raw, (1, 0))


@onnx_function(name="NestedSymbolicNativeContraction")
def _named_outer(x: Any, weight: Any) -> Any:
    return _named_restore(_named_raw(x, weight))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_nested_function_nonleading_formals_outputs_and_actual_native_body(
    compiled: bool, dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(dtype == "float64"):
        weight = _payload(dtype, (64, 128), right=True)
        bias = _payload(dtype, (64,), right=True)
        before = weight.tobytes(), bias.tobytes()

        def fn(x: Any) -> Any:
            result = (
                jax.jit(_named_outer)(x, weight)
                if compiled
                else _named_outer(x, weight)
            )
            return result, result + bias, result, x

        feeds = [(_payload(dtype, (b, 128)),) for b in (2, 5)]
        raw = [_oracle(weight, x, _ORIENTATION, dtype) for (x,) in feeds]
        wanted = [
            (r.T.copy(), np.add(r.T, bias), r.T.copy(), x)
            for r, (x,) in zip(raw, feeds, strict=True)
        ]
        warmed = jax.jit(_fresh(fn))
        for batch_index, (values, expected) in enumerate(
            zip(feeds, wanted, strict=True)
        ):
            outputs = tuple(np.asarray(v).copy() for v in warmed(*values))
            _save(tmp_path / f"pre-export-warmed-{batch_index}.npz", outputs)
            for actual, target in zip(outputs, expected, strict=True):
                _same(actual, target)
        model, _, built = _observe(
            fn,
            _specs([("B", 128)], dtype),
            feeds,
            wanted,
            tmp_path,
            output_shapes=[("B", 64)] * 3 + [("B", 128)],
            named=True,
        )
        assert len(built) == len(model.functions) == 3
        serialized = {(f.domain, f.name): f for f in model.functions}
        child_primitives = {
            ONNX_FUNCTION_PLUGIN_REGISTRY[
                f"{fn.__module__}.{fn.__qualname__}"
            ]._source_primitive
            for fn in (_named_raw, _named_restore)
        }
        outer_keys = [
            (definition.domain, definition.name)
            for definition, closed in built
            if any(e.primitive in child_primitives for e in _body(closed).eqns)
        ]
        (outer_key,) = outer_keys
        assert {
            (node.domain, node.op_type)
            for node in model.graph.node
            if (node.domain, node.op_type) in serialized
        } == {outer_key}
        assert {
            (node.domain, node.op_type)
            for node in serialized[outer_key].node
            if (node.domain, node.op_type) in serialized
        } == set(serialized) - {outer_key}
        for batch_index, (values, expected) in enumerate(
            zip(feeds, wanted, strict=True)
        ):
            for role, call in (
                ("eager", fn),
                ("same-warmed-jit", warmed),
                ("fresh-jit", jax.jit(_fresh(fn))),
            ):
                outputs = tuple(np.asarray(v).copy() for v in call(*values))
                _save(tmp_path / f"post-export-{batch_index}-{role}.npz", outputs)
                for actual, target in zip(outputs, expected, strict=True):
                    _same(actual, target)
        observed_layouts = set()
        for index, (definition, closed) in enumerate(built):
            body = _body(closed)
            manifest = typed_trace_manifest(closed)
            (tmp_path / f"function-{index}-trace.json").write_text(
                json.dumps(manifest, indent=2)
            )
            validate_source_jaxpr(body, 23)
            assert not body.constvars and not closed.consts
            carriers = [
                e.primitive for e in body.eqns if e.primitive in child_primitives
            ]
            if carriers:
                assert set(carriers) == child_primitives and len(carriers) == 2
            function = serialized[(definition.domain, definition.name)]
            infos = {v.name: v for v in function.value_info}
            for names, variables in (
                (function.input, body.invars),
                (function.output, body.outvars),
            ):
                for name, variable in zip(names, variables, strict=True):
                    shape = normalized_shape_dimensions(variable.aval.shape)
                    _port(infos[name], variable.aval.dtype, shape)
                    observed_layouts.add(shape)
            symbols = [
                atomic_symbolic_dimension(dim)
                for v in (*body.invars, *body.outvars)
                for dim in v.aval.shape
                if not isinstance(dim, int)
            ]
            assert symbols and all(
                label == "B" and scope is symbols[0][1] for label, scope in symbols
            )
            inputs = tuple(shape for _, shape, _ in _signature(body.invars))
            evaluator = jaxpr_as_fun(closed)
            for batch_index, ((x,), r) in enumerate(zip(feeds, raw, strict=True)):
                arguments = (r,) if inputs == ((64, "B"),) else (x, weight)
                expected = (
                    r if _signature(body.outvars)[0][1] == (64, "B") else r.T.copy()
                )
                for role, call in (
                    ("eager", evaluator),
                    ("fresh-jit", jax.jit(_fresh(evaluator))),
                ):
                    try:
                        outputs = tuple(np.asarray(v).copy() for v in call(*arguments))
                    except NotImplementedError as error:
                        first_line = str(error).splitlines()[0]
                        expected_errors = {
                            f"MLIR translation rule for primitive '{primitive.name}' not found for platform cpu"
                            for primitive in carriers
                        }
                        if (
                            role != "fresh-jit"
                            or not carriers
                            or first_line not in expected_errors
                        ):
                            raise
                        # Actual captured outer Function carriers have no native
                        # MLIR rule. Keep this evidence gap separate from the
                        # ordinary original-function/JIT and ORT comparisons.
                        (
                            tmp_path / f"function-{index}-{batch_index}-{role}-gap.json"
                        ).write_text(
                            json.dumps(
                                {
                                    "status": "NOT_ESTABLISHED",
                                    "role": role,
                                    "function": [definition.domain, definition.name],
                                    "native_trace_sha256": hashlib.sha256(
                                        json.dumps(manifest, sort_keys=True).encode()
                                    ).hexdigest(),
                                    "child_primitives": [p.name for p in carriers],
                                    "exception_type": type(error).__name__,
                                    "message": str(error),
                                    "obligation": "Whole captured carrier native JIT remains unfinished; original eager/warmed/fresh JIT and leaf captured-body JIT retain exact checks",
                                },
                                indent=2,
                            )
                        )
                        continue
                    _save(
                        tmp_path / f"function-{index}-{batch_index}-{role}.npz", outputs
                    )
                    assert len(outputs) == 1
                    _same(outputs[0], expected)
            assert typed_trace_manifest(closed) == manifest
        assert (64, "B") in observed_layouts and ("B", 64) in observed_layouts
        assert (weight.tobytes(), bias.tobytes()) == before


@pytest.mark.parametrize(
    "tail", [(3,), (2, 3), (2, 3, 4)], ids=["rank-two", "rank-three", "rank-four"]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zero_static_output_tails_keep_positive_input_binding(
    tail: Any, dtype: str, tmp_path: Path
) -> None:
    with jax.enable_x64(dtype == "float64"):
        weight = np.empty((tail[-1], 0), dtype)
        dimensions = (((len(tail),), (0,)), ((), ()))

        def fn(x: Any) -> Any:
            return jax.lax.dot_general(x, weight, dimensions)

        shape = ("B", *tail)
        feeds = [(_payload(dtype, (b, *tail)),) for b in (2, 5)]
        wanted = [(np.empty((b, *tail[:-1], 0), dtype),) for b in (2, 5)]
        _observe(
            fn,
            _specs([shape], dtype),
            feeds,
            wanted,
            tmp_path,
            output_shapes=[("B", *tail[:-1], 0)],
        )


@pytest.mark.parametrize("kind", ["M-zero", "internal-K-zero"])
def test_zero_free_axis_and_composed_zero_contraction_keep_exact_positive_zero(
    kind: str, tmp_path: Path
) -> None:
    empty_left, empty_right = np.empty((0, 3), np.float32), np.empty((3, 0), np.float32)
    continuation = np.empty((0, 2), np.float32)

    def fn(x: Any) -> Any:
        if kind == "M-zero":
            return jax.lax.transpose(
                jax.lax.dot_general(empty_left, x, _ORIENTATION), (1, 0)
            )
        empty = jax.lax.dot_general(x, empty_right, _MATRIX)
        return jax.lax.dot_general(empty, continuation, _MATRIX)

    feeds = [(_payload(np.float32, (b, 3)),) for b in (2, 5)]
    width = 0 if kind == "M-zero" else 2
    wanted = [(np.zeros((b, width), np.float32),) for b in (2, 5)]
    _observe(
        fn,
        _specs([("B", 3)], np.float32),
        feeds,
        wanted,
        tmp_path,
        output_shapes=[("B", width)],
    )


def _native_layout_trace() -> Any:
    weight = _payload(np.float32, (4, 3), right=True)

    def fn(x: Any) -> Any:
        return jax.lax.transpose(jax.lax.dot_general(weight, x, _ORIENTATION), (1, 0))

    return jax.make_jaxpr(fn)(*_specs([("B", 3)], np.float32))


def _isolated_graph(closed: Any) -> Any:
    """Preserve native avals and Var alias topology without editing the source trace."""
    body = _body(closed)
    variables: dict[Any, Any] = {}

    def clone(var: Any) -> Any:
        if not isinstance(var, Var):
            return var
        if var not in variables:
            variables[var] = Var(var.aval)
        return variables[var]

    return SimpleNamespace(
        invars=[clone(v) for v in body.invars],
        constvars=[clone(v) for v in body.constvars],
        outvars=[clone(v) for v in body.outvars],
        effects=body.effects,
        eqns=[
            SimpleNamespace(
                primitive=e.primitive,
                params=dict(e.params),
                invars=[clone(v) for v in e.invars],
                outvars=[clone(v) for v in e.outvars],
                effects=e.effects,
                ctx=e.ctx,
            )
            for e in body.eqns
        ],
    )


def _rebind(var: Any, **changes: Any) -> tuple[Any, Any]:
    original = var.aval
    state = original.shape, original.dtype, original.weak_type
    replacement = original.update(**changes)
    assert type(replacement) is jax.core.ShapedArray and replacement is not original
    # Rebind the same isolated native Var. Never write fields of a shared aval.
    var.aval = replacement
    assert (original.shape, original.dtype, original.weak_type) == state
    return original, state


@pytest.mark.parametrize("owner", ["dot_general", "transpose"])
@pytest.mark.parametrize(
    "fault",
    [
        "scope",
        "shape",
        "dtype",
        "weakness",
        "axes",
        "context",
        "effects",
        "identity",
        "unknown-param",
    ],
)
def test_native_symbolic_equations_reject_corruption_after_valid_baseline(
    owner: str, fault: str
) -> None:
    original = _native_layout_trace()
    original_manifest = typed_trace_manifest(original)
    graph = _isolated_graph(original)
    validate_source_jaxpr(_body(original), 23)
    validate_source_jaxpr(graph, 23)
    eqn = next(e for e in graph.eqns if e.primitive.name == owner)
    audit = None
    if fault in {"scope", "shape", "dtype", "weakness"}:
        shape = tuple(eqn.outvars[0].aval.shape)
        if fault == "scope":
            (foreign,) = export.symbolic_shape("B")
            shape = tuple(dim if isinstance(dim, int) else foreign for dim in shape)
            audit = _rebind(eqn.outvars[0], shape=shape)
        elif fault == "shape":
            shape = tuple(5 if dim == 4 else dim for dim in shape)
            audit = _rebind(eqn.outvars[0], shape=shape)
        elif fault == "dtype":
            audit = _rebind(eqn.outvars[0], dtype=np.float64)
        else:
            audit = _rebind(eqn.outvars[0], weak_type=True)
    elif fault == "axes":
        eqn.params["dimension_numbers" if owner == "dot_general" else "permutation"] = (
            (((1,), (0,)), ((), ())) if owner == "dot_general" else (0, 1)
        )
    elif fault == "context":
        eqn.ctx = SimpleNamespace(manager=object())
    elif fault == "effects":
        eqn.effects = {object()}
    elif fault == "identity":
        # A same-name primitive is only a hostile identity control, never evidence.
        eqn.primitive = Primitive(owner)
    else:
        eqn.params["unreviewed_semantics"] = object()
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(graph, 23)
    assert typed_trace_manifest(original) == original_manifest
    if audit is not None:
        aval, state = audit
        assert (aval.shape, aval.dtype, aval.weak_type) == state
    validate_source_jaxpr(_body(original), 23)


def test_original_transpose_evaluator_is_frozen_against_current_method_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _native_layout_trace()
    graph = _isolated_graph(original)
    validate_source_jaxpr(graph, 23)
    eqn = next(e for e in graph.eqns if e.primitive is jax.lax.transpose_p)
    called = []

    def forged(*args: Any, **kwargs: Any) -> Any:
        called.append(True)
        return eqn.outvars[0].aval, frozenset()

    with monkeypatch.context() as patch:
        patch.setitem(vars(eqn.primitive), "abstract_eval", forged)
        validate_source_jaxpr(graph, 23)
        held, state = _rebind(
            eqn.outvars[0], shape=tuple(reversed(eqn.outvars[0].aval.shape))
        )
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(graph, 23)
        assert not called
        assert (held.shape, held.dtype, held.weak_type) == state
    validate_source_jaxpr(_body(original), 23)

    with monkeypatch.context() as patch:
        patch.setitem(vars(eqn.primitive), "multiple_results", True)
        with pytest.raises(SourcePrecisionError):
            validate_source_jaxpr(_body(original), 23)
    validate_source_jaxpr(_body(original), 23)


@pytest.mark.parametrize("owner", ["add", "neg", "rev", "squeeze", "slice", "reshape"])
def test_unrelated_owners_do_not_inherit_nonleading_symbolic_admission(
    owner: str,
) -> None:
    tail = (1, 3) if owner == "squeeze" else (2, 3) if owner == "reshape" else (3,)
    specs = _specs([("B", *tail)], np.float32)
    order = (
        (1, 2, 0) if owner == "squeeze" else (1, 0, 2) if owner == "reshape" else (1, 0)
    )

    def baseline(x: Any) -> Any:
        return jax.lax.transpose(x, order)

    # This is an actual native transpose, carrying the same formal B/scope.
    validate_source_jaxpr(_body(jax.make_jaxpr(baseline)(*specs)), 23)

    def fn(x: Any) -> Any:
        moved = baseline(x)
        if owner == "add":
            return jax.lax.add(moved, moved)
        if owner == "neg":
            return jax.lax.neg(moved)
        if owner == "rev":
            return jax.lax.rev(moved, (0,))
        if owner == "squeeze":
            return jax.lax.squeeze(moved, (0,))
        if owner == "slice":
            return jax.lax.slice(moved, (0, 0), (2, moved.shape[1]))
        return jax.lax.reshape(moved, (*moved.shape, 1))

    trace = jax.make_jaxpr(fn)(*specs)
    assert any(e.primitive.name == owner for e in _body(trace).eqns)
    with pytest.raises(SourcePrecisionError, match="no audited owner") as caught:
        validate_source_jaxpr(_body(trace), 23)
    assert caught.value.primitive == owner


def _lowering_fixture(eqn: Any) -> tuple[Any, Any, Any, Any]:
    from jax2onnx.converter.ir_builder import IRBuilder

    builder = IRBuilder(
        opset=23, enable_double_precision=False, precision_policy=PrecisionPolicy.SOURCE
    )
    values = [
        ir.val(
            f"input{i}",
            dtype=ir.DataType.FLOAT,
            shape=normalized_shape_dimensions(v.aval.shape),
        )
        for i, v in enumerate(eqn.invars)
    ]
    output = ir.val(
        "output",
        dtype=ir.DataType.FLOAT,
        shape=normalized_shape_dimensions(eqn.outvars[0].aval.shape),
    )
    bound, allocations = [], []

    def allocate(*args: Any, **kwargs: Any) -> Any:
        allocations.append(True)
        return output

    ctx = SimpleNamespace(
        precision_policy=PrecisionPolicy.SOURCE,
        builder=builder,
        fresh_name=builder.fresh_name,
        require_value_for_var=lambda var: values[eqn.invars.index(var)],
        allocate_value_for_var=allocate,
        bind_value_for_var=lambda var, value: bound.append((var, value)),
    )
    return ctx, values, output, (bound, allocations)


@pytest.mark.parametrize("owner", ["dot_general", "transpose"])
@pytest.mark.parametrize("port", ["input-zero", "input-last", "output"])
@pytest.mark.parametrize("fault", ["dtype", "shape", "missing", "symbol"])
def test_canonical_physical_ports_reject_corruption_before_emission(
    owner: str, port: str, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _native_layout_trace()
    eqn = next(e for e in _body(original).eqns if e.primitive.name == owner)
    site = _Site("native-symbolic-physical-baseline", 23, owner)
    validator = _validate_dot if owner == "dot_general" else _validate_tensor_layout
    validator(
        eqn,
        [_tensor(v, site) for v in eqn.invars],
        [_tensor(v, site) for v in eqn.outvars],
        site,
    )
    lower = (
        DotGeneralPlugin()._lower_source
        if owner == "dot_general"
        else TransposePlugin().lower
    )
    valid, valid_values, valid_output, (valid_bound, _) = _lowering_fixture(eqn)
    assert all(
        representation_for_aval(var.aval).matches_ir_value(value)
        for var, value in zip(
            (*eqn.invars, *eqn.outvars), (*valid_values, valid_output), strict=True
        )
    )
    lower(valid, eqn)
    assert len(valid_bound) == 1 and valid_bound[0][1] is valid_output
    assert tuple(valid.builder.graph)
    ctx, values, output, (bound, allocations) = _lowering_fixture(eqn)
    selected = output if port == "output" else values[0 if port == "input-zero" else -1]
    before = [(value.type, value.shape) for value in (*values, output)]
    if fault == "dtype":
        selected.type = ir.TensorType(ir.DataType.DOUBLE)
    elif fault == "missing":
        selected.shape = None
    else:
        shape = tuple(selected.shape)
        selected.shape = ir.Shape(
            tuple("Other" if isinstance(dim, ir.SymbolicDim) else dim for dim in shape)
            if fault == "symbol"
            else (*shape[:-1], 7)
        )
        if fault == "symbol" and selected.shape == ir.Shape(shape):
            # The first dot operand is static: injecting a label is still a mismatch.
            selected.shape = ir.Shape(("Other", *shape[1:]))
    mutated = [(value.type, value.shape) for value in (*values, output)]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid physical descriptor reached node emission")

    monkeypatch.setattr(ctx.builder, "op", forbidden)
    with pytest.raises(ValueError, match="representation|physical.*dtype/shape"):
        lower(ctx, eqn)
    assert not bound
    assert len(allocations) == int(port == "output")
    assert mutated == [(value.type, value.shape) for value in (*values, output)]
    assert before != mutated


def _zero_artifact() -> tuple[Any, Any]:
    weight = np.empty((3, 0), np.float32)

    def fn(x: Any) -> Any:
        return jax.lax.dot_general(x, weight, _MATRIX)

    model = ui._to_onnx(
        fn, _specs([("B", 3)], np.float32), precision_policy=PrecisionPolicy.SOURCE
    )
    onnx.checker.check_model(model, full_check=True)
    sample = SourceSample(
        (_payload(np.float32, (2, 3)),), (np.empty((2, 0), np.float32),), _BITWISE
    )
    validate_source_samples(
        model, samples=[sample], optimization="disabled"
    ).require_sample_acceptance()
    return model, sample


def _adapt_zero_output(model: Any, operation: str) -> None:
    """Change the actual output tensor through a schema-valid operation."""
    output = model.graph.output[0]
    public_name, internal_name = output.name, output.name + "_before_adapter"
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == public_name:
                node.input[index] = internal_name
        for index, name in enumerate(node.output):
            if name == public_name:
                node.output[index] = internal_name
    for value in model.graph.value_info:
        if value.name == public_name:
            value.name = internal_name
    if operation == "Cast":
        node = onnx.helper.make_node(
            "Cast", [internal_name], [public_name], to=onnx.TensorProto.DOUBLE
        )
        output.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif operation == "Pad":
        pads = onnx.numpy_helper.from_array(
            np.asarray([0, 0, 0, 1], np.int64), name="output_adapter_pads"
        )
        model.graph.initializer.append(pads)
        node = onnx.helper.make_node("Pad", [internal_name, pads.name], [public_name])
        output.type.tensor_type.shape.dim[1].dim_value = 1
    else:
        node = onnx.helper.make_node(
            "Transpose", [internal_name], [public_name], perm=[1, 0]
        )
        output.type.tensor_type.shape.dim[0].dim_value = 0
        output.type.tensor_type.shape.dim[1].dim_param = "B"
    model.graph.node.append(node)


@pytest.mark.parametrize(
    "fault",
    [
        "output-symbol",
        "output-concrete",
        "output-missing",
        "output-tail",
        "output-dtype",
        "report-traced",
        "report-exported",
        "report-symbol",
        "input-zero-tail",
        "no-input-binding",
        "nonleading-output",
    ],
)
def test_zero_tail_output_requires_exact_report_artifact_and_input_B_binding(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnxruntime as ort

    model, sample = _zero_artifact()
    report = get_precision_report(model)
    assert report is not None
    metadata = json.loads(precision_report_to_json(report))
    tensor = model.graph.output[0].type.tensor_type
    if fault == "output-symbol":
        tensor.shape.dim[0].dim_param = "Other"
    elif fault == "output-concrete":
        tensor.shape.dim[0].dim_value = 2
    elif fault == "output-missing":
        tensor.shape.dim[0].Clear()
    elif fault == "output-tail":
        _adapt_zero_output(model, "Pad")
    elif fault == "output-dtype":
        _adapt_zero_output(model, "Cast")
    elif fault in {"report-traced", "report-exported", "report-symbol"}:
        keys = (
            ("traced_outputs", "exported_outputs")
            if fault == "report-symbol"
            else (
                ("traced_outputs",)
                if fault == "report-traced"
                else ("exported_outputs",)
            )
        )
        for key in keys:
            metadata[key][0]["shape"][0 if fault == "report-symbol" else 1] = (
                "Other" if fault == "report-symbol" else 1
            )
    elif fault in {"input-zero-tail", "no-input-binding"}:
        axis, dimension = (1, 0) if fault == "input-zero-tail" else (0, 2)
        model.graph.input[0].type.tensor_type.shape.dim[axis].dim_value = dimension
        for key in ("supplied", "normalized", "traced", "exported"):
            metadata["inputs"][0][key]["shape"][axis] = dimension
    else:
        _adapt_zero_output(model, "Transpose")
        # Nonleading output now has a positive companion. Preserve this old ID
        # as a real traced/exported-axis mismatch without layout provenance.
        metadata["exported_outputs"][0]["shape"] = [0, "B"]
    for item in model.metadata_props:
        if item.key == PRECISION_METADATA_KEY:
            item.value = json.dumps(metadata)
    onnx.checker.check_model(model, full_check=True)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid empty-output declaration reached a runtime session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(model, samples=[sample], optimization="disabled")
    assert any(
        c.category == "source-shape-contract" and c.status is CheckStatus.FAILED
        for c in result.checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


@pytest.mark.parametrize(
    "fault", ["B-zero", "reference-B", "reference-tail", "input-K-zero"]
)
def test_empty_outputs_cannot_hide_invalid_sample_bindings(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnxruntime as ort

    model, sample = _zero_artifact()
    inputs, wanted = sample.inputs, sample.expected_outputs
    if fault == "B-zero":
        inputs, wanted = (np.empty((0, 3), np.float32),), (
            np.empty((0, 0), np.float32),
        )
    elif fault == "reference-B":
        wanted = (np.empty((5, 0), np.float32),)
    elif fault == "reference-tail":
        wanted = (np.empty((2, 0, 1), np.float32),)
    else:
        inputs = (np.empty((2, 0), np.float32),)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("invalid empty sample reached a runtime session")

    monkeypatch.setattr(ort, "InferenceSession", forbidden)
    result = validate_source_samples(
        model, samples=[SourceSample(inputs, wanted, _BITWISE)], optimization="all"
    )
    assert result.samples[0].status is CheckStatus.FAILED
    assert any(
        c.category == "sample-input" and c.status is CheckStatus.FAILED
        for c in result.samples[0].checks
    )


def test_actual_empty_runtime_output_must_retain_the_input_bound_positive_B(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import onnxruntime as ort

    model, sample = _zero_artifact()
    constructor = ort.InferenceSession

    def session(*args: Any, **kwargs: Any) -> Any:
        actual = constructor(*args, **kwargs)
        actual.run = lambda *a, **k: [np.empty((5, 0), np.float32)]
        return actual

    monkeypatch.setattr(ort, "InferenceSession", session)
    result = validate_source_samples(model, samples=[sample], optimization="disabled")
    assert any(
        c.category == "sample-output" and c.status is CheckStatus.FAILED
        for c in result.samples[0].checks
    )
    with pytest.raises(SampleAcceptanceError):
        result.require_sample_acceptance()


@pytest.mark.parametrize("dtype,x64", _WIDTHS)
def test_native_nonleading_final_export_keeps_sample_boundary_unfinished(
    dtype: str,
    x64: bool,
    tmp_path: Path,
) -> None:
    """Retain the historical test ID; native nonleading output now requires sample acceptance.

    The frozen preceding packet retains its unavailable-boundary expectation.
    This incremental positive does not imply that expectation was executed.
    """
    with jax.enable_x64(x64):
        weight = _payload(dtype, (64, 128), right=True)

        def fn(x: Any) -> Any:
            return jax.lax.dot_general(weight, x, _ORIENTATION)

        feeds = [(_payload(dtype, (batch, 128)),) for batch in (2, 5)]
        expected = [
            (_oracle(weight, value, _ORIENTATION, dtype),) for (value,) in feeds
        ]
        model, trace, _ = _observe(
            fn,
            _specs([("B", 128)], dtype),
            feeds,
            expected,
            tmp_path,
            output_shapes=[(64, "B")],
        )
        dot = next(e for e in _body(trace).eqns if e.primitive is jax.lax.dot_general_p)
        assert dot.params["dimension_numbers"] == _ORIENTATION
        assert _signature(dot.outvars) == ((dtype, (64, "B"), False),)
        _port(model.graph.output[0], dtype, (64, "B"))


@pytest.mark.parametrize(
    "geometry", ["contracting-B", "repeated-output-B", "multiple-symbols"]
)
def test_unfinished_native_symbolic_geometry_retains_real_native_evidence(
    geometry: str,
    tmp_path: Path,
) -> None:
    validate_source_jaxpr(_body(_native_layout_trace()), 23)
    batch, other = export.symbolic_shape("B,C")
    assert batch.scope is other.scope
    if geometry == "contracting-B":
        shapes, dimensions = ((batch, 3), (batch, 4)), (((0,), (0,)), ((), ()))
    else:
        shapes, dimensions = (
            (batch, 3),
            (other if geometry == "multiple-symbols" else batch, 3),
        ), (((1,), (1,)), ((), ()))

    def fn(left: Any, right: Any) -> Any:
        return jax.lax.dot_general(left, right, dimensions)

    trace = jax.make_jaxpr(fn)(
        *(jax.ShapeDtypeStruct(shape, np.float32) for shape in shapes)
    )
    (tmp_path / "unfinished-native-trace.txt").write_text(str(trace))
    for size in (2, 5):
        concrete = [
            tuple(size if not isinstance(dim, int) else dim for dim in shape)
            for shape in shapes
        ]
        values = (
            _payload(np.float32, concrete[0]),
            _payload(np.float32, concrete[1], right=True),
        )
        wanted = _oracle(*values, dimensions, np.float32)
        for role, call in (("eager", fn), ("fresh-jit", jax.jit(_fresh(fn)))):
            output = np.asarray(call(*values)).copy()
            _save(tmp_path / f"{size}-{role}.npz", (output,))
            _same(output, wanted)
    with pytest.raises(SourcePrecisionError):
        validate_source_jaxpr(_body(trace), 23)


@pytest.mark.parametrize("owner", ["add", "neg", "rev", "reshape", "broadcast_in_dim"])
def test_unrelated_owners_do_not_inherit_zero_tail_intermediate_admission(
    owner: str,
) -> None:
    weight = np.empty((3, 0), np.float32)
    specs = _specs([("B", 3)], np.float32)

    def baseline(x: Any) -> Any:
        return jax.lax.dot_general(x, weight, _MATRIX)

    validate_source_jaxpr(_body(jax.make_jaxpr(baseline)(*specs)), 23)

    def fn(x: Any) -> Any:
        empty = baseline(x)
        if owner == "add":
            return jax.lax.add(empty, empty)
        if owner == "neg":
            return jax.lax.neg(empty)
        if owner == "rev":
            return jax.lax.rev(empty, (0,))
        if owner == "reshape":
            return jax.lax.reshape(empty, (*empty.shape, 1))
        return jax.lax.broadcast_in_dim(empty, (empty.shape[0], 2, 0), (0, 2))

    trace = jax.make_jaxpr(fn)(*specs)
    assert any(e.primitive.name == owner for e in _body(trace).eqns)
    with pytest.raises(SourcePrecisionError, match="no audited owner") as caught:
        validate_source_jaxpr(_body(trace), 23)
    assert caught.value.primitive == owner


@pytest.mark.parametrize("dtype", ["bool", "int32", "uint32", "float16", "bfloat16"])
def test_exact_nonleading_transpose_preserves_other_existing_real_storage_types(
    dtype: str,
    tmp_path: Path,
) -> None:
    def fn(x: Any) -> Any:
        return jax.lax.transpose(x, (1, 0))

    specs = _specs([("B", 3)], np.dtype(dtype))
    trace = jax.make_jaxpr(fn)(*specs)
    validate_source_jaxpr(_body(trace), 23)
    (tmp_path / "native-trace.txt").write_text(str(trace))
    for batch in (2, 5):
        if dtype == "bool":
            value = np.resize(np.asarray([False, True, True], np.bool_), (batch, 3))
        elif dtype == "int32":
            value = np.resize(
                np.asarray([-(2**31), -1, 0, 1, 2**31 - 1], np.int32), (batch, 3)
            )
        elif dtype == "uint32":
            value = np.resize(np.asarray([0, 2**31, 2**32 - 1], np.uint32), (batch, 3))
        else:
            value = _payload(np.dtype(dtype), (batch, 3))
        wanted = value.T.copy()
        for role, call in (("eager", fn), ("fresh-jit", jax.jit(_fresh(fn)))):
            result = np.asarray(call(value)).copy()
            _save(tmp_path / f"{batch}-{role}.npz", (result,))
            _same(result, wanted)
    model = ui._to_onnx(fn, specs, precision_policy=PrecisionPolicy.SOURCE)
    onnx.checker.check_model(model, full_check=True)
    report = get_precision_report(model)
    assert report is not None
    assert report.exported_outputs[0].shape == (3, "B")
    assert report.exported_outputs[0].dtype == dtype
    _port(model.graph.input[0], np.dtype(dtype), ("B", 3))
    _port(model.graph.output[0], np.dtype(dtype), (3, "B"))
    (tmp_path / "transpose.onnx").write_bytes(model.SerializeToString())
    (tmp_path / "transpose-report.json").write_text(precision_report_to_json(report))
    assert {c.category: c.status for c in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
