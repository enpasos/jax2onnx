# tests/extra_tests/converter/test_source_dropout_parameters.py

"""Actual framework Dropout operands; stochastic RNG equivalence is not claimed."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from numbers import Real
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import equinox as eqx
from flax.errors import ModifyScopeVariableError
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import onnx_function
from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import ShapedArray
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.flax.nnx.dropout import DropoutPlugin
from jax2onnx.plugins.flax.linen.spectral_norm import _SpectralNormDense
from jax2onnx.plugins.plugin_system import (
    _DynamicParamRef,
    _StaticParam,
    import_all_plugins,
)
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter import source_precision_diagnostics as diagnostics


class DropoutDiagnosticValidator(diagnostics.DiagnosticValidator):
    """The existing finite diagnostic families plus three exact Dropout schemas."""

    _rates = {
        "nnx.dropout": "rate",
        "linen.dropout": "rate",
        "eqx.nn.dropout": "p",
    }

    def __call__(self, jaxpr: Any, opset: int) -> None:
        rules = {
            **diagnostics._RULES,
            **{
                name: (2, 1, frozenset({rate, "call_time"}))
                for name, rate in self._rates.items()
            },
        }
        with patch.object(diagnostics, "_RULES", rules):
            super().__call__(jaxpr, opset)

    def _ordinary(self, eqn: Any, site: str, opset: int) -> None:
        name = eqn.primitive.name
        if name in self._rates:
            if len(eqn.invars) != 2 or len(eqn.outvars) != 1:
                self._fail(
                    "Dropout requires two operands and one output", site, opset, name
                )
            value, flag = (self._tensor(var, site, opset) for var in eqn.invars)
            output = self._tensor(eqn.outvars[0], site, opset)
            if (
                value[0] not in {"float16", "bfloat16", "float32", "float64"}
                or output != value
            ):
                self._fail(
                    "Dropout must preserve its floating input signature",
                    site,
                    opset,
                    name,
                )
            if flag[:2] != ("bool", ()):
                self._fail("Dropout flag must be BOOL[]", site, opset, name)
            rate_name = self._rates[name]
            if set(eqn.params) != {rate_name, "call_time"}:
                self._fail(
                    "Unreviewed Dropout parameter schema",
                    site,
                    opset,
                    name,
                    pending=True,
                )
            rate = eqn.params[rate_name]
            if (
                isinstance(rate, (bool, np.bool_))
                or not isinstance(rate, Real)
                or not np.isfinite(rate)
                or not 0 <= rate < 1
            ):
                self._fail(
                    "Dropout rate must be finite and in [0, 1)", site, opset, name
                )
            if type(eqn.params["call_time"]) is not bool:
                self._fail("Dropout call_time must be a Python BOOL", site, opset, name)
        super()._ordinary(eqn, site, opset)

    def _function(self, eqn: Any, site: str, opset: int) -> None:
        reserved = {"__jax2onnx_source_trace", "instance_key"}
        dynamic = []
        for name, value in eqn.params.items():
            if name in reserved:
                continue
            if isinstance(value, _DynamicParamRef):
                if (
                    type(value.index) is not int
                    or not 0 <= value.index < len(eqn.invars)
                    or self._tensor(eqn.invars[value.index], site, opset)[:2]
                    != ("bool", ())
                ):
                    self._fail(
                        "Function keyword is not an actual BOOL[] operand",
                        site,
                        opset,
                        eqn.primitive.name,
                    )
                dynamic.append(value.index)
            elif isinstance(value, _StaticParam) and type(value.value) is bool:
                continue
            elif (
                isinstance(value, _StaticParam)
                and eqn.primitive.name == f"onnx_fn::{__name__}._dropout_named"
                and name == "target"
                and callable(value.value)
                and value.value.__module__ == __name__
                and value.value.__qualname__.startswith("_target.<locals>.")
            ):
                # Only this test factory's three explicit framework adapters.
                continue
            else:
                self._fail(
                    "Unreviewed Function keyword parameter",
                    site,
                    opset,
                    eqn.primitive.name,
                    pending=True,
                )
        if sorted(dynamic) != list(
            range(len(eqn.invars) - len(dynamic), len(eqn.invars))
        ):
            self._fail(
                "Function keyword operands are not the exact trailing bindings",
                site,
                opset,
                eqn.primitive.name,
            )
        checked = SimpleNamespace(
            primitive=eqn.primitive,
            invars=eqn.invars,
            outvars=eqn.outvars,
            params={
                name: value for name, value in eqn.params.items() if name in reserved
            },
        )
        super()._function(checked, site, opset)


@pytest.fixture(autouse=True)
def _configuration() -> Iterator[None]:
    previous = {
        name: getattr(jax.config, name)
        for name in (
            "jax_enable_x64",
            "jax_platforms",
            "jax_default_matmul_precision",
            "jax_numpy_dtype_promotion",
            "jax_numpy_rank_promotion",
        )
    }
    import_all_plugins()
    try:
        yield
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


@onnx_function
def _dropout_named(
    x: Any, *, target: Callable[..., Any], deterministic: Any = True
) -> Any:
    return target(x, deterministic)


def _target(framework: str, rate: float) -> Callable[..., Any]:
    if framework == "nnx":
        block = nnx.Dropout(rate, deterministic=True, rngs=nnx.Rngs(31))
        return lambda x, flag: block(x, deterministic=flag)
    if framework == "linen":
        # This unchanged Linen module receives a fixed external bridge key.
        block = nn.Dropout(rate)
        key = jax.random.key(31)
        return lambda x, flag: block.apply(
            {}, x, deterministic=flag, rngs={"dropout": key}
        )
    block = eqx.nn.Dropout(p=rate, inference=True)
    key = jax.random.key(31)
    return lambda x, flag: block(x, inference=flag, key=key)


def _model(target: Callable[..., Any], scope: str) -> Callable[..., Any]:
    def model(x: Any, *, runtime: Any = True) -> Any:
        if scope == "root":
            return target(x, runtime)
        if scope == "function":
            return _dropout_named(x, target=target, deterministic=runtime)
        if scope == "if":
            return jax.lax.cond(
                jax.lax.reduce_sum(x, axes=(0,)) > np.float32(0),
                lambda value: _dropout_named(
                    value, target=target, deterministic=runtime
                ),
                lambda value: -_dropout_named(
                    -value, target=target, deterministic=runtime
                ),
                x,
            )
        return jax.lax.fori_loop(
            0,
            2,
            lambda index, value: _dropout_named(
                value, target=target, deterministic=runtime
            ),
            x,
        )

    return model


class DropoutFlow:
    """Expand actual Function/If/Loop bindings before following control dependencies."""

    def __init__(self, model: onnx.ModelProto):
        self.functions = {(fn.domain, fn.name): fn for fn in model.functions}
        self.edges: dict[str, set[str]] = {}
        self.constants: dict[str, np.ndarray] = {}
        self.controls: list[dict[str, Any]] = []
        self.outputs = self._graph(model.graph, "root", {})

    def _edge(self, target: str, *sources: str) -> None:
        self.edges.setdefault(target, set()).update(
            source for source in sources if source
        )

    def _graph(
        self,
        graph: Any,
        scope: str,
        outer: dict[str, str],
        arguments: list[str] | None = None,
    ) -> list[str]:
        function = isinstance(graph, onnx.FunctionProto)
        inputs = (
            list(graph.input) if function else [value.name for value in graph.input]
        )
        outputs = (
            list(graph.output) if function else [value.name for value in graph.output]
        )
        initializers = [] if function else list(graph.initializer)
        names = (
            set(inputs)
            | {value.name for value in initializers}
            | {name for node in graph.node for name in node.output if name}
        )
        env = {**outer, **{name: f"{scope}/{name}" for name in names}}

        def resolve(name: str) -> str:
            assert not name or name in env, f"Unbound serialized operand {scope}/{name}"
            return env[name] if name else ""

        if arguments is not None:
            assert len(inputs) == len(arguments)
            for name, argument in zip(inputs, arguments, strict=True):
                self._edge(resolve(name), argument)
        for value in initializers:
            self.constants[resolve(value.name)] = onnx.numpy_helper.to_array(value)
        for index, node in enumerate(graph.node):
            ins = [resolve(name) for name in node.input]
            outs = [resolve(name) for name in node.output]
            path = f"{scope}/{index}:{node.op_type}"
            key = node.domain, node.op_type
            if key in self.functions:
                returned = self._graph(self.functions[key], path, {}, ins)
                for output, returned_value in zip(outs, returned, strict=True):
                    self._edge(output, returned_value)
            elif node.op_type == "If":
                branches = [
                    self._graph(attr.g, path + "/" + attr.name, env)
                    for attr in node.attribute
                    if attr.type == onnx.AttributeProto.GRAPH
                ]
                assert len(branches) == 2
                for position, output in enumerate(outs):
                    self._edge(
                        output, ins[0], *(branch[position] for branch in branches)
                    )
            elif node.op_type == "Loop":
                body = next(attr.g for attr in node.attribute if attr.name == "body")
                returned = self._graph(
                    body, path + "/body", env, [path + "/iteration", *ins[1:]]
                )
                assert len(body.input) == len(ins)
                assert len(returned) == len(outs) + 1
                for formal, returned_value in zip(
                    body.input[1:], returned, strict=True
                ):
                    self._edge(path + "/body/" + formal.name, returned_value)
                for output, returned_value in zip(outs, returned[1:], strict=True):
                    self._edge(output, returned_value, *ins[:2])
            else:
                for output in outs:
                    self._edge(output, *ins)
                if node.op_type == "Constant":
                    attribute = next(
                        attr for attr in node.attribute if attr.name == "value"
                    )
                    self.constants[outs[0]] = onnx.numpy_helper.to_array(attribute.t)
                elif (
                    node.op_type in {"Identity", "Cast", "Not"}
                    and ins[0] in self.constants
                ):
                    value = self.constants[ins[0]]
                    self.constants[outs[0]] = (
                        np.logical_not(value) if node.op_type == "Not" else value
                    )
                if node.op_type == "Dropout":
                    assert len(ins) == 3
                    producer = next(
                        (item for item in graph.node if node.input[2] in item.output),
                        None,
                    )
                    self.controls.append(
                        dict(
                            path=path,
                            training=ins[2],
                            training_operator=(
                                producer.op_type if producer is not None else None
                            ),
                            outputs=outs,
                            rate=float(self.constants[ins[1]]),
                        )
                    )
        return [resolve(name) for name in outputs]

    def reach(self, roots: list[str]) -> set[str]:
        seen: set[str] = set()
        pending = list(roots)
        while pending:
            value = pending.pop()
            if value not in seen:
                seen.add(value)
                pending.extend(self.edges.get(value, ()))
        return seen


def _export(
    model: Callable[..., Any], params: dict[str, Any]
) -> tuple[onnx.ModelProto, DropoutDiagnosticValidator]:
    observer = DropoutDiagnosticValidator()
    artifact = ui._to_onnx(
        model,
        [jax.ShapeDtypeStruct((32,), np.float32)],
        input_params=params,
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=observer,
        input_names=["features"],
        output_names=["result"],
    )
    onnx.checker.check_model(artifact, full_check=True)
    report = get_precision_report(artifact)
    assert report is not None and report.profile == "source-diagnostic-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.UNVERIFIED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    return artifact, observer


def _session(model: onnx.ModelProto, optimize: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
@pytest.mark.parametrize("scope", ["root", "function", "if", "loop"])
@pytest.mark.parametrize("rate", [0.5, 0.0], ids=["positive-rate", "zero-rate"])
def test_dropout_uses_real_named_operand_through_nested_scope(
    framework: str, scope: str, rate: float
) -> None:
    model = _model(_target(framework, rate), scope)
    original = np.asarray(model(jnp.ones(32, np.float32), runtime=True))
    np.testing.assert_array_equal(original, np.ones(32, np.float32), strict=True)
    proto, observer = _export(model, {"runtime": True})
    assert observer.scopes
    assert [value.name for value in proto.graph.input] == ["features", "runtime"]
    flag = proto.graph.input[1].type.tensor_type
    assert flag.elem_type == onnx.TensorProto.BOOL and not flag.shape.dim
    flow = DropoutFlow(proto)
    assert flow.controls
    live = flow.reach(flow.outputs)
    assert all(
        any(output in live for output in control["outputs"])
        for control in flow.controls
    )
    dynamic = [
        control
        for control in flow.controls
        if "root/runtime" in flow.reach([control["training"]])
    ]
    if framework == "eqx" and rate == 0:
        assert not dynamic  # Existing p==0 normalization is flag-insensitive.
    else:
        assert len(dynamic) == len(flow.controls)
    assert all(control["training_operator"] == "Not" for control in dynamic)
    assert all(control["rate"] == rate for control in flow.controls)
    for optimize in (False, True):
        session = _session(proto, optimize)
        for inference in (True, False):
            (actual,) = session.run(
                None,
                {"features": np.ones(32, np.float32), "runtime": np.asarray(inference)},
            )
            assert actual.dtype == np.dtype(np.float32) and actual.shape == (32,)
            assert np.isfinite(actual).all()
            if inference or rate == 0:
                np.testing.assert_array_equal(actual, original, strict=True)
            else:
                # Sample properties only; neither a differing-output nor RNG-mask oracle.
                allowed = [0, 4] if scope == "loop" else [0, 2]
                assert np.isin(actual, np.asarray(allowed, np.float32)).all()
    assert get_precision_report(proto).checks[-1].status is CheckStatus.UNVERIFIED


@pytest.mark.parametrize("framework", ["nnx", "linen", "eqx"])
@pytest.mark.parametrize("local", ["default", True, False])
def test_dropout_nested_default_and_literal_override_ignore_same_named_root_sample(
    framework: str, local: Any
) -> None:
    target = _target(framework, 0.5)

    def model(x: Any, *, deterministic: Any = False) -> Any:
        if local == "default":
            return _dropout_named(x, target=target)
        return _dropout_named(x, target=target, deterministic=local)

    proto, _ = _export(model, {"deterministic": False})
    flow = DropoutFlow(proto)
    assert flow.controls
    for control in flow.controls:
        assert "root/deterministic" not in flow.reach([control["training"]])
        expected = False if local == "default" else not local
        assert bool(flow.constants[control["training"]]) is expected


@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_original_nnx_dropout_rng_counter_prerequisite(rate: float) -> None:
    assert nnx.Dropout.__call__.__module__.startswith("flax.nnx.")
    block = nnx.Dropout(rate, rngs=nnx.Rngs(17))

    def count() -> int:
        return int(np.asarray(block.rngs.count.value))

    before = count()
    inference = np.asarray(block(jnp.ones(32, np.float32), deterministic=True))
    assert count() == before
    np.testing.assert_array_equal(inference, np.ones(32, np.float32), strict=True)
    training = np.asarray(block(jnp.ones(32, np.float32), deterministic=False))
    assert count() == before + int(rate > 0)
    assert np.isin(training, [0, 2] if rate else [1]).all()


def test_original_nnx_training_requires_rng_state() -> None:
    with pytest.raises(ValueError, match="rngs"):
        nnx.Dropout(0.5)(jnp.ones(32, np.float32), deterministic=False)


def test_original_linen_fixed_external_key_is_repeatable() -> None:
    assert nn.Dropout.__call__.__module__.startswith("flax.linen.")
    target = _target("linen", 0.5)
    x = jnp.ones(32, np.float32)
    np.testing.assert_array_equal(target(x, True), x, strict=True)
    first, repeated = np.asarray(target(x, False)), np.asarray(target(x, False))
    np.testing.assert_array_equal(first, repeated, strict=True)
    assert np.isin(first, [0, 2]).all()


def test_original_equinox_missing_key_and_external_key_preparation() -> None:
    assert eqx.nn.Dropout.__call__.__module__.startswith("equinox.")
    block = eqx.nn.Dropout(p=0.5)
    x = jnp.ones(32, np.float32)
    np.testing.assert_array_equal(block(x, inference=np.asarray(True)), x, strict=True)
    with pytest.raises(RuntimeError, match="key"):
        block(x, inference=np.asarray(False))
    training = np.asarray(block(x, inference=False, key=jax.random.key(9)))
    assert training.dtype == np.dtype(np.float32) and training.shape == (32,)
    assert np.isin(training, [0, 2]).all()


def test_original_spectral_norm_statistics_require_mutable_state() -> None:
    block = _SpectralNormDense(features=4, n_steps=1)
    x = jnp.arange(8, dtype=np.float32).reshape(1, 8)
    variables = block.init(jax.random.key(17), x, update_stats=False)
    fixed = block.apply(variables, x, update_stats=False)
    assert fixed.shape == (1, 4) and np.isfinite(np.asarray(fixed)).all()
    with pytest.raises(ModifyScopeVariableError):
        block.apply(variables, x, update_stats=True)
    _, updated = block.apply(variables, x, update_stats=True, mutable=["batch_stats"])
    before = jax.tree_util.tree_leaves(variables["batch_stats"])
    after = jax.tree_util.tree_leaves(updated["batch_stats"])
    assert any(
        not np.array_equal(np.asarray(old), np.asarray(new))
        for old, new in zip(before, after, strict=True)
    )


def test_dropout_validator_keeps_unknown_registered_operations_pending() -> None:
    original_rules = diagnostics._RULES
    graph = jax.make_jaxpr(jax.lax.exp)(jax.ShapeDtypeStruct((2,), np.float32))
    with pytest.raises(
        SourcePrecisionError, match="no explicit diagnostic structural family"
    ) as caught:
        DropoutDiagnosticValidator()(graph, 23)
    assert caught.value.category == "profile-exclusion"
    assert diagnostics._RULES is original_rules
    assert "nnx.dropout" not in diagnostics._RULES


@pytest.mark.parametrize(
    "defect",
    [
        "arity",
        "flag-dtype",
        "flag-shape",
        "output",
        "params",
        "rate-nan",
        "rate-one",
        "call-time",
    ],
)
def test_dropout_validator_rejects_malformed_explicit_schema(defect: str) -> None:
    def var(shape: tuple[int, ...], dtype: Any) -> Any:
        return SimpleNamespace(aval=ShapedArray(shape, dtype))

    eqn = SimpleNamespace(
        primitive=DropoutPlugin._PRIM,
        invars=[var((32,), np.float32), var((), np.bool_)],
        outvars=[var((32,), np.float32)],
        params={"rate": 0.5, "call_time": True},
    )
    if defect == "arity":
        eqn.invars.pop()
    elif defect == "flag-dtype":
        eqn.invars[1] = var((), np.int32)
    elif defect == "flag-shape":
        eqn.invars[1] = var((1,), np.bool_)
    elif defect == "output":
        eqn.outvars[0] = var((32,), np.float16)
    elif defect == "params":
        eqn.params["extra"] = 1
    elif defect == "call-time":
        eqn.params["call_time"] = 1
    else:
        eqn.params["rate"] = np.nan if defect == "rate-nan" else 1.0
    with pytest.raises(SourcePrecisionError):
        DropoutDiagnosticValidator()._ordinary(eqn, "malformed-dropout", 23)
