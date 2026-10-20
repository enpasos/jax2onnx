# tests/extra_tests/converter/test_source_scientific_fp64.py

"""Native scientific FP64 contracts and the unchanged issue52 SOURCE payload.

The original payload, helper, three inputs, opset21 and rtol1e-5/atol1e-6 are
fixed. Structural admission is separate from both CPU optimization profiles and
from the independently captured original eager and JIT numerical references.
"""

from collections.abc import Callable, Iterator
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import traceback
from typing import Any

import jax
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import Primitive, Var, scan_arity
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import (
    PrecisionPolicy,
    active_precision_policy,
    precision_policy_scope,
)
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.precision import (
    _array_evidence,
    _compare,
    CheckStatus,
    SampleAcceptanceError,
    SampleCriteria,
    SourceSample,
    get_precision_report,
    validate_source_samples,
)
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)


_CRITERIA = SampleCriteria("tolerant", rtol=1e-5, atol=1e-6)
_ISSUE52_HASHES = {
    "helpers/issue52_scatter_payload_fixture.py": (
        "1a371a5653454bd3b26a55b7d0aa6c8069904b752cf242430badd009cd058033"
    ),
    "payloads/issue52_feedforward_payload.npz": (
        "c87923703463d7d6129d1c053a3f1d651312ee296fcbed47aaca698f5e6083f8"
    ),
}


def _configuration() -> tuple[dict[str, Any], tuple[Any, ...]]:
    globals_ = {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
        "jax_platforms": jax.config.jax_platforms,
    }
    return globals_, tuple(getattr(jax.config, name) for name in globals_)


@pytest.fixture(autouse=True)
def _restore_configuration() -> Iterator[None]:
    before = _configuration()
    policy = active_precision_policy()
    try:
        with (
            jax.enable_x64(True),
            jax.default_matmul_precision(None),
            jax.numpy_dtype_promotion("standard"),
            jax.numpy_rank_promotion("allow"),
        ):
            yield
    finally:
        # The immutable issue52 helper enables x64 at import. Import it only
        # inside this fixture and restore the caller's actual global settings.
        for name, value in before[0].items():
            jax.config.update(name, value)
        assert _configuration() == before
        assert active_precision_policy() is policy


def _body(value: Any) -> Any:
    inner = getattr(value, "jaxpr", value)
    return inner if inner is not value else value


def _signature(variables: Any) -> tuple[Any, ...]:
    return tuple(
        (np.dtype(var.aval.dtype).name, tuple(var.aval.shape), var.aval.weak_type)
        for var in variables
    )


def _scopes(value: Any) -> Iterator[Any]:
    seen: set[int] = set()

    def visit(item: Any) -> Iterator[Any]:
        body = _body(item)
        if all(hasattr(body, name) for name in ("eqns", "invars", "outvars")):
            if id(body) in seen:
                return
            seen.add(id(body))
            yield body
            for equation in body.eqns:
                yield from visit(equation.params)
        elif isinstance(item, dict):
            for child in item.values():
                yield from visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                yield from visit(child)

    yield from visit(value)


def _graphs(graph: onnx.GraphProto) -> Iterator[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from _graphs(child)


def _native_source_trace(
    fn: Callable[..., Any], values: Any, *, opset: int = 21
) -> Any:
    specs = [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values]
    native = jax.make_jaxpr(fn)(*specs)
    before = typed_trace_manifest(native)
    with precision_policy_scope(PrecisionPolicy.SOURCE), _activate_plugin_worlds():
        activated = jax.make_jaxpr(fn)(*specs)
    assert typed_trace_manifest(activated) == before
    native_equations = [equation for body in _scopes(native) for equation in body.eqns]
    source_equations = [
        equation for body in _scopes(activated) for equation in body.eqns
    ]
    assert len(source_equations) == len(native_equations)
    assert all(
        actual.primitive is expected.primitive
        for actual, expected in zip(source_equations, native_equations, strict=True)
    )
    validate_source_jaxpr(_body(native), opset)
    validate_source_jaxpr(_body(activated), opset)
    assert typed_trace_manifest(native) == before
    return native


def _capture_native(
    fn: Callable[..., Any], values: Any, *, native_dir: Path | None = None
) -> dict[str, dict[str, Any]]:
    """Attempt both unpatched native roles before any SOURCE admission checks."""
    observations: dict[str, dict[str, Any]] = {}

    # A new wrapper gives this observation its own JIT callable/cache identity.
    # It invokes the original function with the exact original input values.
    def fresh_jit_target(*arguments: Any) -> Any:
        return fn(*arguments)

    for role in ("eager", "jit"):
        observation: dict[str, Any] = {
            "attempted": True,
            "returned": False,
            "raised": False,
            "status": "attempted",
            "stage": "invoke",
            "arrays": [],
            "exception": None,
        }
        observations[role] = observation
        try:
            result = (
                fn(*values) if role == "eager" else jax.jit(fresh_jit_target)(*values)
            )
            observation["stage"] = "materialize"
            for value in jax.tree.leaves(result):
                observation["arrays"].append(np.asarray(value).copy())
            observation.update(returned=True, status="returned", stage="complete")
        except Exception as exc:
            observation.update(
                raised=True,
                status="raised",
                exception={
                    "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    # Save both returned arrays and actual exception states before any SOURCE
    # trace/parity/profile assertion. Original-model tests keep these in tmp_path.
    if native_dir is not None:
        native_dir.mkdir(parents=True, exist_ok=True)
        metadata = {}
        for role, observation in observations.items():
            np.savez(
                native_dir / f"native-{role}.npz",
                **{
                    f"output_{index}": value
                    for index, value in enumerate(observation["arrays"])
                },
            )
            metadata[role] = {
                **{key: value for key, value in observation.items() if key != "arrays"},
                "array_file": f"native-{role}.npz",
                "outputs": [
                    {"dtype": value.dtype.str, "shape": list(value.shape)}
                    for value in observation["arrays"]
                ],
            }
        (native_dir / "native-observations.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return observations


def _export(
    fn: Callable[..., Any],
    values: Any,
    *,
    opset: int = 21,
    native_dir: Path | None = None,
) -> tuple[onnx.ModelProto, Any, list[np.ndarray], list[np.ndarray]]:
    observations = _capture_native(fn, values, native_dir=native_dir)
    # A failed eager observation never suppresses JIT, and neither failure is
    # waived or replaced by another reference. Preserve all states in the failure.
    assert all(
        observation["status"] == "returned" for observation in observations.values()
    ), observations
    eager, compiled = observations["eager"]["arrays"], observations["jit"]["arrays"]
    native = _native_source_trace(fn, values, opset=opset)
    before = _configuration()
    policy = active_precision_policy()
    model = ui._to_onnx(
        fn,
        [jax.ShapeDtypeStruct(value.shape, value.dtype) for value in values],
        precision_policy=PrecisionPolicy.SOURCE,
        opset=opset,
    )
    assert _configuration() == before
    assert active_precision_policy() is policy
    assert isinstance(model, onnx.ModelProto)
    model = onnx.load_model_from_string(model.SerializeToString())
    onnx.checker.check_model(model, full_check=True)
    assert {item.version for item in model.opset_import if item.domain == ""} == {opset}
    report = get_precision_report(model)
    assert report is not None and report.profile == "source-static-v1"
    assert {check.category: check.status for check in report.checks} == {
        "representation": CheckStatus.VALIDATED,
        "lowering": CheckStatus.VALIDATED,
        "runtime-execution": CheckStatus.UNVERIFIED,
        "numerical-acceptance": CheckStatus.UNVERIFIED,
    }
    assert tuple(
        (value.dtype, value.shape, value.weak_type) for value in report.traced_outputs
    ) == _signature(_body(native).outvars)
    return model, native, eager, compiled


def _accept(
    model: onnx.ModelProto,
    values: Any,
    eager: list[np.ndarray],
    compiled: list[np.ndarray],
) -> None:
    feeds = {
        port.name: np.asarray(value)
        for port, value in zip(model.graph.input, values, strict=True)
    }
    original_report = get_precision_report(model)
    observations = []
    for optimization in ("disabled", "all"):
        for reference, expected in (("eager", eager), ("jit", compiled)):
            evidence = validate_source_samples(
                model,
                samples=[SourceSample(feeds, expected, _CRITERIA)],
                optimization=optimization,
            )
            observations.append((optimization, reference, evidence))
    assert get_precision_report(model) == original_report
    # Observe every profile/reference pair before making any acceptance claim.
    # A failed comparison retains FAILED evidence and the complete observation
    # set in the assertion, including successful or failed later profiles.
    assert all(
        evidence.status is CheckStatus.VALIDATED for _, _, evidence in observations
    ), observations
    for _, _, evidence in observations:
        evidence.require_sample_acceptance()


def _scatter(kind: str) -> Callable[..., Any]:
    operation = {
        "scatter": jax.lax.scatter,
        "scatter-add": jax.lax.scatter_add,
        "scatter-mul": jax.lax.scatter_mul,
    }[kind]
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0, 1, 2, 3),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1,),
    )

    def target(operand: Any, indices: Any, updates: Any) -> Any:
        return operation(
            operand,
            indices,
            updates,
            dimensions,
            indices_are_sorted=True,
            unique_indices=True,
            mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        )

    return target


def _scatter_values() -> tuple[np.ndarray, ...]:
    return (
        np.linspace(0.25, 0.75, 5 * 210, dtype=np.float64).reshape(5, 210, 1, 1),
        np.asarray([7], dtype=np.int64),
        np.linspace(1.00000000001, 1.12500000003, 25, dtype=np.float64).reshape(
            5, 5, 1, 1
        ),
    )


def _checkpoint(left: Any, right: Any) -> Any:
    def target(x: Any, y: Any) -> Any:
        return x * y + x, y

    return jax.checkpoint(target)(left, right)


def _scan(length: int) -> Callable[..., Any]:
    def target(field: Any) -> Any:
        def step(carry: Any, unused: Any) -> Any:
            value, counter = carry
            return (value + np.float64(0.03125000001), counter + 1), value

        return jax.lax.scan(step, (field, 0), None, length=length)

    return target


def _equation(equation: Any, **changes: Any) -> Any:
    values = {
        "primitive": equation.primitive,
        "invars": list(equation.invars),
        "outvars": list(equation.outvars),
        "params": dict(equation.params),
        "effects": equation.effects,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _one_equation(native: Any) -> tuple[Any, Any]:
    body = _body(native)
    assert len(body.eqns) == 1
    return body, body.eqns[0]


def _corrupted_graph(
    body: Any, equation: Any, inputs: dict[int, Any] | None = None
) -> Any:
    replacements = inputs or {}
    equation = _equation(
        equation,
        invars=[replacements.get(id(var), var) for var in equation.invars],
    )
    return SimpleNamespace(
        invars=[replacements.get(id(var), var) for var in body.invars],
        constvars=body.constvars,
        outvars=equation.outvars,
        eqns=[equation],
        effects=body.effects,
    )


def _changed_var(var: Any, **changes: Any) -> Any:
    return SimpleNamespace(aval=var.aval.update(**changes))


def _rejected(graph: Any, primitives: set[str]) -> None:
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(graph, 21)
    assert caught.value.opset == 21
    assert caught.value.primitive in primitives
    assert caught.value.category in {"lowering-defect", "profile-exclusion"}


def test_issue52_helper_and_payload_are_the_original_immutable_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative, expected in _ISSUE52_HASHES.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("kind", ["scatter", "scatter-add", "scatter-mul"])
def test_native_fp64_full_window_scatter_at_original_opset21(kind: str) -> None:
    values = _scatter_values()
    assert np.any(values[2] != values[2].astype(np.float32).astype(np.float64))
    model, native, eager, compiled = _export(_scatter(kind), values)
    _, equation = _one_equation(native)
    assert equation.primitive.name == kind
    assert _signature(equation.invars) == (
        ("float64", (5, 210, 1, 1), False),
        ("int64", (1,), False),
        ("float64", (5, 5, 1, 1), False),
    )
    assert any(node.op_type == "ScatterND" for node in model.graph.node)
    _accept(model, values, eager, compiled)


@pytest.mark.parametrize("kind", ["scatter", "scatter-add", "scatter-mul"])
@pytest.mark.parametrize(
    "corruption",
    [
        "identity",
        "input-arity",
        "result-dtype",
        "result-shape",
        "result-weakness",
        "index-dtype",
        "index-rank",
        "window-too-large",
        "dimension-numbers",
        "mode",
        "sorted-flag",
        "unique-flag",
        "update-constants",
    ],
)
def test_scatter_contract_corruption_is_rejected_before_lowering(
    kind: str, corruption: str
) -> None:
    native = jax.make_jaxpr(_scatter(kind))(*_scatter_values())
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    before = typed_trace_manifest(native)
    equation = _equation(original)
    replacements = {}
    if corruption == "identity":
        equation.primitive = Primitive(kind)
    elif corruption == "input-arity":
        equation.invars = equation.invars[:-1]
    elif corruption.startswith("result-"):
        changes = {
            "result-dtype": {"dtype": np.dtype(np.float32)},
            "result-shape": {"shape": (5, 211, 1, 1)},
            "result-weakness": {"weak_type": True},
        }[corruption]
        equation.outvars = [_changed_var(original.outvars[0], **changes)]
    elif corruption in {"index-dtype", "index-rank", "window-too-large"}:
        index = 2 if corruption == "window-too-large" else 1
        changes = {
            "index-dtype": {"dtype": np.dtype(np.float64)},
            "index-rank": {"shape": ()},
            "window-too-large": {"shape": (5, 211, 1, 1)},
        }[corruption]
        replacements[id(original.invars[index])] = _changed_var(
            original.invars[index], **changes
        )
    elif corruption == "dimension-numbers":
        equation.params["dimension_numbers"] = jax.lax.ScatterDimensionNumbers(
            update_window_dims=(0, 0, 2, 3),
            inserted_window_dims=(),
            scatter_dims_to_operand_dims=(1,),
        )
    else:
        key, value = {
            "mode": ("mode", "promise_in_bounds"),
            "sorted-flag": ("indices_are_sorted", 1),
            "unique-flag": ("unique_indices", 1),
            "update-constants": ("update_consts", (np.float64(1),)),
        }[corruption]
        equation.params[key] = value
    _rejected(_corrupted_graph(body, equation, replacements), {kind})
    assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("kind", ["scatter-add", "scatter-mul"])
@pytest.mark.parametrize(
    "corruption", ["missing", "wrong-reducer", "reducer-constants"]
)
def test_scatter_reducer_must_be_the_exact_native_scalar_operation(
    kind: str, corruption: str
) -> None:
    native = jax.make_jaxpr(_scatter(kind))(*_scatter_values())
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    before = typed_trace_manifest(native)
    equation = _equation(original)
    reducer = equation.params["update_jaxpr"]
    reducer_body = _body(reducer)
    if corruption == "missing":
        equation.params["update_jaxpr"] = None
    elif corruption == "wrong-reducer":
        operation = jax.lax.mul_p if kind == "scatter-add" else jax.lax.add_p
        changed = reducer_body.eqns[0].replace(primitive=operation)
        equation.params["update_jaxpr"] = reducer_body.replace(eqns=[changed])
    else:
        extra = Var(reducer_body.invars[0].aval)
        equation.params["update_jaxpr"] = reducer_body.replace(constvars=[extra])
    _rejected(_corrupted_graph(body, equation), {kind})
    assert typed_trace_manifest(native) == before


def test_scatter_set_requires_unique_indices() -> None:
    native = jax.make_jaxpr(_scatter("scatter"))(*_scatter_values())
    body, original = _one_equation(native)
    equation = _equation(original)
    equation.params["unique_indices"] = False
    _rejected(_corrupted_graph(body, equation), {"scatter"})


def test_native_pure_fp64_remat_keeps_multiple_results_and_aliases() -> None:
    values = (
        np.asarray([0.25, 1.00000000001, -0.5], np.float64),
        np.asarray([1.12500000003, 0.75, 2.25], np.float64),
    )
    model, native, eager, compiled = _export(_checkpoint, values)
    _, equation = _one_equation(native)
    assert equation.primitive.name == "remat2"
    assert len(equation.invars) == len(equation.outvars) == 2
    assert (
        _body(equation.params["jaxpr"]).outvars[1]
        is _body(equation.params["jaxpr"]).invars[1]
    )
    _accept(model, values, eager, compiled)


@pytest.mark.parametrize(
    "corruption",
    [
        "identity",
        "input-arity",
        "output-arity",
        "result-dtype",
        "result-shape",
        "result-weakness",
        "body-input-arity",
        "body-output-arity",
        "body-constants",
        "differentiated",
        "policy",
        "prevent-cse",
    ],
)
def test_remat_contract_corruption_is_rejected_before_inlining(corruption: str) -> None:
    spec = jax.ShapeDtypeStruct((3,), np.float64)
    native = jax.make_jaxpr(_checkpoint)(spec, spec)
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    before = typed_trace_manifest(native)
    equation = _equation(original)
    child = _body(equation.params["jaxpr"])
    if corruption == "identity":
        equation.primitive = Primitive("remat2")
    elif corruption == "input-arity":
        equation.invars = equation.invars[:-1]
    elif corruption == "output-arity":
        equation.outvars = equation.outvars[:-1]
    elif corruption.startswith("result-"):
        changes = {
            "result-dtype": {"dtype": np.dtype(np.float32)},
            "result-shape": {"shape": (4,)},
            "result-weakness": {"weak_type": True},
        }[corruption]
        equation.outvars[0] = _changed_var(original.outvars[0], **changes)
    elif corruption == "body-input-arity":
        equation.params["jaxpr"] = child.replace(invars=child.invars[:-1])
    elif corruption == "body-output-arity":
        equation.params["jaxpr"] = child.replace(outvars=child.outvars[:-1])
    elif corruption == "body-constants":
        equation.params["jaxpr"] = child.replace(constvars=[Var(child.invars[0].aval)])
    else:
        key, value = {
            "differentiated": ("differentiated", True),
            "policy": ("policy", object()),
            "prevent-cse": ("prevent_cse", 1),
        }[corruption]
        equation.params[key] = value
    _rejected(_corrupted_graph(body, equation), {"remat2"})
    assert typed_trace_manifest(native) == before


def test_remat_child_still_requires_recursive_source_admission() -> None:
    def target(value: Any) -> Any:
        return jax.checkpoint(jax.lax.erf)(value)

    native = jax.make_jaxpr(target)(jax.ShapeDtypeStruct((3,), np.float64))
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(_body(native), 21)
    assert caught.value.primitive == "erf"
    assert "remat" in caught.value.location


@pytest.mark.parametrize("kind", ["scatter", "remat"])
@pytest.mark.parametrize("corruption", ["effects", "placement"])
def test_scientific_carriers_reject_effects_and_placement(
    kind: str, corruption: str
) -> None:
    function, values = (
        (_scatter("scatter-add"), _scatter_values())
        if kind == "scatter"
        else (_checkpoint, (np.ones(3, np.float64), np.ones(3, np.float64)))
    )
    native = jax.make_jaxpr(function)(*values)
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    equation = _equation(original)
    if corruption == "effects":
        equation.effects = (object(),)
    else:
        equation.params["out_sharding"] = object()
    _rejected(_corrupted_graph(body, equation), {original.primitive.name})


@pytest.mark.parametrize("length", [0, 1, 3])
def test_native_fp64_no_xs_scan21_preserves_weak_int64_carry_and_stacked_axis(
    length: int,
) -> None:
    values = (np.linspace(0.25, 0.75, 30, dtype=np.float64).reshape(5, 6),)
    model, native, eager, compiled = _export(_scan(length), values)
    _, equation = _one_equation(native)
    assert equation.primitive.name == "scan"
    assert scan_arity(equation.params, len(equation.invars)) == (0, 2, 0)
    assert _signature(equation.outvars) == (
        ("float64", (5, 6), False),
        ("int64", (), True),
        ("float64", (length, 5, 6), False),
    )
    _accept(model, values, eager, compiled)


@pytest.mark.parametrize("kind", ["with-xs", "reverse", "unroll"])
def test_scan21_does_not_admit_unreviewed_scan_routes(kind: str) -> None:
    def target(value: Any, sequence: Any) -> Any:
        def step(carry: Any, element: Any) -> Any:
            increment = element if kind == "with-xs" else np.float64(0.25)
            return carry + increment, carry

        return jax.lax.scan(
            step,
            value,
            sequence if kind == "with-xs" else None,
            length=3,
            reverse=kind == "reverse",
            unroll=2 if kind == "unroll" else 1,
        )

    native = jax.make_jaxpr(target)(
        jax.ShapeDtypeStruct((5, 6), np.float64),
        jax.ShapeDtypeStruct((3, 5, 6), np.float64),
    )
    _rejected(_body(native), {"scan"})


@pytest.mark.parametrize(
    "corruption",
    [
        "identity",
        "result-dtype",
        "result-shape",
        "result-weakness",
        "length-bool",
        "body-output-arity",
    ],
)
def test_scan21_retains_exact_typed_native_contract(corruption: str) -> None:
    native = jax.make_jaxpr(_scan(3))(jax.ShapeDtypeStruct((5, 6), np.float64))
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    before = typed_trace_manifest(native)
    equation = _equation(original)
    if corruption == "identity":
        equation.primitive = Primitive("scan")
    elif corruption.startswith("result-"):
        changes = {
            "result-dtype": {"dtype": np.dtype(np.float32)},
            "result-shape": {"shape": (5, 7)},
            "result-weakness": {"weak_type": True},
        }[corruption]
        equation.outvars[0] = _changed_var(original.outvars[0], **changes)
    elif corruption == "length-bool":
        equation.params["length"] = True
    else:
        closed = equation.params["jaxpr"]
        child = _body(closed)
        # Preserve the installed ClosedJaxpr representation while dropping one
        # stacked result; the carrier/body role mismatch must remain visible.
        if child is closed:
            equation.params["jaxpr"] = child.replace(outvars=child.outvars[:-1])
        else:
            from jax2onnx._compat.jax import ClosedJaxpr

            equation.params["jaxpr"] = ClosedJaxpr(
                child.replace(outvars=child.outvars[:-1]), closed.consts
            )
    _rejected(_corrupted_graph(body, equation), {"scan"})
    assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("kind", ["elementary", "scatter", "remat", "scan"])
def test_scientific_fp64_extension_preserves_opset18_exclusions(kind: str) -> None:
    fn, values = {
        "elementary": (jax.lax.neg, (np.asarray(1.00000000001, np.float64),)),
        "scatter": (_scatter("scatter-add"), _scatter_values()),
        "remat": (_checkpoint, (np.ones(3, np.float64), np.ones(3, np.float64))),
        "scan": (_scan(1), (np.ones((5, 6), np.float64),)),
    }[kind]
    native = jax.make_jaxpr(fn)(*values)
    with pytest.raises(SourcePrecisionError) as caught:
        validate_source_jaxpr(_body(native), 18)
    assert caught.value.opset == 18
    assert caught.value.category == "profile-exclusion"


@pytest.mark.parametrize("operation", ["min", "max"])
@pytest.mark.parametrize("optimization", ["disabled", "all"])
@pytest.mark.parametrize("control", ["zero", "nan"])
def test_fp64_minmax21_preserves_zero_polarity_and_nan_classes(
    operation: str, optimization: str, control: str
) -> None:
    # These controls use exact finite bits and NaN classification, not the
    # issue52 tolerance.
    pairs = {
        "zero": ([0.0, -0.0, 0.0, -0.0], [-0.0, 0.0, 0.0, -0.0]),
        "nan": ([np.nan, 1.0, np.nan], [1.0, np.nan, np.nan]),
    }
    values = tuple(np.asarray(value, np.float64) for value in pairs[control])
    function = jax.lax.min if operation == "min" else jax.lax.max
    model, _, eager, compiled = _export(function, values)
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if optimization == "disabled"
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    actual = session.run(
        None,
        {
            port.name: value
            for port, value in zip(session.get_inputs(), values, strict=True)
        },
    )[0]
    for reference in (eager[0], compiled[0]):
        assert actual.shape == reference.shape and actual.dtype == reference.dtype
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(reference))
        finite = ~np.isnan(reference)
        np.testing.assert_array_equal(
            actual[finite].view(np.uint64), reference[finite].view(np.uint64)
        )


def _capture_original_cpu_targets(
    model: onnx.ModelProto, values: Any, target_dir: Path
) -> list[dict[str, Any]]:
    """Save both original FP64 CPU observations before acceptance assertions."""
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact = model.SerializeToString()
    (target_dir / "source.onnx").write_bytes(artifact)
    np.savez(
        target_dir / "inputs.npz",
        **{f"input_{index}": np.asarray(value) for index, value in enumerate(values)},
    )
    observations = []
    for optimization in ("disabled", "all"):
        graph_path = target_dir / f"{optimization}-runtime-graph.onnx"
        arrays_path = target_dir / f"{optimization}-actual.npz"
        observation: dict[str, Any] = {
            "optimization": optimization,
            "returned": False,
            "arrays": [],
            "exception": None,
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
            "runtime_version": ort.__version__,
            "jax_version": jax.__version__,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "runtime_graph": graph_path.name,
            "array_file": arrays_path.name,
        }
        observations.append(observation)
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                if optimization == "disabled"
                else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.optimized_model_filepath = str(graph_path)
            session = ort.InferenceSession(
                artifact, options, providers=["CPUExecutionProvider"]
            )
            observation["providers"] = session.get_providers()
            feeds = {
                port.name: np.asarray(value)
                for port, value in zip(session.get_inputs(), values, strict=True)
            }
            observation["arrays"] = [
                np.asarray(value).copy() for value in session.run(None, feeds)
            ]
            observation["returned"] = True
        except Exception as exc:
            observation["exception"] = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        np.savez(
            arrays_path,
            **{
                f"output_{index}": value
                for index, value in enumerate(observation["arrays"])
            },
        )
    (target_dir / "observations-before-assertions.json").write_text(
        json.dumps(
            [
                {
                    **{key: value for key, value in item.items() if key != "arrays"},
                    "outputs": [
                        {"dtype": value.dtype.str, "shape": list(value.shape)}
                        for value in item["arrays"]
                    ],
                }
                for item in observations
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return observations


@pytest.mark.parametrize("operation", ["min", "max"])
def test_fp64_minmax21_subnormal_source_selection_and_native_failed_diagnostic(
    operation: str, record_property: Any, tmp_path: Path
) -> None:
    # Preserve the original subnormal source, values, opset and CPU profiles.
    # Correct operand selection and failed native-bitwise qualification are
    # independent obligations; the latter remains an actual FAILED sample.
    tiny = np.nextafter(np.float64(0), np.float64(1))
    values = tuple(
        np.asarray(value, np.float64)
        for value in ([tiny, -tiny], [2 * tiny, -2 * tiny])
    )
    snapshots = tuple(value.tobytes() for value in values)
    function = jax.lax.min if operation == "min" else jax.lax.max
    model, _, eager, compiled = _export(
        function, values, native_dir=tmp_path / "native"
    )
    original_report = get_precision_report(model)
    targets = _capture_original_cpu_targets(model, values, tmp_path / "targets")
    observations = []
    for target in targets:
        for role, reference in (("eager", eager), ("jit", compiled)):
            evidence = validate_source_samples(
                model,
                samples=[SourceSample(values, reference, SampleCriteria("bitwise"))],
                optimization=target["optimization"],
            )
            observations.append((target, role, evidence))
    # Retain every actual profile/reference report before any aggregate check.
    records = [
        {
            "operation": operation,
            "optimization": target["optimization"],
            "reference": role,
            "native_words": [value.view(np.uint64).tolist() for value in reference],
            "actual_words": [
                value.view(np.uint64).tolist() for value in target["arrays"]
            ],
            "artifact_sha256": evidence.artifact_sha256,
            "status": evidence.status,
            "target": None if evidence.target is None else asdict(evidence.target),
            "checks": [asdict(check) for check in evidence.checks],
            "samples": [asdict(sample) for sample in evidence.samples],
        }
        for target, role, evidence in observations
        for reference in (eager if role == "eager" else compiled,)
    ]
    serialized = json.dumps(records, indent=2, sort_keys=True)
    (tmp_path / "native-bitwise-failed-reports.json").write_text(serialized + "\n")
    record_property("source-selection-and-native-failed", serialized)
    sign = 1 << 63
    assert [value.view(np.uint64).tolist() for value in values] == [
        [1, sign + 1],
        [2, sign + 2],
    ]
    # Positive inputs select the smaller/larger word; negative inputs select
    # the larger/smaller magnitude. No native result constructs this oracle.
    selected = (
        (values[0][0], values[1][1])
        if operation == "min"
        else (values[1][0], values[0][1])
    )
    expected_words = [1, sign + 2] if operation == "min" else [2, sign + 1]
    assert np.asarray(selected, np.float64).view(np.uint64).tolist() == expected_words
    for reference in (eager, compiled):
        assert len(reference) == 1
        assert reference[0].dtype == np.dtype(np.float64)
        assert reference[0].shape == (2,)
        assert reference[0].view(np.uint64).tolist() == [0, sign]
    assert all(target["returned"] for target in targets), targets
    for target in targets:
        assert target["providers"] == ["CPUExecutionProvider"]
        assert len(target["arrays"]) == 1
        actual = target["arrays"][0]
        assert actual.dtype == np.dtype(np.float64) and actual.shape == (2,)
        assert actual.view(np.uint64).tolist() == expected_words
    assert get_precision_report(model) == original_report
    for target, role, evidence in observations:
        assert evidence.artifact_sha256 == target["artifact_sha256"]
        assert evidence.target is not None
        assert evidence.target.runtime_version == ort.__version__
        assert evidence.target.providers == ("CPUExecutionProvider",)
        assert evidence.target.intra_op_threads == evidence.target.inter_op_threads == 1
        assert evidence.target.execution_mode == "ORT_SEQUENTIAL"
        assert evidence.target.session_config == ()
        assert evidence.target.optimization == (
            "ORT_DISABLE_ALL"
            if target["optimization"] == "disabled"
            else "ORT_ENABLE_ALL"
        )
        assert all(check.status is CheckStatus.VALIDATED for check in evidence.checks)
        assert evidence.status is CheckStatus.FAILED
        assert len(evidence.samples) == 1
        sample = evidence.samples[0]
        assert sample.inputs == tuple(
            _array_evidence(str(index), value) for index, value in enumerate(values)
        )
        assert sample.references == tuple(
            _array_evidence(str(index), value)
            for index, value in enumerate(eager if role == "eager" else compiled)
        )
        assert sample.outputs == (_array_evidence("0", target["arrays"][0]),)
        assert sample.criteria == SampleCriteria("bitwise")
        assert sample.status is CheckStatus.FAILED
        assert [(check.category, check.status) for check in sample.checks] == [
            ("runtime-execution", CheckStatus.VALIDATED),
            ("numerical-acceptance", CheckStatus.FAILED),
        ]
        with pytest.raises(SampleAcceptanceError, match="bitwise"):
            evidence.require_sample_acceptance()
    assert tuple(value.tobytes() for value in values) == snapshots


@pytest.mark.parametrize("corruption", ["dtype", "shape", "value", "nan"])
def test_original_issue52_criteria_reject_corrupted_target_results(
    corruption: str,
) -> None:
    reference = np.asarray([0.25, 1.12500000001], np.float64)
    changed = reference.copy()
    if corruption == "dtype":
        changed = changed.astype(np.float32)
    elif corruption == "shape":
        changed = changed.reshape(1, 2)
    elif corruption == "value":
        changed[1] += 0.01
    else:
        changed[1] = np.nan
    assert _compare(reference.copy(), reference, _CRITERIA) is None
    assert _compare(changed, reference, _CRITERIA) is not None


def _shape(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    assert value.type.tensor_type.HasField("shape")
    assert all(dim.HasField("dim_value") for dim in value.type.tensor_type.shape.dim)
    return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)


def _issue52_loop_boundaries(model: onnx.ModelProto, native_scans: Any) -> None:
    allowed_types = {
        onnx.TensorProto.DOUBLE,
        onnx.TensorProto.BOOL,
        onnx.TensorProto.INT32,
        onnx.TensorProto.INT64,
    }
    for graph in _graphs(model.graph):
        # The actual native payload has no FP32/FP16 arithmetic. Boundary
        # restoration after an internal downgrade would violate SOURCE too.
        for value in (*graph.input, *graph.output, *graph.value_info):
            assert value.type.tensor_type.elem_type in allowed_types
        for value in (*graph.input, *graph.output):
            _shape(value)
        for initializer in graph.initializer:
            assert initializer.data_type in allowed_types
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    assert attribute.t.data_type in allowed_types
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    assert all(
                        tensor.data_type in allowed_types
                        for tensor in attribute.tensors
                    )
            if node.op_type == "Cast":
                assert (
                    next(
                        attribute.i
                        for attribute in node.attribute
                        if attribute.name == "to"
                    )
                    in allowed_types
                )
    loops = []

    def visit(graph: onnx.GraphProto, owner: Any = None) -> None:
        for node in graph.node:
            if node.op_type == "Loop":
                loops.append((graph, node, owner))
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    visit(
                        attribute.g,
                        (
                            (node, attribute.g)
                            if node.op_type == "Loop" and attribute.name == "body"
                            else None
                        ),
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        visit(child)

    visit(model.graph)
    assert len(loops) == 2
    expected_by_stacked = {}
    for equation in native_scans:
        num_consts, num_carry, num_xs = scan_arity(
            equation.params, len(equation.invars)
        )
        assert num_xs == 0
        stacked = len(equation.outvars) - num_carry
        assert stacked not in expected_by_stacked
        expected_by_stacked[stacked] = (equation, num_consts, num_carry)
    assert set(expected_by_stacked) == {0, 2}

    def expected_signature(variables: Any) -> tuple[Any, ...]:
        return tuple(
            (
                onnx.helper.np_dtype_to_tensor_dtype(np.dtype(var.aval.dtype)),
                tuple(var.aval.shape),
            )
            for var in variables
        )

    def physical_signature(values: Any) -> tuple[Any, ...]:
        return tuple(
            (value.type.tensor_type.elem_type, _shape(value)) for value in values
        )

    roles = set()
    for graph, node, owner in loops:
        body = next(
            attribute.g for attribute in node.attribute if attribute.name == "body"
        )
        states = len(node.input) - 2
        stacked = len(node.output) - states
        equation, num_consts, num_carry = expected_by_stacked[stacked]
        roles.add((states, stacked))
        # This existing emitter transports native scan constants as unchanged
        # leading Loop states, then the actual mutable carry. Both native scans
        # therefore have ten physical states, with distinct 5/5 and 6/4 roles.
        assert states == num_consts + num_carry == 10
        assert len(body.input) == len(node.input)
        assert len(body.output) == len(node.output) + 1
        assert body.input[0].type.tensor_type.elem_type == onnx.TensorProto.INT64
        assert body.input[1].type.tensor_type.elem_type == onnx.TensorProto.BOOL
        assert body.output[0].type.tensor_type.elem_type == onnx.TensorProto.BOOL
        assert _shape(body.input[0]) == _shape(body.input[1]) == ()
        assert _shape(body.output[0]) == ()
        native_body = _body(equation.params["jaxpr"])
        assert physical_signature(body.input[2:]) == expected_signature(
            native_body.invars
        )
        infos = {
            value.name: value
            for value in (*graph.input, *graph.output, *graph.value_info)
        }
        initializers = {value.name: value for value in graph.initializer}

        def initial_signature(name: str) -> tuple[Any, ...]:
            signatures = []
            if name in infos:
                signatures.append(physical_signature([infos[name]])[0])
            if name in initializers:
                value = initializers[name]
                signatures.append((value.data_type, tuple(value.dims)))
            assert signatures, name
            assert all(signature == signatures[0] for signature in signatures)
            return signatures[0]

        assert initial_signature(node.input[0]) == (onnx.TensorProto.INT64, ())
        assert initial_signature(node.input[1]) == (onnx.TensorProto.BOOL, ())
        assert tuple(initial_signature(name) for name in node.input[2:]) == (
            expected_signature(native_body.invars)
        )
        if owner is not None:
            parent_loop, parent_body = owner
            parent_states = len(parent_loop.input) - 2
            parent_stacked = len(parent_loop.output) - parent_states
            parent_equation = expected_by_stacked[parent_stacked][0]
            parent_native = _body(parent_equation.params["jaxpr"])
            parent_bindings = {
                id(variable): value.name
                for variable, value in zip(
                    parent_native.invars, parent_body.input[2:], strict=True
                )
            }

            def bind_remat_inputs(native_body: Any) -> None:
                # The original inner scan is inside an inlined remat2 body.
                # Its formal variables have fresh identities; transport the
                # enclosing bindings through that exact positional interface.
                for remat in native_body.eqns:
                    if remat.primitive.name != "remat2":
                        continue
                    child = _body(remat.params["jaxpr"])
                    assert expected_signature(remat.invars) == expected_signature(
                        child.invars
                    )
                    for argument, formal in zip(
                        remat.invars, child.invars, strict=True
                    ):
                        assert id(argument) in parent_bindings
                        parent_bindings[id(formal)] = parent_bindings[id(argument)]
                    bind_remat_inputs(child)

            bind_remat_inputs(parent_native)
            parent_producers = {
                name: child for child in graph.node for name in child.output
            }
            for variable, name in zip(equation.invars, node.input[2:], strict=True):
                assert id(variable) in parent_bindings
                seen = set()
                while (
                    name in parent_producers
                    and parent_producers[name].op_type == "Identity"
                ):
                    assert name not in seen
                    seen.add(name)
                    identity = parent_producers[name]
                    assert len(identity.input) == len(identity.output) == 1
                    name = identity.input[0]
                assert name == parent_bindings[id(variable)]
        constant_inputs = body.input[2 : 2 + num_consts]
        constant_outputs = body.output[1 : 1 + num_consts]
        assert (
            physical_signature(constant_inputs)
            == physical_signature(constant_outputs)
            == expected_signature(native_body.invars[:num_consts])
        )
        producers = {name: child for child in body.node for name in child.output}

        def identity_source(name: str) -> str:
            seen = set()
            while name in producers and producers[name].op_type == "Identity":
                assert name not in seen
                seen.add(name)
                identity = producers[name]
                assert len(identity.input) == len(identity.output) == 1
                name = identity.input[0]
            return name

        for incoming, outgoing in zip(constant_inputs, constant_outputs, strict=True):
            assert identity_source(outgoing.name) == incoming.name
        carry_inputs = body.input[2 + num_consts :]
        carry_outputs = body.output[1 + num_consts : 1 + states]
        assert physical_signature(carry_inputs) == expected_signature(
            native_body.invars[num_consts:]
        )
        assert physical_signature(carry_outputs) == expected_signature(
            native_body.outvars[:num_carry]
        )
        assert physical_signature(body.output[1 + states :]) == expected_signature(
            native_body.outvars[num_carry:]
        )
        expected_outputs = expected_signature(
            native_body.invars[:num_consts]
        ) + expected_signature(equation.outvars)
        graph_inputs = {value.name for value in graph.input}
        graph_outputs = {value.name for value in graph.output}
        internal_infos = {value.name for value in graph.value_info}
        graph_producers = {}
        for child in graph.node:
            for name in child.output:
                assert name not in graph_producers
                graph_producers[name] = child
        consumers = {
            name
            for child_graph in _graphs(graph)
            for child in child_graph.node
            for name in child.input
        }
        for index, (name, expected) in enumerate(
            zip(node.output, expected_outputs, strict=True)
        ):
            assert graph_producers[name] is node
            value = infos[name]
            unused_constant = (
                owner is not None
                and index < num_consts
                and name in internal_infos
                and name not in graph_inputs | graph_outputs | consumers
            )
            if not unused_constant:
                assert physical_signature([value])[0] == expected
                continue
            # Native constant roles and unchanged body identities were proved
            # above. Only this unused internal annotation may be rank-only.
            tensor = value.type.tensor_type
            dtype, shape = expected
            assert tensor.elem_type == dtype and tensor.HasField("shape")
            dimensions = tensor.shape.dim
            assert len(dimensions) == len(shape)
            assert all(dim.HasField("dim_value") for dim in dimensions) or all(
                dim.WhichOneof("value") is None for dim in dimensions
            )
            for dimension, extent in zip(dimensions, shape, strict=True):
                if dimension.HasField("dim_value"):
                    assert dimension.dim_value == extent
    assert roles == {(10, 2), (10, 0)}
    scatters = [
        node
        for graph in _graphs(model.graph)
        for node in graph.node
        if node.op_type == "ScatterND"
    ]
    assert scatters
    reductions = {
        next(
            (
                attribute.s
                for attribute in node.attribute
                if attribute.name == "reduction"
            ),
            b"none",
        )
        for node in scatters
    }
    assert {b"none", b"add", b"mul"} <= reductions


def _issue52_loop_boundary_corruption_controls(
    model: onnx.ModelProto, native_scans: Any
) -> None:
    """Reject corruptions of the actual exported boundaries without rerunning it."""
    original = model.SerializeToString()
    for corruption in (
        "known-extent",
        "dtype",
        "rank",
        "missing-shape",
        "symbol",
        "mixed-dimensions",
        "body-input",
        "body-output",
        "mutable-carry",
        "used-output",
        "public-output",
        "outer-output",
        "initial-shape",
        "initial-dtype",
        "initial-identity",
        "constant-identity",
    ):
        changed = onnx.load_model_from_string(original)
        outer = next(node for node in changed.graph.node if node.op_type == "Loop")
        parent = next(attr.g for attr in outer.attribute if attr.name == "body")
        inner = next(node for node in parent.node if node.op_type == "Loop")
        body = next(attr.g for attr in inner.attribute if attr.name == "body")
        # The real second constant is INT64[201,6] and its unused inner result
        # has rank two. Select by the verified native slot, never by its name.
        annotation = next(
            value for value in parent.value_info if value.name == inner.output[1]
        )
        dimensions = annotation.type.tensor_type.shape.dim
        assert len(dimensions) == 2
        if corruption == "known-extent":
            dimensions[0].dim_value = 202
            dimensions[1].dim_value = 6
        elif corruption == "dtype":
            annotation.type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
        elif corruption == "rank":
            dimensions.add()
        elif corruption == "missing-shape":
            annotation.type.tensor_type.ClearField("shape")
        elif corruption == "symbol":
            dimensions[0].dim_param = "unproved"
        elif corruption == "mixed-dimensions":
            dimensions[0].dim_value = 201
            dimensions[1].Clear()
        elif corruption in {
            "body-input",
            "body-output",
            "mutable-carry",
            "outer-output",
        }:
            value = {
                "body-input": body.input[3],
                "body-output": body.output[2],
                "mutable-carry": parent.output[6],
                "outer-output": next(
                    value
                    for value in changed.graph.value_info
                    if value.name == outer.output[0]
                ),
            }[corruption]
            assert len(value.type.tensor_type.shape.dim) > 0
            for dimension in value.type.tensor_type.shape.dim:
                dimension.Clear()
        elif corruption == "used-output":
            parent.node.append(
                onnx.helper.make_node("Identity", [annotation.name], ["used-constant"])
            )
            for dimension in dimensions:
                dimension.Clear()
        elif corruption == "public-output":
            parent.output.append(annotation)
            for dimension in parent.output[-1].type.tensor_type.shape.dim:
                dimension.Clear()
        elif corruption in {"initial-shape", "initial-dtype"}:
            initial = next(
                value
                for value in changed.graph.initializer
                if value.name == outer.input[2]
            )
            if corruption == "initial-shape":
                initial.dims[0] = 202
            else:
                initial.data_type = onnx.TensorProto.INT32
        elif corruption == "initial-identity":
            # Both BOOL[201,6] slots have the right shape/type. Swapping only
            # one incoming value must still fail the native variable wiring.
            inner.input[4] = inner.input[6]
        else:
            outgoing = body.output[3].name
            identity = next(node for node in body.node if outgoing in node.output)
            assert identity.op_type == "Identity"
            identity.input[0] = body.input[6].name
        with pytest.raises(AssertionError):
            _issue52_loop_boundaries(changed, native_scans)
    assert model.SerializeToString() == original


@pytest.mark.filterwarnings("ignore:.*Removing initializer.*:UserWarning")
def test_original_issue52_fp64_payload_source21_against_eager_and_jit_both_cpu_profiles(
    tmp_path: Path,
) -> None:
    # Import the unmodified helper within the configuration-restoring fixture.
    from tests.extra_tests.helpers import issue52_scatter_payload_fixture as original

    fixture_root = Path(original.__file__).resolve().parents[1]
    for relative, expected in _ISSUE52_HASHES.items():
        assert (
            hashlib.sha256((fixture_root / relative).read_bytes()).hexdigest()
            == expected
        )
    closed, prim0, initial_time, time_step = original._load_payload()
    values = (prim0, initial_time, time_step)
    snapshots = [
        (np.asarray(value).dtype.str, tuple(value.shape), np.asarray(value).tobytes())
        for value in values
    ]
    assert all(np.asarray(value).dtype == np.dtype(np.float64) for value in values)
    fn = original._feed_forward_fn(closed)
    original_manifest = typed_trace_manifest(closed)
    model, native, eager, compiled = _export(
        fn, values, native_dir=tmp_path / "issue52-native"
    )
    targets = _capture_original_cpu_targets(model, values, tmp_path / "issue52-targets")
    assert len(model.graph.input) == 3
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        for value in model.graph.input
    )
    assert [_shape(value) for value in model.graph.input] == [
        tuple(value.shape) for value in values
    ]
    assert [
        (value.type.tensor_type.elem_type, _shape(value))
        for value in model.graph.output
    ] == [
        (onnx.TensorProto.DOUBLE, tuple(variable.aval.shape))
        for variable in _body(native).outvars
    ]
    assert all(value.dtype == np.dtype(np.float64) for value in eager + compiled)
    equations = [equation for body in _scopes(native) for equation in body.eqns]
    scans = [equation for equation in equations if equation.primitive.name == "scan"]
    assert sorted(
        (equation.params["length"], scan_arity(equation.params, len(equation.invars)))
        for equation in scans
    ) == [(1, (5, 5, 0)), (100, (6, 4, 0))]
    remats = [equation for equation in equations if equation.primitive.name == "remat2"]
    assert len(remats) == 1
    assert len(remats[0].invars) == 10 and len(remats[0].outvars) == 7
    assert {"scatter", "scatter-add", "scatter-mul"} <= {
        equation.primitive.name for equation in equations
    }
    # Distinct native field5, stencil6 and outer stacked1 roles survive tracing;
    # neither hints nor post-export stamps may merge those physical dimensions.
    shapes = {
        tuple(var.aval.shape) for equation in equations for var in equation.outvars
    }
    assert {(5, 210, 1, 1), (201, 6), (5, 201, 6, 1, 1), (1, 5, 210, 1, 1)} <= shapes
    _issue52_loop_boundaries(model, scans)
    _issue52_loop_boundary_corruption_controls(model, scans)
    assert all(target["returned"] for target in targets), targets
    _accept(model, values, eager, compiled)
    assert typed_trace_manifest(closed) == original_manifest
    assert [
        (np.asarray(value).dtype.str, tuple(value.shape), np.asarray(value).tobytes())
        for value in values
    ] == snapshots
    for relative, expected in _ISSUE52_HASHES.items():
        assert (
            hashlib.sha256((fixture_root / relative).read_bytes()).hexdigest()
            == expected
        )


@pytest.mark.parametrize("failure", ["eager", "jit", "both", "source-trace"])
def test_native_capture_retains_both_roles_before_source_assertions(
    failure: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Failure-path control for observation ordering only; this never substitutes
    # a result in the original scientific model or a target acceptance check.
    calls = []
    original = np.asarray([1.00000000001], np.float64)

    def target(value: Any) -> Any:
        calls.append("eager")
        if failure in {"eager", "both"}:
            raise RuntimeError("deliberate eager capture error")
        return value

    def fake_jit(fresh_target: Callable[..., Any]) -> Callable[..., Any]:
        assert fresh_target is not target

        def compiled(value: Any) -> Any:
            calls.append("jit")
            if failure in {"jit", "both"}:
                raise RuntimeError("deliberate JIT capture error")
            return value

        return compiled

    def source_trace(*arguments: Any, **keywords: Any) -> Any:
        calls.append("source")
        raise AssertionError("deliberate SOURCE trace assertion")

    monkeypatch.setattr(jax, "jit", fake_jit)
    monkeypatch.setitem(globals(), "_native_source_trace", source_trace)
    with pytest.raises(AssertionError):
        _export(target, (original,), native_dir=tmp_path)
    assert calls == (
        ["eager", "jit", "source"] if failure == "source-trace" else ["eager", "jit"]
    )
    observations = json.loads((tmp_path / "native-observations.json").read_text())
    for role in ("eager", "jit"):
        expected_raised = failure in {role, "both"}
        observation = observations[role]
        assert observation["attempted"] is True
        assert observation["raised"] is expected_raised
        assert observation["returned"] is not expected_raised
        assert observation["status"] == ("raised" if expected_raised else "returned")
        with np.load(
            tmp_path / observation["array_file"], allow_pickle=False
        ) as arrays:
            if expected_raised:
                assert not arrays.files
                assert observation["exception"]["type"] == "builtins.RuntimeError"
                assert role.lower() in observation["exception"]["message"].lower()
                assert "RuntimeError" in observation["exception"]["traceback"]
            else:
                assert arrays.files == ["output_0"]
                np.testing.assert_array_equal(
                    arrays["output_0"].view(np.uint64), original.view(np.uint64)
                )
                assert observation["exception"] is None


def test_original_masked_gather_trig_source23_fresh_native_both_cpu_profiles(
    tmp_path: Path,
) -> None:
    from tests.extra_tests.test_masked_gather_trig import _masked_gather_trig

    # Keep the original function, exact original data, native INT64 indices,
    # original opset23, and original native-first tolerance orientation.
    values = (
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([0, 2], dtype=np.int64),
    )
    snapshots = [(value.dtype.str, value.shape, value.tobytes()) for value in values]
    model, native, eager, compiled = _export(
        _masked_gather_trig, values, opset=23, native_dir=tmp_path / "masked-native"
    )
    original_file = Path(__file__).resolve().parents[1] / "test_masked_gather_trig.py"
    assert hashlib.sha256(original_file.read_bytes()).hexdigest() == (
        "7b808616d943f6dd9f63ca2f89867e491ecde379c8c6c57b5fa94b035683188c"
    )
    assert _signature(_body(native).invars) == (
        ("float64", (4, 3), False),
        ("int64", (2,), False),
    )
    assert _signature(_body(native).outvars) == (("float64", (2, 3), False),)
    assert len(eager) == len(compiled) == len(model.graph.output) == 1
    assert _shape(model.graph.input[0]) == (4, 3)
    assert _shape(model.graph.input[1]) == (2,)
    assert _shape(model.graph.output[0]) == (2, 3)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    assert model.graph.input[1].type.tensor_type.elem_type == onnx.TensorProto.INT64
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
    observations = {}
    for optimization in ("disabled", "all"):
        observation: dict[str, Any] = {
            "attempted": True,
            "returned": False,
            "raised": False,
            "status": "attempted",
            "arrays": [],
            "providers": None,
            "exception": None,
        }
        observations[optimization] = observation
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                if optimization == "disabled"
                else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                model.SerializeToString(), options, providers=["CPUExecutionProvider"]
            )
            observation["providers"] = session.get_providers()
            outputs = session.run(
                None,
                {
                    port.name: value
                    for port, value in zip(session.get_inputs(), values, strict=True)
                },
            )
            for value in outputs:
                observation["arrays"].append(np.asarray(value).copy())
            observation.update(returned=True, status="returned")
        except Exception as exc:
            observation.update(
                raised=True,
                status="raised",
                exception={
                    "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        np.savez(
            tmp_path / f"masked-target-{optimization}.npz",
            **{
                f"output_{index}": value
                for index, value in enumerate(observation["arrays"])
            },
        )
    (tmp_path / "masked-target-observations.json").write_text(
        json.dumps(
            {
                optimization: {
                    key: value for key, value in observation.items() if key != "arrays"
                }
                for optimization, observation in observations.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    # Both CPU profiles and their returned arrays/raised states are retained
    # before either comparison. An actual exception or mismatch stays a failure.
    assert all(
        observation["status"] == "returned" for observation in observations.values()
    ), observations
    for observation in observations.values():
        assert observation["providers"] == ["CPUExecutionProvider"]
        assert len(observation["arrays"]) == 1
        actual = observation["arrays"][0]
        for reference in (eager[0], compiled[0]):
            assert reference.shape == actual.shape == (2, 3)
            assert reference.dtype == actual.dtype == np.dtype(np.float64)
            np.testing.assert_allclose(reference, actual, rtol=5e-8, atol=1e-12)
    assert [
        (value.dtype.str, value.shape, value.tobytes()) for value in values
    ] == snapshots
    assert hashlib.sha256(original_file.read_bytes()).hexdigest() == (
        "7b808616d943f6dd9f63ca2f89867e491ecde379c8c6c57b5fa94b035683188c"
    )


@pytest.mark.parametrize("kind", ["scatter", "scatter-add", "scatter-mul"])
@pytest.mark.parametrize(
    "mode,start,window_start",
    [
        ("clip", -1, 0),
        ("clip", 0, 0),
        ("clip", 3, 3),
        ("clip", 4, 3),
        ("drop", -1, None),
        ("drop", 0, 0),
        ("drop", 3, 3),
        ("drop", 4, None),
    ],
)
def test_native_fp64_scatter_modes_preserve_whole_window_boundaries(
    kind: str, mode: str, start: int, window_start: int | None
) -> None:
    operation = {
        "scatter": jax.lax.scatter,
        "scatter-add": jax.lax.scatter_add,
        "scatter-mul": jax.lax.scatter_mul,
    }[kind]
    native_mode = {
        "clip": jax.lax.GatherScatterMode.CLIP,
        "drop": jax.lax.GatherScatterMode.FILL_OR_DROP,
    }[mode]
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(0, 1, 2, 3),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1,),
    )

    def target(operand: Any, indices: Any, updates: Any) -> Any:
        return operation(
            operand,
            indices,
            updates,
            dimensions,
            indices_are_sorted=True,
            unique_indices=True,
            mode=native_mode,
        )

    operand = np.arange(1, 13, dtype=np.float64).reshape(2, 6, 1, 1)
    indices = np.asarray([start], dtype=np.int64)
    updates = np.asarray(
        [
            [2.00000000001, 3.00000000003, 5.00000000007],
            [7.00000000011, 11.00000000013, 13.00000000017],
        ],
        dtype=np.float64,
    ).reshape(2, 3, 1, 1)
    values = (operand, indices, updates)
    assert np.any(updates != updates.astype(np.float32).astype(np.float64))
    # The valid starts are exactly 0..3. The explicit cases above require
    # clipping the start of the entire three-cell window, or dropping the
    # entire partial-overlap window. No individual update element is clipped
    # or retained. Each cell has at most one add/multiply, with no overlap.
    expected = operand.copy()
    if window_start is not None:
        window = expected[:, window_start : window_start + 3, :, :]
        if kind == "scatter":
            window[...] = updates
        elif kind == "scatter-add":
            window[...] += updates
        else:
            window[...] *= updates

    model, native, eager, compiled = _export(target, values)
    _, equation = _one_equation(native)
    assert equation.primitive.name == kind
    assert equation.params["mode"] is native_mode
    assert _signature(equation.invars) == (
        ("float64", (2, 6, 1, 1), False),
        ("int64", (1,), False),
        ("float64", (2, 3, 1, 1), False),
    )
    _accept(model, values, eager, compiled)
    for reference in (eager, compiled):
        assert len(reference) == 1
        assert reference[0].shape == operand.shape
        assert reference[0].dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(reference[0], expected)


@pytest.mark.parametrize("parameters", [{}, {"out_dtype": None}])
def test_scatter_multiply_accepts_only_canonical_unspecified_output_dtype(
    parameters: dict[str, Any],
) -> None:
    native = jax.make_jaxpr(_scatter("scatter-mul"))(*_scatter_values())
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    before = typed_trace_manifest(native)
    equation = _equation(original)
    reducer = _body(equation.params["update_jaxpr"])
    changed = reducer.eqns[0].replace(params=parameters)
    equation.params["update_jaxpr"] = reducer.replace(eqns=[changed])
    validate_source_jaxpr(_corrupted_graph(body, equation), 21)
    assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("kind", ["scatter-add", "scatter-mul"])
@pytest.mark.parametrize(
    "corruption", ["out-dtype", "foreign-parameter", "operand-order", "result-identity"]
)
def test_scatter_scalar_reducer_rejects_parameter_and_variable_corruption(
    kind: str, corruption: str
) -> None:
    native = jax.make_jaxpr(_scatter(kind))(*_scatter_values())
    body, original = _one_equation(native)
    validate_source_jaxpr(body, 21)
    before = typed_trace_manifest(native)
    equation = _equation(original)
    reducer = _body(equation.params["update_jaxpr"])
    inner = reducer.eqns[0]
    if corruption == "out-dtype":
        changed = inner.replace(params={"out_dtype": np.dtype(np.float64)})
    elif corruption == "foreign-parameter":
        changed = inner.replace(params={**inner.params, "foreign": None})
    elif corruption == "operand-order":
        changed = inner.replace(invars=list(reversed(inner.invars)))
    else:
        changed = inner.replace(outvars=[Var(inner.outvars[0].aval)])
    equation.params["update_jaxpr"] = reducer.replace(eqns=[changed])
    _rejected(_corrupted_graph(body, equation), {kind})
    assert typed_trace_manifest(native) == before


@pytest.mark.parametrize("kind", ["scatter", "scatter-add", "scatter-mul"])
def test_native_fp64_scatter_drop_selects_entire_mixed_valid_windows(kind: str) -> None:
    operation = {
        "scatter": jax.lax.scatter,
        "scatter-add": jax.lax.scatter_add,
        "scatter-mul": jax.lax.scatter_mul,
    }[kind]
    dimensions = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3, 4),
        inserted_window_dims=(),
        scatter_dims_to_operand_dims=(1,),
    )
    mode = jax.lax.GatherScatterMode.FILL_OR_DROP

    def target(operand: Any, indices: Any, updates: Any) -> Any:
        return operation(
            operand,
            indices,
            updates,
            dimensions,
            indices_are_sorted=True,
            unique_indices=True,
            mode=mode,
        )

    operand = np.arange(1, 13, dtype=np.float64).reshape(2, 6, 1, 1)
    indices = np.asarray([[-1], [0], [3], [4]], dtype=np.int64)
    updates = (np.arange(1, 25, dtype=np.float64) + np.float64(1e-10)).reshape(
        4, 2, 3, 1, 1
    )
    assert np.any(updates != updates.astype(np.float32).astype(np.float64))
    values = (operand, indices, updates)
    expected = operand.copy()
    # N=4 original windows, K=2 valid windows. Only rows 1 and 2 survive;
    # their starts 0 and 3 form disjoint complete three-cell windows.
    for row, start in ((1, 0), (2, 3)):
        window = expected[:, start : start + 3, :, :]
        if kind == "scatter":
            window[...] = updates[row]
        elif kind == "scatter-add":
            window[...] += updates[row]
        else:
            window[...] *= updates[row]

    model, native, eager, compiled = _export(target, values)
    _, equation = _one_equation(native)
    assert equation.primitive.name == kind
    assert equation.params["mode"] is mode
    assert _signature(equation.invars) == (
        ("float64", (2, 6, 1, 1), False),
        ("int64", (4, 1), False),
        ("float64", (4, 2, 3, 1, 1), False),
    )
    _accept(model, values, eager, compiled)
    for reference in (eager, compiled):
        assert len(reference) == 1
        assert reference[0].shape == operand.shape
        assert reference[0].dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(reference[0], expected)
