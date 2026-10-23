# tests/extra_tests/converter/test_source_model_native_bindings_exports.py

"""Original five full models: retained native roles, symbolic artifact and CPU profiles.

Finite FP32 sample criterion frozen before outputs at 67cd821a... and independently
accepted at 69673bf6... . Raw None criteria/report statuses are never replaced.
Failures retain the actual earliest event; no source or numerical acceptance waiver.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import partial
import hashlib
import inspect
from importlib import metadata as import_metadata
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from jax2onnx import user_interface as ui
from jax2onnx._compat.jax import (
    native_context_snapshot,
    native_transformed_execution_context,
    require_native_context_restored,
)
from jax2onnx.converter import conversion_api
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.plugins import plugin_system
from jax2onnx.plugins.plugin_system import FunctionPlugin
from jax2onnx.precision import (
    CheckStatus,
    _artifact_bytes,
    get_precision_report,
    precision_report_to_json,
)
from scripts.source_precision_inventory import canonical_contract
from tests.extra_tests.converter.source_integer_structure import (
    check_integer_source_structure,
    check_integer_callback_counterpart,
    has_integer_source_family,
)
from tests.extra_tests.converter.source_native_input_usage import NativeInputUsage
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.source_precision_support import (
    SourceProvenance,
    _input_data,
    raw_source_case,
    raw_source_scenarios,
    source_configuration,
)
from tests.extra_tests.converter.test_source_fixed_parameter_models import (
    _CONTRACTS,
    _instantiate,
    _state,
)
from tests.extra_tests.converter.test_source_model_native_bindings import (
    _PLUGINS,
    _contexts,
    _fresh,
)

# Immutable criterion-only review does not approve code/application/execution.
_CRITERION_SHA256 = "67cd821a4c40e0464f09a450e7c66afac99a0cef26b8569b17bd842f51c60c51"
_CRITERION_REVIEW_SHA256 = (
    "69673bf6eb4497483d5cf47370fb81a5afd2117a7a6408c70a16677059ed438b"
)
_INDICES = (0, 1, 3, 4, 20)
_OUTPUTS = {0: (1024, 3144), 1: (1024, 768), 3: (1024, 768), 4: (1024, 768), 20: (4,)}
_RTOL = _ATOL = 1e-5
_ROLES = ("eager", "jit-first-for-sample", "same-jit-warmed")


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, default=repr) + "\n")


def _array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        weak_type=bool(getattr(value, "weak_type", False)),
        bytes=array.nbytes,
        sha256=_sha(array.tobytes()),
    )


def _state_binding(target: Any) -> dict[str, Any]:
    arrays, tree = _state(target)
    module = target if isinstance(target, nnx.Module) else target._model
    rngs = getattr(target, "_rngs", None)
    rng_leaves, rng_tree = jax.tree.flatten(rngs)
    rng_binding = []
    for leaf in rng_leaves:
        if hasattr(leaf, "dtype") and jax.dtypes.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            rng_binding.append(
                dict(
                    identity=id(leaf),
                    logical_dtype=str(leaf.dtype),
                    key_data=_array(jax.random.key_data(leaf)),
                )
            )
        else:
            rng_binding.append(dict(identity=id(leaf), value=repr(leaf)))
    return dict(
        bridge_rng_tree=str(rng_tree),
        bridge_rngs=rng_binding,
        tree=tree,
        arrays={name: _array(value) for name, value in arrays.items()},
        identities=[
            (list(path), type(value).__module__, type(value).__qualname__, id(value))
            for path, value in nnx.iter_graph(module)
            if isinstance(value, (nnx.Module, nnx.Variable))
        ],
        graphdef=repr(nnx.split(module)[0]),
        module_identity=id(module),
        target_identity=id(target),
    )


def _configuration() -> dict[str, Any]:
    values = jax.config.values
    holders = jax.config._value_holders
    assert set(values) == set(holders)

    def typed(value: Any) -> tuple[str, str]:
        return f"{type(value).__module__}.{type(value).__qualname__}", repr(value)

    return dict(
        keys=sorted(values),
        holders={k: id(v) for k, v in holders.items()},
        values={k: typed(v) for k, v in values.items()},
        local={
            k: typed(v.get_local())
            for k, v in holders.items()
            if hasattr(v, "get_local")
        },
        global_values={
            k: typed(v.get_global())
            for k, v in holders.items()
            if hasattr(v, "get_global")
        },
    )


def _require_output(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    assert jax.tree.structure(value) == jax.tree.structure(
        jnp.asarray(0, dtype=jnp.float32)
    ), "Original single-array output tree required"
    array = np.asarray(value)
    assert array.dtype == np.dtype(
        "float32"
    ), "Finite full-owner criterion applies only to actual FP32 output"
    assert array.shape == shape
    assert np.isfinite(array).all()
    return array


def _compare(actual: Any, reference: Any, shape: tuple[int, ...]) -> dict[str, Any]:
    a, r = _require_output(actual, shape), _require_output(reference, shape)
    # Float64 arithmetic represents FP32 inputs exactly; reference sets scaling.
    difference = np.abs(a.astype(np.float64) - r.astype(np.float64))
    bound = _ATOL + _RTOL * np.abs(r.astype(np.float64))
    assert np.all(difference <= bound), "Frozen reference-scaled component bound failed"
    return dict(
        max_absolute_difference=float(difference.max(initial=0)),
        all_components=True,
        criterion_sha256=_CRITERION_SHA256,
    )


def _native_wrapper(target: Any, *, spectral: bool) -> tuple[Any, Any]:
    if not spectral:
        assert isinstance(target, nnx.Module)

        def call(module: Any, x: Any, deterministic: Any) -> Any:
            return module(x, deterministic=deterministic)

        return nnx.jit(call), target
    assert isinstance(target._model, nnx.Module)
    bridge, rngs = type(target), target._rngs

    def call_bridge(module: Any, x: Any) -> Any:
        # Same original bridge convention and actual module/RNG owner, never a new factory/model.
        return bridge(module, rngs)(x, update_stats=False)

    return nnx.jit(call_bridge), target._model


@pytest.mark.parametrize("index", _INDICES)
@pytest.mark.parametrize("caller_x64", [False, True], ids=["raw-x64-off", "raw-x64-on"])
def test_original_full_model_native_bindings_cpu_gate(
    index: int,
    caller_x64: bool,
    tmp_path: Path,
) -> None:
    origin, ordinal, digest = _CONTRACTS[index]
    case = raw_source_case(origin, ordinal)
    unresolved: list[str] = []
    contract = canonical_contract(
        {
            k: v
            for k, v in case.items()
            if k not in {"testcase", "description", "context", "component"}
        },
        unresolved=unresolved,
    )
    assert not unresolved
    assert (
        _sha(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
        == digest
    )
    scenarios = raw_source_scenarios(case, SourceProvenance(origin, ordinal))
    scenario = scenarios[int(caller_x64)]
    assert scenario.input_specs is not None
    assert scenario.criteria is None
    root = Path(os.environ.get("JAX2ONNX_NATIVE_BINDINGS_EVIDENCE_DIR", tmp_path))
    folder = root / f"{index:02d}-{'x64-on' if caller_x64 else 'x64-off'}"
    folder.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = dict(
        origin=origin,
        ordinal=ordinal,
        contract=contract,
        contract_sha256=digest,
        criterion_sha256=_CRITERION_SHA256,
        criterion_review_sha256=_CRITERION_REVIEW_SHA256,
        raw_criteria=None,
        original_raw_report_status="UNCHANGED; supplementary samples do not relabel prior observations",
        events=[],
        status="PARTIAL",
        source_hashes={},
        native={},
        cpu={},
        comparisons={},
    )
    source_paths = [Path(ui.__file__), Path(plugin_system.__file__)]
    source_paths += [Path(inspect.getfile(plugin)) for plugin in _PLUGINS]
    # Original implementation and every leaf/helper read by this owner are pinned by ROOT's full source-inputs receipt as well.
    source_paths += [
        Path(__file__),
        Path(__file__).with_name("test_source_model_native_bindings.py"),
    ]
    record["source_hashes"] = {
        str(p.resolve()): _sha(p.read_bytes()) for p in source_paths
    }
    loaded = dict(
        jax=jax.__version__,
        numpy=np.__version__,
        onnx=onnx.__version__,
        ort=ort.__version__,
        ort_distribution=import_metadata.version("onnxruntime"),
    )
    assert loaded["ort"] == loaded["ort_distribution"]
    record["versions"] = loaded

    def checkpoint() -> None:
        _json(folder / "receipt.json", record)

    checkpoint()
    spectral = index == 20
    controls = (
        [dict(case["input_params"])]
        if spectral
        else [dict(case["input_params"]), {"deterministic": False}]
    )
    params = dict(case["input_params"])
    try:
        with source_configuration(scenario), native_transformed_execution_context():
            target = _instantiate(case["callable"])
            export_target = partial(target, update_stats=False) if spectral else target
            export_params = None if spectral else params
            if spectral:
                assert export_target.func is target and export_target.args == ()
                assert export_target.keywords == params == {"update_stats": False}
                assert export_target.keywords["update_stats"] is False
            record["export_callable_binding"] = dict(
                original_target_identity=id(target),
                actual_callable_identity=id(export_target),
                preparation=(
                    "same-original-target-static-False-partial"
                    if spectral
                    else "original-target-runtime-BOOLs"
                ),
                original_params=params,
                actual_input_params=export_params,
                static_only=spectral,
            )
            baseline, configuration = _state_binding(target), _configuration()
            native_context = native_context_snapshot()
            held_config_holders = dict(jax.config._value_holders)
            record["original_state_binding"] = baseline
            record["complete_configuration"] = configuration
            record["native_context"] = repr(native_context)
            wrapper, module = _native_wrapper(target, spectral=spectral)
            record["jit_wrapper_identity"] = id(wrapper)
            record["jit_module_identity"] = id(module)
            held_graph = tuple(
                (tuple(path), value)
                for path, value in nnx.iter_graph(module)
                if isinstance(value, (nnx.Module, nnx.Variable))
            )
            invocation = 0
            feeds: dict[int, tuple[Any, ...]] = {}
            references: dict[tuple[int, bool | None, str], Any] = {}

            def guard() -> None:
                if spectral:
                    assert export_target.func is target and export_target.args == ()
                    assert export_target.keywords == params == {"update_stats": False}
                    assert export_target.keywords["update_stats"] is False
                    assert export_params is None
                else:
                    assert export_target is target and export_params is params
                assert _state_binding(target) == baseline
                assert _configuration() == configuration
                current_holders = jax.config._value_holders
                assert set(current_holders) == set(held_config_holders)
                assert all(
                    current_holders[k] is v for k, v in held_config_holders.items()
                )
                current_graph = tuple(
                    (tuple(path), value)
                    for path, value in nnx.iter_graph(module)
                    if isinstance(value, (nnx.Module, nnx.Variable))
                )
                assert len(current_graph) == len(held_graph)
                assert all(
                    p == q and a is b
                    for (p, a), (q, b) in zip(current_graph, held_graph, strict=True)
                )
                require_native_context_restored(native_context)
                assert {
                    str(p.resolve()): _sha(p.read_bytes()) for p in source_paths
                } == record["source_hashes"]
                for batch, arrays in feeds.items():
                    assert [_array(a) for a in arrays] == record["feeds"][str(batch)]

            def observe(
                stage: str,
                fn: Callable[[], Any],
                *,
                ort_return: bool = False,
                postcheck: Callable[[], None] | None = None,
            ) -> Any:
                event: dict[str, Any] = dict(stage=stage, status="BEFORE_CALL")
                record["events"].append(event)
                checkpoint()
                try:
                    guard()
                except Exception as exc:
                    event.update(
                        status="REJECTED_BEFORE_CALL",
                        precheck_failure=dict(
                            type=type(exc).__name__, message=str(exc)
                        ),
                    )
                    checkpoint()
                    raise
                try:
                    result = fn()
                except Exception as exc:
                    event.update(
                        status="ACTUAL_EXCEPTION",
                        exception_type=type(exc).__name__,
                        exception=str(exc),
                    )
                    checkpoint()
                    raise
                # Retain actual returns before any output/context/state assertion or cleanup.
                event.update(
                    status="ACTUAL_RETURN", tree=str(jax.tree.structure(result))
                )
                checkpoint()
                leaves = jax.tree.leaves(result)
                event["arrays"] = []
                for n, leaf in enumerate(leaves):
                    path = folder / f"event-{len(record['events']):03d}-array-{n}.npy"
                    np.save(path, np.asarray(leaf), allow_pickle=False)
                    event["arrays"].append(
                        dict(
                            path=path.name,
                            sha256=_sha(path.read_bytes()),
                            actual=_array(leaf),
                        )
                    )
                checkpoint()
                try:
                    guard()
                    if postcheck is not None:
                        postcheck()
                    if ort_return:
                        assert type(result) is list and len(result) == 1
                        return result[0]
                    return result
                except Exception as exc:
                    event["postcheck_failure"] = dict(
                        type=type(exc).__name__, message=str(exc)
                    )
                    checkpoint()
                    raise

            record["feeds"] = {}
            for batch in (2, 5):
                shapes = tuple(
                    tuple(batch if d == "B" else d for d in shape)
                    for shape in case["input_shapes"]
                )
                _, arguments = _input_data(
                    case,
                    replace(scenario, concrete_shapes=shapes),
                    (case["context"], case["component"], case["testcase"]),
                )
                feeds[batch] = arguments
                record["feeds"][str(batch)] = [_array(a) for a in arguments]
                for n, argument in enumerate(arguments):
                    np.save(
                        folder / f"feed-B{batch}-{n}.npy",
                        np.asarray(argument),
                        allow_pickle=False,
                    )
                for control in controls:
                    flag = None if spectral else control["deterministic"]
                    key = f"B{batch}-{'staticFalse' if spectral else flag}"
                    sample: dict[str, Any] = dict(control=control, roles={})
                    record["native"][key] = sample
                    for role in _ROLES:

                        def fn() -> Any:
                            if role == "eager":
                                return target(*arguments, **control)
                            if spectral:
                                return wrapper(module, *arguments)
                            return wrapper(
                                module, *arguments, jnp.asarray(flag, dtype=jnp.bool_)
                            )

                        output = observe(f"native-{key}-{role}", fn)
                        shape = (batch, *_OUTPUTS[index])
                        _require_output(output, shape)
                        references[(batch, flag, role)] = output
                        invocation += role != "eager"
                        sample["roles"][role] = dict(
                            actual=_array(output),
                            wrapper_identity=None if role == "eager" else id(wrapper),
                            wrapper_invocation=None if role == "eager" else invocation,
                            is_wrapper_first_invocation=role != "eager"
                            and invocation == 1,
                            role_label=(
                                "first JIT call for this retained sample"
                                if role == "jit-first-for-sample"
                                else role
                            ),
                        )
                        checkpoint()
                    # Retain native disagreement; never choose one role as a favorable CPU reference.
                    sample["role_disagreements"] = {
                        role: dict(
                            byte_equal=_array(references[(batch, flag, role)])
                            == _array(references[(batch, flag, "eager")])
                        )
                        for role in _ROLES[1:]
                    }

            b = jax_export.symbolic_shape("B")[0]
            symbolic_specs = tuple(
                jax.ShapeDtypeStruct(
                    tuple(b if d == "B" else d for d in shape),
                    spec.dtype,
                    weak_type=spec.weak_type,
                )
                for shape, spec in zip(
                    case["input_shapes"], scenario.input_specs, strict=True
                )
            )
            native_trace = jax.make_jaxpr(_fresh(lambda *xs: target(*xs, **params)))(
                *symbolic_specs
            )
            _json(
                folder / "original-native-symbolic-trace.json",
                typed_trace_manifest(native_trace),
            )
            record["original_native_contexts"] = _contexts(native_trace)
            guard()
            with (
                precision_policy_scope(PrecisionPolicy.SOURCE),
                _activate_plugin_worlds(),
            ):
                full_world = jax.make_jaxpr(_fresh(lambda *xs: target(*xs, **params)))(
                    *symbolic_specs
                )
                _json(
                    folder / "activated-full-world-symbolic-trace.json",
                    typed_trace_manifest(full_world),
                )
                if spectral:
                    assert typed_trace_manifest(full_world) == typed_trace_manifest(
                        native_trace
                    ), "Original SpectralNorm full-world inline computation changed"
                    assert _contexts(full_world) == _contexts(native_trace)
                else:
                    # Keep legitimate named Function boundaries above and in the actual export.
                    # For the exact computation counterpart only, use the existing captured-
                    # original bypass for every actual registered Function in this same world.
                    function_names = frozenset(
                        plugin.name
                        for plugin in plugin_system.PLUGIN_REGISTRY.values()
                        if isinstance(plugin, FunctionPlugin)
                    )
                    assert function_names
                    previous_build = plugin_system._IN_FUNCTION_BUILD.get()
                    token = plugin_system._IN_FUNCTION_BUILD.set(
                        previous_build | function_names
                    )
                    record["inlined_counterpart_function_names"] = sorted(
                        function_names
                    )
                    try:
                        inlined = jax.make_jaxpr(
                            _fresh(lambda *xs: target(*xs, **params))
                        )(*symbolic_specs)
                        _json(
                            folder / "activated-inlined-original-symbolic-trace.json",
                            typed_trace_manifest(inlined),
                        )
                    finally:
                        plugin_system._IN_FUNCTION_BUILD.reset(token)
                        assert plugin_system._IN_FUNCTION_BUILD.get() is previous_build
                    assert typed_trace_manifest(inlined) == typed_trace_manifest(
                        native_trace
                    ), "Original GPT full-world typed computation changed"
                    assert _contexts(inlined) == _contexts(native_trace)
                    assert [_array(v) for v in inlined.consts] == [
                        _array(v) for v in native_trace.consts
                    ]
                    record["full_world_native_typed_counterpart"] = (
                        "EXACT_ORIGINAL_INLINE_TYPED_TRACE_CONTEXT_CONSTANTS"
                    )
                    checkpoint()
            guard()
            native_input_usage = NativeInputUsage()
            bodies: list[Any] = []
            original_body_trace = FunctionPlugin._trace_function_body
            original_build = FunctionPlugin._build_function
            original_emit = FunctionPlugin._emit_function_call
            original_finalize = conversion_api._build_and_finalize_ir_model
            built: list[Any] = []

            def retain_body(*args: Any, **kwargs: Any) -> Any:
                body = original_body_trace(*args, **kwargs)
                native_input_usage.retain_body(args[0], body)
                bodies.append(body)
                _json(
                    folder / f"source-function-body-{len(bodies)}.json",
                    typed_trace_manifest(body),
                )
                return body

            def retain_build(*args: Any, **kwargs: Any) -> Any:
                with native_input_usage.build(*args, **kwargs) as frame:
                    definition = original_build(*args, **kwargs)
                    native_input_usage.retain_definition(frame, definition)
                built.append(definition)
                _json(
                    folder / f"source-function-definition-{len(built)}.json",
                    dict(
                        domain=definition.domain,
                        name=definition.name,
                        overload=definition.overload,
                    ),
                )
                return definition

            def retain_emit(*args: Any, **kwargs: Any) -> Any:
                return native_input_usage.emit(original_emit, *args, **kwargs)

            def retain_finalize(*args: Any, **kwargs: Any) -> Any:
                model = original_finalize(*args, **kwargs)
                native_input_usage.retain_model(model)
                return model

            export_event: dict[str, Any] = dict(
                stage="export-original-source-callable", status="BEFORE_CALL"
            )
            record["events"].append(export_event)
            checkpoint()
            try:
                guard()
            except Exception as exc:
                export_event.update(
                    status="REJECTED_BEFORE_CALL",
                    precheck_failure=dict(type=type(exc).__name__, message=str(exc)),
                )
                checkpoint()
                raise
            try:
                with (
                    patch.object(FunctionPlugin, "_trace_function_body", retain_body),
                    patch.object(FunctionPlugin, "_build_function", retain_build),
                    patch.object(
                        FunctionPlugin, "_emit_function_call", staticmethod(retain_emit)
                    ),
                    patch.object(
                        conversion_api, "_build_and_finalize_ir_model", retain_finalize
                    ),
                ):
                    try:
                        artifact = ui._to_onnx(
                            export_target,
                            list(scenario.input_specs),
                            input_params=export_params,
                            opset=23,
                            return_mode="proto",
                            precision_policy=PrecisionPolicy.SOURCE,
                            **{
                                key: case[key]
                                for key in (
                                    "inputs_as_nchw",
                                    "outputs_as_nchw",
                                    "input_names",
                                    "output_names",
                                    "export_mode",
                                    "normalization_mode",
                                )
                                if key in case
                            },
                        )
                    except Exception as exc:
                        export_event.update(
                            status="ACTUAL_EXCEPTION",
                            exception_type=type(exc).__name__,
                            exception=str(exc),
                        )
                        checkpoint()
                        raise
                    export_event.update(
                        status="ACTUAL_RETURN",
                        returned_type=f"{type(artifact).__module__}.{type(artifact).__qualname__}",
                        returned_identity=id(artifact),
                    )
                    checkpoint()
                    try:
                        payload = artifact.SerializeToString(deterministic=True)
                        artifact_path = folder / "model.onnx"
                        artifact_path.write_bytes(payload)
                        export_event.update(
                            artifact_sha256=_sha(payload),
                            artifact_path=artifact_path.name,
                        )
                    except Exception as exc:
                        export_event["postcheck_failure"] = dict(
                            phase="actual-artifact-retention",
                            type=type(exc).__name__,
                            message=str(exc),
                        )
                        checkpoint()
                        raise
            except Exception as exc:
                if export_event["status"] == "BEFORE_CALL":
                    export_event.update(
                        status="REJECTED_BEFORE_CALL",
                        precheck_failure=dict(
                            type=type(exc).__name__, message=str(exc)
                        ),
                    )
                elif export_event["status"] == "ACTUAL_RETURN":
                    export_event.setdefault(
                        "postcheck_failure",
                        dict(
                            phase="export-observer-restoration",
                            type=type(exc).__name__,
                            message=str(exc),
                        ),
                    )
                checkpoint()
                raise
            try:
                record["function_definitions"] = [
                    (definition.domain, definition.name) for definition in built
                ]
                checkpoint()
                guard()
                assert FunctionPlugin._trace_function_body is original_body_trace
                assert FunctionPlugin._build_function is original_build
                assert FunctionPlugin._emit_function_call is original_emit
                assert conversion_api._build_and_finalize_ir_model is original_finalize
                assert _artifact_bytes(artifact) == (payload, _sha(payload))
                record["artifact_sidecars"] = (
                    []
                )  # Proto export must contain inline payloads; helper rejects external references.
                onnx.checker.check_model(artifact, full_check=True)
                report = get_precision_report(artifact)
                assert report is not None and report.profile == "source-static-v1"
                (folder / "original-report.json").write_text(
                    precision_report_to_json(report)
                )
                checks = {check.category: check.status for check in report.checks}
                assert (
                    checks["runtime-execution"]
                    is checks["numerical-acceptance"]
                    is CheckStatus.UNVERIFIED
                )
                assert len(artifact.graph.output) == 1
                dims = artifact.graph.output[0].type.tensor_type.shape.dim
                assert dims[0].dim_param == "B"
                assert [d.dim_value for d in dims[1:]] == list(_OUTPUTS[index])
                callback = case.get("post_check_onnx_graph")
                callback_result = None if callback is None else bool(callback(artifact))
                record["new_original_callback_observation"] = dict(
                    executed=callback is not None,
                    result=callback_result,
                    historical_raw_results="PRESERVED; this is a separate invocation",
                    criterion=None,
                )
                record["native_input_usage_correspondence"] = native_input_usage.seal(
                    artifact
                )
                if has_integer_source_family(native_trace):
                    record["source_integer_structural_amendment"] = {
                        "semantic_structure": check_integer_source_structure(
                            native_trace,
                            artifact,
                            require_function=True,
                            native_input_usage=native_input_usage,
                        ),
                        "callback_counterpart": check_integer_callback_counterpart(
                            callback, artifact, native_input_usage=native_input_usage
                        ),
                    }
                checkpoint()
                assert artifact.SerializeToString(deterministic=True) == payload
                guard()
            except Exception as exc:
                export_event["postcheck_failure"] = dict(
                    phase="artifact-validation",
                    type=type(exc).__name__,
                    message=str(exc),
                )
                checkpoint()
                raise
            for mode, optimization in (
                ("disabled", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
                ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
            ):
                options = ort.SessionOptions()
                options.graph_optimization_level = optimization
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                session_event: dict[str, Any] = dict(
                    stage=f"session-{mode}", status="BEFORE_CALL"
                )
                record["events"].append(session_event)
                checkpoint()
                guard()
                try:
                    session = ort.InferenceSession(
                        payload,
                        sess_options=options,
                        providers=["CPUExecutionProvider"],
                    )
                except Exception as exc:
                    session_event.update(
                        status="ACTUAL_EXCEPTION",
                        exception_type=type(exc).__name__,
                        exception=str(exc),
                    )
                    checkpoint()
                    raise
                session_event.update(
                    status="ACTUAL_RETURN", session_identity=id(session)
                )
                checkpoint()
                try:
                    session.disable_fallback()
                    actual_options = session.get_session_options()
                    assert session.get_providers() == ["CPUExecutionProvider"]
                    assert actual_options.graph_optimization_level == optimization
                    assert (
                        actual_options.execution_mode
                        == ort.ExecutionMode.ORT_SEQUENTIAL
                    )
                    assert (
                        actual_options.intra_op_num_threads
                        == actual_options.inter_op_num_threads
                        == 1
                    )
                except Exception as exc:
                    session_event["postcheck_failure"] = dict(
                        type=type(exc).__name__, message=str(exc)
                    )
                    checkpoint()
                    raise
                record["cpu"][mode] = dict(
                    session_identity=id(session),
                    providers=session.get_providers(),
                    optimization=str(actual_options.graph_optimization_level),
                    runtime_version=ort.__version__,
                    installation=loaded,
                    artifact_sha256=_sha(payload),
                    runs={},
                )

                def require_runtime() -> None:
                    assert {name: _array(a) for name, a in feed.items()} == entry[
                        "feed"
                    ]
                    assert _sha(artifact_path.read_bytes()) == _sha(payload)
                    assert artifact.SerializeToString(deterministic=True) == payload
                    options_after = session.get_session_options()
                    assert options_after.graph_optimization_level == optimization
                    assert (
                        options_after.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
                    )
                    assert (
                        options_after.intra_op_num_threads
                        == options_after.inter_op_num_threads
                        == 1
                    )
                    assert session.get_providers() == ["CPUExecutionProvider"]
                    assert ort.__version__ == loaded["ort_distribution"]

                for batch in (2, 5):
                    for control in controls:
                        flag = None if spectral else control["deterministic"]
                        key = f"B{batch}-{'staticFalse' if spectral else flag}"
                        feed = {}
                        for port in session.get_inputs():
                            if port.name == case.get("input_names", ["in_0"])[0]:
                                feed[port.name] = np.asarray(feeds[batch][0])
                            else:
                                assert (
                                    port.name in control
                                    and port.type == "tensor(bool)"
                                    and port.shape == []
                                )
                                feed[port.name] = np.asarray(
                                    control[port.name], dtype=np.bool_
                                )
                        assert len(feed) == len(session.get_inputs())
                        entry = dict(
                            feed={name: _array(a) for name, a in feed.items()},
                            comparisons={},
                        )
                        record["cpu"][mode]["runs"][key] = entry
                        actual = observe(
                            f"cpu-{mode}-{key}",
                            lambda runtime_session=session: runtime_session.run(
                                None, feed
                            ),
                            ort_return=True,
                            postcheck=require_runtime,
                        )
                        for role in _ROLES:
                            entry["comparisons"][role] = _compare(
                                actual,
                                references[(batch, flag, role)],
                                (batch, *_OUTPUTS[index]),
                            )
                        checkpoint()
            if has_integer_source_family(native_trace):
                assert (
                    record["source_integer_structural_amendment"][
                        "callback_counterpart"
                    ]["other_callback_obligations"]
                    == "preserved and passed"
                )
            else:
                assert (
                    callback_result is not False
                ), "Original graph callback failed; supplementary finite numerics do not waive it"
            guard()
            record["status"] = "PASS_FINITE_ORIGINAL_FIVE_OWNER_SUBGATE"
            record["product_full_B_F_acceptance"] = (
                "UNCHANGED; this finite subgate alone is insufficient"
            )
            checkpoint()
    except Exception as exc:
        record["status"] = "FAIL"
        record["terminal_exception"] = dict(type=type(exc).__name__, message=str(exc))
        checkpoint()
        raise


def test_frozen_five_owner_output_contract_rejects_corruption() -> None:
    shape = (2, 4)
    reference = np.zeros(shape, dtype=np.float32)
    for damaged in (
        np.full(shape, 1, dtype=np.float32),
        np.full(shape, np.nan, dtype=np.float32),
        np.full(shape, np.inf, dtype=np.float32),
        np.zeros(shape, dtype=np.float64),
        np.zeros((1, 4), dtype=np.float32),
        (reference,),
        [reference],
    ):
        with pytest.raises(AssertionError):
            _compare(damaged, reference, shape)
    # Reference-scaled asymmetric predicate; swapping reference is forbidden.
    assert _RTOL == _ATOL == 1e-5
    assert (
        _CRITERION_SHA256
        == "67cd821a4c40e0464f09a450e7c66afac99a0cef26b8569b17bd842f51c60c51"
    )


def test_reference_scaled_direction_and_all_native_roles_are_required() -> None:
    shape = (2, 4)
    zero = np.zeros(shape, dtype=np.float32)
    edge = np.nextafter(np.float32(1e-5), np.float32(np.inf))
    actual = np.full(shape, edge, dtype=np.float32)
    assert np.all(
        np.abs(actual.astype(np.float64))
        <= _ATOL + _RTOL * np.abs(actual.astype(np.float64))
    )
    with pytest.raises(AssertionError):
        _compare(actual, zero, shape)
    _compare(zero, zero, shape)
    with pytest.raises(AssertionError):
        _compare(zero, np.ones(shape, dtype=np.float32), shape)
    with pytest.raises(AssertionError):
        _compare(zero, np.full(shape, np.nan, dtype=np.float32), shape)
