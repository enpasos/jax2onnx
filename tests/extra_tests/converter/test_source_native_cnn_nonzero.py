# tests/extra_tests/converter/test_source_native_cnn_nonzero.py

"""A priori finite nonzero samples of the unchanged original full CNN.

These receipts neither replace raw criteria=None nor activate public SOURCE.
The legacy callback failure and export report are retained independently.
"""

from __future__ import annotations

from builtins import ExceptionGroup
from collections import Counter
import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import inspect
from pathlib import Path
import struct
import time
import resource
from typing import Any
from unittest.mock import patch

from flax import nnx
import jax
import numpy as np
import onnx
import onnx_ir as ir
import pytest

from jax2onnx import user_interface as ui
from jax2onnx.converter.conversion_api import _activate_plugin_worlds
from jax2onnx.converter.precision_policy import PrecisionPolicy, precision_policy_scope
from jax2onnx.plugins.plugin_system import FunctionPlugin, _SourceFunctionTrace
from jax2onnx.precision import CheckStatus, get_precision_report
from tests.extra_tests.converter import source_cnn_acceptance as capture
from tests.extra_tests.converter import source_cnn_roundoff as exact
from tests.extra_tests.converter import source_precision_support as source
from tests.extra_tests.converter import test_source_native_cnn_pool as previous
from tests.extra_tests.converter.source_precision_diagnostics import (
    typed_trace_manifest,
)
from tests.extra_tests.converter.test_source_native_conv import _trace


_ORIGIN = "example::examples.nnx::CNN"
_CNN_SHA256 = "d2ab49cecada8847c81cb5e51a885399636e5c7c8c415f24a110850f28d4ff6f"


def _corrupt_actual_artifact_copies(
    proto: Any,
    parameters: dict[str, np.ndarray],
    contract: dict[str, Any],
    batch: int | str,
    named: bool,
    directory: Path,
) -> None:
    """Corrupt the actual saved artifact, preserving the original bytes."""
    directory.mkdir(parents=True, exist_ok=True)
    original = proto.SerializeToString()
    results = {}
    for family in (
        "weight",
        "pool-divisor",
        "pool-axes",
        "flatten-wiring",
        "pool-layout",
        "relu-extra-data",
        "selection-arm",
        "selection-axis",
        "selection-indices",
    ):
        damaged = copy.deepcopy(proto)
        scopes = [
            ("graph", damaged.graph),
            *((function.name, function) for function in damaged.functions),
        ]
        changed = False
        for owner, scope in scopes:
            if family.startswith("selection-"):
                final_outputs = {
                    tuple(item["outputs"])
                    for item in contract["operations"]
                    if item["owner"] == owner
                    and item["op"] == "Squeeze"
                    and item["result_stage"] == "conv1.relu"
                }
                producers = {name: node for node in scope.node for name in node.output}
                for final in scope.node:
                    if tuple(final.output) not in final_outputs:
                        continue
                    selected = producers[final.input[0]]
                    if family == "selection-arm":
                        choices = producers[selected.input[0]]
                        first, second = choices.input
                        choices.input[0], choices.input[1] = second, first
                    elif family == "selection-axis":
                        for attribute in selected.attribute:
                            if attribute.name == "axis":
                                attribute.i = 1
                    else:
                        indices = producers[producers[selected.input[1]].input[0]]
                        mask = indices.input[0]
                        inverted = mask + "__wrong_selection_mask"
                        index = next(
                            index
                            for index, node in enumerate(scope.node)
                            if tuple(node.output) == tuple(indices.output)
                        )
                        indices.input[0] = inverted
                        scope.node.insert(
                            index, onnx.helper.make_node("Not", [mask], [inverted])
                        )
                    changed = True
                    break
            elif family in ("pool-layout", "relu-extra-data"):
                stage = "conv1.blocks" if family == "pool-layout" else "conv1.relu"
                matches = {
                    tuple(item["outputs"]): item
                    for item in contract["operations"]
                    if item["owner"] == owner
                    and item["result_stage"] == stage
                    and item["op"]
                    in (("Reshape",) if family == "pool-layout" else ("Max", "Relu"))
                }
                for index, node in enumerate(scope.node):
                    if tuple(node.output) not in matches:
                        continue
                    info = matches[tuple(node.output)]
                    if family == "pool-layout":
                        block = node.output[0]
                        swapped = block + "__wrong_pool_layout"
                        for consumer in scope.node:
                            for position, name in enumerate(consumer.input):
                                if name == block:
                                    consumer.input[position] = swapped
                        scope.node.insert(
                            index + 1,
                            onnx.helper.make_node(
                                "Transpose", [block], [swapped], perm=[0, 3, 2, 1, 4, 5]
                            ),
                        )
                    else:
                        original_input = (
                            scope.input[0] if owner != "graph" else scope.input[0].name
                        )
                        if info["result_layout"] == "NCHW":
                            broadcast = node.output[0] + "__original_input_nchw"
                            scope.node.insert(
                                index,
                                onnx.helper.make_node(
                                    "Transpose",
                                    [original_input],
                                    [broadcast],
                                    perm=[0, 3, 1, 2],
                                ),
                            )
                        else:
                            assert info["result_layout"] == "NHWC"
                            broadcast = original_input
                        # A source-preserving lowerer may use unary Relu. Form
                        # its equivalent binary Max first, then add the bad Data.
                        if node.op_type == "Relu":
                            zero = node.output[0] + "__relu_zero"
                            scope.node.insert(
                                index,
                                onnx.helper.make_node(
                                    "Constant",
                                    [],
                                    [zero],
                                    value=onnx.numpy_helper.from_array(
                                        np.array(0, np.float32)
                                    ),
                                ),
                            )
                            node.op_type = "Max"
                            node.input.append(zero)
                        node.input.append(broadcast)
                    changed = True
                    break
            elif family == "flatten-wiring":
                flatten = {
                    tuple(item["outputs"])
                    for item in contract["operations"]
                    if item["owner"] == owner and item["result_stage"] == "flatten"
                }
                for node in scope.node:
                    if tuple(node.output) in flatten:
                        node.input[0] = (
                            scope.input[0] if owner != "graph" else scope.input[0].name
                        )
                        changed = True
                        break
            else:
                tensors = list(scope.initializer) if owner == "graph" else []
                tensors.extend(
                    attribute.t
                    for node in scope.node
                    if node.op_type == "Constant"
                    for attribute in node.attribute
                    if attribute.name == "value"
                )
                for tensor in tensors:
                    value = onnx.numpy_helper.to_array(tensor).copy()
                    if (
                        family == "weight"
                        and value.dtype == np.dtype("float32")
                        and value.shape in ((3, 3, 1, 32), (32, 1, 3, 3))
                    ):
                        value.reshape(-1)[0] = np.nextafter(
                            value.reshape(-1)[0], np.float32(np.inf)
                        )
                    elif (
                        family == "pool-divisor"
                        and value.dtype == np.dtype("float32")
                        and value.size == 1
                        and value.item() == 4
                    ):
                        value[...] = np.float32(8)
                    elif (
                        family == "pool-axes"
                        and value.dtype == np.dtype("int64")
                        and value.shape == (2,)
                        and tuple(value) == (2, 4)
                    ):
                        value[...] = [1, 4]
                    else:
                        continue
                    tensor.CopyFrom(
                        onnx.numpy_helper.from_array(value, name=tensor.name)
                    )
                    changed = True
                    break
            if changed:
                break
        assert changed, f"actual artifact {family} corruption was not exercised"
        (directory / f"{family}.onnx").write_bytes(damaged.SerializeToString())
        if family in ("pool-layout", "relu-extra-data") or family.startswith(
            "selection-"
        ):
            onnx.checker.check_model(damaged, full_check=True)
        try:
            capture.ArtifactContract(parameters, batch, named=named).check(damaged)
        except ValueError as exc:
            results[family] = {
                "status": "rejected",
                "detail": str(exc),
                "sha256": hashlib.sha256(damaged.SerializeToString()).hexdigest(),
            }
        else:
            raise AssertionError(
                f"actual artifact {family} corruption escaped structural rejection"
            )
        assert proto.SerializeToString() == original
    capture.write_json(directory / "receipt.json", results)


def _check_observations(
    observations: dict[str, dict[str, Any]], envelopes: dict[int, exact.Envelope]
) -> None:
    for name, observation in observations.items():
        if "session_contract" in observation:
            session = observation["session_contract"]
            assert "load_exception" not in session
            assert (
                session["providers"] == ["CPUExecutionProvider"]
                and session["runtime_graph_exists"]
            )
            assert session["artifact_sha256"] == session["reloaded_artifact_sha256"]
            assert session["denormal_override"] == "absent; no config entry is added"
            assert session["existing_denormal_entry"] is None
        assert observation["returned"], (name, observation)
        assert observation["exception"] is observation["guard_exception"] is None, (
            name,
            observation,
        )
        assert len(observation["arrays"]) == 1
        array = observation["arrays"][0]
        assert array.dtype == np.dtype("float32") and array.dtype.isnative
        batch = int(name.split("-", 1)[0][1:])
        envelopes[batch].check(
            array.shape,
            array.dtype.name,
            array.astype("<f4", copy=False).tobytes(order="C"),
        )


@pytest.mark.parametrize("ordinal", [0, 1])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("named", [False, True])
def test_original_full_nonzero_cnn_frozen_envelope_all_native_and_cpu_roles(
    ordinal: int, x64: bool, named: bool, tmp_path: Path, record_property: Any
) -> None:
    metadata = source.raw_source_case(_ORIGIN, ordinal)
    scenario = source.raw_source_scenarios(
        metadata, source.SourceProvenance(_ORIGIN, ordinal)
    )[int(x64)]
    scenario = replace(scenario, concrete_shapes=((3, 28, 28, 1),))
    recipe = metadata["callable"]
    identity = (
        str(metadata["context"]),
        str(metadata["component"]),
        str(metadata["testcase"]),
    )
    previous_model = previous._CNN
    previous_body = FunctionPlugin._trace_function_body
    outer_settings = capture.settings_snapshot()
    try:
        with source.source_configuration(scenario):
            try:
                specs, arguments = source._input_data(metadata, scenario, identity)
                model = recipe.instantiate()
            except Exception as exc:
                capture.write_json(
                    tmp_path / "construction-unavailable.json",
                    {
                        "exception": capture.exception_record(exc),
                        "unavailable_roles": ["eager", "fresh-jit", "disabled", "all"],
                        "reason": "original source/input construction did not return",
                    },
                )
                raise
            previous._CNN = model
            source_errors = {}
            source_bytes = b""
            try:
                source_file = Path(inspect.getfile(type(model)))
                source_bytes = source_file.read_bytes()
                (tmp_path / "unchanged-original-cnn.py").write_bytes(source_bytes)
            except Exception as exc:
                source_errors["source_text"] = capture.exception_record(exc)
            parameters = {}
            try:
                parameters = capture.parameters(model)
            except Exception as exc:
                source_errors["parameters"] = capture.exception_record(exc)
            state, state_arrays = {}, {}
            try:
                state, state_arrays = capture.source_snapshot(model)
            except Exception as exc:
                source_errors["graph_state"] = capture.exception_record(exc)
            settings = capture.settings_snapshot()
            stack = capture.stack_snapshot()
            feeds = {3: np.array(arguments[0], copy=True)}
            for batch in () if ordinal == 0 else (2, 5):
                _, extra = source._input_data(
                    metadata,
                    replace(scenario, concrete_shapes=((batch, 28, 28, 1),)),
                    identity,
                )
                feeds[batch] = np.array(extra[0], copy=True)
            for value in feeds.values():
                value.setflags(write=False)
            # Complete state, full parameters and exact original B3 feed are
            # persisted before any native evaluation, trace or export.
            np.savez(tmp_path / "original-state.npz", **state_arrays)
            np.savez(tmp_path / "original-parameters.npz", **parameters)
            np.savez(
                tmp_path / "original-inputs.npz",
                **{f"B{batch}": value for batch, value in feeds.items()},
            )
            capture.write_json(
                tmp_path / "original-source.json",
                {
                    "origin": _ORIGIN,
                    "ordinal": ordinal,
                    "constructor_seed": 0,
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "required_source_sha256": _CNN_SHA256,
                    "baseline_exceptions": source_errors,
                    "state": state,
                    "settings": settings,
                    "stack": stack,
                    "feeds": {
                        str(batch): capture.array_record(value)
                        for batch, value in feeds.items()
                    },
                    "raw_criteria": None,
                    "raw_callback": f"{metadata['post_check_onnx_graph'].__module__}.{metadata['post_check_onnx_graph'].__qualname__}",
                    "opset": 23,
                    "named": named,
                    "raw_options": {
                        name: metadata.get(name)
                        for name in (
                            "input_shapes",
                            "expected_output_shapes",
                            "run_only_f32_variant",
                            "run_only_dynamic",
                            "opset_version",
                            "input_params",
                            "inputs_as_nchw",
                            "outputs_as_nchw",
                            "export_mode",
                            "skip_numeric_validation",
                        )
                    },
                },
            )

            feed_records = {
                batch: capture.array_record(value) for batch, value in feeds.items()
            }

            def complete_guard() -> None:
                errors = []
                for label, expected, snapshot in (
                    (
                        "graph/state/full parameters",
                        state,
                        lambda: capture.source_snapshot(model)[0],
                    ),
                    ("caller/global settings", settings, capture.settings_snapshot),
                    (
                        "immutable feeds",
                        feed_records,
                        lambda: {
                            batch: capture.array_record(value)
                            for batch, value in feeds.items()
                        },
                    ),
                ):
                    try:
                        capture.require_unchanged(expected, snapshot(), label)
                    except Exception as exc:
                        errors.append(exc)
                if errors:
                    raise ExceptionGroup(
                        "original CNN source/state guards failed", errors
                    )

            native = capture.capture_native(
                model, feeds, tmp_path / "native", complete_guard
            )
            traces: dict[str, Any] = {}
            manifests: dict[str, Any] = {}
            # Independent trace attempts follow ALL native attempts, including
            # exceptions. Manifests are written before structural assertions.
            for role in ("native", "source"):
                try:
                    if role == "source":
                        with (
                            precision_policy_scope(PrecisionPolicy.SOURCE),
                            _activate_plugin_worlds(),
                        ):
                            traces[role] = _trace(model, specs)
                    else:
                        traces[role] = _trace(model, specs)
                    manifests[role] = typed_trace_manifest(traces[role])
                except Exception as exc:
                    manifests[role] = {"exception": capture.exception_record(exc)}
                capture.write_json(
                    tmp_path / f"{role}-typed-manifest.json", manifests[role]
                )
            bodies = []

            def observe_body(
                plugin: Any, callee: Any, equation: Any, captures: Any, **kwargs: Any
            ) -> Any:
                body = previous_body(plugin, callee, equation, captures, **kwargs)
                if callee is previous._named_cnn:
                    trace = kwargs["source_trace"]
                    bodies.append(
                        {
                            "trace_type": type(trace).__name__,
                            "source_trace_settings": {
                                name: getattr(trace, name)
                                for name in (
                                    "x64_enabled",
                                    "default_matmul_precision",
                                    "numpy_dtype_promotion",
                                    "numpy_rank_promotion",
                                    "debug_key_reuse",
                                )
                            },
                            "precision_policy": kwargs["precision_policy"].value,
                            "dynamic_capture_count": len(captures.dynamic),
                            "static_kwargs": dict(captures.static_kwargs),
                            "manifest": typed_trace_manifest(body),
                        }
                    )
                    capture.write_json(tmp_path / "actual-named-bodies.json", bodies)
                return body

            def named_call(value: Any) -> Any:
                return previous._named_cnn(value)

            export_started = time.monotonic()
            export_peak_before = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            proto = None
            blob = None
            export_error = None
            try:
                with patch.object(FunctionPlugin, "_trace_function_body", observe_body):
                    artifact = ui._to_onnx(
                        named_call if named else model,
                        specs,
                        opset=23,
                        precision_policy=PrecisionPolicy.SOURCE,
                        return_mode="ir",
                    )
                proto = ir.to_proto(artifact)
                blob = proto.SerializeToString()
                (tmp_path / "unchanged-original-source.onnx").write_bytes(blob)
            except Exception as exc:
                capture.write_json(
                    tmp_path / "export-exception.json", capture.exception_record(exc)
                )
                export_error = capture.exception_record(exc)
            artifact_sha = (
                hashlib.sha256(blob).hexdigest() if blob is not None else None
            )
            capture.write_json(
                tmp_path / "export.json",
                {
                    "artifact_sha256": artifact_sha,
                    "exception": export_error,
                    "export_wall_seconds": time.monotonic() - export_started,
                    "process_peak_rss_bytes_before": export_peak_before,
                    "process_peak_rss_bytes_after": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                    * 1024,
                    "numerical_receipt": "separate finite envelope; original criteria=None",
                },
            )
            remaining = capture.capture_oracles_and_targets(
                parameters,
                feeds,
                proto if blob is not None else None,
                tmp_path,
                complete_guard,
                artifact_sha256=artifact_sha,
                export_error=export_error,
            )
            envelopes = remaining["envelopes"]
            targets = remaining["targets"]
            oracle_errors = remaining["oracle_exceptions"]
            target_capture_error = remaining["target_capture_exception"]
            # All available native arrays, target arrays and optimized graphs
            # are frozen before callbacks or acceptance/control assertions.
            capture.write_json(
                tmp_path / "observation-capture-complete.json",
                {
                    "native_attempts": len(native),
                    "target_attempts": len(targets),
                    "export_exception": export_error,
                    "oracle_exceptions": oracle_errors,
                    "target_capture_exception": target_capture_error,
                },
            )
            report = None
            report_error = None
            if proto is not None:
                try:
                    report = get_precision_report(proto)
                except Exception as exc:
                    report_error = capture.exception_record(exc)
            capture.write_json(
                tmp_path / "actual-product-report.json",
                {
                    "report": repr(report),
                    "exception": report_error,
                    "unavailable": proto is None,
                    "raw_criteria": None,
                },
            )
            callback: dict[str, Any] = {
                "attempted": proto is not None and blob is not None
            }
            if callback["attempted"]:
                try:
                    callback["result"] = metadata["post_check_onnx_graph"](proto)
                except Exception as exc:
                    callback["exception"] = capture.exception_record(exc)
            else:
                callback["unavailable"] = "recorded original artifact export failed"
            capture.write_json(tmp_path / "original-raw-callback.json", callback)
            assert (
                export_error is None and proto is not None and blob is not None
            ), export_error
            assert target_capture_error is None, target_capture_error
            assert not oracle_errors, oracle_errors
            assert type(model) is recipe._ctor
            assert (
                f"{type(model).__module__}.{type(model).__qualname__}"
                == "jax2onnx.plugins.examples.nnx.cnn.CNN"
            )
            assert hashlib.sha256(source_bytes).hexdigest() == _CNN_SHA256
            assert not source_errors, source_errors
            assert metadata["testcase"] == (
                "simple_cnn_static" if ordinal == 0 else "simple_cnn"
            )
            assert metadata["input_shapes"] == [(3 if ordinal == 0 else "B", 28, 28, 1)]
            assert metadata["expected_output_shapes"] == [
                (3 if ordinal == 0 else "B", 10)
            ]
            assert metadata["run_only_f32_variant"] is True
            assert metadata.get("run_only_dynamic", False) is bool(ordinal)
            assert metadata.get("opset_version", 23) == 23
            assert recipe._dtype is recipe._instance is None
            assert recipe._args == () and set(recipe._kwargs) == {"rngs"}
            rng_recipe = recipe._kwargs["rngs"]
            assert rng_recipe.metadata["rng_trace"].kind == "nnx_rng"
            assert rng_recipe.metadata["rng_trace"].seed == 0
            assert scenario.criteria is None and metadata.get("input_params", {}) == {}
            complete_guard()
            assert proto.SerializeToString() == blob
            assert FunctionPlugin._trace_function_body is previous_body
            assert stack["native_backend"] == "cpu"
            assert (
                "exception" not in manifests["native"]
                and manifests["source"] == manifests["native"]
            )
            capture.write_json(
                tmp_path / "native-trace-contract.json",
                capture.trace_contract(traces["native"], 3 if ordinal == 0 else "B"),
            )
            assert Counter(
                item["sha256"] for item in state["parameters"].values()
            ) <= Counter(
                item["sha256"] for item in manifests["native"]["const_payloads"]
            )
            if named:
                assert bodies and all(
                    body["trace_type"] == _SourceFunctionTrace.__name__
                    and body["manifest"] == manifests["native"]
                    and body["dynamic_capture_count"] == 0
                    and body["static_kwargs"] == {}
                    and body["precision_policy"] == PrecisionPolicy.SOURCE.value
                    and body["source_trace_settings"]
                    == {
                        key: settings["effective"][key]
                        for key in body["source_trace_settings"]
                    }
                    for body in bodies
                )
            onnx.checker.check_model(proto, full_check=True)
            contract = capture.ArtifactContract(
                parameters, 3 if ordinal == 0 else "B", named=named
            ).check(proto)
            capture.write_json(tmp_path / "actual-artifact-contract.json", contract)
            _corrupt_actual_artifact_copies(
                proto,
                parameters,
                contract,
                3 if ordinal == 0 else "B",
                named,
                tmp_path / "corruption-controls",
            )
            assert report_error is None, report_error
            assert report is not None and report.profile == "source-static-v1"
            assert {check.category: check.status for check in report.checks} == {
                "representation": CheckStatus.VALIDATED,
                "lowering": CheckStatus.VALIDATED,
                "runtime-execution": CheckStatus.UNVERIFIED,
                "numerical-acceptance": CheckStatus.UNVERIFIED,
            }
            # Corrupt a separate actual model clone; the captured original
            # instance and all original parameter arrays remain unchanged.
            changed_model = nnx.clone(model)
            clone_before = capture.source_snapshot(changed_model)[0]
            changed_weight = np.array(changed_model.conv1.kernel[...], copy=True)
            changed_weight.reshape(-1)[0] = np.nextafter(
                changed_weight.reshape(-1)[0], np.float32(np.inf)
            )
            changed_model.conv1.kernel[...] = changed_weight
            with pytest.raises(ValueError, match="changed"):
                capture.require_unchanged(
                    clone_before,
                    capture.source_snapshot(changed_model)[0],
                    "actual cloned source state",
                )
            with jax.numpy_rank_promotion(
                "raise"
                if settings["effective"]["numpy_rank_promotion"] != "raise"
                else "allow"
            ):
                with pytest.raises(ExceptionGroup, match="source/state guards failed"):
                    complete_guard()
            complete_guard()
            capture.write_json(
                tmp_path / "actual-source-guard-controls.json",
                {
                    "changed_actual_model_clone_state": "rejected",
                    "changed_effective_caller_setting": "rejected",
                    "original_instance_after_controls": "unchanged",
                    "numerical_comparator": "independent from mandatory source/state guard",
                },
            )
            assert len(native) == len(targets) == 4 * len(feeds)
            # All actual vectors and runtime graphs exist before the first
            # numerical comparison. The raw callback has its separate receipt.
            _check_observations(native, envelopes)
            _check_observations(targets, envelopes)
            for batch, envelope in envelopes.items():
                source_values = [
                    value["arrays"][0]
                    for name, value in native.items()
                    if name.startswith(f"B{batch}-")
                ]
                target_values = [
                    value["arrays"][0]
                    for name, value in targets.items()
                    if name.startswith(f"B{batch}-")
                ]
                for left in source_values:
                    for right in (*source_values, *target_values):
                        envelope.check_pair(
                            left.astype("<f4", copy=False).tobytes(),
                            right.astype("<f4", copy=False).tobytes(),
                        )
                for role, observations in (("native", native), ("target", targets)):
                    for kind in (
                        ("eager", "fresh-jit")
                        if role == "native"
                        else ("disabled", "all")
                    ):
                        first = observations[f"B{batch}-{kind}-0"]["arrays"][0]
                        second = observations[f"B{batch}-{kind}-1"]["arrays"][0]
                        assert (
                            first.dtype == second.dtype
                            and first.shape == second.shape
                            and first.tobytes() == second.tobytes()
                        ), (role, kind, "repeated execution drift")
            assert callback == {"attempted": True, "result": False}
            assert (
                proto.SerializeToString() == blob
                and get_precision_report(proto) == report
            )
            assert (
                scenario.criteria is None
                and metadata["callable"] is recipe
                and recipe._instance is None
            )
            complete_guard()
            capture.write_json(
                tmp_path / "finite-sample-receipt.json",
                {
                    "status": "accepted recorded finite vectors",
                    "criterion": exact.CRITERION,
                    "artifact_sha256": artifact_sha,
                    "native_attempts": len(native),
                    "target_attempts": len(targets),
                    "batches": list(feeds),
                    "raw_criteria": None,
                    "raw_callback": "failed-retained",
                    "product_export_report_numerics": "UNVERIFIED",
                    "provider_algorithm_theorem": "UNVERIFIED",
                    "whole_model_or_public_acceptance": False,
                },
            )
            record_property("cnn_finite_nonzero_criterion", exact.CRITERION)
            record_property("raw_callback_status", "failed-retained")
            record_property(
                "public_SOURCE_activation", "not authorized by finite samples"
            )
    finally:
        previous._CNN = previous_model
        assert FunctionPlugin._trace_function_body is previous_body
        capture.require_unchanged(
            outer_settings, capture.settings_snapshot(), "outer caller/global settings"
        )


def _tensor(
    values: list[float], shape: tuple[int, ...], *, parameter: bool = False
) -> exact.Dyadics:
    return exact.Dyadics.fp32(
        shape, struct.pack("<" + "f" * len(values), *values), parameter=parameter
    )


def _fraction(value: int, exponent: int) -> Fraction:
    return (
        Fraction(value << exponent)
        if exponent >= 0
        else Fraction(value, 1 << -exponent)
    )


def _rn32(value: Fraction) -> Fraction:
    """Independent monotone word search for bounded RN-even scalar controls."""
    sign = -1 if value < 0 else 1
    magnitude = abs(value)

    def decode(word: int) -> Fraction:
        numerator, exponent = exact.fp32_word(word)
        return _fraction(numerator, exponent)

    lower, upper = 0, 0x7F7FFFFF
    assert magnitude <= decode(upper)
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if decode(middle) <= magnitude:
            lower = middle
        else:
            upper = middle - 1
    if decode(lower) == magnitude:
        return sign * magnitude
    above = lower + 1
    distance_lower, distance_upper = (
        magnitude - decode(lower),
        decode(above) - magnitude,
    )
    chosen = (
        lower
        if distance_lower < distance_upper
        or (distance_lower == distance_upper and lower % 2 == 0)
        else above
    )
    return sign * decode(chosen)


@pytest.mark.parametrize(
    "value,expected",
    [
        (Fraction(1) + Fraction(1, 1 << 24), Fraction(1)),
        (Fraction(1) + Fraction(3, 1 << 24), Fraction(1) + Fraction(1, 1 << 22)),
        (Fraction(1, 1 << 150), Fraction(0)),
        (Fraction(3, 1 << 150), Fraction(1, 1 << 148)),
    ],
)
def test_cnn_oracle_rn32_ties_and_underflow_controls(
    value: Fraction, expected: Fraction
) -> None:
    assert _rn32(value) == expected


def test_cnn_oracle_fma_order_and_daz_controls_do_not_assume_a_provider_algorithm() -> (
    None
):
    a, b = Fraction(1) + Fraction(1, 1 << 23), Fraction(1) - Fraction(1, 1 << 23)
    assert a * b - 1 == Fraction(-1, 1 << 46)
    assert _rn32(_rn32(a * b) - 1) == 0
    assert _rn32(a * b - 1) == Fraction(-1, 1 << 46)
    assert _rn32(_rn32(Fraction(1 << 24) + 1) - (1 << 24)) == 0
    assert _rn32(_rn32(Fraction(1 << 24) - (1 << 24)) + 1) == 1
    # A flushed subnormal input may create a normal-sized output difference;
    # the incoming tau term must be multiplied by the original normal weight.
    tiny, weight = Fraction(1, 1 << 149), Fraction(1 << 100)
    assert tiny * weight == Fraction(1, 1 << 49)
    assert tiny * weight <= Fraction(1, 1 << 126) * weight


def test_cnn_exact_oracle_small_conv_relu_pool_dot_hand_composition() -> None:
    incoming = exact.Envelope(
        _tensor([0.5, -0.25, 0.375, -0.125], (1, 2, 2, 1)), (0,) * 4
    )
    weights = _tensor([2, -1], (1, 1, 1, 2), parameter=True)
    bias = _tensor([0.25, 0.125], (2,), parameter=True)
    budget = exact.Budget(10)
    convolved = exact.affine(
        incoming,
        weights,
        bias,
        [(index,) for index in range(4)],
        (1, 2, 2, 2),
        1,
        budget,
    )
    pooled = exact.pool(exact.relu(convolved))
    assert [
        _fraction(value, pooled.center.exponent) for value in pooled.center.values
    ] == [Fraction(9, 16), Fraction(5, 32)]
    result = exact.affine(
        pooled,
        _tensor([-2, 4], (2, 1), parameter=True),
        _tensor([0.125], (1,), parameter=True),
        [(0, 1)],
        (1, 1),
        2,
        budget,
    )
    assert _fraction(result.center.values[0], result.center.exponent) == Fraction(-3, 8)
    assert exact.ceil_div(16, 3) == 6 and exact.ceil_div(6, 3) == 2


def test_cnn_pool_formula_and_relu_clipping_are_independently_rational() -> None:
    tau = Fraction(1, 1 << 126)
    incoming = exact.Envelope(
        _tensor([0.5, -0.25, 0.375, -0.125], (1, 2, 2, 1)), (0,) * 4
    )
    result = exact.pool(incoming)
    expected = (5 + (13 * exact.U - 20) * tau) / (4 * (exact.U - 4))
    actual = _fraction(result.errors[0], exact.ERROR_EXPONENT)
    assert expected <= actual < expected + Fraction(1, 1 << 192)
    assert _fraction(result.center.values[0], result.center.exponent) == Fraction(1, 8)
    clipped = exact.relu(exact.Envelope(_tensor([-1.0, 1.0], (2,)), (0, 0)))
    assert clipped.errors == (0, exact.TAU)


@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan", "inf", "outside"])
def test_cnn_exact_comparator_rejects_invalid_or_outside_values(
    corruption: str,
) -> None:
    envelope = exact.Envelope(_tensor([1], (1, 1)), (1 << (192 - 23),))
    shape, dtype, value = (1, 1), "float32", 1.0
    if corruption == "dtype":
        dtype = "float64"
    elif corruption == "shape":
        shape = (1,)
    elif corruption == "nan":
        value = float("nan")
    elif corruption == "inf":
        value = float("inf")
    else:
        value = 1 + 2**-22
    with pytest.raises(ValueError):
        envelope.check(shape, dtype, struct.pack("<f", value))
    envelope.check((1, 1), "float32", struct.pack("<f", 1 + 2**-23))
    envelope.check_pair(struct.pack("<f", 1 - 2**-23), struct.pack("<f", 1 + 2**-23))


@pytest.mark.parametrize(
    "family",
    ["subnormal-parameter", "overflow", "gamma", "budget", "nondiscriminating"],
)
def test_cnn_oracle_domain_and_cost_gates_are_unconditional(family: str) -> None:
    with pytest.raises(ValueError):
        if family == "subnormal-parameter":
            exact.Dyadics.fp32((1,), struct.pack("<I", 1), parameter=True)
        elif family == "overflow":
            exact.overflow_gate((1 << 24) - 1, 104, 2, 2)
        elif family == "gamma":
            exact.overflow_gate(0, 0, exact.U, 0)
        elif family == "budget":
            exact.Budget(1).step(2)
        else:
            envelope = exact.Envelope(_tensor([0], (1, 1)), (0,))
            if not envelope.rejects_zero():
                raise ValueError("NONDISCRIMINATING")


@pytest.mark.parametrize("family", ["state", "settings"])
def test_cnn_state_and_settings_changes_rejected_even_with_equal_values(
    family: str,
) -> None:
    envelope = exact.Envelope(_tensor([1], (1, 1)), (0,))
    envelope.check((1, 1), "float32", struct.pack("<f", 1))
    with pytest.raises(ValueError, match="changed"):
        capture.require_unchanged({family: "original"}, {family: "changed"}, family)


def test_cnn_capture_attempts_both_native_roles_and_retains_each_exception(
    tmp_path: Path,
) -> None:
    calls = []

    def model(value: Any) -> Any:
        calls.append("eager")
        raise RuntimeError("independent eager failure")

    def compiled(fn: Any) -> Any:
        def fail(*values: Any) -> Any:
            calls.append("jit")
            raise RuntimeError("independent fresh-JIT failure")

        return fail

    with patch.object(nnx, "jit", compiled):
        observations = capture.capture_native(
            model, {3: np.ones((3, 28, 28, 1), np.float32)}, tmp_path, lambda: None
        )
    assert calls == ["eager", "jit", "eager", "jit"]
    assert len(observations) == 4 and all(
        not item["returned"] and item["exception"] is not None
        for item in observations.values()
    )
    assert len(list(tmp_path.glob("*.npz"))) == len(list(tmp_path.glob("*.json"))) == 4


@pytest.mark.parametrize("layer,size,channels", [("conv1", 14, 32), ("conv2", 7, 64)])
def test_cnn_artifact_contract_rejects_equal_shape_pool_axis_relabeling(
    layer: str, size: int, channels: int
) -> None:
    contract = capture.ArtifactContract({}, 3, named=False)
    original = capture.Data(
        f"{layer}.blocks", (3, size, 2, size, 2, channels), "NhrwcC"
    )
    axes = np.array([2, 4], np.int64)
    expected = capture.Data(f"{layer}.sum", (3, size, size, channels), "NHWC")
    assert (
        contract.operation("ReduceSum", [original, axes], {"keepdims": 0}) == expected
    )
    swapped = contract.operation("Transpose", [original], {"perm": [0, 3, 2, 1, 4, 5]})
    assert swapped.shape == original.shape and swapped.layout != original.layout
    with pytest.raises(ValueError, match="axes/order"):
        contract.operation("ReduceSum", [swapped, axes], {"keepdims": 0})


@pytest.mark.parametrize("operator", ["Max", "Relu"])
def test_cnn_artifact_contract_rejects_extra_dynamic_relu_operand(
    operator: str,
) -> None:
    contract = capture.ArtifactContract({}, 3, named=False)
    affine = capture.Data("conv1.affine", (3, 32, 28, 28), "NCHW")
    original_input = capture.Data("input", (3, 1, 28, 28), "NCHW")
    inputs = [affine, np.array(0, np.float32)] if operator == "Max" else [affine]
    assert contract.operation(operator, inputs, {}) == capture.Data(
        "conv1.relu", affine.shape, affine.layout
    )
    with pytest.raises(ValueError, match="clipping|arity"):
        contract.operation(operator, [*inputs, original_input], {})


@pytest.mark.parametrize("failure", ["guard", "disabled-session", "all-session"])
def test_cnn_target_capture_freezes_both_modes_before_rejecting_failures(
    failure: str, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    calls = []

    class Options:
        def get_session_config_entry(self, key: str) -> None:
            raise RuntimeError("entry absent")

    runtime = SimpleNamespace(SessionOptions=Options, __version__="fake-capture-only")
    model = SimpleNamespace(SerializeToString=lambda: b"capture-only-artifact")
    feeds = {
        3: np.ones((3, 28, 28, 1), np.float32),
        2: np.ones((2, 28, 28, 1), np.float32),
    }

    def session(proto: Any, cell: Path, optimized: bool) -> Any:
        mode = "all" if optimized else "disabled"
        calls.append(("session", mode))
        options = runtime.SessionOptions()
        if failure == f"{mode}-session":
            raise RuntimeError(f"independent {mode} load failure")
        (cell / "source_dot.onnx").write_bytes(proto.SerializeToString())
        Path(options.optimized_model_filepath).write_bytes(
            f"actual {mode} graph".encode()
        )

        def run(names: Any, arguments: Any) -> Any:
            feed = arguments["x"]
            calls.append(("run", mode, feed.shape[0]))
            return [np.ones((feed.shape[0], 10), np.float32)]

        return SimpleNamespace(
            disable_fallback=lambda: None,
            get_providers=lambda: ["CPUExecutionProvider"],
            get_provider_options=lambda: {},
            get_inputs=lambda: [SimpleNamespace(name="x")],
            run=run,
        )

    def guard() -> None:
        if failure == "guard":
            raise ValueError("independent changed source/settings guard")

    with (
        patch.dict("sys.modules", {"onnxruntime": runtime}),
        patch.object(capture, "_session", session),
    ):
        observations = capture.capture_targets(model, feeds, tmp_path, guard)
    assert [call for call in calls if call[0] == "session"] == [
        ("session", "disabled"),
        ("session", "all"),
    ]
    assert len(observations) == 8
    for name, observation in observations.items():
        failed_mode = failure == f"{name.split('-')[1]}-session"
        assert observation["returned"] is not failed_mode
        assert (observation["guard_exception"] is not None) is (failure == "guard")
        assert (tmp_path / f"{name}.npz").is_file()
        assert (tmp_path / f"{name}.json").is_file()
        if not failed_mode:
            assert len(observation["arrays"]) == 1
            assert observation["session_contract"]["runtime_graph_exists"]
    failed = {
        name: observation
        for name, observation in observations.items()
        if observation["exception"] is not None
        or observation["guard_exception"] is not None
    }
    assert len(failed) == (8 if failure == "guard" else 4)
    with pytest.raises(AssertionError):
        _check_observations(failed, {})


@pytest.mark.parametrize("failure", ["first-oracle", "all-oracles"])
def test_cnn_failed_apriori_setup_retains_other_setups_and_target_capture(
    failure: str, tmp_path: Path
) -> None:
    calls = []
    feeds = {
        3: np.ones((3, 28, 28, 1), np.float32),
        2: np.ones((2, 28, 28, 1), np.float32),
    }
    model = object()

    def setup(parameters: Any, feed: Any, directory: Path, artifact_sha: Any) -> Any:
        batch = feed.shape[0]
        calls.append(f"oracle-B{batch}")
        directory.mkdir()
        (directory / "attempted.json").write_text("frozen a priori attempt")
        if batch == 3 or failure == "all-oracles":
            raise ValueError(
                f"original B{batch} oracle domain/budget/nonvacuity failure"
            )
        return f"frozen-envelope-B{batch}"

    def target_capture(proto: Any, values: Any, directory: Path, guard: Any) -> Any:
        assert proto is model and values is feeds
        assert calls == ["oracle-B3", "oracle-B2"]
        directory.mkdir()
        (directory / "captured.json").write_text("independent target attempt")
        calls.append("target-capture")
        return {"independent-target-observation": {"returned": True}}

    with (
        patch.object(capture, "oracle", setup),
        patch.object(capture, "capture_targets", target_capture),
    ):
        result = capture.capture_oracles_and_targets(
            {},
            feeds,
            model,
            tmp_path,
            lambda: None,
            artifact_sha256="frozen-artifact",
            export_error=None,
        )
    assert calls == ["oracle-B3", "oracle-B2", "target-capture"]
    assert set(result["oracle_exceptions"]) == (
        {3, 2} if failure == "all-oracles" else {3}
    )
    assert set(result["envelopes"]) == (set() if failure == "all-oracles" else {2})
    assert result["target_capture_exception"] is None
    assert result["targets"]["independent-target-observation"]["returned"]
    assert (tmp_path / "oracle-B3-exception.json").is_file()
    assert (tmp_path / "target/captured.json").is_file()
    with pytest.raises(AssertionError):
        assert not result["oracle_exceptions"], result["oracle_exceptions"]


@pytest.mark.parametrize(
    "corruption",
    [
        "extra-max-data",
        "arm",
        "axis",
        "indices",
        "unowned",
        "zero-sign",
        "axes",
        "dtype",
        "shared-cse",
    ],
)
def test_cnn_signed_zero_relu_composite_preserves_exact_owned_selection(
    corruption: str,
) -> None:
    from types import SimpleNamespace

    def node(operator: str, inputs: list[str], output: str, **attributes: Any) -> Any:
        return onnx.helper.make_node(operator, inputs, [output], **attributes)

    nodes = [
        node("Max", ["affine", "zero"], "base"),
        node("Equal", ["affine", "zero"], "left-zero"),
        node("Equal", ["zero", "zero"], "right-zero"),
        node("And", ["left-zero", "right-zero"], "ties"),
        node("Add", ["affine", "zero"], "correction"),
        node("Unsqueeze", ["base", "axes"], "base-choice"),
        node("Unsqueeze", ["correction", "axes"], "correction-choice"),
        node("Concat", ["base-choice", "correction-choice"], "choices", axis=0),
        node("Cast", ["ties"], "indices", to=7),
        node("Unsqueeze", ["indices", "axes"], "index-choice"),
        node("GatherElements", ["choices", "index-choice"], "selected", axis=0),
        node("Squeeze", ["selected", "axes"], "result"),
    ]
    graph = SimpleNamespace(
        node=nodes,
        input=[SimpleNamespace(name=name) for name in ("affine", "zero", "axes")],
        initializer=[],
        output=[SimpleNamespace(name="result")],
        value_info=[],
    )
    contract = capture.ArtifactContract({}, 3, named=False)
    contract.functions = {}
    environment = {
        "affine": capture.Data("conv1.affine", (3, 28, 28, 32), "NHWC"),
        "zero": np.array(0, np.float32),
        "axes": np.array([0], np.int64),
    }
    selections = contract.relu_selections(graph)
    assert contract.selected_relu(
        selections["result"], environment, graph
    ) == capture.Data("conv1.relu", (3, 28, 28, 32), "NHWC")
    assert contract.scope(graph, list(environment.values()), "graph") == [
        capture.Data("conv1.relu", (3, 28, 28, 32), "NHWC")
    ]
    if corruption == "shared-cse":
        duplicated = copy.deepcopy(nodes)
        for duplicate in duplicated:
            if duplicate.output[0] == "right-zero":
                continue
            for index, name in enumerate(duplicate.input):
                if name not in ("zero", "axes", "right-zero"):
                    duplicate.input[index] = name + "-second"
            duplicate.output[0] += "-second"
            nodes.append(duplicate)
        graph.input.append(SimpleNamespace(name="affine-second"))
        graph.output.append(SimpleNamespace(name="result-second"))
        outputs = contract.scope(
            graph,
            [
                *environment.values(),
                capture.Data("conv2.affine", (3, 14, 14, 64), "NHWC"),
            ],
            "graph",
        )
        assert outputs == [
            capture.Data("conv1.relu", (3, 28, 28, 32), "NHWC"),
            capture.Data("conv2.relu", (3, 14, 14, 64), "NHWC"),
        ]
        return
    if corruption == "extra-max-data":
        nodes[0].input.append("original-input")
    elif corruption == "arm":
        nodes[7].input[0], nodes[7].input[1] = "correction-choice", "base-choice"
    elif corruption == "axis":
        nodes[10].attribute[0].i = 1
    elif corruption == "indices":
        nodes[8].input[0] = "inverted-ties"
        nodes.insert(8, node("Not", ["ties"], "inverted-ties"))
    elif corruption == "unowned":
        nodes.append(node("Identity", ["base"], "outside-use"))
    elif corruption == "zero-sign":
        environment["zero"] = np.array(-0.0, np.float32)
    elif corruption == "axes":
        environment["axes"] = np.array([1], np.int64)
    else:
        environment["zero"] = np.array(0, np.float64)
    with pytest.raises(ValueError, match="signed-zero ReLU"):
        selections = contract.relu_selections(graph)
        contract.selected_relu(selections["result"], environment, graph)
        contract.scope(graph, list(environment.values()), "graph")
