# tests/extra_tests/converter/source_cnn_acceptance.py

"""Narrow original-CNN capture and actual-artifact structural qualification.

No callback, report or shared validation API is replaced. Actual arrays are
written separately from receipts, and failed attempts remain failed attempts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import time
import traceback
from typing import Any, Callable
from unittest.mock import patch

from flax import nnx
import jax
import numpy as np
import onnx
from onnx import numpy_helper

from tests.extra_tests.converter import source_cnn_roundoff as exact
from tests.extra_tests.converter.test_source_dot_precision import _session


_ORACLES: dict[str, tuple[exact.Envelope, dict[str, Any]]] = {}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def exception_record(exc: Exception) -> dict[str, str]:
    return {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def array_record(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "nbytes": array.nbytes,
    }


def parameters(model: Any) -> dict[str, np.ndarray]:
    result = {}
    for name in ("conv1", "conv2", "linear1", "linear2"):
        layer = getattr(model, name)
        for kind in ("kernel", "bias"):
            value = np.array(getattr(layer, kind)[...], copy=True)
            value.setflags(write=False)
            result[f"{name}.{kind}"] = value
    if set(result) != set(exact.PARAMETER_SHAPES) or any(
        value.dtype != np.dtype("float32")
        or value.shape != exact.PARAMETER_SHAPES[name]
        for name, value in result.items()
    ):
        raise ValueError("original full eight-parameter CNN geometry changed")
    return result


def source_snapshot(model: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    graph, state = nnx.split(model)
    arrays = {}
    leaves = []
    for index, (path, value) in enumerate(
        jax.tree_util.tree_flatten_with_path(state)[0]
    ):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        key = f"state_{index}"
        arrays[key] = array
        leaves.append(
            {
                "path": [str(part) for part in path],
                "storage": key,
                **array_record(array),
            }
        )
    # GraphDef retains actual classes, static attributes and callable ownership;
    # its full repr is captured, compared on the same instance after each role.
    document = {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "graph_definition": repr(graph),
        "state": leaves,
        "parameters": {
            name: array_record(value) for name, value in parameters(model).items()
        },
    }
    return document, arrays


def settings_snapshot() -> dict[str, Any]:
    return {
        "effective": {
            "x64_enabled": bool(jax.config.x64_enabled),
            "default_matmul_precision": jax.config.jax_default_matmul_precision,
            "numpy_dtype_promotion": jax.config.jax_numpy_dtype_promotion,
            "numpy_rank_promotion": jax.config.jax_numpy_rank_promotion,
            "debug_key_reuse": bool(jax.config.jax_debug_key_reuse),
            "platforms": jax.config.jax_platforms,
        },
        "global": {
            "x64_enabled": jax.enable_x64.get_global(),
            "default_matmul_precision": jax.default_matmul_precision.get_global(),
            "numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
            "numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
            "debug_key_reuse": jax.debug_key_reuse.get_global(),
        },
    }


def stack_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "flax": version("flax"),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnx_ir": version("onnx-ir"),
        "native_backend": jax.default_backend(),
        "native_devices": [str(device) for device in jax.devices()],
    }


def require_unchanged(expected: Any, current: Any, label: str) -> None:
    if current != expected:
        raise ValueError(f"original CNN {label} changed")


def trace_contract(closed: Any, batch: int | str) -> dict[str, Any]:
    """Bind the oracle geometry to the actual unpatched native typed trace."""
    leaves = []

    def signature(var: Any) -> tuple[str, tuple[Any, ...]]:
        return np.dtype(var.aval.dtype).name, tuple(
            int(size) if isinstance(size, (int, np.integer)) else str(size)
            for size in var.aval.shape
        )

    def walk(value: Any, ancestors: frozenset[int] = frozenset()) -> None:
        body = getattr(value, "jaxpr", value)
        if id(body) in ancestors:
            raise ValueError("cyclic CNN native trace")
        ancestors = ancestors | {id(body)}
        if body.effects:
            raise ValueError("effectful CNN native trace")
        for equation in body.eqns:
            if equation.effects:
                raise ValueError("effectful CNN native equation")
            children = []
            for parameter in equation.params.values():
                candidate = getattr(parameter, "jaxpr", parameter)
                if hasattr(candidate, "eqns"):
                    children.append(parameter)
            if children:
                if (
                    equation.primitive.name not in ("jit", "pjit", "custom_jvp_call")
                    or len(children) != 1
                ):
                    raise ValueError("unreviewed CNN native nested arithmetic")
                walk(children[0], ancestors)
            else:
                leaves.append(equation)

    body = getattr(closed, "jaxpr", closed)
    if [signature(var) for var in body.invars] != [("float32", (batch, 28, 28, 1))] or [
        signature(var) for var in body.outvars
    ] != [("float32", (batch, 10))]:
        raise ValueError("CNN native typed input/output changed")
    walk(closed)
    allowed = {
        "conv_general_dilated",
        "dot_general",
        "reduce_window_sum",
        "div",
        "reshape",
        "broadcast_in_dim",
        "add",
        "max",
        "convert_element_type",
        "copy",
    }
    if any(equation.primitive.name not in allowed for equation in leaves):
        raise ValueError("unreviewed arithmetic in original CNN typed trace")
    convolutions = [
        equation
        for equation in leaves
        if equation.primitive.name == "conv_general_dilated"
    ]
    dots = [equation for equation in leaves if equation.primitive.name == "dot_general"]
    windows = [
        equation
        for equation in leaves
        if equation.primitive.name == "reduce_window_sum"
    ]
    divisions = [equation for equation in leaves if equation.primitive.name == "div"]
    if (len(convolutions), len(dots), len(windows), len(divisions)) != (2, 2, 2, 2):
        raise ValueError("original CNN contraction/pool counts changed")
    for equation, ci, co, size in zip(
        convolutions, (1, 32), (32, 64), (28, 14), strict=True
    ):
        if [signature(var) for var in (*equation.invars, *equation.outvars)] != [
            ("float32", (batch, size, size, ci)),
            ("float32", (3, 3, ci, co)),
            ("float32", (batch, size, size, co)),
        ]:
            raise ValueError("original CNN native convolution tensors changed")
        p = equation.params
        dims = p["dimension_numbers"]
        if (
            tuple(p["window_strides"]) != (1, 1)
            or tuple(map(tuple, p["padding"])) != ((1, 1), (1, 1))
            or tuple(p["lhs_dilation"]) != (1, 1)
            or tuple(p["rhs_dilation"]) != (1, 1)
            or p["feature_group_count"] != 1
            or p["batch_group_count"] != 1
            or tuple(dims.lhs_spec) != (0, 3, 1, 2)
            or tuple(dims.rhs_spec) != (3, 2, 0, 1)
            or tuple(dims.out_spec) != (0, 3, 1, 2)
        ):
            raise ValueError("original CNN native SAME convolution geometry changed")
    for equation, fanin, channels in zip(dots, (3136, 256), (256, 10), strict=True):
        if [signature(var) for var in (*equation.invars, *equation.outvars)] != [
            ("float32", (batch, fanin)),
            ("float32", (fanin, channels)),
            ("float32", (batch, channels)),
        ] or equation.params["dimension_numbers"] != (((1,), (0,)), ((), ())):
            raise ValueError("original CNN native dot geometry changed")
    for equation, size, channels in zip(windows, (28, 14), (32, 64), strict=True):
        if [signature(var) for var in (*equation.invars, *equation.outvars)] != [
            ("float32", (batch, size, size, channels)),
            ("float32", (batch, size // 2, size // 2, channels)),
        ]:
            raise ValueError("original CNN native pool tensors changed")
        p = equation.params
        if (
            tuple(p["window_dimensions"]) != (1, 2, 2, 1)
            or tuple(p["window_strides"]) != (1, 2, 2, 1)
            or tuple(map(tuple, p["padding"])) != ((0, 0),) * 4
            or tuple(p["base_dilation"]) != (1,) * 4
            or tuple(p["window_dilation"]) != (1,) * 4
        ):
            raise ValueError("original CNN native disjoint pool geometry changed")
    for equation in divisions:
        divisor = equation.invars[1]
        if (
            signature(divisor) != ("float32", ())
            or not hasattr(divisor, "val")
            or np.asarray(divisor.val).item() != 4
        ):
            raise ValueError("original CNN native separate literal Div4 changed")
    maxima = [equation for equation in leaves if equation.primitive.name == "max"]
    if len(maxima) != 3:
        raise ValueError("original CNN native ReLU count changed")
    for equation in maxima:
        literals = [var for var in equation.invars if hasattr(var, "val")]
        if (
            len(literals) != 1
            or signature(literals[0]) != ("float32", ())
            or np.asarray(literals[0].val).item() != 0
        ):
            raise ValueError("original CNN native ReLU clipping changed")
    flatten = [
        equation
        for equation in leaves
        if equation.primitive.name == "reshape"
        and signature(equation.outvars[0]) == ("float32", (batch, 3136))
    ]
    if (
        len(flatten) != 1
        or signature(flatten[0].invars[0]) != ("float32", (batch, 7, 7, 64))
        or flatten[0].params.get("dimensions") not in (None, (0, 1, 2, 3))
    ):
        raise ValueError("original CNN native NHWC flatten order changed")
    for equation in leaves:
        if equation.primitive.name in ("convert_element_type", "copy") and signature(
            equation.invars[0]
        ) != signature(equation.outvars[0]):
            raise ValueError("original CNN native precision changed")
    return {
        "leaf_primitives": [equation.primitive.name for equation in leaves],
        "fanin": [9, 288, 3136, 256],
        "input": [batch, 28, 28, 1],
        "output": [batch, 10],
        "provider_algorithm_theorem": "UNVERIFIED",
    }


def capture_attempt(
    call: Callable[[], Any],
    directory: Path,
    name: str,
    *,
    guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "attempted": True,
        "returned": False,
        "exception": None,
        "guard_exception": None,
        "arrays": [],
    }
    started = time.monotonic()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    arrays = {}
    try:
        values = jax.tree_util.tree_leaves(jax.device_get(call()))
        for index, value in enumerate(values):
            array = np.array(value, copy=True)
            array.setflags(write=False)
            arrays[f"output_{index}"] = array
            result["arrays"].append(array)
        result["returned"] = True
    except Exception as exc:
        result["exception"] = exception_record(exc)
    # Persist actual outputs even if the post-call state/settings guard fails.
    np.savez(directory / f"{name}.npz", **arrays)
    if guard is not None:
        try:
            guard()
        except Exception as exc:
            result["guard_exception"] = exception_record(exc)
    result["wall_seconds"] = time.monotonic() - started
    result["process_peak_rss_bytes_before"] = peak_before
    result["process_peak_rss_bytes_after"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    write_json(
        directory / f"{name}.json",
        {
            **{key: value for key, value in result.items() if key != "arrays"},
            "outputs": [array_record(array) for array in result["arrays"]],
        },
    )
    return result


def capture_native(
    model: Any, feeds: dict[int, np.ndarray], directory: Path, guard: Callable[[], None]
) -> dict[str, dict[str, Any]]:
    observations = {}
    for batch, feed in feeds.items():
        # A fresh function AND fresh NNX wrapper per attempt prevents reuse of
        # a preceding wrapper's compiled callable as the fresh-JIT evidence.
        for repeat in range(2):

            def fresh(original: Any, value: Any) -> Any:
                return original(value)

            observations[f"B{batch}-eager-{repeat}"] = capture_attempt(
                lambda: model(jax.device_put(feed)),
                directory,
                f"B{batch}-eager-{repeat}",
                guard=guard,
            )
            observations[f"B{batch}-fresh-jit-{repeat}"] = capture_attempt(
                lambda: nnx.jit(fresh)(model, jax.device_put(feed)),
                directory,
                f"B{batch}-fresh-jit-{repeat}",
                guard=guard,
            )
    return observations


def oracle(
    parameters_: dict[str, np.ndarray],
    feed: np.ndarray,
    directory: Path,
    artifact_sha256: str | None,
) -> exact.Envelope:
    parameter_bytes = {
        name: (value.shape, value.astype("<f4", copy=False).tobytes(order="C"))
        for name, value in parameters_.items()
    }
    feed_bytes = (feed.shape, feed.astype("<f4", copy=False).tobytes(order="C"))
    key = exact.identity(parameter_bytes, feed_bytes)
    directory.mkdir(parents=True, exist_ok=True)
    if key in _ORACLES:
        envelope, receipt = _ORACLES[key]
        receipt = {**receipt, "cache_hit": True, "fresh_setup_wall_seconds": 0.0}
    else:
        budget = exact.Budget(feed.shape[0] * exact.PRODUCTS_PER_IMAGE)
        receipt = {
            "cache_hit": False,
            "budget": {
                "max_products": budget.max_products,
                "max_seconds": budget.max_seconds,
                "max_integer_bits": budget.max_integer_bits,
            },
            "setup_status": "attempted",
        }
        peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        try:
            decoded = {
                name: exact.Dyadics.fp32(shape, payload, parameter=True)
                for name, (shape, payload) in parameter_bytes.items()
            }
            envelope = exact.original_cnn(
                decoded, exact.Dyadics.fp32(*feed_bytes), budget
            )
            receipt.update(
                setup_status="returned",
                products=budget.products,
                stages=budget.stages,
                fresh_setup_wall_seconds=time.monotonic() - budget.started,
                process_peak_rss_bytes_before=peak_before,
                process_peak_rss_bytes_after=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            )
        except Exception as exc:
            receipt.update(
                setup_status="failed",
                exception=exception_record(exc),
                products=budget.products,
                stages=budget.stages,
                fresh_setup_wall_seconds=time.monotonic() - budget.started,
                process_peak_rss_bytes_before=peak_before,
                process_peak_rss_bytes_after=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
            )
            write_json(directory / "setup.json", receipt)
            raise
        _ORACLES[key] = (envelope, receipt)
    document = envelope.document()
    serialized = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    if len(serialized) > 4_000_000:
        raise ValueError("CNN final envelope serialization budget exceeded")
    (directory / "exact-envelope.json").write_bytes(serialized)
    write_json(
        directory / "setup.json",
        {
            **receipt,
            "parameter_feed_criterion_sha256": key,
            "artifact_sha256": artifact_sha256,
            "envelope_sha256": hashlib.sha256(serialized).hexdigest(),
            "frozen_before_any_target_session": True,
            "provider_algorithm_theorem": "UNVERIFIED",
            "rejects_all_zero": envelope.rejects_zero(),
        },
    )
    if not envelope.rejects_zero():
        raise ValueError(
            "NONDISCRIMINATING a priori CNN envelope; no target-driven repair permitted"
        )
    return envelope


@dataclass(frozen=True)
class Data:
    stage: str
    shape: tuple[Any, ...]
    layout: str


class ArtifactContract:
    """Interpret only shape/provenance, never evaluate model arithmetic."""

    def __init__(
        self, parameters_: dict[str, np.ndarray], batch: int | str, *, named: bool
    ) -> None:
        self.parameters = parameters_
        self.batch, self.named = batch, named
        self.used: Counter[str] = Counter()
        self.operations: list[dict[str, Any]] = []
        self.function_calls: list[str] = []
        self.constant_owners: list[dict[str, Any]] = []

    def parameter(self, value: Any, name: str, *, convolution: bool = False) -> None:
        expected = self.parameters[name]
        if convolution:
            expected = expected.transpose(3, 2, 0, 1)
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != expected.dtype
            or value.shape != expected.shape
            or value.tobytes(order="C") != expected.tobytes(order="C")
        ):
            raise ValueError(f"actual artifact changed original parameter {name}")
        self.used[name] += 1

    def bias(self, value: Any, name: str, target: Data | None = None) -> None:
        expected = self.parameters[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != expected.dtype
            or value.size != expected.size
            or tuple(size for size in value.shape if size != 1) != expected.shape
        ):
            raise ValueError("actual artifact bias broadcast geometry changed")
        if target is not None:
            shape = (1,) * (len(target.shape) - value.ndim) + value.shape
            if len(shape) != len(target.shape) or any(
                size != (expected.size if target.layout[index] == "C" else 1)
                for index, size in enumerate(shape)
            ):
                raise ValueError("actual artifact bias broadcast axis changed")
        self.parameter(value.reshape(expected.shape), name)

    @staticmethod
    def dimensions(value: Any) -> tuple[Any, ...]:
        return tuple(
            int(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param
            for dim in value.type.tensor_type.shape.dim
        )

    def check(self, model: Any) -> dict[str, Any]:
        if (
            dict((item.domain, item.version) for item in model.opset_import).get("")
            != 23
        ):
            raise ValueError("original CNN opset23 required")
        for ports, shape in (
            (model.graph.input, (self.batch, 28, 28, 1)),
            (model.graph.output, (self.batch, 10)),
        ):
            if (
                len(ports) != 1
                or ports[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                or self.dimensions(ports[0]) != shape
            ):
                raise ValueError("original CNN physical port signature changed")
        self.functions = {
            (function.domain, function.name): function for function in model.functions
        }
        output = self.scope(
            model.graph, [Data("input", (self.batch, 28, 28, 1), "NHWC")], "graph"
        )
        if output != [Data("linear2.affine", (self.batch, 10), "NC")]:
            raise ValueError("actual artifact changed original CNN output wiring")
        if self.used != Counter({name: 1 for name in exact.PARAMETER_SHAPES}):
            raise ValueError(
                "actual artifact did not bind every original parameter role exactly once"
            )
        if self.named and self.function_calls != ["OriginalCNNSource"]:
            raise ValueError("named original CNN body/constant ownership changed")
        if not self.named and self.function_calls:
            raise ValueError("unexpected artifact function ownership")
        return {
            "parameter_roles": dict(self.used),
            "operations": self.operations,
            "function_calls": self.function_calls,
            "constant_owners": self.constant_owners,
            "batch": self.batch,
            "source_semantics": "two SAME Conv+bias/ReLU/sum4/Div4; NHWC flatten; Linear/ReLU/Linear",
            "provider_algorithm_theorem": "UNVERIFIED",
        }

    def scope(self, graph: Any, arguments: list[Any], owner: str) -> list[Any]:
        function = owner != "graph"
        names = list(graph.input) if function else [value.name for value in graph.input]
        if len(names) != len(arguments):
            raise ValueError("CNN function input arity changed")
        environment = dict(zip(names, arguments, strict=True))
        if not function:
            for initializer in graph.initializer:
                environment[initializer.name] = numpy_helper.to_array(
                    initializer
                ).copy()
                self.constant_owners.append(
                    {
                        "owner": owner,
                        "name": initializer.name,
                        **array_record(environment[initializer.name]),
                    }
                )
        selections = self.relu_selections(graph)
        for node in graph.node:
            selection = (
                selections.get(node.output[0]) if len(node.output) == 1 else None
            )
            if selection is not None and node.output[0] != selection["base"]:
                if node.output[0] == selection["final"]:
                    outputs = [self.selected_relu(selection, environment, graph)]
                    environment[node.output[0]] = outputs[0]
                else:
                    # Only the complete, owned composite below can expose Data.
                    # Intermediate selection values have no generic meaning here.
                    outputs = [None]
                self.operations.append(
                    {
                        "owner": owner,
                        "op": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                        "result_stage": (
                            outputs[0].stage if isinstance(outputs[0], Data) else None
                        ),
                        "result_layout": (
                            outputs[0].layout if isinstance(outputs[0], Data) else None
                        ),
                    }
                )
                continue
            values = [environment[name] if name else None for name in node.input]
            attributes = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            if (node.domain, node.op_type) in self.functions:
                if node.op_type != "OriginalCNNSource" or owner != "graph":
                    raise ValueError("unexpected nested function in original CNN")
                self.function_calls.append(node.op_type)
                outputs = self.scope(
                    self.functions[(node.domain, node.op_type)], values, node.op_type
                )
            elif node.domain in ("", "ai.onnx"):
                outputs = [self.operation(node.op_type, values, attributes)]
            else:
                raise ValueError("unexpected original CNN operator domain")
            if len(outputs) != len(node.output):
                raise ValueError("original CNN node output arity changed")
            environment.update(zip(node.output, outputs, strict=True))
            self.operations.append(
                {
                    "owner": owner,
                    "op": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "result_stage": (
                        outputs[0].stage if isinstance(outputs[0], Data) else None
                    ),
                    "result_layout": (
                        outputs[0].layout if isinstance(outputs[0], Data) else None
                    ),
                }
            )
            if node.op_type == "Constant":
                self.constant_owners.append(
                    {"owner": owner, "name": node.output[0], **array_record(outputs[0])}
                )
        names = (
            list(graph.output) if function else [value.name for value in graph.output]
        )
        return [environment[name] for name in names]

    @staticmethod
    def relu_selections(graph: Any) -> dict[str, dict[str, Any]]:
        """Recognize only the actual SOURCE Max signed-zero selection composite.

        SSA ownership prevents extra inputs, exchanged arms or external uses of
        an internal value. CSE may share the constant-only Equal(0,0) across
        otherwise independent ReLUs; the complete union still owns every use.
        """
        producers = {}
        consumers: dict[str, Counter[str]] = {}
        for node in graph.node:
            for name in node.output:
                if name in producers:
                    raise ValueError("original CNN duplicate artifact producer")
                producers[name] = node
            for name in node.input:
                if name:
                    consumers.setdefault(name, Counter())[node.output[0]] += 1
        patterns = []
        owned_uses: dict[str, Counter[str]] = {}
        for final in graph.node:
            if final.op_type != "Squeeze" or not final.input:
                continue
            parent = producers.get(final.input[0])
            if parent is None or parent.op_type != "GatherElements":
                continue
            nodes = {}

            def take(
                name: str, operator: str, arity: int, attributes: dict[str, Any]
            ) -> Any:
                node = producers.get(name)
                if (
                    node is None
                    or node.domain not in ("", "ai.onnx")
                    or node.op_type != operator
                    or len(node.input) != arity
                    or list(node.output) != [name]
                ):
                    raise ValueError(
                        "original CNN signed-zero ReLU selection producer/arity changed"
                    )
                actual = {
                    attribute.name: onnx.helper.get_attribute_value(attribute)
                    for attribute in node.attribute
                }
                if actual != attributes:
                    raise ValueError(
                        "original CNN signed-zero ReLU selection attributes changed"
                    )
                nodes[name] = node
                return node

            squeeze = take(final.output[0], "Squeeze", 2, {})
            axes = squeeze.input[1]
            selected = take(squeeze.input[0], "GatherElements", 2, {"axis": 0})
            choices = take(selected.input[0], "Concat", 2, {"axis": 0})
            base_choice = take(choices.input[0], "Unsqueeze", 2, {})
            correction_choice = take(choices.input[1], "Unsqueeze", 2, {})
            base = take(base_choice.input[0], "Max", 2, {})
            correction = take(correction_choice.input[0], "Add", 2, {})
            lhs, rhs = base.input
            if list(correction.input) != [lhs, rhs]:
                raise ValueError("original CNN signed-zero ReLU correction arm changed")
            index_choice = take(selected.input[1], "Unsqueeze", 2, {})
            indices = take(index_choice.input[0], "Cast", 1, {"to": 7})
            ties = take(indices.input[0], "And", 2, {})
            left_zero = take(ties.input[0], "Equal", 2, {})
            right_zero = take(ties.input[1], "Equal", 2, {})
            zero = left_zero.input[1]
            if (
                list(left_zero.input) != [lhs, zero]
                or list(right_zero.input) != [rhs, zero]
                or any(
                    node.input[1] != axes
                    for node in (base_choice, correction_choice, index_choice)
                )
            ):
                raise ValueError(
                    "original CNN signed-zero ReLU mask/axis wiring changed"
                )
            pattern = {
                "base": base.output[0],
                "final": squeeze.output[0],
                "lhs": lhs,
                "rhs": rhs,
                "zero": zero,
                "axes": axes,
                "nodes": nodes,
                "signatures": {
                    base.output[0]: (1, "data"),
                    left_zero.output[0]: (9, "data"),
                    right_zero.output[0]: (9, "scalar"),
                    ties.output[0]: (9, "data"),
                    correction.output[0]: (1, "data"),
                    base_choice.output[0]: (1, "choice"),
                    correction_choice.output[0]: (1, "choice"),
                    choices.output[0]: (1, "choices"),
                    indices.output[0]: (7, "data"),
                    index_choice.output[0]: (7, "choice"),
                    selected.output[0]: (1, "choice"),
                    squeeze.output[0]: (1, "data"),
                },
            }
            patterns.append(pattern)
            for name, node in nodes.items():
                if name == pattern["final"]:
                    continue
                uses = Counter(
                    consumer.output[0]
                    for consumer in nodes.values()
                    for input_name in consumer.input
                    if input_name == name
                )
                # Shared Equal(0,0) legitimately contributes a different And
                # use in each composite after common-subexpression elimination.
                owned_uses.setdefault(name, Counter()).update(uses)
        graph_outputs = {
            value if isinstance(value, str) else value.name for value in graph.output
        }
        for name, uses in owned_uses.items():
            if name in graph_outputs or consumers.get(name, Counter()) != uses:
                raise ValueError(
                    "original CNN signed-zero ReLU internal ownership changed"
                )
        result = {}
        for pattern in patterns:
            for name in pattern["nodes"]:
                if name in result and producers[name].op_type != "Equal":
                    raise ValueError("original CNN signed-zero ReLU composites overlap")
                result[name] = pattern
        return result

    def selected_relu(
        self, selection: dict[str, Any], environment: dict[str, Any], graph: Any
    ) -> Data:
        def positive_zero(value: Any) -> bool:
            return (
                isinstance(value, np.ndarray)
                and value.dtype == np.dtype("float32")
                and value.shape == ()
                and value.tobytes() == b"\0\0\0\0"
            )

        if not positive_zero(environment[selection["rhs"]]) or not positive_zero(
            environment[selection["zero"]]
        ):
            raise ValueError(
                "original CNN signed-zero ReLU scalar FP32 positive zero changed"
            )
        axes = environment[selection["axes"]]
        if (
            not isinstance(axes, np.ndarray)
            or axes.dtype != np.dtype("int64")
            or axes.shape != (1,)
            or axes.item() != 0
        ):
            raise ValueError("original CNN signed-zero ReLU selection axes changed")
        result = self.operation(
            "Max", [environment[selection["lhs"]], environment[selection["rhs"]]], {}
        )
        shapes = {
            "scalar": (),
            "data": result.shape,
            "choice": (1, *result.shape),
            "choices": (2, *result.shape),
        }
        for value in getattr(graph, "value_info", ()):
            if value.name not in selection["signatures"]:
                continue
            dtype, shape = selection["signatures"][value.name]
            if (
                value.type.tensor_type.elem_type != dtype
                or self.dimensions(value) != shapes[shape]
            ):
                raise ValueError(
                    "original CNN signed-zero ReLU annotated type/shape changed"
                )
        return result

    def operation(self, op: str, values: list[Any], attributes: dict[str, Any]) -> Any:
        def allowed(*names: str) -> None:
            if set(attributes) - set(names):
                raise ValueError(f"unexpected {op} attributes")

        if op == "Constant":
            allowed("value")
            return numpy_helper.to_array(attributes["value"]).copy()
        if op == "Identity":
            allowed()
            return values[0]
        if op == "Transpose":
            allowed("perm")
            permutation = tuple(attributes["perm"])
            if isinstance(values[0], Data):
                x = values[0]
                if len(permutation) != len(x.shape) or sorted(permutation) != list(
                    range(len(x.shape))
                ):
                    raise ValueError("invalid CNN data permutation")
                return Data(
                    x.stage,
                    tuple(x.shape[index] for index in permutation),
                    "".join(x.layout[index] for index in permutation),
                )
            return values[0].transpose(permutation)
        if op == "Shape":
            allowed("start", "end")
            if not isinstance(values[0], Data) or attributes:
                raise ValueError("CNN requires the complete actual data shape")
            return np.asarray(values[0].shape, dtype=object)
        if op == "Gather":
            allowed("axis")
            if isinstance(values[0], Data) or attributes.get("axis", 0) != 0:
                raise ValueError("only original shape Gather is supported")
            return np.take(values[0], values[1], axis=0)
        if op in ("Unsqueeze", "Squeeze"):
            allowed()
            if isinstance(values[0], Data):
                raise ValueError("unexpected CNN data squeeze")
            axes = tuple(int(axis) for axis in values[1].reshape(-1))
            return (
                np.expand_dims(values[0], axes)
                if op == "Unsqueeze"
                else np.squeeze(values[0], axes)
            )
        if op == "Concat":
            allowed("axis")
            if (
                any(isinstance(value, Data) for value in values)
                or attributes.get("axis", 0) != 0
            ):
                raise ValueError("only original shape Concat is supported")
            return np.concatenate(values, axis=0)
        if op == "Reshape":
            allowed("allowzero")
            if attributes.get("allowzero", 0) != 0:
                raise ValueError("CNN reshape allowzero changed")
            target = tuple(values[1].reshape(-1).tolist())
            x = values[0]
            if not isinstance(x, Data):
                return x.reshape(target)
            if target and target[0] == 0:
                target = (x.shape[0], *target[1:])
            if x.stage in ("conv1.relu", "conv2.relu") and x.layout == "NHWC":
                _, h, w, c = x.shape
                expected = (self.batch, h // 2, 2, w // 2, 2, c)
                if target != expected:
                    raise ValueError("original CNN pool block wiring changed")
                return Data(x.stage.replace(".relu", ".blocks"), expected, "NhrwcC")
            if x == Data("conv2.pool", (self.batch, 7, 7, 64), "NHWC") and target in (
                (self.batch, 3136),
                (self.batch, -1),
            ):
                return Data("flatten", (self.batch, 3136), "NC")
            raise ValueError("original CNN flatten/reshape order changed")
        if op == "Conv":
            allowed("kernel_shape", "strides", "pads", "dilations", "group", "auto_pad")
            x = values[0]
            names = {"input": ("conv1", 1, 32, 28), "conv1.pool": ("conv2", 32, 64, 14)}
            if not isinstance(x, Data) or x.stage not in names or x.layout != "NCHW":
                raise ValueError("original convolution input/layout changed")
            name, ci, co, size = names[x.stage]
            if (
                x.shape != (self.batch, ci, size, size)
                or attributes.get("kernel_shape", [3, 3]) != [3, 3]
                or attributes.get("strides", [1, 1]) != [1, 1]
                or attributes.get("dilations", [1, 1]) != [1, 1]
                or attributes.get("group", 1) != 1
            ):
                raise ValueError("original convolution geometry changed")
            if attributes.get("auto_pad", b"NOTSET") not in (
                b"NOTSET",
                b"",
            ) or attributes.get("pads", [0, 0, 0, 0]) != [1, 1, 1, 1]:
                raise ValueError("original SAME convolution padding changed")
            self.parameter(values[1], name + ".kernel", convolution=True)
            stage = name + ".raw"
            if len(values) == 3:
                self.bias(values[2], name + ".bias")
                stage = name + ".affine"
            elif len(values) != 2:
                raise ValueError("CNN convolution arity changed")
            return Data(stage, (self.batch, co, size, size), "NCHW")
        if op == "MatMul":
            allowed()
            x = values[0]
            if x == Data("flatten", (self.batch, 3136), "NC"):
                name, width = "linear1", 256
            elif x == Data("linear1.relu", (self.batch, 256), "NC"):
                name, width = "linear2", 10
            else:
                raise ValueError("original CNN linear wiring changed")
            self.parameter(values[1], name + ".kernel")
            return Data(name + ".raw", (self.batch, width), "NC")
        if op == "Add":
            allowed()
            if isinstance(values[1], Data):
                values = list(reversed(values))
            x = values[0]
            if not isinstance(x, Data) or not x.stage.endswith(".raw"):
                raise ValueError("unexpected original CNN Add")
            self.bias(values[1], x.stage.replace(".raw", ".bias"), x)
            return Data(x.stage.replace(".raw", ".affine"), x.shape, x.layout)
        if op in ("Max", "Relu"):
            allowed()
            x = next((value for value in values if isinstance(value, Data)), None)
            if x is None or x.stage not in (
                "conv1.affine",
                "conv2.affine",
                "linear1.affine",
            ):
                raise ValueError("original CNN ReLU wiring changed")
            expected = {
                "conv1.affine": (
                    Data("conv1.affine", (self.batch, 28, 28, 32), "NHWC"),
                    Data("conv1.affine", (self.batch, 32, 28, 28), "NCHW"),
                ),
                "conv2.affine": (
                    Data("conv2.affine", (self.batch, 14, 14, 64), "NHWC"),
                    Data("conv2.affine", (self.batch, 64, 14, 14), "NCHW"),
                ),
                "linear1.affine": (Data("linear1.affine", (self.batch, 256), "NC"),),
            }
            if x not in expected[x.stage]:
                raise ValueError("original CNN ReLU layout/shape changed")
            if op == "Max":
                constants = [value for value in values if not isinstance(value, Data)]
                if (
                    len(values) != 2
                    or sum(isinstance(value, Data) for value in values) != 1
                    or len(constants) != 1
                    or not isinstance(constants[0], np.ndarray)
                    or constants[0].dtype != np.dtype("float32")
                    or constants[0].shape != ()
                    or constants[0].size != 1
                    or constants[0].item() != 0
                ):
                    raise ValueError("original CNN ReLU clipping changed")
            elif len(values) != 1:
                raise ValueError("original CNN Relu arity changed")
            return Data(x.stage.replace(".affine", ".relu"), x.shape, x.layout)
        if op == "ReduceSum":
            allowed("keepdims", "noop_with_empty_axes")
            x = values[0]
            if (
                len(values) != 2
                or not isinstance(x, Data)
                or x
                not in (
                    Data("conv1.blocks", (self.batch, 14, 2, 14, 2, 32), "NhrwcC"),
                    Data("conv2.blocks", (self.batch, 7, 2, 7, 2, 64), "NhrwcC"),
                )
                or not isinstance(values[1], np.ndarray)
                or values[1].dtype != np.dtype("int64")
                or values[1].shape != (2,)
                or tuple(values[1]) != (2, 4)
                or attributes.get("keepdims", 1) != 0
                or attributes.get("noop_with_empty_axes", 0) != 0
            ):
                raise ValueError("original CNN pool sum axes/order changed")
            shape = tuple(
                size for index, size in enumerate(x.shape) if index not in (2, 4)
            )
            return Data(x.stage.replace(".blocks", ".sum"), shape, "NHWC")
        if op == "Div":
            allowed()
            x, divisor = values
            if (
                not isinstance(x, Data)
                or x.stage not in ("conv1.sum", "conv2.sum")
                or not isinstance(divisor, np.ndarray)
                or divisor.dtype != np.dtype("float32")
                or divisor.size != 1
                or divisor.item() != 4
            ):
                raise ValueError("original CNN separate Div4 changed")
            return Data(x.stage.replace(".sum", ".pool"), x.shape, x.layout)
        raise ValueError(f"unreviewed original CNN artifact operator {op}")


def capture_targets(
    model: Any, feeds: dict[int, np.ndarray], directory: Path, guard: Callable[[], None]
) -> dict[str, dict[str, Any]]:
    import onnxruntime as ort  # type: ignore[import-untyped]

    observations = {}
    original_options = ort.SessionOptions
    blob = model.SerializeToString()
    for optimization in ("disabled", "all"):
        cell = directory / optimization
        cell.mkdir(parents=True, exist_ok=True)
        graph_path = cell / "actual-runtime-graph.onnx"
        options_receipt: dict[str, Any] = {}
        session_started = time.monotonic()
        peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

        def options() -> Any:
            value = original_options()
            value.optimized_model_filepath = str(graph_path)
            options_receipt["denormal_override"] = "absent; no config entry is added"
            try:
                options_receipt["existing_denormal_entry"] = (
                    value.get_session_config_entry("session.set_denormal_as_zero")
                )
            except RuntimeError:
                options_receipt["existing_denormal_entry"] = None
            return value

        session = None
        try:
            # Reuse the existing checked save/reload and CPU session builder.
            # Only the runtime graph output path is added; no FTZ override.
            with patch.object(ort, "SessionOptions", options):
                session = _session(model, cell, optimization == "all")
            session.disable_fallback()
            options_receipt.update(
                providers=session.get_providers(),
                provider_options=session.get_provider_options(),
                runtime_version=ort.__version__,
                architecture=platform.machine(),
                intra_op_threads=1,
                inter_op_threads=1,
                optimization=optimization,
                artifact_sha256=hashlib.sha256(blob).hexdigest(),
                reloaded_artifact_sha256=hashlib.sha256(
                    (cell / "source_dot.onnx").read_bytes()
                ).hexdigest(),
            )
            options_receipt["runtime_graph_exists"] = graph_path.is_file()
            if graph_path.is_file():
                options_receipt["runtime_graph_sha256"] = hashlib.sha256(
                    graph_path.read_bytes()
                ).hexdigest()
        except Exception as exc:
            options_receipt["load_exception"] = exception_record(exc)
        options_receipt.update(
            session_construction_attempts=1,
            requested_run_attempts=2 * len(feeds),
            session_wall_seconds=time.monotonic() - session_started,
            process_peak_rss_bytes_before=peak_before,
            process_peak_rss_bytes_after=resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
        )
        write_json(cell / "session.json", options_receipt)
        for batch, feed in feeds.items():
            for repeat in range(2):

                def execute() -> Any:
                    if session is None:
                        raise RuntimeError("recorded CPU session load failed")
                    return session.run(None, {session.get_inputs()[0].name: feed})

                label = f"B{batch}-{optimization}-{repeat}"
                observations[label] = capture_attempt(
                    execute, directory, label, guard=guard
                )
                observations[label]["session_contract"] = options_receipt
        # Both CPU profiles are attempted even if either construction/run failed.
    return observations


def capture_oracles_and_targets(
    parameters_: dict[str, np.ndarray],
    feeds: dict[int, np.ndarray],
    model: Any,
    directory: Path,
    guard: Callable[[], None],
    *,
    artifact_sha256: str | None,
    export_error: dict[str, str] | None,
) -> dict[str, Any]:
    """Freeze every a priori setup attempt before independent CPU captures.

    Returned exceptions are evidence, never a passing acceptance status. The
    original-CNN caller must reject each nonempty failure field after capture.
    """
    envelopes = {}
    oracle_errors = {}
    for batch, value in feeds.items():
        try:
            envelopes[batch] = oracle(
                parameters_, value, directory / f"oracle-B{batch}", artifact_sha256
            )
        except Exception as exc:
            oracle_errors[batch] = exception_record(exc)
            write_json(
                directory / f"oracle-B{batch}-exception.json", oracle_errors[batch]
            )
    targets = {}
    target_error = None
    if model is not None and artifact_sha256 is not None:
        try:
            targets = capture_targets(model, feeds, directory / "target", guard)
        except Exception as exc:
            target_error = exception_record(exc)
            write_json(directory / "target-capture-exception.json", target_error)
    else:
        write_json(
            directory / "target-unavailable.json",
            {
                "reason": "original artifact export failed",
                "export_exception": export_error,
                "unavailable_modes": ["disabled", "all"],
                "unavailable_batches": list(feeds),
                "requested_attempts": 4 * len(feeds),
            },
        )
    return {
        "envelopes": envelopes,
        "targets": targets,
        "oracle_exceptions": oracle_errors,
        "target_capture_exception": target_error,
    }
