# tests/extra_tests/converter/precision_test_support.py

"""Shared bounded source-policy harness for regressions and internal experiments.

Reuse the real tracing, plugins, optimizer, and serialization pipeline for a
small, explicitly bounded source-dtype experiment using private core policy.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
import onnx
import onnx_ir as ir

from jax2onnx.converter import conversion_api
from jax2onnx.converter.ir_postprocess import postprocess_ir_model
from jax2onnx.converter.precision_policy import PrecisionPolicy


class PrototypeLimit(ValueError):
    """Outside the experiment's audited subset, not an ONNX support verdict."""


class RepresentationLimit(ValueError):
    """The selected ONNX representation cannot express this operation."""


_PRIMITIVES = frozenset(
    {
        "add",
        "sub",
        "mul",
        "neg",
        "div",
        "sqrt",
        "jax.numpy.sqrt",
        "broadcast_in_dim",
        "dot_general",
        "convert_element_type",
        "reduce_sum",
        "bitcast_convert_type",
        "gt",
        "cond",
    }
)
_FLOATS = frozenset({"float16", "bfloat16", "float32", "float64"})


def _validate_aval(aval: Any, location: str) -> None:
    dtype = np.dtype(aval.dtype)
    if dtype.name not in _FLOATS | {"uint32", "bool", "int32"}:
        raise PrototypeLimit(f"{location}: dtype {dtype} is outside the PoC subset")
    if any(not isinstance(dim, (int, np.integer)) for dim in aval.shape):
        raise PrototypeLimit(f"{location}: only static shapes are covered by the PoC")


def _validate_trace(jaxpr: Any, opset: int) -> None:
    if jaxpr.effects:
        raise PrototypeLimit("Effects are outside the PoC subset")
    for var in (*jaxpr.invars, *jaxpr.constvars, *jaxpr.outvars):
        _validate_aval(var.aval, "graph boundary")
    for index, eqn in enumerate(jaxpr.eqns):
        primitive = eqn.primitive.name
        location = f"equation {index} ({primitive})"
        if primitive not in _PRIMITIVES and not primitive.startswith("onnx_fn::"):
            raise PrototypeLimit(
                f"{location}: outside the PoC subset; loops, JIT regions, and "
                "other primitives need separate integration work"
            )
        for var in (*eqn.invars, *eqn.outvars):
            _validate_aval(var.aval, location)
        if primitive.startswith("onnx_fn::"):
            # Actual function bodies are retraced by their plugin and validated
            # by the same scope callback before their equations are lowered.
            continue
        if primitive == "cond":
            branches = eqn.params.get("branches", ())
            if len(branches) != 2:
                raise PrototypeLimit(f"{location}: only two-branch cond is covered")
            for branch in branches:
                _validate_trace(getattr(branch, "jaxpr", branch), opset)
            continue
        if primitive == "gt":
            if any(np.dtype(var.aval.dtype).name not in _FLOATS for var in eqn.invars):
                raise PrototypeLimit(
                    f"{location}: only floating comparisons are covered"
                )
            continue
        if primitive == "convert_element_type" and (
            np.dtype(eqn.invars[0].aval.dtype) == np.dtype(np.bool_)
            and np.dtype(eqn.outvars[0].aval.dtype) == np.dtype(np.int32)
        ):
            continue  # JAX's selector conversion for the supported cond.
        if primitive == "bitcast_convert_type":
            source = np.dtype(eqn.invars[0].aval.dtype)
            target = np.dtype(eqn.outvars[0].aval.dtype)
            if source.itemsize != target.itemsize:
                raise RepresentationLimit(
                    f"{location}: {source} -> {target} changes word width; "
                    "ONNX BitCast requires equal widths. No alternative "
                    "representation is implemented in this experiment."
                )
            if opset < 26:
                raise RepresentationLimit(
                    f"{location}: {source} -> {target} needs ONNX BitCast at opset "
                    f"26; selected opset is {opset}. Explicitly select 26 if your "
                    "deployment supports it; the PoC does not upgrade it."
                )
        elif any(
            np.dtype(var.aval.dtype).name not in _FLOATS
            for var in (*eqn.invars, *eqn.outvars)
        ):
            raise PrototypeLimit(f"{location}: only floating arithmetic is covered")


def export_source_dtypes(
    fn: Callable[..., Any],
    inputs: Sequence[jax.ShapeDtypeStruct],
    *,
    opset: int = 23,
) -> onnx.ModelProto:
    """Export the bounded experiment using the caller's current JAX configuration.

    Only typed, static inputs and the listed primitives are covered. This checks
    the ONNX model, not runtime kernels or general numerical equivalence.
    """
    if opset not in (23, 26):
        raise PrototypeLimit("The PoC covers only opsets 23 and 26")
    for index, spec in enumerate(inputs):
        if not isinstance(spec, jax.ShapeDtypeStruct):
            raise PrototypeLimit("The PoC requires typed jax.ShapeDtypeStruct inputs")
        _validate_aval(spec, f"input {index}")
        if np.dtype(spec.dtype).name not in _FLOATS:
            raise PrototypeLimit("The PoC accepts only floating graph inputs")
        if np.dtype(spec.dtype) == np.dtype(np.float64) and not jax.config.x64_enabled:
            raise ValueError(
                f"input {index} requests float64, but JAX x64 is disabled; "
                "configure JAX before creating arrays and exporting"
            )

    model = conversion_api._convert_to_ir(
        fn=fn,
        inputs=inputs,
        input_params=None,
        model_name="precision_source_dtype_experiment",
        opset=opset,
        enable_double_precision=False,
        record_primitive_calls_file=None,
        protective_clone=True,
        strict_optimizer_failures=True,
        precision_policy=PrecisionPolicy.SOURCE,
        jaxpr_validator=_validate_trace,
    )
    # The common path checked these boundary types against the source Jaxpr.
    # Retain them through postprocessing and protobuf serialization as well.
    input_dtypes = tuple(value.dtype for value in model.graph.inputs)
    output_dtypes = tuple(value.dtype for value in model.graph.outputs)
    postprocess_ir_model(model, promote_to_double=False)
    proto = ir.to_proto(model)
    onnx.checker.check_model(proto, full_check=True)
    for label, expected_dtypes, values in (
        ("input", input_dtypes, proto.graph.input),
        ("output", output_dtypes, proto.graph.output),
    ):
        for expected, value in zip(expected_dtypes, values, strict=True):
            if expected is None or value.type.tensor_type.elem_type != int(expected):
                raise RuntimeError(
                    f"Final {label} {value.name} lost source type {expected}"
                )
    return proto
