# internal_docs/precision/poc/numerical_contract.py

"""Bounded numerical-contract experiments, not a general exporter.

Run with ``python -m internal_docs.precision.poc.numerical_contract``. Models
remain in memory. Hand-built ONNX graphs below are feasibility controls and
never evidence that the public exporter implements their source semantics.
"""

import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from internal_docs.precision.poc.source_export import export_source_dtypes
from jax2onnx import to_onnx
from tests.extra_tests.converter.precision_numerical_support import (
    normalization_input,
    normalization_reference,
    selected_variance_normalization,
)
from tests.extra_tests.converter.precision_runtime_support import (
    SampleAcceptanceError,
    check_cpu_sample,
    require_sample_acceptance,
)


def preferred_dot(x: jax.Array, w: jax.Array) -> jax.Array:
    """FP16 operands with JAX's requested FP32 preferred element type."""
    return jax.lax.dot_general(
        x,
        w,
        dimension_numbers=(((2,), (1,)), ((0,), (0,))),
        preferred_element_type=jnp.float32,
    )


def explicit_float32_dot(x: jax.Array, w: jax.Array) -> jax.Array:
    """A changed source representation to test, not an automatic lowering."""
    return preferred_dot(x.astype(jnp.float32), w.astype(jnp.float32))


def square_sum(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    square = x * x
    return square, jax.lax.reduce_sum(square, axes=(0,))


def explicit_rounding_square_sum(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Make the intermediate FP16 rounding observable to the tested runtime.

    This source candidate changes the arithmetic to explicit FP32 Mul and Sum,
    retaining the intermediate FP16 boundary and original result dtypes. Its
    equivalence is established only for this experiment's underflow sample.
    """
    wider = x.astype(jnp.float32)
    square = (wider * wider).astype(jnp.float16)
    total = jax.lax.reduce_sum(square.astype(jnp.float32), axes=(0,))
    return square, total.astype(jnp.float16)


def _control_model(
    name: str,
    nodes: Sequence[onnx.NodeProto],
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    initializers: Sequence[onnx.TensorProto],
) -> onnx.ModelProto:
    model = helper.make_model(
        helper.make_graph(nodes, name, inputs, outputs, initializer=initializers),
        opset_imports=[helper.make_opsetid("", 23)],
        ir_version=11,
        producer_name="internal_numerical_feasibility_control",
    )
    onnx.checker.check_model(model, full_check=True)
    return model


def normalization_control() -> onnx.ModelProto:
    """Hand-built fixed (2,16) ONNX feasibility control, not source export."""
    node = helper.make_node
    tensor = helper.make_tensor_value_info
    nodes = [
        node("ReduceSum", ["x", "axis"], ["sum32"], keepdims=1),
        node("Div", ["sum32", "count32"], ["mean32"]),
        node("Sub", ["x", "mean32"], ["centered"]),
        node("Mul", ["centered", "centered"], ["squares"]),
        node("Cast", ["squares"], ["squares64"], to=TensorProto.DOUBLE),
        node("ReduceSum", ["squares64", "axis"], ["variance_sum64"], keepdims=1),
        node("Div", ["variance_sum64", "count64"], ["variance64"]),
        node("Cast", ["variance64"], ["variance"], to=TensorProto.FLOAT),
        node("Add", ["variance", "epsilon32"], ["stabilized32"]),
        node("Sqrt", ["stabilized32"], ["denominator32"]),
        node("Div", ["centered", "denominator32"], ["normalized"]),
    ]
    return _control_model(
        "normalization_feasibility_control",
        nodes,
        [tensor("x", TensorProto.FLOAT, [2, 16])],
        [
            tensor("centered", TensorProto.FLOAT, [2, 16]),
            tensor("squares", TensorProto.FLOAT, [2, 16]),
            tensor("variance", TensorProto.FLOAT, [2, 1]),
            tensor("normalized", TensorProto.FLOAT, [2, 16]),
        ],
        [
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis"),
            numpy_helper.from_array(np.asarray(16, dtype=np.float32), "count32"),
            numpy_helper.from_array(np.asarray(16, dtype=np.float64), "count64"),
            numpy_helper.from_array(np.asarray(1e-5, dtype=np.float32), "epsilon32"),
        ],
    )


def unfused_square_sum_control() -> onnx.ModelProto:
    """Hand-built Mul -> ReduceSum control; an unfused graph can still differ."""
    tensor = helper.make_tensor_value_info
    return _control_model(
        "unfused_square_sum_feasibility_control",
        [
            helper.make_node("Mul", ["x", "x"], ["square"]),
            helper.make_node("ReduceSum", ["square", "axis"], ["total"], keepdims=0),
        ],
        [tensor("x", TensorProto.FLOAT16, [8])],
        [
            tensor("square", TensorProto.FLOAT16, [8]),
            tensor("total", TensorProto.FLOAT16, []),
        ],
        [numpy_helper.from_array(np.asarray([0], dtype=np.int64), "axis")],
    )


def run_cpu(
    model: onnx.ModelProto, inputs: Sequence[np.ndarray], *, optimize: bool
) -> tuple[np.ndarray, ...]:
    """Kernel failures propagate; no unsupported-kernel success classification."""
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    assert session.get_providers() == ["CPUExecutionProvider"]
    feeds = {
        spec.name: value
        for spec, value in zip(session.get_inputs(), inputs, strict=True)
    }
    return tuple(session.run(None, feeds))


def _arrays(tree: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(value) for value in jax.tree.leaves(tree))


def _assert_equal(
    actual: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
    *,
    close_final: bool = False,
) -> None:
    assert len(actual) == len(expected)
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        assert value.dtype == reference.dtype
        assert value.shape == reference.shape
        # Only the final normalized values allow sqrt/division variation. In
        # particular variance 8388608 must never pass in place of 8388609.
        if close_final and index == len(actual) - 1:
            np.testing.assert_allclose(value, reference, rtol=3e-7, atol=1e-7)
        else:
            np.testing.assert_array_equal(value, reference)


def rounding_status(outputs: Sequence[np.ndarray]) -> str:
    """Accept agreement or the known diagnostic, rejecting unfamiliar results.

    Tests do not require an ORT defect to persist. A future runtime producing
    the JAX result is accepted and reported as numerical agreement.
    """
    square, total = outputs
    assert square.dtype == total.dtype == np.float16
    assert square.shape == (8,) and total.shape == ()
    np.testing.assert_array_equal(square, np.zeros(8, dtype=np.float16))
    if total == np.float16(0):
        return "accepted: sample matches JAX"
    if total == np.float16(2**-23):
        return "diagnostic: zero FP16 square output, nonzero sum (2**-23)"
    raise AssertionError(f"Unexpected FP16 intermediate result: {total!r}")


@dataclass(frozen=True)
class Observation:
    name: str
    origin: str
    model: onnx.ModelProto
    eager: tuple[np.ndarray, ...]
    compiled: tuple[np.ndarray, ...]
    reference: tuple[np.ndarray, ...]
    disabled: tuple[np.ndarray, ...]
    enabled: tuple[np.ndarray, ...]
    semantic_status: str
    disabled_status: str
    enabled_status: str


def _observe(
    name: str,
    origin: str,
    fn: Callable[..., Any],
    inputs: Sequence[np.ndarray],
    model: onnx.ModelProto,
    *,
    kind: str = "agreement",
) -> Observation:
    eager = _arrays(fn(*(jnp.asarray(value) for value in inputs)))
    compiled = _arrays(jax.jit(fn)(*(jnp.asarray(value) for value in inputs)))
    close = kind == "normalization"
    _assert_equal(compiled, eager, close_final=close)
    feeds = {
        spec.name: value for spec, value in zip(model.graph.input, inputs, strict=True)
    }
    reference = tuple(ReferenceEvaluator(model).run(None, feeds))
    disabled = run_cpu(model, inputs, optimize=False)
    enabled = run_cpu(model, inputs, optimize=True)
    status = "accepted: sample matches JAX"
    semantic_status = "sample agreement; general equivalence unverified"
    if kind == "dot_gap":
        _assert_equal(eager, (np.asarray([[[2049]]], dtype=np.float32),))
        expected = (np.asarray([[[2048]]], dtype=np.float16),)
        for values in (reference, disabled, enabled):
            _assert_equal(values, expected)
        status = "diagnostic: JAX FP32 2049 became FP16 2048"
        semantic_status = "known result-type/representation gap in public export"
    elif kind == "rounding":
        expected = (np.zeros(8, dtype=np.float16), np.asarray(0, dtype=np.float16))
        _assert_equal(eager, expected)
        _assert_equal(reference, expected)
        semantic_status = (
            "typed graph and ReferenceEvaluator agree; ORT checked separately"
        )
        return Observation(
            name,
            origin,
            model,
            eager,
            compiled,
            reference,
            disabled,
            enabled,
            semantic_status,
            rounding_status(disabled),
            rounding_status(enabled),
        )
    else:
        for values in (reference, disabled, enabled):
            _assert_equal(values, eager, close_final=close)
        if close:
            independent = normalization_reference(inputs[0])
            for values in (eager, compiled, reference, disabled, enabled):
                _assert_equal(values, independent, close_final=True)
            semantic_status = "sample agreement with independent staged reference"
    return Observation(
        name,
        origin,
        model,
        eager,
        compiled,
        reference,
        disabled,
        enabled,
        semantic_status,
        status,
        status,
    )


def run_experiments() -> tuple[Observation, ...]:
    """Model preparation owns x64; always restore the caller's global setting."""
    previous = bool(jax.config.x64_enabled)
    try:
        jax.config.update("jax_enable_x64", True)
        x = np.asarray([[[2048, 1]]], dtype=np.float16)
        w = np.ones((1, 2, 1), dtype=np.float16)
        specs = [jax.ShapeDtypeStruct(v.shape, v.dtype) for v in (x, w)]
        baseline = to_onnx(preferred_dot, specs, enable_double_precision=False)
        candidate = to_onnx(explicit_float32_dot, specs, enable_double_precision=False)
        normalization = normalization_input()
        normalization_source = export_source_dtypes(
            selected_variance_normalization,
            [jax.ShapeDtypeStruct(normalization.shape, normalization.dtype)],
        )
        dot_source = export_source_dtypes(preferred_dot, specs)
        small = np.full(8, 2**-13, dtype=np.float16)
        small_specs = [jax.ShapeDtypeStruct(small.shape, small.dtype)]
        return (
            _observe(
                "preferred FP32 batched dot",
                "public export baseline",
                preferred_dot,
                (x, w),
                baseline,
                kind="dot_gap",
            ),
            _observe(
                "explicit FP32 operands",
                "changed JAX source + public export",
                explicit_float32_dot,
                (x, w),
                candidate,
            ),
            _observe(
                "selected FP64 variance",
                "hand-built ONNX feasibility control; NOT export",
                selected_variance_normalization,
                (normalization,),
                normalization_control(),
                kind="normalization",
            ),
            _observe(
                "FP16 square then sum",
                "existing source-dtype integration PoC",
                square_sum,
                (small,),
                export_source_dtypes(square_sum, small_specs),
                kind="rounding",
            ),
            _observe(
                "unfused FP16 square then sum",
                "hand-built ONNX feasibility control; NOT export",
                square_sum,
                (small,),
                unfused_square_sum_control(),
                kind="rounding",
            ),
            _observe(
                "explicit FP16 intermediate boundary",
                "changed JAX source + existing integration PoC",
                explicit_rounding_square_sum,
                (small,),
                export_source_dtypes(explicit_rounding_square_sum, small_specs),
            ),
            _observe(
                "selected FP64 variance through source export",
                "actual JAX callable + bounded source export",
                selected_variance_normalization,
                (normalization,),
                normalization_source,
                kind="normalization",
            ),
            _observe(
                "preferred FP32 result through source export",
                "unchanged FP16-input JAX callable + bounded source export",
                preferred_dot,
                (x, w),
                dot_source,
            ),
        )
    finally:
        jax.config.update("jax_enable_x64", previous)


def graph_summary(model: onnx.ModelProto) -> str:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {
        value.name: TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        for value in (
            *inferred.graph.input,
            *inferred.graph.value_info,
            *inferred.graph.output,
        )
    }
    return " -> ".join(
        f"{node.op_type}[{','.join(types[name] for name in node.output)}]"
        for node in inferred.graph.node
    )


def _describe(values: Sequence[np.ndarray]) -> str:
    return "; ".join(
        f"{value.dtype}{value.shape}: "
        + (
            str(value.tolist())
            if value.size <= 8
            else f"first={value.flat[0]}, nonzero={np.count_nonzero(value)}"
        )
        for value in values
    )


def main() -> None:
    print(
        f"Python {platform.python_version()}, JAX/JAXLIB {jax.__version__}/{jax.lib.__version__}, "
        f"NumPy {np.__version__}, ONNX {onnx.__version__}, ONNX IR {onnx_ir.__version__}, "
        f"ORT {ort.__version__}; {platform.machine()}; JAX devices {jax.devices()}"
    )
    print(
        "CPUExecutionProvider; one thread; ORT_DISABLE_ALL and ORT_ENABLE_ALL; opset 23"
    )
    print(
        "Schema validity, source contract, kernel execution, and numerical acceptance are separate checks."
    )
    for case in run_experiments():
        print(f"\n{case.name}: {case.origin}")
        print(
            f"  representation: ONNX full checker passed; {graph_summary(case.model)}"
        )
        print(f"  source contract: {case.semantic_status}")
        print(f"  JAX eager: {_describe(case.eager)}; JIT agrees on this sample")
        print(f"  ONNX ReferenceEvaluator: {_describe(case.reference)}")
        print(
            f"  ORT disabled: {_describe(case.disabled)}; kernel executed; {case.disabled_status}"
        )
        print(
            f"  ORT enabled:  {_describe(case.enabled)}; kernel executed; {case.enabled_status}"
        )
        if case.name == "FP16 square then sum":
            sample = check_cpu_sample(
                case.model,
                [np.full(8, 2**-13, dtype=np.float16)],
                case.eager,
                optimize=False,
            )
            try:
                require_sample_acceptance(sample)
            except SampleAcceptanceError:
                print("  required CPU sample acceptance: rejected (numerical failure)")
            else:
                print("  required CPU sample acceptance: accepted for this sample")
    print(
        "\nNormalization and preferred-FP32 dot now have bounded end-to-end source samples."
    )
    print(
        "FP32 dot operands/results do not prove an accumulator algorithm, native FP32 hardware, or speed."
    )
    print(
        "An unfused graph can show the FP16 mismatch; these observations do not identify fusion as its cause."
    )
    print(
        "Only these fixed CPU samples are checked; no arbitrary shapes/types/backends or bitwise guarantee."
    )


if __name__ == "__main__":
    main()
