# tests/extra_tests/converter/precision_ux_support.py

"""Internal calling-convention experiment, not a proposed supported API.

``precision="source"`` is a discussion aid. Shape-only source inputs default to
FP32 in this candidate; neither the keyword nor that default is a public API
decision. Runtime configuration/execution stays outside the export facade.
Concurrent calls and arbitrary model side effects are outside this experiment.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeAlias, cast
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import onnx

from tests.extra_tests.converter.precision_test_support import (
    PrototypeLimit,
    RepresentationLimit,
    export_source_dtypes,
)
from jax2onnx import to_onnx
from jax2onnx.user_interface import UserInputSpec


class _Omitted(Enum):
    VALUE = "omitted"


OMITTED = _Omitted.VALUE
Route: TypeAlias = Literal["legacy-omitted", "legacy-false", "legacy-true", "source"]
Input: TypeAlias = jax.ShapeDtypeStruct | np.ndarray | jax.Array | Sequence[int | str]


class FailureCategory(Enum):
    OPTIONS = "conflicting-or-invalid-options"
    INPUT = "input-specification"
    JAX_CONFIG = "jax-configuration"
    TRACE = "model-tracing"
    PROTOTYPE = "prototype-scope"
    REPRESENTATION = "onnx-representation"
    EXPORT = "export-failure"


@dataclass(frozen=True)
class InputSummary:
    kind: Literal["shape-only", "typed-spec", "array"]
    shape: tuple[Any, ...]
    dtype: str | None
    weak_type: bool | None


@dataclass(frozen=True)
class CapturedWarning:
    category: str
    message: str


@dataclass(frozen=True)
class CallReport:
    route: Route
    ambient_x64: bool
    after_x64: bool
    input_specs: tuple[InputSummary, ...]
    source_input_dtypes: tuple[str, ...]
    source_input_weak_types: tuple[bool, ...]
    tracing_x64: bool | None
    traced_inputs: tuple[str, ...]
    traced_input_weak_types: tuple[bool, ...]
    traced_outputs: tuple[str, ...]
    model_inputs: tuple[str, ...]
    model_outputs: tuple[str, ...]
    warnings: tuple[CapturedWarning, ...]


@dataclass(frozen=True)
class ExportResult:
    model: onnx.ModelProto
    report: CallReport


class MigrationError(ValueError):
    """An experiment failure with a limited, evidence-based category."""

    def __init__(
        self,
        category: FailureCategory,
        message: str,
        report: CallReport | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.report = report


def _route(precision: str | None, legacy: bool | _Omitted) -> Route:
    if precision not in (None, "source"):
        raise MigrationError(
            FailureCategory.OPTIONS,
            "This internal experiment accepts only precision='source' or omission.",
        )
    if legacy is not OMITTED and type(legacy) is not bool:
        raise MigrationError(
            FailureCategory.OPTIONS,
            "enable_double_precision must be a bool or omitted.",
        )
    if precision == "source":
        if legacy is not OMITTED:
            raise MigrationError(
                FailureCategory.OPTIONS,
                "precision='source' conflicts with an explicitly supplied "
                "enable_double_precision, including False; omit the legacy argument.",
            )
        return "source"
    if legacy is OMITTED:
        return "legacy-omitted"
    return "legacy-true" if legacy else "legacy-false"


def _summarize_input(value: Input) -> InputSummary:
    if isinstance(value, jax.ShapeDtypeStruct):
        return InputSummary(
            "typed-spec",
            tuple(value.shape),
            np.dtype(value.dtype).name,
            bool(value.weak_type),
        )
    if isinstance(value, (jax.Array, np.ndarray)):
        return InputSummary(
            "array",
            tuple(value.shape),
            np.dtype(value.dtype).name,
            bool(getattr(value, "weak_type", False)),
        )
    if isinstance(value, (tuple, list)):
        return InputSummary("shape-only", tuple(value), None, None)
    raise MigrationError(
        FailureCategory.INPUT,
        "Use shape tuples/lists, jax.ShapeDtypeStruct, NumPy arrays, or JAX arrays "
        "in this bounded experiment.",
    )


def _source_inputs(
    summaries: tuple[InputSummary, ...], ambient_x64: bool
) -> tuple[jax.ShapeDtypeStruct, ...]:
    specs = []
    for index, summary in enumerate(summaries):
        if any(
            not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 0
            for dim in summary.shape
        ):
            raise MigrationError(
                FailureCategory.INPUT,
                f"input {index}: this source experiment requires static, nonnegative "
                "integer dimensions; use a fixed shape for the comparison.",
            )
        # An array's dtype is what exists now, not evidence of earlier intent.
        dtype = np.dtype(summary.dtype or np.float32)
        if dtype == np.dtype(np.float64) and not ambient_x64:
            raise MigrationError(
                FailureCategory.JAX_CONFIG,
                f"input {index} requests float64, but JAX x64 is disabled; enable "
                "JAX x64 before preparing FP64 inputs and exporting, or explicitly "
                "choose an FP32 input if that is the intended calculation.",
            )
        specs.append(
            jax.ShapeDtypeStruct(
                tuple(map(int, summary.shape)),
                dtype,
                weak_type=bool(summary.weak_type),
            )
        )
    return tuple(specs)


def _model_dtypes(values: Any) -> tuple[str, ...]:
    return tuple(
        onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
        for value in values
    )


def export_candidate(
    fn: Callable[..., Any],
    inputs: Sequence[Input],
    *,
    precision: Literal["source"] | None = None,
    enable_double_precision: bool | _Omitted = OMITTED,
    opset: int = 23,
) -> ExportResult:
    """Compare real legacy calls with the bounded source exporter.

    Normalize source inputs once at this boundary. Legacy normalization remains
    owned by the actual public entry point, which receives the original inputs.
    The observer records this export's trace without independently tracing again.
    """
    route = _route(precision, enable_double_precision)
    ambient = bool(jax.config.x64_enabled)
    summaries = tuple(map(_summarize_input, inputs))
    source_specs = _source_inputs(summaries, ambient) if route == "source" else ()
    # The public structural protocol declares writable shape/dtype attributes;
    # the concrete array types accepted at this boundary expose properties.
    public_inputs = cast(Sequence[UserInputSpec], inputs)
    traced_inputs: tuple[str, ...] = ()
    traced_input_weak_types: tuple[bool, ...] = ()
    traced_outputs: tuple[str, ...] = ()
    tracing_x64: bool | None = None
    tracing = False

    def observed(*xs: Any) -> Any:
        nonlocal traced_inputs, traced_input_weak_types, traced_outputs
        nonlocal tracing_x64, tracing
        tracing = True
        tracing_x64 = bool(jax.config.x64_enabled)
        traced_inputs = tuple(np.dtype(x.dtype).name for x in xs)
        traced_input_weak_types = tuple(bool(x.weak_type) for x in xs)
        result = fn(*xs)
        traced_outputs = tuple(
            np.dtype(jnp.result_type(value)).name for value in jax.tree.leaves(result)
        )
        tracing = False
        return result

    def report(model: onnx.ModelProto | None) -> CallReport:
        captured = tuple(
            dict.fromkeys(
                CapturedWarning(w.category.__name__, str(w.message))
                for w in recorded_warnings
            )
        )
        return CallReport(
            route=route,
            ambient_x64=ambient,
            after_x64=bool(jax.config.x64_enabled),
            input_specs=summaries,
            source_input_dtypes=tuple(
                np.dtype(spec.dtype).name for spec in source_specs
            ),
            source_input_weak_types=tuple(
                bool(spec.weak_type) for spec in source_specs
            ),
            tracing_x64=tracing_x64,
            traced_inputs=traced_inputs,
            traced_input_weak_types=traced_input_weak_types,
            traced_outputs=traced_outputs,
            model_inputs=_model_dtypes(model.graph.input) if model is not None else (),
            model_outputs=(
                _model_dtypes(model.graph.output) if model is not None else ()
            ),
            warnings=captured,
        )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        try:
            if route == "source":
                model = export_source_dtypes(observed, source_specs, opset=opset)
            elif route == "legacy-omitted":
                model = to_onnx(observed, public_inputs, opset=opset)
            else:
                model = to_onnx(
                    observed,
                    public_inputs,
                    opset=opset,
                    enable_double_precision=route == "legacy-true",
                )
        except Exception as exc:
            category = FailureCategory.EXPORT
            if tracing:
                category = FailureCategory.TRACE
            elif isinstance(exc, RepresentationLimit):
                category = FailureCategory.REPRESENTATION
            elif isinstance(exc, PrototypeLimit):
                category = FailureCategory.PROTOTYPE
            raise MigrationError(category, str(exc), report(None)) from exc
    return ExportResult(model, report(model))


def compact_report(report: CallReport) -> str:
    """Types and warnings from this export; no runtime support claim."""
    provided = ", ".join(
        f"{spec.kind}:{spec.shape}:{spec.dtype or 'unspecified'}"
        f":weak_type={spec.weak_type if spec.weak_type is not None else 'unspecified'}"
        for spec in report.input_specs
    )
    lines = [
        f"{report.route}; x64 ambient={report.ambient_x64}, "
        f"trace={report.tracing_x64}, after={report.after_x64}",
        f"  supplied=[{provided}]; source-normalized={report.source_input_dtypes}",
        f"  source-normalized weak_type={report.source_input_weak_types}",
        f"  trace {report.traced_inputs} -> {report.traced_outputs}; "
        f"ONNX {report.model_inputs} -> {report.model_outputs}",
        f"  trace input weak_type={report.traced_input_weak_types}",
    ]
    lines.extend(
        f"  {warning.category}: {' '.join(warning.message.split())}"
        for warning in report.warnings
    )
    return "\n".join(lines)


def _run_cpu(model: onnx.ModelProto, value: np.ndarray) -> np.ndarray:
    # Runtime execution is explicitly separate from the calling-convention PoC.
    import onnxruntime as ort  # type: ignore[import-untyped]

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    actual: np.ndarray = session.run(None, {session.get_inputs()[0].name: value})[0]
    return actual


def main() -> None:
    """Print a small migration comparison; write no model artifacts."""

    def exact_region(x: jax.Array) -> jax.Array:
        y = x.astype(jnp.float32)
        return (y + 1) - y

    previous = bool(jax.config.x64_enabled)
    print(f"Internal UX candidate; JAX {jax.__version__}; shape-only source => FP32")
    try:
        for ambient in (False, True):
            jax.config.update("jax_enable_x64", ambient)
            for label, supplied in (
                ("shape-only", [(1,)]),
                ("typed FP64", [jax.ShapeDtypeStruct((1,), np.float64)]),
                ("concrete FP32", [np.asarray([1e8], dtype=np.float32)]),
            ):
                for options in (
                    {},
                    {"enable_double_precision": False},
                    {"enable_double_precision": True},
                    {"precision": "source"},
                ):
                    print(f"\n{label}; options={options}")
                    try:
                        result = export_candidate(exact_region, supplied, **options)
                    except MigrationError as exc:
                        print(f"  {exc.category.value}: {exc}")
                        if exc.report is not None:
                            print(compact_report(exc.report))
                    else:
                        print(compact_report(result.report))
        jax.config.update("jax_enable_x64", True)
        spec = jax.ShapeDtypeStruct((1,), np.float64)
        x: np.ndarray = np.asarray([1e8], dtype=np.float64)
        print("\nSeparate CPU sample; ORT_DISABLE_ALL; one thread:")
        for options in ({"enable_double_precision": True}, {"precision": "source"}):
            result = export_candidate(exact_region, [spec], **options)
            actual = _run_cpu(result.model, x)
            print(f"  {result.report.route}: {actual.tolist()} {actual.dtype}")
        jax.config.update("jax_enable_x64", False)
        print("\nFP32 input with a model-requested FP64 cast under disabled x64:")
        result = export_candidate(
            lambda value: value.astype(jnp.float64),
            [jax.ShapeDtypeStruct((1,), np.float32)],
            precision="source",
        )
        print(compact_report(result.report))
    finally:
        jax.config.update("jax_enable_x64", previous)


if __name__ == "__main__":
    main()
