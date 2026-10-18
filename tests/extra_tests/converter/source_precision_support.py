# tests/extra_tests/converter/source_precision_support.py

"""Independent original-JAX preparation for the private SOURCE migration lane.

Raw metadata identifies existing cases, domains, factories and graph checks. This
helper does not use the legacy generated parameter variants or allclose adapters.
Only the audited initial profile can receive an audited export result here; other
functionality stays explicit migration work. Runtime acceptance is sample-only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
from numbers import Integral
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir

from jax2onnx import user_interface
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.precision import (
    CheckStatus,
    PrecisionCheck,
    SampleCriteria,
    SourceSample,
    SourceValidationReport,
    TensorProperties,
    validate_source_samples,
    get_precision_report,
)


class FactoryDtypePolicy(Enum):
    PRESERVE = "preserve-original-factory"


PRESERVE_FACTORY_DTYPE = FactoryDtypePolicy.PRESERVE


@dataclass(frozen=True)
class SourceProvenance:
    origin: str
    ordinal: int

    def __post_init__(self) -> None:
        if not self.origin or self.ordinal < 0:
            raise ValueError(
                "Source provenance needs a registration and nonnegative ordinal"
            )


@dataclass(frozen=True)
class SourceScenario:
    name: str
    caller_x64_enabled: bool
    model_dtype: Any = PRESERVE_FACTORY_DTYPE
    input_specs: tuple[jax.ShapeDtypeStruct, ...] | None = None
    input_values: tuple[Any, ...] | None = None
    concrete_shapes: tuple[tuple[int, ...], ...] | None = None
    default_input_dtype: Any = np.float32
    default_matmul_precision: str | None = None
    criteria: SampleCriteria | None = None
    provenance: SourceProvenance | None = None
    require_reviewed_criteria: bool = False

    @property
    def cell_identity(self) -> tuple[str, int, str] | None:
        if self.provenance is None:
            return None
        return (self.provenance.origin, self.provenance.ordinal, self.name)


@dataclass(frozen=True)
class PreparedSourceCase:
    identity: tuple[str, str, str]
    scenario: SourceScenario
    fn: Callable[..., Any]
    metadata: Mapping[str, Any]
    input_specs: tuple[jax.ShapeDtypeStruct, ...]
    arguments: tuple[jax.Array, ...]
    traced_inputs: tuple[TensorProperties, ...]
    traced_outputs: tuple[TensorProperties, ...]
    reference_outputs: tuple[np.ndarray, ...]
    output_tree: str
    criteria: SampleCriteria | None


@dataclass(frozen=True)
class SourceCaseResult:
    stage: str
    category: Literal[
        "audited-profile",
        "migration-gap",
        "representation-limit",
        "implementation-gap",
        "runtime-evidence",
    ]
    audit: PrecisionCheck
    prepared: PreparedSourceCase | None = None
    model: onnx.ModelProto | None = None
    validation: SourceValidationReport | None = None

    @property
    def numerical_status(self) -> CheckStatus:
        if self.validation is None:
            return CheckStatus.UNVERIFIED
        checks = self.validation.checks + tuple(
            check for sample in self.validation.samples for check in sample.checks
        )
        numerical = [
            check.status for check in checks if check.category == "numerical-acceptance"
        ]
        if CheckStatus.FAILED in numerical:
            return CheckStatus.FAILED
        return (
            CheckStatus.VALIDATED
            if numerical
            and all(status is CheckStatus.VALIDATED for status in numerical)
            else CheckStatus.UNVERIFIED
        )


class SourcePreparationError(ValueError):
    """The declared original-source inputs/reference could not be prepared faithfully."""


@contextmanager
def source_configuration(scenario: SourceScenario) -> Iterator[None]:
    previous_x64 = bool(jax.config.x64_enabled)
    previous_precision = jax.config.jax_default_matmul_precision
    try:
        jax.config.update("jax_enable_x64", scenario.caller_x64_enabled)
        jax.config.update(
            "jax_default_matmul_precision", scenario.default_matmul_precision
        )
        yield
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        jax.config.update("jax_default_matmul_precision", previous_precision)


def is_source_float(dtype: Any) -> bool:
    return np.dtype(dtype).name in {"float16", "bfloat16", "float32", "float64"}


def _properties(aval: Any) -> TensorProperties:
    return TensorProperties(
        None,
        tuple(
            int(dim) if isinstance(dim, Integral) else str(dim) for dim in aval.shape
        ),
        np.dtype(aval.dtype).name,
        bool(getattr(aval, "weak_type", False)),
    )


def _criteria(
    metadata: Mapping[str, Any], scenario: SourceScenario
) -> SampleCriteria | None:
    if scenario.criteria is not None:
        return scenario.criteria
    if scenario.require_reviewed_criteria:
        return None
    if "rtol" in metadata or "atol" in metadata:
        if "rtol" not in metadata or "atol" not in metadata:
            raise SourcePreparationError(
                "Source tolerances need explicit rtol and atol; select a reviewed criterion for this case"
            )
        return SampleCriteria(
            "tolerant", rtol=float(metadata["rtol"]), atol=float(metadata["atol"])
        )
    # Legacy f32/f64 harness defaults and transformed expected dtypes are not
    # source criteria. A caller can explicitly select a reviewed alternative.
    return SampleCriteria("exact")


def _typed_input_spec(value: Any) -> jax.ShapeDtypeStruct | None:
    """Read supplied typed properties before JAX can canonicalize their dtype."""
    if not isinstance(value, (np.ndarray, np.generic, jax.Array)):
        return None
    return jax.ShapeDtypeStruct(
        tuple(value.shape),
        np.dtype(value.dtype),
        weak_type=bool(getattr(value, "weak_type", False)),
    )


def raw_source_scenarios(
    metadata: Mapping[str, Any],
    provenance: SourceProvenance,
    *,
    criteria: SampleCriteria | None = None,
) -> tuple[SourceScenario, SourceScenario]:
    """Freeze one original input recipe for both raw caller configurations.

    This does not construct models, trace JAX, or infer a numerical criterion.
    Shape-only metadata uses a strong FP32 input unless it explicitly declares
    a dtype. Untyped values without shapes need a separately reviewed recipe.
    """
    supplied = metadata.get("input_values")
    shapes = metadata.get("input_shapes")
    dtypes = metadata.get("input_dtypes")
    if shapes is not None:
        if dtypes is not None and len(shapes) != len(dtypes):
            raise SourcePreparationError("input_shapes and input_dtypes lengths differ")
        specs = tuple(
            jax.ShapeDtypeStruct(
                tuple(shape), np.float32 if dtypes is None else dtypes[index]
            )
            for index, shape in enumerate(shapes)
        )
    elif supplied is not None:
        typed_specs = tuple(_typed_input_spec(value) for value in supplied)
        if any(spec is None for spec in typed_specs):
            raise SourcePreparationError(
                "Untyped raw values without declared shapes need a reviewed fixed input recipe"
            )
        specs = tuple(spec for spec in typed_specs if spec is not None)
    else:
        raise SourcePreparationError("Raw metadata must declare inputs, including []")

    def snapshot(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            copied = value.copy()
            copied.setflags(write=False)
            return copied
        if isinstance(value, np.generic):
            return value.copy()
        if (
            isinstance(value, (jax.Array, bool, int, float, complex, str))
            or value is None
        ):
            return value
        if isinstance(value, (list, tuple)):
            return tuple(snapshot(item) for item in value)
        raise SourcePreparationError("Raw input value needs a reviewed snapshot recipe")

    values = None if supplied is None else tuple(snapshot(value) for value in supplied)
    scenarios = tuple(
        SourceScenario(
            name,
            enabled,
            input_specs=specs,
            input_values=values,
            criteria=criteria,
            provenance=provenance,
            require_reviewed_criteria=True,
        )
        for name, enabled in (("raw-x64-off", False), ("raw-x64-on", True))
    )
    return scenarios[0], scenarios[1]


def _require_preserved_dtype(dtype: Any) -> None:
    declared = np.dtype(dtype)
    canonical = np.dtype(jax.dtypes.canonicalize_dtype(declared))
    if canonical != declared:
        raise SourcePreparationError(
            f"Source input dtype {declared} would canonicalize to {canonical} under the declared caller JAX configuration; no implicit feed cast is allowed"
        )


def _argument(value: Any, spec: jax.ShapeDtypeStruct) -> jax.Array:
    supplied = _typed_input_spec(value)
    typed = supplied is not None
    if supplied is not None:
        if np.dtype(supplied.dtype) != np.dtype(spec.dtype):
            raise SourcePreparationError(
                f"Supplied typed input dtype {supplied.dtype} differs from declared {spec.dtype}; no implicit feed cast is allowed"
            )
        _require_preserved_dtype(supplied.dtype)
    _require_preserved_dtype(spec.dtype)
    if spec.weak_type:
        if typed and not bool(getattr(value, "weak_type", False)):
            array = np.asarray(value)
            if array.shape:
                raise SourcePreparationError(
                    "Weak source arguments need a supplied weak array or a scalar literal"
                )
            value = array.item()
        argument = jnp.asarray(value)
    else:
        # A typed value must already have the declared dtype; only untyped
        # Python literals receive the explicitly declared source input dtype.
        argument = jnp.asarray(value) if typed else jnp.asarray(value, dtype=spec.dtype)
    if np.dtype(argument.dtype) != np.dtype(spec.dtype):
        raise SourcePreparationError(
            f"Source input dtype {argument.dtype} differs from declared {spec.dtype}; no implicit feed cast is allowed"
        )
    if bool(argument.weak_type) != bool(spec.weak_type):
        raise SourcePreparationError(
            "Source input weak_type differs from its declaration"
        )
    return argument


def _input_data(
    metadata: Mapping[str, Any],
    scenario: SourceScenario,
    identity: tuple[str, str, str],
) -> tuple[tuple[jax.ShapeDtypeStruct, ...], tuple[jax.Array, ...]]:
    supplied = (
        scenario.input_values
        if scenario.input_values is not None
        else metadata.get("input_values")
    )
    specs = scenario.input_specs
    if specs is None:
        shapes = metadata.get("input_shapes")
        dtypes = metadata.get("input_dtypes")
        if shapes is not None:
            if dtypes is not None and len(shapes) != len(dtypes):
                raise SourcePreparationError(
                    "input_shapes and input_dtypes lengths differ"
                )
            specs = tuple(
                jax.ShapeDtypeStruct(
                    tuple(shape),
                    (
                        dtypes[index]
                        if dtypes is not None
                        else scenario.default_input_dtype
                    ),
                )
                for index, shape in enumerate(shapes)
            )
        elif supplied is not None:
            inferred = []
            for value in supplied:
                typed_spec = _typed_input_spec(value)
                if typed_spec is not None:
                    inferred.append(typed_spec)
                else:
                    # Only genuinely untyped literals use JAX's caller-defined
                    # defaults. Typed values never pass through this conversion.
                    argument = jnp.asarray(value)
                    inferred.append(
                        jax.ShapeDtypeStruct(
                            argument.shape,
                            argument.dtype,
                            weak_type=bool(argument.weak_type),
                        )
                    )
            specs = tuple(inferred)
        else:
            raise SourcePreparationError(
                "Raw metadata or scenario must declare source inputs, including [] for zero-input cases"
            )
    if scenario.concrete_shapes is not None and len(scenario.concrete_shapes) != len(
        specs
    ):
        raise SourcePreparationError(
            "Concrete sample shape count differs from source inputs"
        )
    for spec in specs:
        _require_preserved_dtype(spec.dtype)
    normalized = tuple(
        conversion_api._as_sds_list(
            specs, False, precision_policy=PrecisionPolicy.SOURCE
        )
    )
    if supplied is None:
        shapes = scenario.concrete_shapes
        if shapes is None:
            if any(
                not all(isinstance(dim, Integral) for dim in spec.shape)
                for spec in specs
            ):
                raise SourcePreparationError(
                    "Symbolic source inputs need explicit concrete sample shapes"
                )
            shapes = tuple(tuple(int(dim) for dim in spec.shape) for spec in specs)
        if len(shapes) != len(specs):
            raise SourcePreparationError(
                "Concrete sample shape count differs from source inputs"
            )
        seed_identity = (
            f"{scenario.provenance.origin}\0{scenario.provenance.ordinal}"
            if scenario.provenance is not None
            else "|".join((*identity, scenario.name))
        )
        seed = hashlib.sha256(seed_identity.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        generated = []
        for shape, spec in zip(shapes, specs, strict=True):
            dtype = np.dtype(spec.dtype)
            if is_source_float(dtype):
                raw = rng.standard_normal(shape) * 0.25
            elif np.issubdtype(dtype, np.integer):
                raw = rng.integers(0, 5, size=shape)
            elif dtype == np.dtype(bool):
                raw = rng.random(shape) > 0.5
            else:
                raise SourcePreparationError(
                    f"No reviewed random input domain for {dtype}; supply explicit input_values"
                )
            generated.append(np.asarray(raw, dtype=dtype))
        supplied = tuple(generated)
    if len(supplied) != len(specs):
        raise SourcePreparationError("Source input value/spec counts differ")
    arguments = tuple(
        _argument(value, spec) for value, spec in zip(supplied, specs, strict=True)
    )
    for index, (argument, spec) in enumerate(zip(arguments, specs, strict=True)):
        concrete = (
            scenario.concrete_shapes[index]
            if scenario.concrete_shapes is not None
            else tuple(argument.shape)
        )
        if tuple(argument.shape) != tuple(concrete) or len(argument.shape) != len(
            spec.shape
        ):
            raise SourcePreparationError(
                "Source input sample shape differs from its declaration"
            )
        for actual, declared in zip(argument.shape, spec.shape, strict=True):
            if isinstance(declared, Integral) and actual != declared:
                raise SourcePreparationError(
                    "Source input sample changed a static dimension"
                )
    return normalized, arguments


def raw_source_case(origin: str, ordinal: int) -> dict[str, Any]:
    """Resolve an inventory provenance ID directly, without generated variants."""
    from scripts.source_precision_inventory import raw_registrations

    registrations, _ = raw_registrations()
    matches = [metadata for name, _, metadata in registrations if name == origin]
    if len(matches) != 1:
        raise SourcePreparationError(
            f"Expected one raw registration for {origin}, found {len(matches)}"
        )
    metadata = matches[0]
    case = dict(metadata["testcases"][ordinal])
    case.update(
        context=metadata.get("context", "default"),
        component=metadata.get("component", "default"),
    )
    return case


def prepare_source_case(
    metadata: Mapping[str, Any], scenario: SourceScenario
) -> PreparedSourceCase:
    """Instantiate once, then independently trace/evaluate the original JAX callable."""
    identity = (
        str(metadata.get("context", "default")),
        str(metadata.get("component", "default")),
        str(metadata.get("testcase", "default")),
    )
    with source_configuration(scenario):
        # Reject contradictory source inputs before invoking a constructor or
        # original callable, so failed preparation cannot mutate model state.
        specs, arguments = _input_data(metadata, scenario, identity)
        target = metadata.get("callable")
        if target is not None and getattr(target, "__jax2onnx_factory__", False):
            recipe = (
                target
                if scenario.model_dtype is PRESERVE_FACTORY_DTYPE
                else target.with_dtype(scenario.model_dtype)
            )
            target = recipe.instantiate()
        if not callable(target):
            raise SourcePreparationError(
                "Raw metadata must supply a callable or recognized constructor recipe"
            )
        params = dict(metadata.get("input_params") or {})

        def original(*values: Any) -> Any:
            return target(*values, **params)

        closed = jax.make_jaxpr(original)(*specs)
        reference = jax.device_get(original(*arguments))
        outputs, tree = jax.tree_util.tree_flatten(reference)
        snapshots = []
        for value in outputs:
            array = np.array(value, copy=True)
            array.setflags(write=False)
            snapshots.append(array)
        traced_outputs = tuple(_properties(var.aval) for var in closed.jaxpr.outvars)
        if len(snapshots) != len(traced_outputs):
            raise SourcePreparationError(
                "Original JAX trace/evaluation output arities differ"
            )
        for value, properties in zip(snapshots, traced_outputs, strict=True):
            if (
                all(isinstance(dim, int) for dim in properties.shape)
                and value.shape != properties.shape
            ):
                raise SourcePreparationError(
                    "Original JAX trace/evaluation output shapes differ"
                )
            if value.dtype.name != properties.dtype:
                raise SourcePreparationError(
                    "Original JAX trace/evaluation output dtypes differ"
                )
        return PreparedSourceCase(
            identity,
            scenario,
            target,
            dict(metadata),
            specs,
            arguments,
            tuple(_properties(var.aval) for var in closed.jaxpr.invars),
            traced_outputs,
            tuple(snapshots),
            str(tree),
            _criteria(metadata, scenario),
        )


def _export_prepared(prepared: PreparedSourceCase) -> onnx.ModelProto:
    model = user_interface._to_onnx(
        fn=prepared.fn,
        inputs=prepared.input_specs,
        model_name="source_migration_oracle",
        opset=int(prepared.metadata.get("opset_version", 23)),
        input_names=prepared.metadata.get("input_names"),
        output_names=prepared.metadata.get("output_names"),
        return_mode="proto",
        precision_policy=PrecisionPolicy.SOURCE,
    )
    if not isinstance(model, onnx.ModelProto):
        raise SourcePreparationError(
            "Shared SOURCE facade did not return its declared protobuf artifact"
        )
    report = get_precision_report(model)
    if report is None:
        raise SourcePreparationError(
            "Shared SOURCE facade did not attach its real export report"
        )
    for independent, observed in (
        (prepared.traced_inputs, tuple(item.traced for item in report.inputs)),
        (prepared.traced_outputs, report.traced_outputs),
    ):
        if tuple(
            (item.dtype, item.shape, item.weak_type) for item in independent
        ) != tuple((item.dtype, item.shape, item.weak_type) for item in observed):
            raise SourcePreparationError(
                "Shared export trace differs from the independent original-JAX signature"
            )
    onnx.checker.check_model(model, full_check=True)
    return model


def _check_exported_signature(
    prepared: PreparedSourceCase, model: onnx.ModelProto
) -> None:
    for label, traced, exported in (
        ("input", prepared.traced_inputs, model.graph.input),
        ("output", prepared.traced_outputs, model.graph.output),
    ):
        if len(traced) != len(exported):
            raise SourcePreparationError(
                f"Exported {label} count differs from the independent original trace"
            )
        for source, value in zip(traced, exported, strict=True):
            dtype = np.dtype(ir.DataType(value.type.tensor_type.elem_type).numpy()).name
            shape = tuple(
                (
                    int(dim.dim_value)
                    if dim.HasField("dim_value")
                    else dim.dim_param or None
                )
                for dim in value.type.tensor_type.shape.dim
            )
            if shape != source.shape:
                raise SourcePreparationError(
                    f"Exported {label} shape {shape} differs from original source {source.shape}"
                )
            if dtype != source.dtype:
                raise SourcePreparationError(
                    f"Exported {label} dtype {dtype} differs from original source {source.dtype}"
                )


def run_source_case(
    metadata: Mapping[str, Any],
    scenario: SourceScenario,
    *,
    optimization: Literal["disabled", "all"] | None = "disabled",
) -> SourceCaseResult:
    prepared = None
    try:
        prepared = prepare_source_case(metadata, scenario)
    except Exception as exc:
        return SourceCaseResult(
            "source-preparation",
            "migration-gap",
            PrecisionCheck("original-source", CheckStatus.FAILED, str(exc)),
        )
    unsupported = [
        name
        for name in (
            "input_params",
            "inputs_as_nchw",
            "outputs_as_nchw",
            "record_primitive_calls_file",
        )
        if prepared.metadata.get(name)
    ]
    if prepared.metadata.get("normalization_mode", "auto") != "auto":
        unsupported.append("normalization_mode")
    if prepared.metadata.get("export_mode", "standard") != "standard":
        unsupported.append("export_mode")
    if unsupported or any(
        not is_source_float(spec.dtype)
        or any(not isinstance(dim, Integral) for dim in spec.shape)
        for spec in prepared.input_specs
    ):
        return SourceCaseResult(
            "profile-admission",
            "migration-gap",
            PrecisionCheck(
                "source-audit",
                CheckStatus.UNVERIFIED,
                f"Outside the initial audited top-level input/options profile: {unsupported}",
            ),
            prepared,
        )
    try:
        with source_configuration(scenario):
            model = _export_prepared(prepared)
        _check_exported_signature(prepared, model)
        callback = metadata.get("post_check_onnx_graph")
        if callback is not None and not callback(model):
            raise AssertionError("Raw metadata structural callback failed")
    except SourcePrecisionError as exc:
        category: Literal[
            "representation-limit", "migration-gap", "implementation-gap"
        ] = (
            "representation-limit"
            if exc.category == "representation-limit"
            else (
                "migration-gap"
                if exc.category == "profile-exclusion"
                else "implementation-gap"
            )
        )
        return SourceCaseResult(
            exc.phase,
            category,
            PrecisionCheck(
                "source-audit",
                (
                    CheckStatus.UNVERIFIED
                    if category == "migration-gap"
                    else CheckStatus.FAILED
                ),
                str(exc),
            ),
            prepared,
        )
    except Exception as exc:
        return SourceCaseResult(
            "source-export",
            "implementation-gap",
            PrecisionCheck("source-audit", CheckStatus.FAILED, str(exc)),
            prepared,
        )
    skipped = bool(metadata.get("skip_numeric_validation")) or prepared.criteria is None
    samples = []
    if not skipped:
        assert prepared.criteria is not None
        samples.append(
            SourceSample(
                tuple(np.asarray(value) for value in prepared.arguments),
                prepared.reference_outputs,
                prepared.criteria,
            )
        )
    validation = validate_source_samples(
        model,
        samples=samples,
        optimization=None if skipped else optimization,
    )
    return SourceCaseResult(
        "artifact-checked" if skipped or optimization is None else "runtime-checked",
        "audited-profile" if skipped or optimization is None else "runtime-evidence",
        PrecisionCheck(
            "source-audit",
            CheckStatus.VALIDATED,
            "Initial audited profile and independent boundary types checked; this is not model-wide numerical acceptance.",
        ),
        prepared,
        model,
        validation,
    )
