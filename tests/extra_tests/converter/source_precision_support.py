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
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from numbers import Integral
from typing import Any, Final, Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx_ir as ir

from jax2onnx import user_interface
from jax2onnx.converter import conversion_api
from jax2onnx.converter.precision_policy import PrecisionPolicy
from jax2onnx.converter.source_profile import (
    SourcePrecisionError,
    validate_source_jaxpr,
)
from jax2onnx.converter.source_representation import (
    representation_for_aval,
    source_shape_contract,
    validate_source_shapes,
)
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

if TYPE_CHECKING:
    from tests.extra_tests.converter.source_precision_corpus_oracles import (
        CorpusStructureEvidence,
        DotCounterpart,
        GatherCounterpart,
        LinearCounterpart,
    )


class FactoryDtypePolicy(Enum):
    PRESERVE = "preserve-original-factory"


PRESERVE_FACTORY_DTYPE: Final[FactoryDtypePolicy] = FactoryDtypePolicy.PRESERVE


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
    input_params: Mapping[str, Any] | None = None
    parameter_values: Mapping[str, Any] | None = None

    @property
    def cell_identity(self) -> tuple[str, int, str] | None:
        if self.provenance is None:
            return None
        return (self.provenance.origin, self.provenance.ordinal, self.name)


@dataclass(frozen=True)
class SourceSettings:
    x64_enabled: bool
    default_matmul_precision: str | None
    jax_platforms: str | None
    numpy_dtype_promotion: str
    numpy_rank_promotion: str

    @classmethod
    def observe(cls) -> SourceSettings:
        return cls(
            bool(jax.config.x64_enabled),
            jax.config.jax_default_matmul_precision,
            jax.config.jax_platforms,
            jax.config.jax_numpy_dtype_promotion,
            jax.config.jax_numpy_rank_promotion,
        )


@dataclass(frozen=True)
class SourceValueEvidence:
    logical: TensorProperties
    logical_sha256: str
    encoding: str
    physical_dtype: str
    physical_shape: tuple[object, ...]
    physical_sha256: str


def value_evidence(value: Any, properties: TensorProperties) -> SourceValueEvidence:
    array = np.asarray(value)
    if array.dtype.name != properties.dtype:
        raise SourcePreparationError("Evidence array dtype differs from original trace")
    representation = representation_for_aval(
        jax.ShapeDtypeStruct(array.shape, array.dtype)
    )
    physical = representation.encode_array(array)

    def digest(payload: np.ndarray) -> str:
        header = json.dumps(
            [payload.dtype.str, payload.shape], separators=(",", ":")
        ).encode()
        return hashlib.sha256(header + b"\0" + payload.tobytes(order="C")).hexdigest()

    return SourceValueEvidence(
        properties,
        digest(array),
        representation.encoding,
        representation.physical_dtype.name,
        representation.physical_shape,
        digest(physical),
    )


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
    settings: SourceSettings
    original_trace_json: str
    input_evidence: tuple[SourceValueEvidence, ...]
    reference_evidence: tuple[SourceValueEvidence, ...]
    original_jaxpr: Any = field(repr=False, compare=False)
    named_specs: Mapping[str, jax.ShapeDtypeStruct] = field(default_factory=dict)
    named_arguments: Mapping[str, jax.Array] = field(default_factory=dict)

    @property
    def all_input_specs(self) -> tuple[jax.ShapeDtypeStruct, ...]:
        return (*self.input_specs, *self.named_specs.values())

    @property
    def all_arguments(self) -> tuple[jax.Array, ...]:
        return (*self.arguments, *self.named_arguments.values())


@dataclass(frozen=True)
class SourceCaseResult:
    stage: str
    category: Literal[
        "audited-profile",
        "migration-gap",
        "representation-limit",
        "implementation-gap",
        "runtime-evidence",
        "audit-pending",
    ]
    audit: PrecisionCheck
    prepared: PreparedSourceCase | None = None
    model: onnx.ModelProto | None = None
    validation: SourceValidationReport | None = None
    export_settings: SourceSettings | None = None
    schema: CheckStatus = CheckStatus.UNVERIFIED
    corpus_structure: CorpusStructureEvidence | None = None
    raw_callback_status: str = "not-executed"

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

    def __init__(self, message: str, *, category: str = "migration-gap") -> None:
        self.category = category
        super().__init__(message)


def _global_source_configuration() -> dict[str, Any]:
    # Effective values can be shadowed by caller contexts. Snapshot the global
    # state separately so restoring it cannot overwrite the caller's state.
    return {
        "jax_enable_x64": jax.enable_x64.get_global(),
        "jax_default_matmul_precision": jax.default_matmul_precision.get_global(),
        "jax_platforms": jax.config.jax_platforms,
        "jax_numpy_dtype_promotion": jax.numpy_dtype_promotion.get_global(),
        "jax_numpy_rank_promotion": jax.numpy_rank_promotion.get_global(),
    }


@contextmanager
def source_configuration(scenario: SourceScenario) -> Iterator[None]:
    previous = _global_source_configuration()
    try:
        with (
            jax.enable_x64(scenario.caller_x64_enabled),
            jax.default_matmul_precision(scenario.default_matmul_precision),
        ):
            settings = SourceSettings.observe()
            yield
            if (
                SourceSettings.observe() != settings
                or _global_source_configuration() != previous
            ):
                raise SourcePreparationError(
                    "Source callable changed caller JAX settings"
                )
    finally:
        for name, value in previous.items():
            jax.config.update(name, value)


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
            f"Source input dtype {declared} would canonicalize to {canonical} under the declared caller JAX configuration; no implicit feed cast is allowed",
            category="configuration-conflict",
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


def _bind_source_shape(
    actual: tuple[int, ...],
    declared: tuple[object, ...],
    bindings: dict[str, int],
    *,
    bind: bool,
) -> tuple[int, ...]:
    shape, _ = source_shape_contract(declared)
    if len(actual) != len(shape):
        raise SourcePreparationError(
            "Source input/reference rank differs from its declaration"
        )
    concrete = []
    for extent, expected in zip(actual, shape, strict=True):
        if (
            not isinstance(extent, Integral)
            or isinstance(extent, (bool, np.bool_))
            or extent < 0
        ):
            raise SourcePreparationError(
                "Concrete sample extents must be nonnegative integers"
            )
        size = int(extent)
        if isinstance(expected, str):
            if size <= 0:
                raise SourcePreparationError("Source sample batch must be positive")
            if expected not in bindings and bind:
                bindings[expected] = size
            if bindings.get(expected) != size:
                raise SourcePreparationError(
                    "Source input/reference must match one shared batch binding"
                )
        elif size != expected:
            raise SourcePreparationError(
                "Source input/reference changed a static dimension"
            )
        concrete.append(size)
    return tuple(concrete)


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
    try:
        normalized = tuple(
            conversion_api._as_sds_list(
                specs, False, precision_policy=PrecisionPolicy.SOURCE
            )
        )
    except ValueError as exc:
        raise SourcePreparationError(str(exc)) from exc
    bindings: dict[str, int] = {}
    shapes = scenario.concrete_shapes
    if shapes is not None:
        shapes = tuple(
            _bind_source_shape(tuple(shape), tuple(spec.shape), bindings, bind=True)
            for shape, spec in zip(shapes, normalized, strict=True)
        )
    if supplied is None:
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
        _bind_source_shape(
            tuple(argument.shape), tuple(normalized[index].shape), bindings, bind=True
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
    metadata: Mapping[str, Any],
    scenario: SourceScenario,
    *,
    settings_observer: Callable[[SourceSettings], None] | None = None,
) -> PreparedSourceCase:
    """Instantiate once, then independently trace/evaluate the original JAX callable."""
    identity = (
        str(metadata.get("context", "default")),
        str(metadata.get("component", "default")),
        str(metadata.get("testcase", "default")),
    )
    with source_configuration(scenario):
        settings = SourceSettings.observe()
        if settings_observer is not None:
            settings_observer(settings)
        # Reject contradictory source inputs before invoking a constructor or
        # original callable, so failed preparation cannot mutate model state.
        specs, arguments = _input_data(metadata, scenario, identity)
        params = dict(
            metadata.get("input_params") or {}
            if scenario.input_params is None
            else scenario.input_params
        )
        names = tuple(params)
        declarations = [
            conversion_api._source_named_input_spec(
                name, value, opset=int(metadata.get("opset_version", 23))
            )
            for name, value in params.items()
        ]
        all_specs = tuple(
            conversion_api._as_sds_list(
                [*specs, *declarations], False, precision_policy=PrecisionPolicy.SOURCE
            )
        )
        positional_count = len(specs)
        specs = all_specs[:positional_count]
        named_specs = dict(zip(names, all_specs[positional_count:], strict=True))
        supplied_params = (
            params if scenario.parameter_values is None else scenario.parameter_values
        )
        if tuple(supplied_params) != names:
            raise SourcePreparationError(
                "Named sample keys/order differ from runtime declarations"
            )
        named_arguments = {
            name: _argument(supplied_params[name], spec)
            for name, spec in named_specs.items()
        }
        all_arguments = (*arguments, *named_arguments.values())

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

        def original(*values: Any) -> Any:
            return target(
                *values[:positional_count],
                **dict(zip(names, values[positional_count:], strict=True)),
            )

        closed = jax.make_jaxpr(original)(*all_specs)
        validate_source_shapes(
            [tuple(spec.shape) for spec in all_specs]
            + [tuple(var.aval.shape) for var in closed.jaxpr.outvars]
        )
        bindings: dict[str, int] = {}
        for spec, argument in zip(all_specs, all_arguments, strict=True):
            _bind_source_shape(
                tuple(argument.shape), tuple(spec.shape), bindings, bind=True
            )
        reference = jax.device_get(original(*all_arguments))
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
            _bind_source_shape(value.shape, properties.shape, bindings, bind=False)
            if value.dtype.name != properties.dtype:
                raise SourcePreparationError(
                    "Original JAX trace/evaluation output dtypes differ"
                )
        from tests.extra_tests.converter.source_precision_diagnostics import (
            typed_trace_manifest,
        )

        traced_inputs = tuple(_properties(var.aval) for var in closed.jaxpr.invars)
        if SourceSettings.observe() != settings:
            raise SourcePreparationError(
                "Original callable changed caller JAX settings during preparation"
            )
        return PreparedSourceCase(
            identity,
            scenario,
            target,
            dict(metadata),
            specs,
            arguments,
            traced_inputs,
            traced_outputs,
            tuple(snapshots),
            str(tree),
            _criteria(metadata, scenario),
            settings,
            json.dumps(
                typed_trace_manifest(closed), sort_keys=True, separators=(",", ":")
            ),
            tuple(
                value_evidence(value, properties)
                for value, properties in zip(all_arguments, traced_inputs, strict=True)
            ),
            tuple(
                value_evidence(value, properties)
                for value, properties in zip(snapshots, traced_outputs, strict=True)
            ),
            closed,
            named_specs,
            named_arguments,
        )


def _export_prepared(
    prepared: PreparedSourceCase,
    *,
    diagnostic_jaxpr_validator: Callable[[Any, int], None] | None = None,
) -> onnx.ModelProto:
    model = user_interface._to_onnx(
        fn=prepared.fn,
        inputs=prepared.input_specs,
        input_params=prepared.named_specs,
        model_name="source_migration_oracle",
        opset=int(prepared.metadata.get("opset_version", 23)),
        input_names=prepared.metadata.get("input_names"),
        output_names=prepared.metadata.get("output_names"),
        return_mode="proto",
        precision_policy=PrecisionPolicy.SOURCE,
        diagnostic_jaxpr_validator=diagnostic_jaxpr_validator,
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
    expected_profile = (
        "source-static-v1"
        if diagnostic_jaxpr_validator is None
        else "source-diagnostic-v1"
    )
    if report.profile != expected_profile:
        raise SourcePreparationError(
            "Shared facade returned a different precision profile"
        )
    if (
        report.caller_x64_enabled != prepared.settings.x64_enabled
        or report.caller_default_matmul_precision
        != prepared.settings.default_matmul_precision
        or report.caller_numpy_dtype_promotion
        != prepared.settings.numpy_dtype_promotion
        or report.caller_numpy_rank_promotion != prepared.settings.numpy_rank_promotion
    ):
        raise SourcePreparationError(
            "Actual export caller settings differ from prepared original settings"
        )
    if diagnostic_jaxpr_validator is not None and any(
        check.category == "lowering" and check.status is not CheckStatus.UNVERIFIED
        for check in report.checks
    ):
        raise SourcePreparationError(
            "A diagnostic artifact cannot validate its lowering audit"
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
            representation = representation_for_aval(
                jax.ShapeDtypeStruct(source.shape, source.dtype)
            )
            dtype = np.dtype(ir.DataType(value.type.tensor_type.elem_type).numpy()).name
            shape = tuple(
                (
                    int(dim.dim_value)
                    if dim.HasField("dim_value")
                    else dim.dim_param or None
                )
                for dim in value.type.tensor_type.shape.dim
            )
            if shape != representation.physical_shape:
                raise SourcePreparationError(
                    f"Exported {label} shape {shape} differs from original source {source.shape}"
                )
            if dtype != representation.physical_dtype.name:
                raise SourcePreparationError(
                    f"Exported {label} dtype {dtype} differs from original source {source.dtype}"
                )


def run_source_case(
    metadata: Mapping[str, Any],
    scenario: SourceScenario,
    *,
    optimization: Literal["disabled", "all"] | None = "disabled",
    diagnostic_jaxpr_validator: Callable[[Any, int], None] | None = None,
    _prepared: PreparedSourceCase | None = None,
    _corpus_counterpart: (
        DotCounterpart | LinearCounterpart | GatherCounterpart | None
    ) = None,
) -> SourceCaseResult:
    prepared = _prepared
    try:
        if prepared is None:
            prepared = prepare_source_case(metadata, scenario)
        elif prepared.scenario != scenario:
            raise SourcePreparationError(
                "Prepared evidence belongs to another scenario"
            )
        body = getattr(prepared.original_jaxpr, "jaxpr", prepared.original_jaxpr)
        if tuple(prepared.named_specs) != tuple(prepared.named_arguments):
            raise SourcePreparationError("Prepared named bindings changed names/order")
        actual_params = (
            metadata.get("input_params") or {}
            if scenario.input_params is None
            else scenario.input_params
        )
        if tuple(actual_params) != tuple(prepared.named_specs):
            raise SourcePreparationError("Prepared named bindings differ from metadata")
        supplied_params = (
            actual_params
            if scenario.parameter_values is None
            else scenario.parameter_values
        )
        if tuple(supplied_params) != tuple(prepared.named_specs):
            raise SourcePreparationError(
                "Named sample keys/order differ from declarations"
            )
        with source_configuration(scenario):
            current_specs = [
                conversion_api._source_named_input_spec(
                    name, value, opset=int(metadata.get("opset_version", 23))
                )
                for name, value in actual_params.items()
            ]
            validate_source_shapes(
                [spec.shape for spec in (*prepared.all_input_specs, *current_specs)]
            )
            for index, (name, spec) in enumerate(prepared.named_specs.items()):
                if _properties(current_specs[index]) != _properties(spec):
                    raise SourcePreparationError(
                        "Named declaration differs from original binding"
                    )
                argument = _argument(supplied_params[name], spec)
                properties = prepared.traced_inputs[len(prepared.input_specs) + index]
                if (
                    value_evidence(argument, properties)
                    != prepared.input_evidence[len(prepared.input_specs) + index]
                ):
                    raise SourcePreparationError(
                        "Named metadata payload differs from original evidence"
                    )
        if _prepared is not None:
            if not (
                len(prepared.all_input_specs)
                == len(prepared.all_arguments)
                == len(prepared.traced_inputs)
                == len(body.invars)
                and len(prepared.traced_outputs)
                == len(prepared.reference_outputs)
                == len(body.outvars)
            ):
                raise SourcePreparationError(
                    "Prepared input/reference counts differ from the original trace"
                )
            validate_source_shapes(
                [tuple(spec.shape) for spec in prepared.all_input_specs]
                + [
                    tuple(var.aval.shape)
                    for var in (*body.invars, *body.constvars, *body.outvars)
                ]
            )
            prepared_bindings: dict[str, int] = {}
            for spec, argument, properties, var in zip(
                prepared.all_input_specs,
                prepared.all_arguments,
                prepared.traced_inputs,
                body.invars,
                strict=True,
            ):
                if (
                    properties != _properties(spec)
                    or properties != _properties(var.aval)
                    or np.dtype(argument.dtype).name != properties.dtype
                    or bool(getattr(argument, "weak_type", False)) != spec.weak_type
                ):
                    raise SourcePreparationError(
                        "Prepared input dtype/weakness differs from its original declaration"
                    )
                _bind_source_shape(
                    tuple(argument.shape),
                    tuple(spec.shape),
                    prepared_bindings,
                    bind=True,
                )
            for properties, reference, var in zip(
                prepared.traced_outputs,
                prepared.reference_outputs,
                body.outvars,
                strict=True,
            ):
                array = np.asarray(reference)
                if (
                    properties != _properties(var.aval)
                    or array.dtype.name != properties.dtype
                ):
                    raise SourcePreparationError(
                        "Prepared reference dtype differs from the original trace"
                    )
                _bind_source_shape(
                    array.shape, properties.shape, prepared_bindings, bind=False
                )
        if len(prepared.all_arguments) != len(prepared.input_evidence):
            raise SourcePreparationError("Prepared input evidence arity changed")
        if (
            tuple(
                value_evidence(value, properties)
                for value, properties in zip(
                    prepared.all_arguments, prepared.traced_inputs, strict=True
                )
            )
            != prepared.input_evidence
        ):
            raise SourcePreparationError(
                "Prepared input payload differs from original evidence"
            )
        if (
            tuple(
                value_evidence(value, properties)
                for value, properties in zip(
                    prepared.reference_outputs, prepared.traced_outputs, strict=True
                )
            )
            != prepared.reference_evidence
        ):
            raise SourcePreparationError(
                "Prepared reference payload differs from original evidence"
            )
    except Exception as exc:
        return SourceCaseResult(
            "source-preparation",
            "migration-gap",
            PrecisionCheck("original-source", CheckStatus.FAILED, str(exc)),
        )
    unsupported = [
        name
        for name in (
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
    symbolic = any(
        any(not isinstance(dim, Integral) for dim in spec.shape)
        for spec in prepared.all_input_specs
    )
    symbolic_excluded = False
    if symbolic:
        try:
            validate_source_jaxpr(
                getattr(prepared.original_jaxpr, "jaxpr", prepared.original_jaxpr),
                int(prepared.metadata.get("opset_version", 23)),
            )
        except SourcePrecisionError as exc:
            # Keep excluded symbolic sources at the existing admission boundary.
            # Other defects retain the facade's normal owning error category.
            symbolic_excluded = exc.category == "profile-exclusion"
    if unsupported or symbolic_excluded:
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
    export_settings = None
    schema_status = CheckStatus.UNVERIFIED
    corpus_structure = None
    raw_callback_status = "not-executed"
    try:
        with source_configuration(scenario):
            export_settings = SourceSettings.observe()
            model = (
                _export_prepared(prepared)
                if diagnostic_jaxpr_validator is None
                else _export_prepared(
                    prepared, diagnostic_jaxpr_validator=diagnostic_jaxpr_validator
                )
            )
        schema_status = CheckStatus.VALIDATED
        _check_exported_signature(prepared, model)
        if _corpus_counterpart is None:
            callback = metadata.get("post_check_onnx_graph")
            raw_callback_status = "absent" if callback is None else "error"
            if callback is not None:
                raw_callback_status = "passed" if callback(model) else "failed"
                if raw_callback_status == "failed":
                    raise AssertionError("Raw metadata structural callback failed")
        else:
            from tests.extra_tests.converter.source_precision_corpus_oracles import (
                DotCounterpart,
                GatherCounterpart,
                LinearCounterpart,
            )

            if type(_corpus_counterpart) not in (
                DotCounterpart,
                LinearCounterpart,
                GatherCounterpart,
            ):
                raise SourcePreparationError(
                    "Corpus structure requires an actual reviewed DotCounterpart, LinearCounterpart or GatherCounterpart"
                )
            if type(_corpus_counterpart) is GatherCounterpart and (
                diagnostic_jaxpr_validator is None or optimization is not None
            ):
                raise SourcePreparationError(
                    "Gather corpus structure requires diagnostic export with optimization=None"
                )
            corpus_structure = _corpus_counterpart.check(prepared, model)
            raw_callback_status = corpus_structure.raw_callback_status
            if not corpus_structure.accepted:
                raise AssertionError(
                    "Reviewed corpus structure failed: "
                    + corpus_structure.source_detail
                    + "; "
                    + corpus_structure.raw_callback_detail
                )
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
            export_settings=export_settings,
            schema=schema_status,
            corpus_structure=corpus_structure,
            raw_callback_status=raw_callback_status,
        )
    except Exception as exc:
        if isinstance(
            exc, (onnx.checker.ValidationError, onnx.shape_inference.InferenceError)
        ):
            schema_status = CheckStatus.FAILED
        return SourceCaseResult(
            "source-export",
            "implementation-gap",
            PrecisionCheck("source-audit", CheckStatus.FAILED, str(exc)),
            prepared,
            export_settings=export_settings,
            schema=schema_status,
            corpus_structure=corpus_structure,
            raw_callback_status=raw_callback_status,
        )
    diagnostic = diagnostic_jaxpr_validator is not None
    skipped = (
        diagnostic
        or bool(metadata.get("skip_numeric_validation"))
        or prepared.criteria is None
    )
    samples = []
    if not skipped:
        assert prepared.criteria is not None
        bindings: dict[str, int] = {}
        for spec, value in zip(
            prepared.all_input_specs, prepared.all_arguments, strict=True
        ):
            _bind_source_shape(
                tuple(value.shape), tuple(spec.shape), bindings, bind=True
            )

        def encode(value: Any, properties: TensorProperties) -> np.ndarray:
            array = np.asarray(value)
            shape = _bind_source_shape(
                array.shape, properties.shape, bindings, bind=False
            )
            return representation_for_aval(
                replace(properties, shape=shape)
            ).encode_array(array)

        samples.append(
            SourceSample(
                {
                    port.name: encode(value, _properties(spec))
                    for port, spec, value in zip(
                        model.graph.input,
                        prepared.all_input_specs,
                        prepared.all_arguments,
                        strict=True,
                    )
                },
                tuple(
                    encode(value, properties)
                    for properties, value in zip(
                        prepared.traced_outputs, prepared.reference_outputs, strict=True
                    )
                ),
                prepared.criteria,
            )
        )
    validation = validate_source_samples(
        model,
        samples=samples,
        optimization=None if skipped else optimization,
    )
    return SourceCaseResult(
        (
            "diagnostic-artifact-checked"
            if diagnostic
            else (
                "artifact-checked"
                if skipped or optimization is None
                else "runtime-checked"
            )
        ),
        (
            "audit-pending"
            if diagnostic
            else (
                "audited-profile"
                if skipped or optimization is None
                else "runtime-evidence"
            )
        ),
        PrecisionCheck(
            "source-audit",
            CheckStatus.UNVERIFIED if diagnostic else CheckStatus.VALIDATED,
            (
                "Diagnostic structure/representation checked; lowering audit_pending."
                if diagnostic
                else "Initial audited profile and independent boundary types checked; this is not model-wide numerical acceptance."
            ),
        ),
        prepared,
        model,
        validation,
        export_settings,
        schema_status,
        corpus_structure,
        raw_callback_status,
    )
