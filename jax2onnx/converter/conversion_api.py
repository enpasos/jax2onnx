# jax2onnx/converter/conversion_api.py

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import logging
import os
import jax
import jax.numpy as jnp
from jax import export as jax_export
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
from onnx_ir.traversal import RecursiveGraphIterator

from jax2onnx.ir_utils import (
    ir_shape_from_dims,
    iter_ir_functions,
    maybe_numpy_dtype,
    numpy_dtype_to_ir,
)
from jax2onnx.plugins import plugin_system as ps2
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    apply_monkey_patches,
    import_all_plugins,
)
from jax2onnx.plugins.jax._autodiff_utils import backfill_missing_transpose_rules
from jax2onnx.plugins._ir_shapes import _as_ir_dim_label

from .ir_context import IRContext
from .ir_builder import _dtype_to_ir
from .ir_optimizations import optimize_graph
from .function_scope import FunctionRegistry
from .precision_policy import PrecisionPolicy, precision_policy_scope
from .source_profile import SourcePrecisionError, validate_source_input_dtype
from .source_representation import representation_for_aval, validate_source_shapes
from . import lowering_dispatch as _lowering_dispatch
from .typing_support import IRBuilderAPI, LoweringContextProtocol

from jax.extend import core as jcore_ext

ShapeDimSpec = Union[int, str]
ShapeTupleSpec = Tuple[ShapeDimSpec, ...]
InputSpec = Union[jax.ShapeDtypeStruct, ShapeTupleSpec]

_ORT_SAFE_IR_VERSION: int = 10
_LOGGER: logging.Logger = logging.getLogger(__name__)
_STRICT_OPTIMIZER_FAILURES_ENV: str = "JAX2ONNX_STRICT_OPTIMIZER_FAILURES"
_NHWC_TO_NCHW_PERM: tuple[int, int, int, int] = (0, 3, 1, 2)
_NCHW_TO_NHWC_PERM: tuple[int, int, int, int] = (0, 2, 3, 1)
_current_eqn_scope: Callable[..., object] = _lowering_dispatch.current_eqn_scope
_staged_lowering_metadata: Callable[..., object] = (
    _lowering_dispatch.staged_lowering_metadata
)


@dataclass(frozen=True)
class _SourceInputBinding:
    trace_index: int
    positional_index: int | None
    keyword_name: str | None
    spec: jax.ShapeDtypeStruct
    var: Any

    @property
    def internal_name(self) -> str:
        return f"in_{self.trace_index}"


@dataclass(frozen=True)
class _TraceResult:
    closed_jaxpr: Any
    jaxpr: Any
    sds_list: list[jax.ShapeDtypeStruct]
    frozen_params: dict[str, object]
    inputs_as_nchw: tuple[int, ...]
    outputs_as_nchw: tuple[int, ...]
    source_inputs: tuple[_SourceInputBinding, ...] = ()


def _source_named_input_spec(
    name: str, value: object, *, opset: int
) -> jax.ShapeDtypeStruct:
    if not isinstance(name, str) or not name:
        raise ValueError("SOURCE input_params names must be nonempty strings")
    if isinstance(value, (jax.ShapeDtypeStruct, jax.Array, np.ndarray, np.generic)):
        return jax.ShapeDtypeStruct(
            tuple(value.shape),
            value.dtype,
            weak_type=bool(getattr(value, "weak_type", False)),
        )
    if type(value) in (bool, int, float, complex):
        # Let JAX derive Python scalar dtype/weakness under the caller's settings.
        # NumPy conversion here would invent a strong platform-default dtype.
        return cast(jax.ShapeDtypeStruct, jax.eval_shape(lambda scalar: scalar, value))
    raise SourcePrecisionError(
        f"named parameter {name!r} requires a tensor or Python scalar declaration",
        phase="input",
        category="profile-exclusion",
        location=f"input_params[{name!r}]",
        opset=opset,
        requirement="a typed array/ShapeDtypeStruct or a native Python numeric scalar",
        action="declare a runtime tensor or bind intended fixed configuration with a closure or partial",
    )


def _source_input_values(
    trace: _TraceResult,
    values: Sequence[ir.Value],
    *,
    require_internal_names: bool = False,
) -> tuple[ir.Value, ...]:
    bindings = trace.source_inputs
    if len(bindings) != len(trace.jaxpr.invars) or len(values) != len(bindings):
        raise RuntimeError(
            "SOURCE finalized input arity differs from its real root bindings"
        )
    for index, (binding, var, value) in enumerate(
        zip(bindings, trace.jaxpr.invars, values, strict=True)
    ):
        if (
            binding.trace_index != index
            or binding.var is not var
            or (require_internal_names and value.name != binding.internal_name)
        ):
            raise RuntimeError(
                "SOURCE finalized input order differs from its real root bindings"
            )
    return tuple(values)


# ---------------------------
# Helpers
# ---------------------------


def _np_float_dtype(enable_double_precision: bool) -> np.dtype[Any]:
    return np.dtype(np.float64 if enable_double_precision else np.float32)


def _maybe_promote_float_array(
    arr: np.ndarray, enable_double_precision: bool
) -> np.ndarray:
    """Promote floating arrays to float64 when double precision is enabled."""
    if not enable_double_precision:
        return arr
    if not np.issubdtype(arr.dtype, np.floating):
        return arr
    if arr.dtype == np.float64:
        return arr
    return arr.astype(np.float64, copy=False)


def _to_ir_dtype_from_np(np_dtype: np.dtype) -> "ir.DataType":
    np_dtype = np.dtype(np_dtype)
    if np_dtype.name == "bfloat16":
        return ir.DataType.BFLOAT16
    if np.issubdtype(np_dtype, np.floating):
        return ir.DataType.DOUBLE if np_dtype == np.float64 else ir.DataType.FLOAT
    if np.issubdtype(np_dtype, np.integer):
        return numpy_dtype_to_ir(np_dtype, default=ir.DataType.INT64)
    if np_dtype == np.dtype(np.bool_):
        return numpy_dtype_to_ir(np_dtype)
    return numpy_dtype_to_ir(np_dtype, default=ir.DataType.FLOAT)


def _to_ir_shape(shape_tuple: Sequence[ShapeDimSpec]) -> "ir.Shape":
    return ir_shape_from_dims(
        shape_tuple,
        parse_integer_like=False,
    )


def _convert_ir_attr(name: str, value: object) -> Optional[Attr]:
    if isinstance(value, Attr):
        return value

    attr_value: object = value
    attr_type: Optional[AttributeType] = None
    if isinstance(value, (np.bool_, np.integer)):
        attr_value = int(value)
    elif isinstance(value, np.floating):
        attr_value = float(value)
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, (bool, np.bool_, int, np.integer)) for x in value):
            attr_value = [int(x) for x in value]
            attr_type = AttributeType.INTS
        elif all(isinstance(x, (float, np.floating)) for x in value):
            attr_value = [float(x) for x in value]
            attr_type = AttributeType.FLOATS
        elif all(isinstance(x, str) for x in value):
            attr_value = list(value)
            attr_type = AttributeType.STRINGS

    try:
        return ir.convenience.convert_attribute(name, cast(Any, attr_value), attr_type)
    except (TypeError, ValueError):
        pass

    try:
        tensor_value = (
            attr_value
            if hasattr(attr_value, "data_type")
            else ir.tensor(np.asarray(attr_value))
        )
        return ir.convenience.convert_attribute(
            name, cast(Any, tensor_value), AttributeType.TENSOR
        )
    except Exception:
        return None


def _as_sds_list(
    inputs: Sequence[InputSpec],
    enable_double_precision: bool,
    *,
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
) -> List["jax.ShapeDtypeStruct"]:
    """Normalize user 'inputs' to ShapeDtypeStructs for abstract tracing."""
    sds_list: List[jax.ShapeDtypeStruct] = []
    source = precision_policy is PrecisionPolicy.SOURCE
    native_scope = (
        validate_source_shapes(
            [
                tuple(spec.shape) if isinstance(spec, jax.ShapeDtypeStruct) else spec
                for spec in inputs
            ]
        )
        if source
        else None
    )

    # 1) gather string symbols
    symnames: list[str] = []
    for spec in inputs:
        dims_iter: Iterable[object]
        if isinstance(spec, jax.ShapeDtypeStruct):
            dims_iter = tuple(spec.shape)
        else:
            dims_iter = spec
        for dim in dims_iter:
            if isinstance(dim, str) and dim not in symnames:
                symnames.append(dim)

    # 2) create symbolic sizes
    name2sym: dict[str, object] = {}
    shared_scope = (
        native_scope
        if native_scope is not None
        else jax_export.SymbolicScope() if symnames else None
    )
    for n in symnames:
        syms = jax_export.symbolic_shape(n, scope=shared_scope)
        if not syms:
            raise ValueError(f"symbolic_shape('{n}') returned no dimensions")
        if len(syms) != 1:
            raise ValueError(
                f"symbolic_shape('{n}') produced {len(syms)} dims; expected 1"
            )
        name2sym[n] = syms[0]

    # 3) build SDS list
    for spec in inputs:
        if isinstance(spec, jax.ShapeDtypeStruct):
            dims_list: List[object] = []
            for dim in tuple(spec.shape):
                if isinstance(dim, str):
                    dims_list.append(name2sym[dim])
                elif isinstance(dim, (int, np.integer)):
                    dims_list.append(int(dim))
                else:
                    dims_list.append(dim)
            sds_list.append(
                jax.ShapeDtypeStruct(
                    tuple(dims_list),
                    spec.dtype,
                    weak_type=(
                        spec.weak_type
                        if precision_policy is PrecisionPolicy.SOURCE
                        else False
                    ),
                )
            )
            continue

        dims_tuple = tuple(
            name2sym[dim] if isinstance(dim, str) else int(dim) for dim in spec
        )
        dt = jnp.float64 if enable_double_precision else jnp.float32
        sds_list.append(jax.ShapeDtypeStruct(dims_tuple, dt))
    if source:
        validate_source_shapes([spec.shape for spec in sds_list])
    return sds_list


def _maybe_dtype(aval: Any) -> Optional[np.dtype[Any]]:
    return maybe_numpy_dtype(getattr(aval, "dtype", None))


def _validate_layout_indices(
    indices: Optional[Sequence[int]],
    *,
    kind: str,
    upper_bound: int,
) -> Tuple[int, ...]:
    if indices is None:
        return ()

    normalized: List[int] = []
    seen: set[int] = set()
    for raw_idx in indices:
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, (int, np.integer)):
            raise ValueError(f"{kind} entries must be integers; got {raw_idx!r}")
        idx = int(raw_idx)
        if idx < 0 or idx >= upper_bound:
            raise ValueError(
                f"{kind} index {idx} is out of range for {upper_bound} traced values"
            )
        if idx in seen:
            raise ValueError(f"{kind} indices must be unique; duplicate {idx} found")
        seen.add(idx)
        normalized.append(idx)
    return tuple(normalized)


def _log_nonfatal_stage_failure(stage: str, exc: BaseException) -> None:
    _LOGGER.warning("%s skipped after %s: %s", stage, type(exc).__name__, exc)
    _LOGGER.debug("Nonfatal export failure in %s", stage, exc_info=exc)


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _resolve_strict_optimizer_failures(
    strict_optimizer_failures: Optional[bool],
) -> bool:
    if strict_optimizer_failures is not None:
        return strict_optimizer_failures
    return _env_flag_enabled(_STRICT_OPTIMIZER_FAILURES_ENV)


def _optimize_graph_with_failure_policy(
    model: ir.Model,
    *,
    strict_optimizer_failures: Optional[bool],
    precision_policy: PrecisionPolicy,
) -> None:
    try:
        optimize_graph(model, precision_policy=precision_policy)
    except Exception as exc:
        if _resolve_strict_optimizer_failures(strict_optimizer_failures):
            raise
        _log_nonfatal_stage_failure("optimize_graph", exc)


# Deprecated compatibility alias for TYPE_CHECKING-only legacy plugin imports.
_IRBuildContext = LoweringContextProtocol


def _function_store_identifier(fn_ir: ir.Function) -> tuple[str, str, str]:
    return fn_ir.identifier()


def _attach_ir_functions(ir_model: ir.Model, ctx: IRContext) -> None:
    ir_funcs = list(ctx.ir_functions)
    if not ir_funcs:
        return

    functions_store = ir_model.functions
    for fn_ir in ir_funcs:
        functions_store[_function_store_identifier(fn_ir)] = fn_ir

    model_imports: Dict[str, int] = dict(ir_model.opset_imports or {})
    model_imports.setdefault("", int(ctx.builder.opset) or 23)
    for fn_ir in ir_funcs:
        dom = (fn_ir.domain or "").strip()
        if dom and dom not in model_imports:
            model_imports[dom] = 1
    ir_model.opset_imports.update(model_imports)


def _iter_graph_values(gr: ir.Graph) -> Iterable[ir.Value]:
    seen: set[int] = set()
    staged: list[ir.Value] = []

    def _queue(values: Iterable[object]) -> None:
        for val in values:
            if val is None:
                continue
            if not isinstance(val, ir.Value):
                continue
            vid = id(val)
            if vid in seen:
                continue
            seen.add(vid)
            staged.append(val)

    def _on_enter(graph_like: object) -> None:
        try:
            _queue(graph_like.inputs)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        try:
            _queue(graph_like.outputs)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass
        try:
            _queue(graph_like.initializers)  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    for node in RecursiveGraphIterator(gr, enter_graph=_on_enter):
        _queue(node.inputs)
        _queue(node.outputs)

    for value in staged:
        yield value


def _normalize_value_shape(val: ir.Value) -> None:
    shape_obj = val.shape
    if shape_obj is None:
        return
    dims_source: Tuple[object, ...] = tuple(shape_obj.dims)

    normalized_dims: List[int | ir.SymbolicDim | None] = []
    dirty = False
    for dim in dims_source:
        normalized_dim: int | ir.SymbolicDim | None
        if isinstance(dim, (int, np.integer)):
            normalized_dim = int(dim)
        elif isinstance(dim, ir.SymbolicDim):
            normalized_dim = dim
        else:
            label = _as_ir_dim_label(dim)
            if isinstance(label, int):
                normalized_dim = int(label)
            elif isinstance(label, str):
                normalized_dim = ir.SymbolicDim(label)
            else:
                normalized_dim = None
        if normalized_dim is not dim:
            dirty = True
        normalized_dims.append(normalized_dim)

    if dirty:
        val.shape = ir.Shape(tuple(normalized_dims))


def _finalize_model_value_shapes(model_proto: ir.Model) -> None:
    for value in _iter_graph_values(model_proto.graph):
        _normalize_value_shape(value)

    for fn in iter_ir_functions(model_proto.functions):
        try:
            fn_graph = fn.graph
        except AttributeError:
            continue
        for value in _iter_graph_values(fn_graph):
            _normalize_value_shape(value)


def _source_finalization_error(
    exc: Exception,
    *,
    stage: str,
    location: str,
    opset: int | None,
    primitive: str | None = None,
) -> SourcePrecisionError:
    return SourcePrecisionError(
        f"{stage} failed after {type(exc).__name__}: {exc}",
        phase="finalization",
        category="lowering-defect",
        location=location,
        primitive=primitive,
        opset=opset,
        requirement="required finalization and explicit node attributes must be applied successfully",
        action="fix the converter or plugin finalization failure before exporting this program",
    )


def _apply_ir_attr_overrides_to_graph(
    gr: ir.Graph,
    overrides: dict[str, dict[str, object]],
    *,
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
    location: str = "graph",
    opset: int | None = None,
) -> None:
    if not overrides:
        return
    name2node: Dict[str, ir.Node] = {
        node.name: node for node in gr.all_nodes() if node.name
    }

    for nm, kv in (overrides or {}).items():
        node = name2node.get(nm)
        if node is None or kv is None:
            continue
        for attr_name, attr_value in kv.items():
            try:
                attr_obj = _convert_ir_attr(attr_name, attr_value)
                if attr_obj is not None:
                    node.attributes[attr_name] = attr_obj
                elif precision_policy is PrecisionPolicy.SOURCE:
                    raise ValueError(
                        f"Explicit attribute {attr_name!r} could not be converted"
                    )
            except Exception as exc:
                if precision_policy is PrecisionPolicy.SOURCE:
                    raise _source_finalization_error(
                        exc,
                        stage="explicit attribute override",
                        location=f"{location}/node {nm}/attribute {attr_name}",
                        opset=opset,
                        primitive=node.op_type,
                    ) from exc
                raise


def _fix_concat_axis_in_graph(gr: ir.Graph) -> None:
    for node in gr.all_nodes():
        if node.op_type != "Concat":
            continue
        if "axis" in node.attributes:
            continue
        node.attributes["axis"] = ir.convenience.convert_attribute("axis", 0)


def _apply_late_ir_attr_overrides(ir_model: ir.Model, ctx: IRContext) -> None:
    graph_location = f"graph {ir_model.graph.name}"
    try:
        _apply_ir_attr_overrides_to_graph(
            ir_model.graph,
            ctx.attr_overrides,
            precision_policy=ctx.precision_policy,
            location=graph_location,
            opset=ctx.opset,
        )
        _fix_concat_axis_in_graph(ir_model.graph)
    except SourcePrecisionError:
        raise
    except Exception as exc:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            raise _source_finalization_error(
                exc,
                stage="graph postprocess",
                location=graph_location,
                opset=ctx.opset,
            ) from exc
        raise

    missing_overrides = object()
    for fn in iter_ir_functions(ir_model.functions):
        try:
            graph_obj = getattr(fn, "graph", None)
            if graph_obj is None:
                continue
            raw_overrides: object = missing_overrides
            if hasattr(fn, "_attr_overrides"):
                raw_overrides = object.__getattribute__(fn, "_attr_overrides")
            if raw_overrides is missing_overrides:
                fn_overrides = dict(ctx.attr_overrides or {})
            elif isinstance(raw_overrides, Mapping):
                fn_overrides = dict(
                    cast(Mapping[str, dict[str, object]], raw_overrides)
                )
            else:
                fn_overrides = {}
            _apply_ir_attr_overrides_to_graph(
                graph_obj,
                fn_overrides,
                precision_policy=ctx.precision_policy,
                location=f"function {fn.name}",
                opset=ctx.opset,
            )
            _fix_concat_axis_in_graph(graph_obj)
        except Exception as exc:
            fn_name = getattr(fn, "name", "<unnamed function>")
            if ctx.precision_policy is PrecisionPolicy.SOURCE:
                if isinstance(exc, SourcePrecisionError):
                    raise
                raise _source_finalization_error(
                    exc,
                    stage="function postprocess",
                    location=f"function {fn_name}",
                    opset=ctx.opset,
                ) from exc
            _log_nonfatal_stage_failure(f"function postprocess for {fn_name}", exc)


def _consume_onnx_function_hits() -> None:
    try:
        ps2._consume_onnx_function_hits()
    except AttributeError:
        pass
    except Exception:
        pass


def _build_and_finalize_ir_model(
    ctx: IRContext,
    *,
    model_name: str,
    protective_clone: bool,
    strict_optimizer_failures: Optional[bool] = None,
) -> ir.Model:
    ir_model = ctx.builder.to_ir_model(
        name=model_name,
        ir_version=_ORT_SAFE_IR_VERSION,
        protective_clone=protective_clone,
    )

    _attach_ir_functions(ir_model, ctx)

    _optimize_graph_with_failure_policy(
        ir_model,
        strict_optimizer_failures=strict_optimizer_failures,
        precision_policy=ctx.precision_policy,
    )

    _apply_late_ir_attr_overrides(ir_model, ctx)
    _consume_onnx_function_hits()

    try:
        _finalize_model_value_shapes(ir_model)
    except Exception as exc:
        if ctx.precision_policy is PrecisionPolicy.SOURCE:
            raise _source_finalization_error(
                exc,
                stage="shape finalization",
                location=f"model {model_name}",
                opset=ctx.opset,
            ) from exc
        _log_nonfatal_stage_failure("finalize_model_value_shapes", exc)

    return ir_model


@contextmanager
def _activate_plugin_worlds() -> Iterator[None]:
    # Ensure plugin registry is populated
    import_all_plugins()
    with ExitStack() as stack:
        # Legacy patches first
        stack.enter_context(apply_monkey_patches())
        # New-style leaf bindings
        leaf_prims: list[jcore_ext.Primitive] = []
        for plugin_instance in PLUGIN_REGISTRY.values():
            if isinstance(plugin_instance, PrimitiveLeafPlugin):
                stack.enter_context(plugin_instance.__class__.plugin_binding())
                prim = getattr(plugin_instance.__class__, "_PRIM", None)
                if isinstance(prim, jcore_ext.Primitive):
                    leaf_prims.append(prim)
        backfill_missing_transpose_rules(leaf_prims)
        yield


@contextmanager
def _force_jax_x64(enable_double_precision: bool) -> Iterator[None]:
    # Config reads include a local override; restore the actual global value.
    previous = jax.enable_x64.get_global()
    try:
        jax.config.update("jax_enable_x64", bool(enable_double_precision))
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _create_ir_context(
    *,
    opset: int,
    enable_double_precision: bool,
    input_specs: Sequence[Any],
    frozen_params: Mapping[str, object],
    record_primitive_calls_file: Optional[str],
    normalization_mode: str = "auto",
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
    jaxpr_validator: Callable[[Any, int], None] | None = None,
) -> IRContext:
    ctx = IRContext(
        opset=opset,
        enable_double_precision=enable_double_precision,
        normalization_mode=normalization_mode,
        input_specs=input_specs,
        precision_policy=precision_policy,
        jaxpr_validator=jaxpr_validator,
    )
    ctx._call_input_param_names = set(frozen_params.keys())
    ctx._call_input_param_literals = dict(frozen_params)

    if record_primitive_calls_file:
        ctx.record_primitive_calls_file = str(record_primitive_calls_file)

    if ctx.get_function_registry() is None:
        ctx.set_function_registry(FunctionRegistry())

    return ctx


def _bind_closed_jaxpr_constants(
    ctx: IRContext,
    jpr: Any,
    consts: Sequence[object],
    *,
    default_float: np.dtype[Any],
    enable_double_precision: bool,
) -> None:
    source_precision = ctx.precision_policy is PrecisionPolicy.SOURCE
    for cv, cval in zip(jpr.constvars, consts, strict=source_precision):
        if source_precision:
            ctx.bind_const_for_var(cv, np.asarray(cval, dtype=cv.aval.dtype))
            continue
        np_c = np.asarray(cval)
        target_dtype = None
        try:
            target_dtype = np.dtype(cv.aval.dtype)
        except AttributeError:
            target_dtype = None
        except TypeError:
            target_dtype = None
        desired_dtype = None
        if target_dtype is not None and target_dtype != np_c.dtype:
            desired_dtype = target_dtype
        elif target_dtype is None and np.issubdtype(np_c.dtype, np.floating):
            desired_dtype = default_float

        if (
            not enable_double_precision
            and desired_dtype is not None
            and np.issubdtype(desired_dtype, np.floating)
            and desired_dtype not in (np.dtype(np.float16), np.dtype(np.float32))
            and desired_dtype.name != "bfloat16"
            and np_c.dtype != np.float64
        ):
            desired_dtype = np.dtype(np.float32)

        if desired_dtype is not None and np_c.dtype != desired_dtype:
            np_c = np_c.astype(desired_dtype, copy=False)

        np_c = _maybe_promote_float_array(np_c, enable_double_precision)
        ctx.bind_const_for_var(cv, np_c)


class _LayoutAdapter:
    def __init__(self, ctx: IRContext, *, enable_double_precision: bool) -> None:
        self.ctx = ctx
        self.enable_double_precision = enable_double_precision

    @staticmethod
    def _require_4d(shape: Sequence[object], *, kind: str, index: int) -> None:
        if len(shape) == 4:
            return
        if kind == "input":
            raise ValueError(
                f"inputs_as_nchw: input {index} has rank {len(shape)}, expected 4 for NCHW handling."
            )
        raise ValueError(
            f"outputs_as_nchw: output {index} has rank {len(shape)}, expected 4."
        )

    def bind_input(self, var: Any, index: int) -> None:
        aval_shape = tuple(var.aval.shape)
        self._require_4d(aval_shape, kind="input", index=index)

        if self.ctx.precision_policy is PrecisionPolicy.SOURCE:
            canonical = representation_for_aval(var.aval)
            boundary = representation_for_aval(var.aval, boundary_layout="NCHW")
            external = ir.val(
                f"in_{index}",
                dtype=boundary.ir_dtype,
                shape=ir_shape_from_dims(
                    boundary.physical_shape, parse_integer_like=False
                ),
            )
            self.ctx.add_graph_input_value(external)
            restored = self.ctx.builder.op(
                "Transpose",
                [external],
                {"perm": ir.AttrInt64s("perm", boundary.input_permutation)},
                output=ir.val(
                    f"in_{index}_nhwc_restored",
                    dtype=canonical.ir_dtype,
                    shape=ir_shape_from_dims(
                        canonical.physical_shape, parse_integer_like=False
                    ),
                ),
            )
            self.ctx.bind_value_for_var_without_origins(var, restored)
            self.ctx.record_symbolic_dim_origins(boundary.physical_shape, external)
            return

        nchw_shape = tuple(aval_shape[p] for p in _NHWC_TO_NCHW_PERM)
        nchw_input_val = ir.Value(
            name=f"in_{index}_nchw",
            type=ir.TensorType(
                _dtype_to_ir(np.dtype(var.aval.dtype), self.enable_double_precision)
            ),
            shape=_to_ir_shape(nchw_shape),
        )
        self.ctx.add_graph_input_value(nchw_input_val)

        transposed = self.ctx.builder.Transpose(
            nchw_input_val,
            perm=list(_NCHW_TO_NHWC_PERM),
            _outputs=[f"in_{index}_nhwc_restored"],
        )
        transposed.type = nchw_input_val.type
        transposed.shape = _to_ir_shape(aval_shape)
        self.ctx.bind_value_for_var_without_origins(var, transposed)
        self.ctx.record_symbolic_dim_origins(nchw_shape, nchw_input_val)

    def bind_inputs(self, jpr: Any, *, inputs_as_nchw: Sequence[int]) -> None:
        nchw_inputs_indices = set(inputs_as_nchw)
        for index, var in enumerate(jpr.invars):
            if index in nchw_inputs_indices:
                self.bind_input(var, index)
            else:
                self.ctx.add_input_for_invar(var, index)

    def bind_output(self, out_var: Any, index: int) -> None:
        val = self.ctx.require_value_for_var(out_var)
        aval_shape = tuple(out_var.aval.shape)
        self._require_4d(aval_shape, kind="output", index=index)

        if self.ctx.precision_policy is PrecisionPolicy.SOURCE:
            boundary = representation_for_aval(out_var.aval, boundary_layout="NCHW")
            external = self.ctx.builder.op(
                "Transpose",
                [val],
                {"perm": ir.AttrInt64s("perm", boundary.output_permutation)},
                output=ir.val(
                    f"out_{index}_nchw_converted",
                    dtype=boundary.ir_dtype,
                    shape=ir_shape_from_dims(
                        boundary.physical_shape, parse_integer_like=False
                    ),
                ),
            )
            self.ctx.add_graph_output_value(external)
            return

        output_type = val.type
        if output_type is None:
            aval_dtype = _maybe_dtype(out_var.aval)
            if aval_dtype is not None:
                ir_dtype = _dtype_to_ir(aval_dtype, self.enable_double_precision)
                output_type = ir.TensorType(ir_dtype)

        output_shape = (
            ir.Shape(tuple(val.shape[p] for p in _NHWC_TO_NCHW_PERM))
            if val.shape is not None
            else None
        )
        output = ir.Value(
            name=f"out_{index}_nchw_converted", type=output_type, shape=output_shape
        )
        builder: IRBuilderAPI = self.ctx.builder
        transposed_out = builder.op(
            "Transpose",
            [val],
            {"perm": ir.AttrInt64s("perm", _NHWC_TO_NCHW_PERM)},
            output=output,
        )

        self.ctx.add_graph_output_value(transposed_out)

    def bind_outputs(self, jpr: Any, *, outputs_as_nchw: Sequence[int]) -> None:
        if not outputs_as_nchw:
            self.ctx.add_outputs_from_vars(jpr.outvars)
            return

        nchw_outputs_indices = set(outputs_as_nchw)
        for index, out_var in enumerate(jpr.outvars):
            if index in nchw_outputs_indices:
                self.bind_output(out_var, index)
            else:
                self.ctx.add_outputs_from_vars([out_var])


def _bind_nchw_input_for_invar(ctx: IRContext, var: Any, index: int) -> None:
    _LayoutAdapter(ctx, enable_double_precision=ctx.enable_double_precision).bind_input(
        var, index
    )


def _bind_jaxpr_inputs(
    ctx: IRContext,
    jpr: Any,
    *,
    inputs_as_nchw: Sequence[int],
) -> None:
    _LayoutAdapter(
        ctx, enable_double_precision=ctx.enable_double_precision
    ).bind_inputs(jpr, inputs_as_nchw=inputs_as_nchw)


def _bind_nchw_output_for_outvar(
    ctx: IRContext,
    out_var: Any,
    index: int,
    *,
    enable_double_precision: bool,
) -> None:
    _LayoutAdapter(ctx, enable_double_precision=enable_double_precision).bind_output(
        out_var, index
    )


def _bind_jaxpr_outputs(
    ctx: IRContext,
    jpr: Any,
    *,
    outputs_as_nchw: Sequence[int],
    enable_double_precision: bool,
) -> None:
    if ctx.precision_policy is PrecisionPolicy.SOURCE:
        for var in jpr.outvars:
            value = ctx.require_value_for_var(var)
            representation = representation_for_aval(var.aval)
            if not representation.matches_ir_value(value):
                raise RuntimeError(
                    f"Source result type {representation.ir_dtype} became {value.dtype}; "
                    f"required shape {representation.physical_shape}, received {value.shape}"
                )
    _LayoutAdapter(ctx, enable_double_precision=enable_double_precision).bind_outputs(
        jpr, outputs_as_nchw=outputs_as_nchw
    )


def _trace_to_jaxpr(
    *,
    fn: Callable[..., Any],
    inputs: Sequence[InputSpec],
    input_params: Optional[Mapping[str, object]],
    enable_double_precision: bool,
    inputs_as_nchw: Optional[Sequence[int]],
    outputs_as_nchw: Optional[Sequence[int]],
    input_names: Optional[Sequence[str]],
    output_names: Optional[Sequence[str]],
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
    opset: int = 23,
) -> _TraceResult:
    source = precision_policy is PrecisionPolicy.SOURCE
    positional_count = len(inputs)
    named_params = dict(input_params or {}) if source else {}
    declarations = list(inputs)
    if source:
        declarations.extend(
            _source_named_input_spec(name, value, opset=opset)
            for name, value in named_params.items()
        )
        for index, spec in enumerate(declarations):
            assert isinstance(spec, jax.ShapeDtypeStruct)
            validate_source_input_dtype(spec.dtype, index=index, opset=opset)
    # Normalize the entire call together so named and positional B declarations
    # share the actual native scope, rather than just its printed spelling.
    sds_list = _as_sds_list(
        declarations, enable_double_precision, precision_policy=precision_policy
    )
    frozen_params: Dict[str, object] = {} if source else dict(input_params or {})

    def _wrapped(*xs: Any) -> Any:
        if source:
            named_values = dict(zip(named_params, xs[positional_count:], strict=True))
            return fn(*xs[:positional_count], **named_values)
        return fn(*xs, **frozen_params)

    with precision_policy_scope(precision_policy), _activate_plugin_worlds():
        try:
            closed = jax.make_jaxpr(_wrapped)(*sds_list)
        except jax.errors.TracerBoolConversionError as exc:
            if source and named_params:
                exc.add_note(
                    "SOURCE named runtime arguments are real traced operands. For pure Python-if "
                    "source, prepare an external jax.lax.cond adapter whose concrete True/False "
                    "branches invoke the complete unchanged original callable; bind intended "
                    "fixed configuration with a closure or functools.partial instead."
                )
            raise
    if os.environ.get("J2O_PRINT_JAXPR", "0") == "1":
        try:
            print(f"JAXPR: {closed.jaxpr.pretty_print()}")
        except Exception:
            pass
    jpr = closed.jaxpr

    if input_names is not None and len(input_names) != positional_count:
        raise ValueError(
            f"input_names length ({len(input_names)}) must match traced positional inputs ({positional_count})."
        )
    if output_names is not None and len(output_names) != len(jpr.outvars):
        raise ValueError(
            f"output_names length ({len(output_names)}) must match traced outputs ({len(jpr.outvars)})."
        )
    validated_inputs_as_nchw = _validate_layout_indices(
        inputs_as_nchw,
        kind="inputs_as_nchw",
        upper_bound=positional_count,
    )
    validated_outputs_as_nchw = _validate_layout_indices(
        outputs_as_nchw,
        kind="outputs_as_nchw",
        upper_bound=len(jpr.outvars),
    )

    source_inputs = (
        tuple(
            _SourceInputBinding(
                index,
                index if index < positional_count else None,
                (
                    None
                    if index < positional_count
                    else tuple(named_params)[index - positional_count]
                ),
                spec,
                var,
            )
            for index, (spec, var) in enumerate(zip(sds_list, jpr.invars, strict=True))
        )
        if source
        else ()
    )
    return _TraceResult(
        closed_jaxpr=closed,
        jaxpr=jpr,
        sds_list=sds_list,
        frozen_params=frozen_params,
        inputs_as_nchw=validated_inputs_as_nchw,
        outputs_as_nchw=validated_outputs_as_nchw,
        source_inputs=source_inputs,
    )


def _lower_jaxpr_equations(ctx: IRContext, jpr: Any) -> None:
    _lowering_dispatch.lower_jaxpr_with_plugins(
        ctx=ctx,
        jaxpr=jpr,
        registry=PLUGIN_REGISTRY,
        source="converter",
    )


def _convert_to_ir(
    *,
    fn: Callable[..., Any],
    inputs: Sequence[InputSpec],
    input_params: Optional[Mapping[str, object]],
    model_name: str,
    opset: int,
    enable_double_precision: bool,
    record_primitive_calls_file: Optional[str],
    protective_clone: bool = True,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    normalization_mode: str = "auto",
    strict_optimizer_failures: Optional[bool] = None,
    precision_policy: PrecisionPolicy = PrecisionPolicy.LEGACY,
    jaxpr_validator: Callable[[Any, int], None] | None = None,
    trace_observer: Callable[[_TraceResult], None] | None = None,
) -> ir.Model:
    """Trace, lower, and finalize through one private conversion pipeline.

    The caller owns any legacy x64 override. SOURCE retains ambient JAX
    configuration and requires a validator for its bounded audited subset.
    """
    if precision_policy is PrecisionPolicy.SOURCE:
        if inputs_as_nchw is not None and len(inputs_as_nchw) == 0:
            inputs_as_nchw = None
        if outputs_as_nchw is not None and len(outputs_as_nchw) == 0:
            outputs_as_nchw = None
        strict_optimizer_failures = True
        if enable_double_precision:
            raise ValueError("Source precision cannot use legacy double promotion")
        if jaxpr_validator is None:
            raise ValueError("Private source conversion requires a subset validator")
        unsupported = {
            "normalization_mode": normalization_mode != "auto",
        }
        for option, supplied in unsupported.items():
            if supplied:
                raise ValueError(f"Private source conversion does not support {option}")
        for index, spec in enumerate(inputs):
            if not isinstance(spec, jax.ShapeDtypeStruct):
                raise ValueError("Private source conversion requires typed inputs")
            validate_source_input_dtype(spec.dtype, index=index, opset=opset)
        validate_source_shapes(
            [cast(jax.ShapeDtypeStruct, spec).shape for spec in inputs]
        )

    trace = _trace_to_jaxpr(
        fn=fn,
        inputs=inputs,
        input_params=input_params,
        enable_double_precision=enable_double_precision,
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        input_names=input_names,
        output_names=output_names,
        precision_policy=precision_policy,
        opset=opset,
    )
    if trace_observer is not None:
        trace_observer(trace)
    jpr = trace.jaxpr
    if jaxpr_validator is not None:
        jaxpr_validator(jpr, opset)
    ctx = _create_ir_context(
        opset=opset,
        enable_double_precision=enable_double_precision,
        normalization_mode=normalization_mode,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=record_primitive_calls_file,
        precision_policy=precision_policy,
        jaxpr_validator=jaxpr_validator,
    )
    _bind_closed_jaxpr_constants(
        ctx,
        jpr,
        trace.closed_jaxpr.consts,
        default_float=_np_float_dtype(enable_double_precision),
        enable_double_precision=enable_double_precision,
    )
    _bind_jaxpr_inputs(ctx, jpr, inputs_as_nchw=trace.inputs_as_nchw)

    _lower_jaxpr_equations(ctx, jpr)
    _bind_jaxpr_outputs(
        ctx,
        jpr,
        outputs_as_nchw=trace.outputs_as_nchw,
        enable_double_precision=enable_double_precision,
    )

    model = _build_and_finalize_ir_model(
        ctx,
        model_name=model_name,
        protective_clone=protective_clone,
        strict_optimizer_failures=strict_optimizer_failures,
    )
    if precision_policy is PrecisionPolicy.SOURCE:
        for label, variables, values, layouts in (
            ("input", jpr.invars, model.graph.inputs, trace.inputs_as_nchw),
            ("output", jpr.outvars, model.graph.outputs, trace.outputs_as_nchw),
        ):
            for index, (var, value) in enumerate(zip(variables, values, strict=True)):
                representation = representation_for_aval(
                    var.aval, boundary_layout="NCHW" if index in layouts else None
                )
                if not representation.matches_ir_value(value):
                    raise RuntimeError(
                        f"Final {label} {value.name} lost source type {representation.ir_dtype} "
                        f"or shape {representation.physical_shape}"
                    )
        _source_input_values(trace, model.graph.inputs, require_internal_names=True)
    return model


def to_onnx(
    *,
    fn: Callable[..., Any],
    inputs: Sequence[InputSpec],
    input_params: Optional[Mapping[str, object]],
    model_name: str,
    opset: int,
    enable_double_precision: bool,
    record_primitive_calls_file: Optional[str],
    protective_clone: bool = True,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    normalization_mode: str = "auto",
    strict_optimizer_failures: Optional[bool] = None,
) -> ir.Model:
    """Build an ONNX-IR model using the legacy export precision rules."""
    with _force_jax_x64(enable_double_precision):
        return _convert_to_ir(
            fn=fn,
            inputs=inputs,
            input_params=input_params,
            model_name=model_name,
            opset=opset,
            enable_double_precision=enable_double_precision,
            record_primitive_calls_file=record_primitive_calls_file,
            protective_clone=protective_clone,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw,
            input_names=input_names,
            output_names=output_names,
            normalization_mode=normalization_mode,
            strict_optimizer_failures=strict_optimizer_failures,
        )
