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
from jax2onnx.utils.debug import RecordedPrimitiveCallLog, save_primitive_calls_log
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
from .lowering_dispatch import (
    get_registered_lowering_plugin,
    identify_lowering_plugin,
    lower_equation_with_plugin,
    make_converter_facade,
)
from .output_binding import get_bound_value
from .typing_support import LoweringContextProtocol

from jax.extend import core as jcore_ext

ShapeDimSpec = Union[int, str]
ShapeTupleSpec = Tuple[ShapeDimSpec, ...]
InputSpec = Union[jax.ShapeDtypeStruct, ShapeTupleSpec]

_ORT_SAFE_IR_VERSION: int = 10
_LOGGER: logging.Logger = logging.getLogger(__name__)
_STRICT_OPTIMIZER_FAILURES_ENV: str = "JAX2ONNX_STRICT_OPTIMIZER_FAILURES"
_NHWC_TO_NCHW_PERM: tuple[int, int, int, int] = (0, 3, 1, 2)
_NCHW_TO_NHWC_PERM: tuple[int, int, int, int] = (0, 2, 3, 1)


@dataclass(frozen=True)
class _TraceResult:
    closed_jaxpr: Any
    jaxpr: Any
    sds_list: list[jax.ShapeDtypeStruct]
    frozen_params: dict[str, object]
    inputs_as_nchw: tuple[int, ...]
    outputs_as_nchw: tuple[int, ...]


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
    if np.issubdtype(np_dtype, np.floating):
        return ir.DataType.DOUBLE if np_dtype == np.float64 else ir.DataType.FLOAT
    if np.issubdtype(np_dtype, np.integer):
        return numpy_dtype_to_ir(np_dtype, default=ir.DataType.INT64)
    if np_dtype == np.dtype(np.bool_):
        return numpy_dtype_to_ir(np_dtype)
    return ir.DataType.FLOAT


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
    inputs: Sequence[InputSpec], enable_double_precision: bool
) -> List["jax.ShapeDtypeStruct"]:
    """Normalize user 'inputs' to ShapeDtypeStructs for abstract tracing."""
    sds_list: List[jax.ShapeDtypeStruct] = []

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
    shared_scope = jax_export.SymbolicScope() if symnames else None
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
            sds_list.append(jax.ShapeDtypeStruct(tuple(dims_list), spec.dtype))
            continue

        dims_tuple = tuple(
            name2sym[dim] if isinstance(dim, str) else int(dim) for dim in spec
        )
        dt = jnp.float64 if enable_double_precision else jnp.float32
        sds_list.append(jax.ShapeDtypeStruct(dims_tuple, dt))
    return sds_list


def _maybe_dtype(aval: Any) -> Optional[np.dtype[Any]]:
    return cast(
        Optional[np.dtype[Any]], maybe_numpy_dtype(getattr(aval, "dtype", None))
    )


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
) -> None:
    try:
        optimize_graph(model)
    except Exception as exc:
        if _resolve_strict_optimizer_failures(strict_optimizer_failures):
            raise
        _log_nonfatal_stage_failure("optimize_graph", exc)


def _aval_log_entry(var: object) -> tuple[tuple[object, ...], str, str]:
    aval = getattr(var, "aval", None)
    if aval is None:
        return ((), "", type(var).__name__)
    raw_shape = getattr(aval, "shape", ())
    try:
        shape = tuple(raw_shape)
    except TypeError:
        shape = ()
    dtype = getattr(aval, "dtype", "")
    return (shape, str(dtype), type(aval).__name__)


def _var_log_name(var: object) -> str:
    try:
        return str(var)
    except Exception:
        return repr(var)


def _bound_value_name(ctx: IRContext, var: object) -> str:
    value = get_bound_value(ctx, var)
    if value is None:
        return ""
    return value.name or ""


def _params_repr(params: Mapping[str, object]) -> str:
    if not params:
        return ""
    lines: list[str] = []
    for key in sorted(params):
        try:
            value_repr = repr(params[key])
        except Exception:
            value_repr = f"<unrepresentable:{type(params[key]).__name__}>"
        lines.append(f"  {key}: {value_repr}")
    return "\n".join(lines)


def _plugin_file_hint(plugin_ref: object, prim_name: str) -> str:
    if plugin_ref is None:
        return prim_name
    return f"{type(plugin_ref).__module__}.{type(plugin_ref).__name__}"


def _primitive_call_record(
    ctx: IRContext,
    eqn: object,
    *,
    eqn_index: int,
    primitive_name: str,
    plugin_ref: object,
) -> RecordedPrimitiveCallLog:
    invars = list(getattr(eqn, "invars", ()))
    outvars = list(getattr(eqn, "outvars", ()))
    params = getattr(eqn, "params", {})
    if not isinstance(params, Mapping):
        params = {}
    return RecordedPrimitiveCallLog(
        sequence_id=eqn_index,
        primitive_name=primitive_name,
        plugin_file_hint=_plugin_file_hint(plugin_ref, primitive_name),
        params=dict(params),
        params_repr=_params_repr(params),
        inputs_aval=[_aval_log_entry(var) for var in invars],
        outputs_aval=[_aval_log_entry(var) for var in outvars],
        inputs_jax_vars=[_var_log_name(var) for var in invars],
        inputs_onnx_names=[_bound_value_name(ctx, var) for var in invars],
        outputs_jax_vars=[_var_log_name(var) for var in outvars],
        outputs_onnx_names=[_bound_value_name(ctx, var) for var in outvars],
    )


def _append_primitive_call_record(
    records: list[RecordedPrimitiveCallLog],
    ctx: IRContext,
    eqn: object,
    *,
    eqn_index: int,
    primitive_name: str,
    plugin_ref: object,
) -> None:
    if not ctx.record_primitive_calls_file:
        return
    records.append(
        _primitive_call_record(
            ctx,
            eqn,
            eqn_index=eqn_index,
            primitive_name=primitive_name,
            plugin_ref=plugin_ref,
        )
    )


def _eqn_jax_traceback(eqn: object) -> str | None:
    source_info = getattr(eqn, "source_info", None)
    if source_info is None:
        return None
    traceback = getattr(source_info, "traceback", None)
    if traceback is None:
        return None
    try:
        return str(traceback)
    except Exception:
        return None


@contextmanager
def _staged_lowering_metadata(
    builder: Any,
    *,
    eqn: object,
    plugin_ref: object,
    primitive_name: str,
) -> Iterator[None]:
    prev_jax_trace = builder.current_jax_traceback
    prev_plugin_id = builder.current_plugin_identifier
    prev_plugin_line = builder.current_plugin_line
    jax_trace: Optional[str] = None
    plugin_identifier: Optional[str] = None
    plugin_line: Optional[str] = None

    if builder.stacktrace_metadata_enabled:
        jax_trace = _eqn_jax_traceback(eqn)
        plugin_identifier, plugin_line = identify_lowering_plugin(
            plugin_ref,
            primitive_name,
        )

    builder.set_current_jax_traceback(jax_trace)
    builder.set_current_plugin_identifier(plugin_identifier, plugin_line)
    try:
        yield
    finally:
        builder.set_current_jax_traceback(prev_jax_trace)
        builder.set_current_plugin_identifier(prev_plugin_id, prev_plugin_line)


@contextmanager
def _current_eqn_scope(ctx: IRContext, eqn: object) -> Iterator[None]:
    previous_eqn = ctx._current_eqn
    ctx._current_eqn = eqn
    try:
        yield
    finally:
        ctx._current_eqn = previous_eqn


# Deprecated compatibility alias for TYPE_CHECKING-only legacy plugin imports.
_IRBuildContext = LoweringContextProtocol


def _function_id(func: ir.Function) -> tuple[str, str, str]:
    return (
        (func.domain or ""),
        (func.name or ""),
        (func.overload or ""),
    )


def _function_store_identifier(fn_ir: ir.Function) -> object:
    identifier: object | None = None
    try:
        identifier_fn = fn_ir.identifier
    except AttributeError:
        identifier_fn = None

    if callable(identifier_fn):
        try:
            identifier = identifier_fn()
        except Exception:
            identifier = None
    if not identifier and hasattr(fn_ir, "id"):
        identifier = object.__getattribute__(fn_ir, "id")
    if not identifier:
        identifier = _function_id(fn_ir)
    return identifier


def _attach_ir_functions(ir_model: ir.Model, ctx: IRContext) -> None:
    ir_funcs = list(ctx.ir_functions)
    if not ir_funcs:
        return

    functions_store = ir_model.functions
    if functions_store is None:
        try:
            ir_model.functions = {}
            functions_store = ir_model.functions
        except Exception:
            ir_model.functions = []
            functions_store = ir_model.functions

    if isinstance(functions_store, dict):
        for fn_ir in ir_funcs:
            functions_store[_function_store_identifier(fn_ir)] = fn_ir
    elif isinstance(functions_store, list):
        existing = {_function_id(func) for func in functions_store}
        for fn_ir in ir_funcs:
            func_id = _function_id(fn_ir)
            if func_id not in existing:
                functions_store.append(fn_ir)
                existing.add(func_id)
    else:
        ir_model.functions = list(ir_funcs)

    model_imports: Dict[str, int] = dict(ir_model.opset_imports or {})
    model_imports.setdefault("", int(ctx.builder.opset) or 23)
    for fn_ir in ir_funcs:
        dom = (fn_ir.domain or "").strip()
        if dom and dom not in model_imports:
            model_imports[dom] = 1
    try:
        ir_model.opset_imports = model_imports
    except Exception:
        try:
            existing_imports = ir_model.opset_imports
            if hasattr(existing_imports, "update"):
                existing_imports.update(model_imports)
        except Exception:
            pass


def _iter_graph_values(gr: ir.Graph) -> Iterable[ir.Value]:
    seen: set[int] = set()
    staged: list[ir.Value] = []

    def _queue(values: Iterable[ir.Value]) -> None:
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
    if isinstance(shape_obj, ir.Shape):
        dims_source: Tuple[object, ...] = tuple(shape_obj.dims)
    elif isinstance(shape_obj, Iterable):
        dims_source = tuple(shape_obj)
    else:
        return

    normalized_dims: List[object] = []
    dirty = False
    for dim in dims_source:
        normalized_dim: object = dim
        if isinstance(dim, (int, np.integer)):
            normalized_dim = int(dim)
        else:
            label = _as_ir_dim_label(dim)
            if isinstance(label, int):
                normalized_dim = int(label)
            elif isinstance(label, str):
                normalized_dim = ir.SymbolicDim(label)
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


def _apply_ir_attr_overrides_to_graph(
    gr: ir.Graph, overrides: dict[str, dict[str, object]]
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
            attr_obj = _convert_ir_attr(attr_name, attr_value)
            if attr_obj is not None:
                node.attributes[attr_name] = attr_obj


def _fix_concat_axis_in_graph(gr: ir.Graph) -> None:
    for node in gr.all_nodes():
        if node.op_type != "Concat":
            continue
        if "axis" in node.attributes:
            continue
        node.attributes["axis"] = ir.convenience.convert_attribute("axis", 0)


def _apply_late_ir_attr_overrides(ir_model: ir.Model, ctx: IRContext) -> None:
    _apply_ir_attr_overrides_to_graph(ir_model.graph, ctx.attr_overrides)
    _fix_concat_axis_in_graph(ir_model.graph)

    for fn in iter_ir_functions(ir_model.functions):
        try:
            graph_obj = getattr(fn, "graph", None)
            if graph_obj is None:
                continue
            overrides_attr: dict[str, dict[str, object]] = {}
            if hasattr(fn, "_attr_overrides"):
                raw_overrides = object.__getattribute__(fn, "_attr_overrides")
                if raw_overrides:
                    overrides_attr = dict(raw_overrides)
            fn_overrides = overrides_attr or dict(ctx.attr_overrides or {})
            _apply_ir_attr_overrides_to_graph(graph_obj, fn_overrides)
            _fix_concat_axis_in_graph(graph_obj)
        except Exception as exc:
            fn_name = getattr(fn, "name", "<unnamed function>")
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
    )

    _apply_late_ir_attr_overrides(ir_model, ctx)
    _consume_onnx_function_hits()

    try:
        _finalize_model_value_shapes(ir_model)
    except Exception as exc:
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
    read_config = jax.config.read if hasattr(jax.config, "read") else None
    if callable(read_config):
        previous = bool(read_config("jax_enable_x64"))
    else:
        previous = (
            bool(jax.config.jax_enable_x64)
            if hasattr(jax.config, "jax_enable_x64")
            else False
        )
    target = bool(enable_double_precision)
    if previous != target:
        jax.config.update("jax_enable_x64", target)
    try:
        yield
    finally:
        if previous != target:
            jax.config.update("jax_enable_x64", previous)


def _create_ir_context(
    *,
    opset: int,
    enable_double_precision: bool,
    input_specs: Sequence[Any],
    frozen_params: Mapping[str, object],
    record_primitive_calls_file: Optional[str],
) -> IRContext:
    ctx = IRContext(
        opset=opset,
        enable_double_precision=enable_double_precision,
        input_specs=input_specs,
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
    for cv, cval in zip(jpr.constvars, consts):
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
            and desired_dtype != np.float32
            and np_c.dtype != np.float64
        ):
            desired_dtype = np.float32

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

        nchw_shape = tuple(aval_shape[p] for p in _NHWC_TO_NCHW_PERM)
        nchw_input_val = ir.Value(
            name=f"in_{index}_nchw",
            type=ir.TensorType(_to_ir_dtype_from_np(np.dtype(var.aval.dtype))),
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
        val = self.ctx.get_value_for_var(out_var)
        aval_shape = tuple(out_var.aval.shape)
        self._require_4d(aval_shape, kind="output", index=index)

        output_type = val.type
        if output_type is None:
            aval_dtype = _maybe_dtype(out_var.aval)
            if aval_dtype is not None:
                ir_dtype = _dtype_to_ir(aval_dtype, self.enable_double_precision)
                output_type = ir.TensorType(ir_dtype)

        transposed_out = self.ctx.builder.Transpose(
            val,
            perm=list(_NHWC_TO_NCHW_PERM),
            _outputs=[f"out_{index}_nchw_converted"],
        )
        if output_type is not None:
            transposed_out.type = output_type

        if isinstance(val.shape, ir.Shape):
            src_dims = val.shape.dims
            transposed_out.shape = ir.Shape(
                tuple(src_dims[p] for p in _NHWC_TO_NCHW_PERM)
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
) -> _TraceResult:
    sds_list = _as_sds_list(inputs, enable_double_precision)
    frozen_params: Dict[str, object] = dict(input_params or {})

    def _wrapped(*xs: Any) -> Any:
        return fn(*xs, **frozen_params)

    with _activate_plugin_worlds():
        closed = jax.make_jaxpr(_wrapped)(*sds_list)
    if os.environ.get("J2O_PRINT_JAXPR", "0") == "1":
        try:
            print(f"JAXPR: {closed.jaxpr.pretty_print()}")
        except Exception:
            pass
    jpr = closed.jaxpr

    if input_names is not None and len(input_names) != len(jpr.invars):
        raise ValueError(
            f"input_names length ({len(input_names)}) must match traced positional inputs ({len(jpr.invars)})."
        )
    if output_names is not None and len(output_names) != len(jpr.outvars):
        raise ValueError(
            f"output_names length ({len(output_names)}) must match traced outputs ({len(jpr.outvars)})."
        )
    validated_inputs_as_nchw = _validate_layout_indices(
        inputs_as_nchw,
        kind="inputs_as_nchw",
        upper_bound=len(jpr.invars),
    )
    validated_outputs_as_nchw = _validate_layout_indices(
        outputs_as_nchw,
        kind="outputs_as_nchw",
        upper_bound=len(jpr.outvars),
    )

    return _TraceResult(
        closed_jaxpr=closed,
        jaxpr=jpr,
        sds_list=sds_list,
        frozen_params=frozen_params,
        inputs_as_nchw=validated_inputs_as_nchw,
        outputs_as_nchw=validated_outputs_as_nchw,
    )


def _lower_jaxpr_equations(ctx: IRContext, jpr: Any) -> None:
    converter = make_converter_facade(ctx)
    ctx._const_folder.install_producers(jpr)
    primitive_call_records: list[RecordedPrimitiveCallLog] = []

    try:
        for eqn_index, eqn in enumerate(jpr.eqns):
            prim_name = eqn.primitive.name
            plugin_ref = get_registered_lowering_plugin(
                PLUGIN_REGISTRY,
                prim_name,
                source="converter",
            )
            builder = ctx.builder
            with (
                _current_eqn_scope(ctx, eqn),
                _staged_lowering_metadata(
                    builder,
                    eqn=eqn,
                    plugin_ref=plugin_ref,
                    primitive_name=prim_name,
                ),
            ):
                lower_equation_with_plugin(
                    plugin_ref,
                    ctx=ctx,
                    eqn=eqn,
                    primitive_name=prim_name,
                    eqn_index=eqn_index,
                    source="converter",
                    converter=converter,
                )
                _append_primitive_call_record(
                    primitive_call_records,
                    ctx,
                    eqn,
                    eqn_index=eqn_index,
                    primitive_name=prim_name,
                    plugin_ref=plugin_ref,
                )
    finally:
        if ctx.record_primitive_calls_file:
            save_primitive_calls_log(
                primitive_call_records, ctx.record_primitive_calls_file
            )


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
    strict_optimizer_failures: Optional[bool] = None,
) -> ir.Model:
    """
    Build an ONNX-IR model in three phases:
    1) Trace JAX to ClosedJaxpr with symbolic shapes.
    2) Lower to onnx_ir (plugins; function bodies allowed).
    3) Run a single IR-wide optimization pass (cross-node cleanups).
    """
    with _force_jax_x64(enable_double_precision):
        default_float = _np_float_dtype(enable_double_precision)
        trace = _trace_to_jaxpr(
            fn=fn,
            inputs=inputs,
            input_params=input_params,
            enable_double_precision=enable_double_precision,
            inputs_as_nchw=inputs_as_nchw,
            outputs_as_nchw=outputs_as_nchw,
            input_names=input_names,
            output_names=output_names,
        )
        closed = trace.closed_jaxpr
        jpr = trace.jaxpr
        sds_list = trace.sds_list
        frozen_params = trace.frozen_params
        validated_inputs_as_nchw = trace.inputs_as_nchw
        validated_outputs_as_nchw = trace.outputs_as_nchw

        ctx = _create_ir_context(
            opset=opset,
            enable_double_precision=enable_double_precision,
            input_specs=sds_list,
            frozen_params=frozen_params,
            record_primitive_calls_file=record_primitive_calls_file,
        )
        _bind_closed_jaxpr_constants(
            ctx,
            jpr,
            closed.consts,
            default_float=default_float,
            enable_double_precision=enable_double_precision,
        )
        _bind_jaxpr_inputs(ctx, jpr, inputs_as_nchw=validated_inputs_as_nchw)

        _lower_jaxpr_equations(ctx, jpr)
        _bind_jaxpr_outputs(
            ctx,
            jpr,
            outputs_as_nchw=validated_outputs_as_nchw,
            enable_double_precision=enable_double_precision,
        )

        return _build_and_finalize_ir_model(
            ctx,
            model_name=model_name,
            protective_clone=protective_clone,
            strict_optimizer_failures=strict_optimizer_failures,
        )
