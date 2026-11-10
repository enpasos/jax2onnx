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
from .ir_optimizations import optimize_graph
from .state_inputs import StateInputs
from .function_scope import FunctionRegistry
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


def _as_sds_list(inputs: Sequence[InputSpec]) -> List["jax.ShapeDtypeStruct"]:
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
            sds_list.append(
                jax.ShapeDtypeStruct(
                    tuple(dims_list), spec.dtype, weak_type=spec.weak_type
                )
            )
            continue

        dims_tuple = tuple(
            name2sym[dim] if isinstance(dim, str) else int(dim) for dim in spec
        )
        sds_list.append(jax.ShapeDtypeStruct(dims_tuple, np.float32))
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
) -> None:
    try:
        optimize_graph(model)
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


def _create_ir_context(
    *,
    opset: int,
    input_specs: Sequence[Any],
    frozen_params: Mapping[str, object],
    record_primitive_calls_file: Optional[str],
    normalization_mode: str = "auto",
    state: Optional[StateInputs] = None,
) -> IRContext:
    ctx = IRContext(
        opset=opset,
        normalization_mode=normalization_mode,
        input_specs=input_specs,
    )
    ctx._state_inputs = state
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
) -> None:
    for cv in jpr.constvars:
        if _maybe_dtype(cv.aval) is None:
            raise TypeError("Constant has no traced dtype")
    ctx.bind_captured_arrays(jpr.constvars, consts)


class _LayoutAdapter:
    def __init__(self, ctx: IRContext) -> None:
        self.ctx = ctx

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
            type=ir.TensorType(numpy_dtype_to_ir(var.aval.dtype)),
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

        output_type = val.type
        if output_type is None:
            output_type = ir.TensorType(numpy_dtype_to_ir(out_var.aval.dtype))

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
    _LayoutAdapter(ctx).bind_input(var, index)


def _bind_jaxpr_inputs(
    ctx: IRContext,
    jpr: Any,
    *,
    inputs_as_nchw: Sequence[int],
) -> None:
    _LayoutAdapter(ctx).bind_inputs(jpr, inputs_as_nchw=inputs_as_nchw)


def _bind_nchw_output_for_outvar(
    ctx: IRContext,
    out_var: Any,
    index: int,
) -> None:
    _LayoutAdapter(ctx).bind_output(out_var, index)


def _bind_jaxpr_outputs(
    ctx: IRContext,
    jpr: Any,
    *,
    outputs_as_nchw: Sequence[int],
) -> None:
    _LayoutAdapter(ctx).bind_outputs(jpr, outputs_as_nchw=outputs_as_nchw)


def _trace_to_jaxpr(
    *,
    fn: Callable[..., Any],
    inputs: Sequence[InputSpec],
    input_params: Optional[Mapping[str, object]],
    inputs_as_nchw: Optional[Sequence[int]],
    outputs_as_nchw: Optional[Sequence[int]],
    input_names: Optional[Sequence[str]],
    output_names: Optional[Sequence[str]],
) -> _TraceResult:
    sds_list = _as_sds_list(inputs)
    native_specs = jax.eval_shape(lambda *values: values, *sds_list)
    for index, (spec, native_spec) in enumerate(zip(sds_list, native_specs)):
        canonical = np.dtype(native_spec.dtype)
        if canonical != np.dtype(spec.dtype):
            raise ValueError(
                f"Input {index} dtype {spec.dtype} would be narrowed to {canonical}; "
                "configure JAX first, for example with jax.enable_x64()."
            )
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
    _lowering_dispatch.lower_jaxpr_with_plugins(
        ctx=ctx,
        jaxpr=jpr,
        registry=PLUGIN_REGISTRY,
        source="converter",
    )


def to_onnx(
    *,
    fn: Callable[..., Any],
    inputs: Sequence[InputSpec],
    input_params: Optional[Mapping[str, object]],
    model_name: str,
    opset: int,
    record_primitive_calls_file: Optional[str],
    protective_clone: bool = True,
    inputs_as_nchw: Optional[Sequence[int]] = None,
    outputs_as_nchw: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    normalization_mode: str = "auto",
    strict_optimizer_failures: Optional[bool] = None,
    state: Optional[StateInputs] = None,
) -> ir.Model:
    """
    Build an ONNX-IR model in three phases:
    1) Trace JAX to ClosedJaxpr with symbolic shapes.
    2) Lower to onnx_ir (plugins; function bodies allowed).
    3) Run a single IR-wide optimization pass (cross-node cleanups).
    """
    trace = _trace_to_jaxpr(
        fn=fn,
        inputs=inputs,
        input_params=input_params,
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        input_names=input_names,
        output_names=output_names,
    )
    ctx = _create_ir_context(
        opset=opset,
        normalization_mode=normalization_mode,
        input_specs=trace.sds_list,
        frozen_params=trace.frozen_params,
        record_primitive_calls_file=record_primitive_calls_file,
        state=state,
    )
    _bind_closed_jaxpr_constants(ctx, trace.jaxpr, trace.closed_jaxpr.consts)
    _bind_jaxpr_inputs(ctx, trace.jaxpr, inputs_as_nchw=trace.inputs_as_nchw)

    _lower_jaxpr_equations(ctx, trace.jaxpr)
    _bind_jaxpr_outputs(ctx, trace.jaxpr, outputs_as_nchw=trace.outputs_as_nchw)

    return _build_and_finalize_ir_model(
        ctx,
        model_name=model_name,
        protective_clone=protective_clone,
        strict_optimizer_failures=strict_optimizer_failures,
    )
