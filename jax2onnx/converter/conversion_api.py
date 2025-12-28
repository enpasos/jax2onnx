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
)
from contextlib import contextmanager, ExitStack
import inspect as _ins
import os
import jax
import jax.numpy as jnp
from jax import export as jax_export
import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
from onnx_ir.traversal import RecursiveGraphIterator

from jax2onnx.plugins import plugin_system as ps2
from jax2onnx.plugins.plugin_system import (
    PLUGIN_REGISTRY,
    PrimitiveLeafPlugin,
    apply_monkey_patches,
    import_all_plugins,
)
from jax2onnx.plugins._ir_shapes import _as_ir_dim_label

from .ir_context import IRContext
from .ir_builder import IRBuilder
from .ir_optimizations import optimize_graph
from .function_scope import FunctionRegistry
from .typing_support import (
    FunctionLowering,
    PrimitiveLowering,
    SymbolicDimOrigin,
)

from jax.extend import core as jcore_ext

_LITERAL_TYPES: tuple[type[jcore_ext.Literal], ...] = (jcore_ext.Literal,)

ShapeDimSpec = Union[int, str]
ShapeTupleSpec = Tuple[ShapeDimSpec, ...]
InputSpec = Union[jax.ShapeDtypeStruct, ShapeTupleSpec]

_ORT_SAFE_IR_VERSION: int = 10


def run_optional_shape_inference(model: "ir.Model") -> "ir.Model":
    """Placeholder for optional shape inference; returns the model unchanged."""
    return model


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
        return {
            np.dtype(np.int64): ir.DataType.INT64,
            np.dtype(np.int32): ir.DataType.INT32,
            np.dtype(np.int16): ir.DataType.INT16,
            np.dtype(np.int8): ir.DataType.INT8,
            np.dtype(np.uint64): ir.DataType.UINT64,
            np.dtype(np.uint32): ir.DataType.UINT32,
            np.dtype(np.uint16): ir.DataType.UINT16,
            np.dtype(np.uint8): ir.DataType.UINT8,
        }.get(np_dtype, ir.DataType.INT64)
    if np_dtype == np.bool_:
        return ir.DataType.BOOL
    return ir.DataType.FLOAT


def _to_ir_shape(shape_tuple: Sequence[ShapeDimSpec]) -> "ir.Shape":
    dims: Tuple[Union[int, str], ...] = tuple(
        int(d) if isinstance(d, (int, np.integer)) else str(d) for d in shape_tuple
    )
    return ir.Shape(dims)


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


# ---------------------------
# Minimal IR Build Context facade (for plugins)
# ---------------------------


class _IRBuildContext:
    def __init__(self, *, opset: int, default_float_dtype: np.dtype):
        self.opset = opset
        self._default_float_dtype = np.dtype(default_float_dtype)
        self._var2val: Dict[Any, ir.Value] = {}
        self._inputs: List[ir.Value] = []
        self._initializers: List[ir.Value] = []
        self._nodes: List[ir.Node] = []
        self._name_counter = 0
        self._symdim_origin: dict[object, SymbolicDimOrigin] = {}
        self._symdim_origin_str: dict[str, SymbolicDimOrigin] = {}

    def fresh_name(self, prefix: str) -> str:
        self._name_counter += 1
        return f"{prefix}_{self._name_counter}"

    def add_node(
        self,
        node: ir.Node,
        inputs: Sequence[ir.Value] | None = None,
        outputs: Sequence[ir.Value] | None = None,
    ) -> ir.Node:
        if inputs is not None:
            node.inputs = list(inputs)
        if outputs is not None:
            node.outputs = list(outputs)
        self._nodes.append(node)
        return node

    def get_value_for_var(
        self,
        var: Any,
        *,
        name_hint: Optional[str] = None,
        prefer_np_dtype: Optional[np.dtype] = None,
    ) -> "ir.Value":
        if _LITERAL_TYPES and isinstance(var, _LITERAL_TYPES):
            arr = np.asarray(var.val)
            if np.issubdtype(arr.dtype, np.floating):
                if prefer_np_dtype is not None:
                    prefer_dt = np.dtype(prefer_np_dtype)
                    if np.issubdtype(prefer_dt, np.floating):
                        target = (
                            self._default_float_dtype
                            if self._default_float_dtype == np.float64
                            else prefer_dt
                        )
                    else:
                        target = prefer_dt
                else:
                    target = self._default_float_dtype
                arr = np.asarray(var.val, dtype=target)
            c_ir = ir.Value(
                name=name_hint or self.fresh_name("const"),
                type=ir.TensorType(_to_ir_dtype_from_np(arr.dtype)),
                shape=_to_ir_shape(arr.shape),
                const_value=ir.tensor(arr),
            )
            self._initializers.append(c_ir)
            return c_ir

        if var in self._var2val:
            return self._var2val[var]

        if not hasattr(var, "aval"):
            raise TypeError(f"Unsupported var type: {type(var)}")
        aval = var.aval
        aval_dtype = aval.dtype
        aval_shape = tuple(aval.shape)
        v = ir.Value(
            name=name_hint or self.fresh_name("v"),
            type=ir.TensorType(_to_ir_dtype_from_np(aval_dtype)),
            shape=_to_ir_shape(aval_shape),
        )
        self._var2val[var] = v
        return v

    def add_input_for_invar(self, var: Any, index: int) -> ir.Value:
        aval = var.aval
        val = ir.Value(
            name=f"in_{index}",
            type=ir.TensorType(_to_ir_dtype_from_np(np.dtype(aval.dtype))),
            shape=_to_ir_shape(tuple(aval.shape)),
        )
        self._var2val[var] = val
        self._inputs.append(val)

        # Track symbolic dim origins
        for ax, d in enumerate(tuple(aval.shape)):
            if not isinstance(d, (int, np.integer)):
                origin = SymbolicDimOrigin(value=val, axis=ax)
                self._symdim_origin[d] = origin
                self._symdim_origin_str[str(d)] = origin
        return val

    def get_symbolic_dim_origin(self, dim: object) -> Optional[SymbolicDimOrigin]:
        if dim in self._symdim_origin:
            return self._symdim_origin[dim]
        return self._symdim_origin_str.get(str(dim))

    def cast_like(
        self, tensor: ir.Value, exemplar: ir.Value, *, name_hint: Optional[str] = None
    ) -> ir.Value:
        out = ir.Value(
            name=self.fresh_name(name_hint or f"{tensor.name}_cast"),
            type=exemplar.type,
            shape=tensor.shape,
        )
        self.add_node(
            ir.Node(
                op_type="CastLike",
                domain="",
                inputs=[tensor, exemplar],
                outputs=[out],
                name=self.fresh_name("CastLike"),
            )
        )
        return out


@contextmanager
def _activate_plugin_worlds() -> Iterator[None]:
    # Ensure plugin registry is populated
    import_all_plugins()
    with ExitStack() as stack:
        # Legacy patches first
        stack.enter_context(apply_monkey_patches())
        # New-style leaf bindings
        for plugin_instance in PLUGIN_REGISTRY.values():
            if isinstance(plugin_instance, PrimitiveLeafPlugin):
                stack.enter_context(plugin_instance.__class__.plugin_binding())
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
) -> ir.Model:
    """
    Build an ONNX-IR model in three phases:
    1) Trace JAX to ClosedJaxpr with symbolic shapes.
    2) Lower to onnx_ir (plugins; function bodies allowed).
    3) Run a single IR-wide optimization pass (cross-node cleanups).
    """
    with _force_jax_x64(enable_double_precision):
        # 1) Abstract inputs
        default_float = _np_float_dtype(enable_double_precision)
        sds_list = _as_sds_list(inputs, enable_double_precision)

        # 2) JAXPR (optionally print for debugging)
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

        # 3) IR context & inputs/consts
        ctx: IRContext = IRContext(
            opset=opset,
            enable_double_precision=enable_double_precision,
            input_specs=sds_list,
        )
        call_param_names = set(frozen_params.keys())
        ctx._call_input_param_names = call_param_names
        ctx._call_input_param_literals = dict(frozen_params)
        # Expose knobs for downstream (optional)

        if record_primitive_calls_file:
            ctx.record_primitive_calls_file = str(record_primitive_calls_file)

        if ctx.get_function_registry() is None:
            ctx.set_function_registry(FunctionRegistry())

        # Map constvars
        for cv, cval in zip(jpr.constvars, closed.consts):
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

        # Inputs
        for i, v in enumerate(jpr.invars):
            ctx.add_input_for_invar(v, i)

        # Lower equations
        class _ConverterFacade:
            builder: IRBuilder
            ctx: IRContext

        converter = _ConverterFacade()
        converter.builder = ctx.builder
        converter.ctx = ctx

        ctx._const_folder.install_producers(jpr)

        for eqn in jpr.eqns:
            prim_name = eqn.primitive.name
            plugin_ref = PLUGIN_REGISTRY.get(prim_name)
            if plugin_ref is None:
                raise NotImplementedError(
                    f"[converter] No plugins registered for primitive '{prim_name}'"
                )
            ctx._current_eqn = eqn
            builder = ctx.builder
            prev_jax_trace = builder.current_jax_traceback
            prev_plugin_id = builder.current_plugin_identifier
            prev_plugin_line = builder.current_plugin_line
            jax_trace: Optional[str] = None
            plugin_identifier: Optional[str] = None
            plugin_line: Optional[str] = None
            if builder.stacktrace_metadata_enabled:
                try:
                    source_info = eqn.source_info
                except AttributeError:
                    source_info = None

                if source_info is not None:
                    try:
                        tb = source_info.traceback
                    except AttributeError:
                        tb = None
                    if tb is not None:
                        try:
                            jax_trace = str(tb)
                        except Exception:
                            jax_trace = None
                try:
                    if isinstance(plugin_ref, PrimitiveLowering):
                        lower_fn = plugin_ref.lower
                        try:
                            func_name = lower_fn.__name__
                        except AttributeError:
                            func_name = "lower"

                        plugin_identifier = (
                            f"{type(plugin_ref).__module__}.{type(plugin_ref).__name__}."
                            f"{func_name}"
                        )
                        try:
                            _, start_line = _ins.getsourcelines(lower_fn)
                            plugin_line = str(start_line)
                        except (OSError, TypeError):
                            plugin_line = None
                    elif isinstance(plugin_ref, FunctionLowering):
                        plugin_identifier = f"{type(plugin_ref).__module__}.{type(plugin_ref).__name__}.get_handler"
                    elif hasattr(plugin_ref, "__class__"):
                        plugin_identifier = (
                            f"{type(plugin_ref).__module__}.{type(plugin_ref).__name__}"
                        )
                    else:
                        plugin_identifier = prim_name
                except Exception:
                    plugin_identifier = prim_name
            builder.set_current_jax_traceback(jax_trace)
            builder.set_current_plugin_identifier(plugin_identifier, plugin_line)
            try:
                if isinstance(plugin_ref, PrimitiveLowering):
                    lower = plugin_ref.lower
                    try:
                        has_params = "params" in _ins.signature(lower).parameters
                    except Exception:
                        has_params = False
                    if has_params:
                        lower(ctx, eqn, eqn.params)
                    else:
                        lower(ctx, eqn)
                elif isinstance(plugin_ref, FunctionLowering):
                    handler = plugin_ref.get_handler(converter)
                    handler(converter, eqn, eqn.params)
                else:
                    raise NotImplementedError(
                        f"[converter] Unsupported plugin type for '{prim_name}'"
                    )
            finally:
                builder.set_current_jax_traceback(prev_jax_trace)
                builder.set_current_plugin_identifier(prev_plugin_id, prev_plugin_line)
        ctx._current_eqn = None

        # Outputs
        ctx.add_outputs_from_vars(jpr.outvars)

        # Build IR model
        ir_model = ctx.builder.to_ir_model(
            name=model_name,
            ir_version=_ORT_SAFE_IR_VERSION,
            protective_clone=protective_clone,
        )

        # Attach any native ir.Functions collected on ctx
        ir_funcs = list(ctx.ir_functions)
        if ir_funcs:
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
                        identifier = (
                            (fn_ir.domain or ""),
                            (fn_ir.name or ""),
                            (fn_ir.overload or ""),
                        )
                    functions_store[identifier] = fn_ir
            elif isinstance(functions_store, list):

                def _fid(func: ir.Function) -> tuple[str, str, str]:
                    return (
                        (func.domain or ""),
                        (func.name or ""),
                        (func.overload or ""),
                    )

                existing = {_fid(func) for func in functions_store}
                for fn_ir in ir_funcs:
                    func_id = _fid(fn_ir)
                    if func_id not in existing:
                        functions_store.append(fn_ir)
                        existing.add(func_id)
            else:
                ir_model.functions = list(ir_funcs)

            # Ensure model-level opset imports cover default "" and each function domain
            model_imports: Dict[str, int] = dict(ir_model.opset_imports or {})
            model_imports.setdefault("", int(ctx.builder.opset) or 21)
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

        # ---- Single IR-wide optimization pass (centralized cleanups) ----
        try:
            optimize_graph(ir_model)
        except Exception as _e:
            import logging as _logging

            _logging.getLogger("jax2onnx.converter.ir_optimizations").debug(
                "optimize_graph skipped: %s", _e
            )

        # ---- Late attribute overrides (polish; not structural rewrites) ----

        def _ir_attr_int(name: str, val: int) -> Attr:
            ival = int(val)
            try:
                return Attr.i(name, ival)
            except AttributeError:
                return Attr(name, AttributeType.INT, ival)

        def _finalize_model_value_shapes(
            model_proto: ir.Model, _ctx: IRContext
        ) -> None:
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

            def _iter_graph_values(gr: ir.Graph) -> Iterable[ir.Value]:
                seen: set[int] = set()
                staged: list[ir.Value] = []

                def _queue(values: Iterable[ir.Value]) -> None:
                    for val in values:
                        if val is None:
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

            for value in _iter_graph_values(model_proto.graph):
                _normalize_value_shape(value)

            function_container = model_proto.functions
            graph_values: Iterable[object]
            if isinstance(function_container, dict):
                graph_values = function_container.values()
            elif isinstance(function_container, Sequence):
                graph_values = function_container
            else:
                graph_values = ()

            for fn in graph_values:
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
                node.name: node for node in gr.nodes if node.name
            }

            def _make_attr(name: str, value: object) -> Optional[Attr]:
                if isinstance(value, Attr):
                    return value

                try:
                    return Attr(name, value)
                except TypeError:
                    pass
                except Exception:
                    pass

                v = value
                if isinstance(v, (bool, np.bool_, int, np.integer)):
                    return Attr(name, AttributeType.INT, int(v))
                if isinstance(v, (float, np.floating)):
                    return Attr(name, AttributeType.FLOAT, float(v))
                if isinstance(v, str):
                    return Attr(name, AttributeType.STRING, v)

                if isinstance(v, (list, tuple)):
                    if all(isinstance(x, (bool, np.bool_, int, np.integer)) for x in v):
                        return Attr(name, AttributeType.INTS, [int(x) for x in v])
                    if all(isinstance(x, (float, np.floating)) for x in v):
                        return Attr(name, AttributeType.FLOATS, [float(x) for x in v])
                    if all(isinstance(x, str) for x in v):
                        return Attr(name, AttributeType.STRINGS, list(v))

                try:
                    tensor_val = (
                        v if hasattr(v, "data_type") else ir.tensor(np.asarray(v))
                    )
                    return Attr(name, AttributeType.TENSOR, tensor_val)
                except Exception:
                    return None

            for nm, kv in (overrides or {}).items():
                node = name2node.get(nm)
                if node is None or kv is None:
                    continue
                for attr_name, attr_value in kv.items():
                    attr_obj = _make_attr(attr_name, attr_value)
                    if attr_obj is not None:
                        node.attributes[attr_name] = attr_obj

        def _fix_concat_axis_in_graph(gr: ir.Graph) -> None:
            for node in gr.all_nodes():
                if node.op_type != "Concat":
                    continue
                if "axis" in node.attributes:
                    continue
                node.attributes["axis"] = _ir_attr_int("axis", 0)

        # Apply overrides/fixes to top graph
        _apply_ir_attr_overrides_to_graph(ir_model.graph, ctx.attr_overrides)
        _fix_concat_axis_in_graph(ir_model.graph)
        # …and to all function bodies (if any)
        function_container = ir_model.functions
        if isinstance(function_container, dict):
            function_values: Iterable[ir.Function] = function_container.values()
        elif isinstance(function_container, Sequence):
            function_values = function_container
        else:
            function_values = []
        for fn in function_values:
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
            except Exception:
                pass

        # Avoid emitting placeholders for onnx_function hits across runs
        try:
            ps2._consume_onnx_function_hits()
        except AttributeError:
            pass
        except Exception:
            pass

        ir_model = run_optional_shape_inference(ir_model)

        try:
            _finalize_model_value_shapes(ir_model, ctx)
        except Exception:
            pass

        return ir_model
