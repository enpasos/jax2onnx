# jax2onnx/converter/ir_builder.py


from __future__ import annotations
import os
import traceback
from collections.abc import MutableSequence, Sequence
from typing import (
    Any,
    Final,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import numpy as np
import onnx_ir as ir
from onnx_ir import Attr, AttributeType
from onnx_ir._tape import Builder as _TapeBuilder

from .ir_clone import clone_graph
from .typing_support import SymbolicDimOrigin


STACKTRACE_METADATA_KEY: Final[str] = "pkg.jax2onnx.stacktrace"
JAX_TRACE_METADATA_KEY: Final[str] = "pkg.jax2onnx.jax_traceback"
JAX_CALLSITE_METADATA_KEY: Final[str] = "pkg.jax2onnx.callsite"
PLUGIN_METADATA_KEY: Final[str] = "pkg.jax2onnx.plugin"
_STACKTRACE_IGNORE_PATTERNS: Final[tuple[str, ...]] = (
    "jax2onnx/converter/ir_builder.py",
    "onnx_ir/_tape.py",
    "traceback.py",
)


def _dtype_to_ir(dtype: Optional[np.dtype], enable_double: bool) -> ir.DataType:
    """
    Map numpy dtype to onnx_ir.DataType.
    Floats are normalized by enable_double flag.
    """
    if dtype is None:
        return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
    key = np.dtype(dtype)
    if np.issubdtype(key, np.floating):
        if key == np.float16:
            return ir.DataType.FLOAT16
        if key == np.float32:
            return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
        if key == np.float64:
            return ir.DataType.DOUBLE
        return ir.DataType.DOUBLE if enable_double else ir.DataType.FLOAT
    try:
        return ir.DataType.from_numpy(key)
    except Exception as e:
        raise TypeError(f"Unsupported dtype: {dtype}") from e


def _value_const_numpy(value: ir.Value) -> np.ndarray[Any, np.dtype[Any]] | None:
    """Return a numpy view of `value` when backed by a constant tensor."""
    const = value.const_value
    if const is None:
        return None
    try:
        array = const.numpy()
    except Exception:
        try:
            array = np.asarray(const)
        except Exception:
            return None
    return cast(np.ndarray[Any, np.dtype[Any]], array)


class _InitializerList(MutableSequence[ir.Value]):
    """List-like view over graph initializers that preserves builder semantics."""

    def __init__(self, graph: ir.Graph):
        self._graph = graph

    def __len__(self) -> int:
        return len(self._graph.initializers)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self) -> Iterator[ir.Value]:
        return iter(self._values())

    def __contains__(self, value: object) -> bool:
        return any(existing is value for existing in self._graph.initializers.values())

    @overload
    def __getitem__(self, index: int) -> ir.Value: ...

    @overload
    def __getitem__(self, index: slice) -> list[ir.Value]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[ir.Value, list[ir.Value]]:
        values = self._values()
        return values[index]

    @overload
    def __setitem__(self, index: int, value: ir.Value) -> None: ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[ir.Value]) -> None: ...

    def __setitem__(
        self, index: Union[int, slice], value: Union[ir.Value, Iterable[ir.Value]]
    ) -> None:
        values = self._values()
        if isinstance(index, slice):
            if not isinstance(value, Iterable):
                raise TypeError("Slice assignment expects an iterable of ir.Value")
            values[index] = list(value)
        else:
            if not isinstance(value, ir.Value):
                raise TypeError("Expected ir.Value for item assignment")
            values[index] = value
        self.replace(values)

    @overload
    def __delitem__(self, index: int) -> None: ...

    @overload
    def __delitem__(self, index: slice) -> None: ...

    def __delitem__(self, index: Union[int, slice]) -> None:
        values = self._values()
        del values[index]
        self.replace(values)

    def insert(self, index: int, value: ir.Value) -> None:
        values = self._values()
        values.insert(index, value)
        self.replace(values)

    def extend(self, values: Iterable[ir.Value]) -> None:
        for value in values:
            self.append(value)

    def append(self, value: ir.Value) -> None:
        self._add(value)

    def remove(self, value: ir.Value) -> None:
        for name, existing in list(self._graph.initializers.items()):
            if existing is value:
                del self._graph.initializers[name]
                return
        raise ValueError("Initializer not found in builder")

    def clear(self) -> None:
        self._graph.initializers.clear()

    def replace(self, values: Iterable[ir.Value]) -> None:
        self.clear()
        self.extend(values)

    def copy(self) -> list[ir.Value]:
        return self._values()

    def pop(self, index: int = -1) -> ir.Value:
        values = self._values()
        value = values.pop(index)
        self.replace(values)
        return value

    def _add(self, value: ir.Value) -> None:
        name = value.name
        if name is None:
            raise ValueError("Initializer values must have a name")

        # Prefer the graph container API to register initializers. It
        # maintains graph invariants better than raw item assignment.
        # Additionally enforce a no-readd policy for duplicate names:
        # - If an initializer with the same name already exists and the
        #   constant payloads are identical, reuse the existing one and do
        #   not overwrite connections.
        # - If a different payload is attempted under the same name, raise.

        if name in self._graph.initializers:
            existing = self._graph.initializers[name]
            if existing is value:
                return

            arr_new = _value_const_numpy(value)
            arr_old = _value_const_numpy(existing)
            if arr_new is not None and arr_old is not None:
                # Normalize dtype for fair comparison: onnx_ir tensors may
                # materialize as float64 via numpy bridge.
                try:
                    arr_new_cast: np.ndarray[Any, np.dtype[Any]] = arr_new.astype(
                        arr_old.dtype, copy=False
                    )
                except Exception:
                    arr_new_cast = arr_new
                same = (
                    arr_new_cast.shape == arr_old.shape
                    and arr_new_cast.dtype == arr_old.dtype
                    and np.array_equal(arr_new_cast, arr_old)
                )
                if same:
                    # Keep using the existing initializer to preserve
                    # value connections; do not re-add.
                    return
                raise ValueError(
                    f"Initializer '{name}' already exists with different data"
                )
            # If payloads are not comparable, be conservative and reject.
            raise ValueError(
                f"Initializer '{name}' already exists and payloads are not comparable"
            )

        # No conflict: register through the container's add method.
        self._graph.initializers.add(value)

    def _values(self) -> list[ir.Value]:
        return list(self._graph.initializers.values())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{list(self)!r}"


class IRBuilder:
    """
    Minimal IR graph assembler for converter.
    Holds a mapping from jaxpr vars to ir.Values, and accumulates nodes/inputs/outputs.
    """

    def __init__(
        self,
        *,
        opset: int,
        enable_double_precision: bool,
        enable_stacktrace_metadata: bool = False,
    ):
        self.opset = opset
        self.enable_double_precision = enable_double_precision
        graph = ir.Graph(
            inputs=[],
            outputs=[],
            nodes=[],
            initializers=[],
            name="main_graph",
            opset_imports={"": self.opset},
        )
        self.graph = graph
        self._inputs: MutableSequence[ir.Value] = graph.inputs
        self._outputs: MutableSequence[ir.Value] = graph.outputs
        self._nodes: ir.Graph = graph
        self._initializers = _InitializerList(graph)
        self._tape_builder = _TapeBuilder(graph)
        self.used_opsets: set[tuple[str, int | None]] = self._tape_builder.used_opsets

        # Intermediate ValueInfo entries (propagated to ir.Graph)
        self._function_mode: bool = False
        self._var2val: dict[Any, ir.Value] = {}
        self._counters: dict[str, int] = {}
        # optional: symbolic dim origins used by some plugins
        self._sym_origin: dict[str, SymbolicDimOrigin] = {}
        self._stacktrace_metadata_enabled: bool = enable_stacktrace_metadata
        self._stacktrace_rel_base: str = os.getcwd()
        self._current_jax_traceback: Optional[str] = None
        self._current_plugin_identifier: Optional[str] = None
        self._current_plugin_line: Optional[str] = None
        self._stacktrace_detail: str = "minimal"

    @property
    def inputs(self) -> MutableSequence[ir.Value]:
        return self._inputs

    @inputs.setter
    def inputs(self, values: Iterable[ir.Value]) -> None:
        self._inputs.clear()
        self._inputs.extend(values)

    @property
    def outputs(self) -> MutableSequence[ir.Value]:
        return self._outputs

    @outputs.setter
    def outputs(self, values: Iterable[ir.Value]) -> None:
        self._outputs.clear()
        self._outputs.extend(values)

    @property
    def nodes(self) -> ir.Graph:
        return self._nodes

    @nodes.setter
    def nodes(self, values: Iterable[ir.Node]) -> None:
        existing = list(self._nodes)
        for node in existing:
            self._nodes.remove(node)
        self._nodes.extend(values)

    @property
    def initializers(self) -> _InitializerList:
        return self._initializers

    @initializers.setter
    def initializers(self, values: Iterable[ir.Value]) -> None:
        self._initializers.replace(values)

    @property
    def stacktrace_metadata_enabled(self) -> bool:
        return self._stacktrace_metadata_enabled

    @property
    def current_jax_traceback(self) -> Optional[str]:
        return self._current_jax_traceback

    @property
    def current_plugin_identifier(self) -> Optional[str]:
        return self._current_plugin_identifier

    @property
    def current_plugin_line(self) -> Optional[str]:
        return self._current_plugin_line

    def set_current_jax_traceback(self, trace: Optional[str]) -> None:
        if not self._stacktrace_metadata_enabled:
            self._current_jax_traceback = None
            return
        if trace is None:
            self._current_jax_traceback = None
            return
        cleaned = trace.strip()
        self._current_jax_traceback = cleaned if cleaned else None

    def set_current_plugin_identifier(
        self, plugin_id: Optional[str], line: Optional[str] = None
    ) -> None:
        if not self._stacktrace_metadata_enabled:
            self._current_plugin_identifier = None
            self._current_plugin_line = None
            return
        if plugin_id is None:
            self._current_plugin_identifier = None
            self._current_plugin_line = None
            return
        cleaned = plugin_id.strip()
        self._current_plugin_identifier = cleaned if cleaned else None
        line_clean = line.strip() if line and line.strip() else None
        self._current_plugin_line = line_clean

    def set_stacktrace_mode(self, mode: str) -> None:
        if mode not in {"minimal", "full"}:
            mode = "minimal"
        self._stacktrace_detail = mode

    # ---------- naming ----------
    def fresh_name(self, base: str) -> str:
        i = self._counters.get(base, 0)
        self._counters[base] = i + 1
        return f"{base}_{i}"

    # ---------- values ----------
    def _make_value(
        self, name: str, shape: Tuple[Any, ...], np_dtype: Optional[np.dtype]
    ) -> ir.Value:
        dtype_enum = _dtype_to_ir(np_dtype, self.enable_double_precision)
        return ir.Value(
            name=name, shape=ir.Shape(shape), type=ir.TensorType(dtype_enum)
        )

    # public helpers for initializers (used by FunctionPlugin)
    def add_initializer_from_scalar(self, name: str, value: Any) -> ir.Value:
        if name in self.graph.initializers:
            # Enforce duplicate policy: identical → reuse; different → error.
            existing = self.graph.initializers[name]

            arr_existing = _value_const_numpy(existing)
            arr_new = np.asarray(value)
            # Respect dtype downcast policy for floats when comparing
            if not self.enable_double_precision and np.issubdtype(
                arr_new.dtype, np.floating
            ):
                arr_new = arr_new.astype(np.float32)
            if arr_existing is not None:
                try:
                    arr_new_cast = arr_new.astype(arr_existing.dtype, copy=False)
                except Exception:
                    arr_new_cast = arr_new
                if (
                    arr_existing.shape == arr_new_cast.shape
                    and arr_existing.dtype == arr_new_cast.dtype
                    and np.array_equal(arr_existing, arr_new_cast)
                ):
                    return existing
                raise ValueError(
                    f"Initializer '{name}' already exists with different data"
                )
            # If we cannot compare payloads, be conservative and reject
            raise ValueError(
                f"Initializer '{name}' already exists and payloads are not comparable"
            )

        arr = np.asarray(value)
        if not self.enable_double_precision and np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        tensor = ir.tensor(arr)
        if self._function_mode:
            v = ir.Value(
                name=name,
                shape=ir.Shape(arr.shape if arr.shape else ()),
                type=ir.TensorType(
                    _dtype_to_ir(arr.dtype, self.enable_double_precision)
                ),
                const_value=tensor,
            )
            attributes = [
                Attr("value", AttributeType.TENSOR, tensor),
            ]
            node = ir.Node(
                op_type="Constant",
                domain="",
                inputs=[],
                outputs=[v],
                name=self.fresh_name("Constant"),
                attributes=attributes,
            )
            self.nodes.append(node)
            self._maybe_attach_stacktrace_to_nodes([node])
            return v
        v = self._tape_builder.initializer(tensor, name=name)
        return v

    def add_initializer_from_array(self, name: str, array: np.ndarray) -> ir.Value:
        return self.add_initializer_from_scalar(name, np.asarray(array))

    # convenient I64 consts for shape ops
    def const_i64(self, name: str, values: Sequence[int]) -> ir.Value:
        arr = np.asarray(values, dtype=np.int64)
        return self.add_initializer_from_array(name, arr)

    # bind graph inputs from specs
    def add_inputs_from_specs(
        self, invars: Sequence[Any], specs: Sequence[Any]
    ) -> None:
        """
        Bind jaxpr invars to graph inputs using the provided input specs.
        """
        for i, (var, spec) in enumerate(zip(invars, specs)):
            if hasattr(spec, "shape"):
                shp = tuple(spec.shape)
                dt = spec.dtype if hasattr(spec, "dtype") else None
            elif isinstance(spec, (tuple, list)):
                shp = tuple(spec)
                dt = None
            else:
                raise TypeError(f"Unsupported spec for graph input: {type(spec)}")
            v = self._make_value(
                name=f"x{i}",
                shape=shp,
                np_dtype=(np.dtype(dt) if dt is not None else None),
            )
            self._var2val[var] = v
            self.inputs.append(v)

    def get_value_for_var(
        self, var: Any, *, name_hint: Optional[str] = None
    ) -> ir.Value:
        """
        Return an ir.Value for a jaxpr var; create it from aval if needed.
        """
        if var in self._var2val:
            return self._var2val[var]
        aval = var.aval if hasattr(var, "aval") else None
        if aval is None:
            raise ValueError(f"Missing aval for var: {var}")
        shp = tuple(aval.shape)
        try:
            np_dt = np.dtype(aval.dtype)
        except Exception:
            np_dt = None
        v = self._make_value(
            name=name_hint or self.fresh_name("v"), shape=shp, np_dtype=np_dt
        )
        self._var2val[var] = v
        return v

    def add_outputs_from_vars(self, outvars: Sequence[Any]) -> None:
        for i, var in enumerate(outvars):
            v = self.get_value_for_var(var, name_hint=f"y{i}")
            self.outputs.append(v)

    # ---------- stacktrace helpers ----------
    def _iter_stacktrace_values(self, obj: Any) -> Iterable[ir.Value]:
        if isinstance(obj, ir.Value):
            yield obj
            return
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for item in obj:
                yield from self._iter_stacktrace_values(item)

    def _collect_nodes_from_result(self, result: Any) -> list[ir.Node]:
        nodes: list[ir.Node] = []
        seen: set[int] = set()
        for value in self._iter_stacktrace_values(result):
            producer = getattr(value, "producer", None)
            if not callable(producer):
                continue
            try:
                node = producer()
            except Exception:
                node = None
            if node is None:
                continue
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            nodes.append(node)
        return nodes

    def _capture_stack_frames(self) -> list[traceback.FrameSummary]:
        frames: list[traceback.FrameSummary] = list(traceback.extract_stack())
        idx = len(frames)
        while idx > 0:
            frame = frames[idx - 1]
            filename = frame.filename.replace("\\", "/")
            if any(pattern in filename for pattern in _STACKTRACE_IGNORE_PATTERNS):
                idx -= 1
                continue
            break
        return frames[:idx]

    def _format_stacktrace(
        self, frames: Sequence[traceback.FrameSummary]
    ) -> Optional[str]:
        if not frames:
            return None
        formatted: list[str] = []
        for frame in frames:
            filename = frame.filename
            if os.path.isabs(filename):
                try:
                    rel = os.path.relpath(filename, self._stacktrace_rel_base)
                except Exception:
                    rel = filename
            else:
                rel = filename
            rel = rel.replace("\\", "/")
            formatted.append(f"{rel}:{frame.lineno} in {frame.name}")
        return "\n".join(formatted)

    def _parse_callsite_entry(self, entry: str) -> tuple[Optional[str], Optional[str]]:
        candidate = entry.strip()
        if not candidate:
            return None, None
        file_part, sep, remainder = candidate.partition(":")
        if not sep:
            return None, None
        line_token = ""
        func_token = ""
        remainder = remainder.strip()
        if "(" in remainder and remainder.endswith(")"):
            before_paren, _, after_paren = remainder.partition("(")
            line_token = before_paren.strip()
            func_token = after_paren[:-1].strip()
        else:
            line_token = remainder.strip()
        digits = "".join(ch for ch in line_token if ch.isdigit())
        line_clean = digits or line_token or None
        func_clean = func_token or os.path.splitext(os.path.basename(file_part))[0]
        if not func_clean:
            func_clean = os.path.basename(file_part)
        if line_clean:
            display = f"{func_clean}:{line_clean}"
        else:
            display = func_clean
        return display, line_clean

    def _extract_callsite(
        self,
        jax_trace: Optional[str],
        frames: Sequence[traceback.FrameSummary],
    ) -> tuple[Optional[str], Optional[str]]:
        if jax_trace:
            for line in jax_trace.splitlines():
                lowered = line.lower()
                if "site-packages/jax" in lowered or "jax/_src/" in lowered:
                    continue
                display, line_no = self._parse_callsite_entry(line)
                if display:
                    return display, line_no
        for frame in reversed(frames):
            path = frame.filename.replace("\\", "/")
            lowered = path.lower()
            if "/site-packages/jax/" in lowered or "jax/_src/" in lowered:
                continue
            if "/jax2onnx/" in lowered:
                continue
            entry = f"{path}:{frame.lineno} ({frame.name})"
            display, line_no = self._parse_callsite_entry(entry)
            if display:
                return display, line_no
        return None, None

    def _extract_plugin_site(
        self, frames: Sequence[traceback.FrameSummary]
    ) -> tuple[Optional[str], Optional[str]]:
        for frame in reversed(frames):
            path = frame.filename.replace("\\", "/")
            if "/jax2onnx/plugins/" not in path:
                continue
            rel = path.split("/jax2onnx/plugins/")[-1]
            rel = rel.replace(".py", "")
            module = rel.replace("/", ".")
            func = frame.name or "lower"
            return f"{module}.{func}", str(frame.lineno)
        return None, None

    def _maybe_attach_stacktrace_to_nodes(self, nodes: Iterable[ir.Node]) -> None:
        if not self._stacktrace_metadata_enabled:
            return
        nodes_list = list(nodes)
        if not nodes_list:
            return
        frames = self._capture_stack_frames()
        detail = self._stacktrace_detail
        stack_str = self._format_stacktrace(frames) if detail == "full" else None
        jax_trace = self._current_jax_traceback
        callsite, line_no = self._extract_callsite(jax_trace, frames)
        plugin_site_name, plugin_site_line = self._extract_plugin_site(frames)
        plugin_id = self._current_plugin_identifier
        plugin_line = plugin_site_line or self._current_plugin_line or line_no
        plugin_display: Optional[str] = None
        if plugin_id:
            parts = plugin_id.split(".")
            short_plugin = ".".join(parts[-2:]) if len(parts) >= 2 else plugin_id
            if plugin_line:
                plugin_display = f"{short_plugin}:{plugin_line}"
            else:
                plugin_display = short_plugin
        elif plugin_site_name:
            plugin_display = (
                f"{plugin_site_name}:{plugin_site_line}"
                if plugin_site_line
                else plugin_site_name
            )
        if not any([stack_str, jax_trace, callsite, plugin_display]):
            return
        for node in nodes_list:
            try:
                metadata = getattr(node, "metadata_props")
            except Exception:
                metadata = None
            if metadata is None:
                continue
            try:
                if callsite and metadata.get(JAX_CALLSITE_METADATA_KEY) is None:
                    metadata[JAX_CALLSITE_METADATA_KEY] = callsite
                if plugin_display and metadata.get(PLUGIN_METADATA_KEY) is None:
                    metadata[PLUGIN_METADATA_KEY] = plugin_display
                if detail == "full":
                    if stack_str and STACKTRACE_METADATA_KEY not in metadata:
                        metadata[STACKTRACE_METADATA_KEY] = stack_str
                    if jax_trace and JAX_TRACE_METADATA_KEY not in metadata:
                        metadata[JAX_TRACE_METADATA_KEY] = jax_trace
            except Exception:
                continue

    def _maybe_attach_stacktrace_from_result(self, result: Any) -> None:
        if not self._stacktrace_metadata_enabled:
            return
        nodes = self._collect_nodes_from_result(result)
        self._maybe_attach_stacktrace_to_nodes(nodes)

    # ---------- nodes ----------
    def add_node_obj(self, node: ir.Node) -> None:
        self.nodes.append(node)
        self._maybe_attach_stacktrace_to_nodes([node])

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value],
        outputs: Sequence[ir.Value],
        attributes: Optional[list[ir.Attr]] = None,
        name: Optional[str] = None,
    ) -> ir.Node:
        n = ir.Node(
            op_type=op_type,
            domain="",
            inputs=list(inputs),
            outputs=list(outputs),
            name=name or self.fresh_name(op_type),
            attributes=(attributes or []),
        )
        self.nodes.append(n)
        self._maybe_attach_stacktrace_to_nodes([n])
        return n

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            )
        tape_builder = object.__getattribute__(self, "_tape_builder")
        try:
            attr = object.__getattribute__(tape_builder, name)
        except AttributeError as err:
            try:
                getattr_hook = object.__getattribute__(
                    type(tape_builder), "__getattr__"
                )
            except AttributeError:
                getattr_hook = None
            if getattr_hook is None:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute {name!r}"
                ) from err
            attr = getattr_hook(tape_builder, name)
        if callable(attr):

            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                result = attr(*args, **kwargs)
                self._maybe_attach_stacktrace_from_result(result)
                return result

            return _wrapped
        return attr

    # ---------- symbolic dim origin ----------
    def record_symbol_origin(self, sym: str, src_val: ir.Value, axis: int) -> None:
        self._sym_origin[sym] = SymbolicDimOrigin(value=src_val, axis=axis)

    def get_symbolic_dim_origin(self, sym: str) -> Optional[SymbolicDimOrigin]:
        return self._sym_origin.get(sym)

    def to_ir_model(
        self,
        *,
        name: str,
        ir_version: int = 11,
        protective_clone: bool = True,
    ) -> ir.Model:
        graph = clone_graph(self.graph) if protective_clone else self.graph
        if name:
            graph.name = name
        return ir.Model(graph, ir_version=ir_version, producer_name="jax2onnx")
