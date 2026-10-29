# jax2onnx/plugins/flax/nnx/_source_function_state.py

"""Public NNX state lifting for the internal SOURCE Function binder."""

from __future__ import annotations

from dataclasses import Field, dataclass, field, fields, replace
import inspect
import marshal
from types import (
    CellType,
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    ModuleType,
)
from typing import Any, Callable, Mapping, cast

from flax import errors, nnx
from flax.core import frozen_dict as flax_frozen_dict
from flax.ids import FlaxId
from flax.linen import module as linen_module
from flax.nnx import graph as nnx_graph, pytreelib
from flax.nnx.bridge import wrappers as nnx_wrappers
from flax.nnx.nn import dtypes as nnx_dtypes
from flax.typing import HashableMapping
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jax2onnx._compat.jax import ShapedArray, Tracer
from jax2onnx.converter.source_profile import SourcePrecisionError
from jax2onnx.plugins.flax.nnx._source_callback import (
    CallbackCapture,
    CallbackGraph,
    CallbackToken,
)
from jax2onnx.plugins.flax.nnx._source_linen_declaration import (
    is_static_linen_declaration,
)


def _failure(message: str, *, defect: bool = False) -> SourcePrecisionError:
    return SourcePrecisionError(
        message,
        phase="profile-validation",
        category="lowering-defect" if defect else "profile-exclusion",
        location="NNX Function state",
        requirement="fixed NNX topology and explicitly carried numeric state owned by the current trace",
        action="prepare a fresh NNX owner with split/merge inside the enclosing traced call",
    )


_NNX_GRAPH_SIGNATURES: tuple[
    tuple[tuple[str, inspect._ParameterKind, object], ...], ...
] = tuple(
    tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    )
    for function in (nnx.split, nnx.iter_graph)
)


def _nnx_graph_protocol() -> bool:
    """Select graph-only or explicit graph mode from exact native API shapes."""
    parameter = inspect.Parameter
    if not (
        type(_NNX_GRAPH_SIGNATURES) is tuple
        and len(_NNX_GRAPH_SIGNATURES) == 2
        and all(type(signature) is tuple for signature in _NNX_GRAPH_SIGNATURES)
        and all(
            type(item) is tuple
            and len(item) == 3
            and type(item[0]) is str
            and any(
                item[1] is kind
                for kind in (
                    parameter.POSITIONAL_ONLY,
                    parameter.POSITIONAL_OR_KEYWORD,
                    parameter.VAR_POSITIONAL,
                    parameter.KEYWORD_ONLY,
                )
            )
            and (item[2] is parameter.empty or item[2] is None)
            for signature in _NNX_GRAPH_SIGNATURES
            for item in signature
        )
    ):
        raise _failure("NNX graph API schema needs an exact adapter", defect=True)
    split = (
        ("node", parameter.POSITIONAL_OR_KEYWORD, parameter.empty),
        ("filters", parameter.VAR_POSITIONAL, parameter.empty),
    )
    iteration = (("node", parameter.POSITIONAL_ONLY, parameter.empty),)
    graph = ("graph", parameter.KEYWORD_ONLY, None)
    if _NNX_GRAPH_SIGNATURES == (split, iteration):
        return False
    if _NNX_GRAPH_SIGNATURES == ((*split, graph), (*iteration, graph)):
        return True
    raise _failure("NNX graph API schema needs an exact adapter", defect=True)


def _split_graph(owner: Any, *, expected_callbacks: CallbackGraph | None = None) -> Any:
    _preflight_native_storage(owner, expected_callbacks=expected_callbacks)
    if _nnx_graph_protocol():
        return nnx.split(owner, graph=True)
    return nnx.split(owner)


def _iter_graph(owner: Any, *, expected_callbacks: CallbackGraph | None = None) -> Any:
    _preflight_native_storage(owner, expected_callbacks=expected_callbacks)
    if _nnx_graph_protocol():
        return nnx.iter_graph(owner, graph=True)
    return nnx.iter_graph(owner)


def _nnx_variable_roles() -> tuple[str, str]:
    fields = nnx.Variable.required_metadata
    if type(fields) is frozenset and all(type(name) is str for name in fields):
        if fields == frozenset({"has_ref", "is_hijax", "is_mutable", "eager_sharding"}):
            return "has_ref", "is_hijax"
        if fields == frozenset({"ref", "hijax", "eager_sharding"}):
            return "ref", "hijax"
    raise _failure("NNX Variable metadata schema needs an exact adapter", defect=True)


def _require_ordinary_variable_metadata(metadata: Any) -> None:
    roles = _nnx_variable_roles()
    if not (
        type(metadata) is dict
        and all(type(name) is str for name in metadata)
        and nnx.Variable.required_metadata <= metadata.keys()
        and all(type(metadata[role]) is bool for role in roles)
    ):
        raise _failure("NNX Variable metadata is malformed", defect=True)
    if any(metadata[role] for role in roles):
        raise _failure("Mutable NNX Function state requires ordinary non-Ref Variables")


@dataclass(frozen=True)
class _StateSpec:
    shape: tuple[Any, ...]
    dtype: Any
    weak_type: bool

    @classmethod
    def of(cls, value: Any) -> _StateSpec:
        aval = jax.typeof(value)
        if not isinstance(aval, ShapedArray):
            raise _failure(
                "NNX Function state must contain ordinary numeric JAX values"
            )
        return cls(tuple(aval.shape), aval.dtype, bool(aval.weak_type))


@dataclass(frozen=True)
class _StateEntry:
    path: tuple[Any, ...]
    tree: jtu.PyTreeDef
    start: int
    stop: int


def _foreign_failure(message: str) -> SourcePrecisionError:
    return _failure(
        f"NNX foreign numeric carrier support remains unfinished: {message}",
        defect=True,
    )


def _static_failure(message: str) -> SourcePrecisionError:
    return _failure(
        f"NNX mutable static payload support remains unfinished: {message}",
        defect=True,
    )


def _capture_callback(
    capture: CallbackCapture, value: FunctionType, path: str
) -> CallbackToken:
    try:
        return capture.capture(value, path)
    except SourcePrecisionError as exc:
        raise _static_failure(
            f"GraphDef.Static payload requires a supported callback adapter: {exc.detail}"
        ) from exc


@dataclass(frozen=True)
class _NativeStaticAnchor:
    kind: str
    name: str


# These are trusted process-global framework anchors, never user callables or
# model-owned objects. Identity snapshots avoid ScalarMeta equality and hashing.
_DTYPE_NAMES: tuple[str, ...] = (
    "bool_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
_DTYPE_ANCHORS: tuple[tuple[ModuleType, str, Any, tuple[tuple[str, Any], ...]], ...] = (
    tuple(
        (
            module,
            name,
            getattr(module, name),
            tuple(vars(getattr(module, name)).items()),
        )
        for module in (np, jnp)
        for name in (*_DTYPE_NAMES, *(("bfloat16",) if module is jnp else ()))
    )
)
_NATIVE_DTYPES: tuple[np.dtype[Any], ...] = tuple(
    np.dtype(getattr(jnp, name)) for name in (*_DTYPE_NAMES, "bfloat16")
)
_BFLOAT16_TYPE: type[Any] = np.dtype(jnp.bfloat16).type
_FUNCTION_ANCHORS: tuple[
    tuple[
        ModuleType,
        str,
        Any,
        CodeType,
        tuple[Any, ...] | None,
        tuple[tuple[str, Any], ...],
    ],
    ...,
] = tuple(
    (
        module,
        name,
        function,
        function.__code__,
        function.__defaults__,
        tuple((function.__kwdefaults__ or {}).items()),
    )
    for module, name, function in (
        (jax.lax, "dot_general", jax.lax.dot_general),
        (jax.lax, "conv_general_dilated", jax.lax.conv_general_dilated),
        (nnx_dtypes, "promote_dtype", nnx_dtypes.promote_dtype),
    )
)
_PRECISION_CLASS: type[Any] = jax.lax.Precision
_PRECISION_ANCHORS: tuple[tuple[str, Any, tuple[tuple[str, Any], ...]], ...] = tuple(
    (
        name,
        getattr(_PRECISION_CLASS, name),
        tuple(vars(getattr(_PRECISION_CLASS, name)).items()),
    )
    for name in ("DEFAULT", "HIGH", "HIGHEST")
)
_NATIVE_RNGS: type[Any] = nnx.Rngs
_NATIVE_TONNX: type[Any] = nnx_wrappers.ToNNX
_NATIVE_LINEAR_GENERAL: type[Any] = nnx.LinearGeneral
_NATIVE_FROZEN_DICT: type[Any] = flax_frozen_dict.FrozenDict
_NATIVE_FALLBACKS: tuple[
    tuple[type[Any], Any, CodeType, tuple[CellType, ...] | None, tuple[Any, ...]], ...
] = tuple(
    (
        owner_type,
        callback,
        callback.__code__,
        callback.__closure__,
        tuple(cell.cell_contents for cell in callback.__closure__ or ()),
    )
    for owner_type in (_NATIVE_RNGS, _NATIVE_TONNX)
    for callback in (inspect.getattr_static(owner_type, "__getattr__"),)
)
_FROZEN_DICT_SLOTS: tuple[tuple[str, Any], ...] = tuple(
    (name, inspect.getattr_static(_NATIVE_FROZEN_DICT, name))
    for name in ("_dict", "_hash")
)
_FROZEN_DICT_INIT: Any = _NATIVE_FROZEN_DICT.__init__
_FROZEN_DICT_INIT_CODE: CodeType = _FROZEN_DICT_INIT.__code__
_FROZEN_DICT_NEW: Any = inspect.getattr_static(_NATIVE_FROZEN_DICT, "__new__")


def _same_native_items(actual: Any, expected: tuple[tuple[str, Any], ...]) -> bool:
    return (
        type(actual) is dict
        and len(actual) == len(expected)
        and all(type(key) is str for key in actual)
        and all(key in actual and actual[key] is item for key, item in expected)
    )


def _native_anchor(
    value: Any, owner_type: Any = None, name: str | None = None
) -> _NativeStaticAnchor | None:
    for index, (module, attribute, anchor, snapshot) in enumerate(_DTYPE_ANCHORS):
        if value is anchor:
            if getattr(module, attribute) is not anchor or not _same_native_items(
                dict(vars(anchor)), snapshot
            ):
                raise _static_failure("native dtype constructor changed")
            return _NativeStaticAnchor("dtype-constructor", str(index))
    if value is _BFLOAT16_TYPE:
        return _NativeStaticAnchor("bfloat16-constructor", "bfloat16")
    if any(type(value) is type(dtype) for dtype in _NATIVE_DTYPES):
        if (
            value.metadata is None
            and value.fields is None
            and value.subdtype is None
            and not value.hasobject
        ):
            for index, dtype in enumerate(_NATIVE_DTYPES):
                if value.str == dtype.str:
                    return _NativeStaticAnchor("numpy-dtype", str(index))
        raise _static_failure("noncanonical dtype configuration needs an adapter")
    for attribute, anchor, snapshot in _PRECISION_ANCHORS:
        if value is anchor:
            if jax.lax.Precision is not _PRECISION_CLASS or not _same_native_items(
                vars(anchor), snapshot
            ):
                raise _static_failure("native Precision configuration changed")
            return _NativeStaticAnchor("precision", attribute)
    for (
        module,
        attribute,
        anchor,
        code,
        defaults,
        keyword_defaults,
    ) in _FUNCTION_ANCHORS:
        if value is not anchor:
            continue
        if attribute == "dot_general":
            valid_role = name == attribute and any(
                owner_type is native for native in (nnx.Linear, nnx.LinearGeneral)
            )
        elif attribute == "conv_general_dilated":
            valid_role = owner_type is nnx.Conv and name == attribute
        else:
            valid_role = (
                name == "promote_dtype"
                and any(
                    owner_type is native
                    for native in (
                        nnx.Linear,
                        nnx.LinearGeneral,
                        nnx.BatchNorm,
                        nnx.Embed,
                        nnx.LayerNorm,
                        nnx.Conv,
                    )
                )
            ) or (
                owner_type is nnx.MultiHeadAttention
                and name
                in ("qkv_promote_dtype", "out_promote_dtype", "ln_promote_dtype")
            )
        if not valid_role:
            raise _static_failure(
                "GraphDef.Static payload requires the exact native callable field role"
            )
        actual_keywords = anchor.__kwdefaults__
        if actual_keywords is None:
            actual_keywords = {}
        if (
            type(anchor) is not FunctionType
            or getattr(module, attribute) is not anchor
            or anchor.__code__ is not code
            or anchor.__defaults__ is not defaults
            or anchor.__closure__ is not None
            or type(anchor.__dict__) is not dict
            or anchor.__dict__
            or not _same_native_items(actual_keywords, keyword_defaults)
        ):
            raise _static_failure("native callable configuration changed")
        _require_inert_static(defaults, "native callable defaults")
        for _, item in keyword_defaults:
            _require_inert_static(item, "native callable defaults")
        return _NativeStaticAnchor("callable", attribute)
    return None


def _resolve_native_anchor(
    value: _NativeStaticAnchor, owner_type: Any, name: str
) -> Any:
    if value.kind == "dtype-constructor":
        candidates = tuple(item[2] for item in _DTYPE_ANCHORS)
    elif value.kind == "numpy-dtype":
        candidates = _NATIVE_DTYPES
    elif value.kind == "bfloat16-constructor" and value.name == "bfloat16":
        return _BFLOAT16_TYPE
    elif value.kind == "precision":
        candidates = tuple(
            item[1] for item in _PRECISION_ANCHORS if item[0] == value.name
        )
        if len(candidates) == 1 and _native_anchor(candidates[0]) == value:
            return candidates[0]
        raise _static_failure("native Precision anchor changed")
    elif value.kind == "callable":
        candidates = tuple(
            item[2] for item in _FUNCTION_ANCHORS if item[1] == value.name
        )
        if (
            len(candidates) == 1
            and _native_anchor(candidates[0], owner_type, name) == value
        ):
            return candidates[0]
        raise _static_failure("native callable anchor changed")
    else:
        raise _static_failure("unknown native static anchor")
    if not value.name.isdecimal() or not 0 <= int(value.name) < len(candidates):
        raise _static_failure("malformed native dtype anchor")
    result = candidates[int(value.name)]
    if _native_anchor(result) != value:
        raise _static_failure("native dtype anchor changed")
    return result


def _map_native_anchors(
    value: Any, owner_type: Any, name: str, *, resolve: bool = False
) -> Any:
    if resolve and type(value) is _NativeStaticAnchor:
        return _resolve_native_anchor(value, owner_type, name)
    if not resolve:
        anchor = _native_anchor(value, owner_type, name)
        if anchor is not None:
            return anchor
    if type(value) is tuple or type(value) is frozenset:
        items = tuple(
            _map_native_anchors(item, None, "", resolve=resolve) for item in value
        )
        if all(before is after for before, after in zip(value, items, strict=True)):
            return value
        return items if type(value) is tuple else frozenset(items)
    return value


def _require_inert_static(value: Any, location: str) -> None:
    """Accept a closed value grammar, without hashing or invoking user hooks.

    An outer frozen/hashable object says nothing about the objects it retains.
    Ordinary dataclasses, arbitrary mappings and callables need their own state
    adapter. Exact literal scalars, recursively inert tuples/frozensets and
    the finite checked native global anchors can be shared with a fresh owner.
    """
    if type(value) in {type(None), bool, int, float, complex, str, bytes}:
        return
    if type(value) in {tuple, frozenset}:
        for item in value:
            _require_inert_static(item, location)
        return
    if _native_anchor(value) is not None:
        return
    raise _static_failure(f"{location} requires an immutable-payload adapter")


def _native_mapping_items(value: Any, location: str) -> tuple[tuple[Any, Any], ...]:
    if type(value) is not HashableMapping:
        raise _static_failure(f"{location} native mapping schema needs an adapter")
    raw = object.__getattribute__(value, "__dict__")
    if (
        type(raw) is not dict
        or any(type(key) is not str for key in raw)
        or tuple(raw) != ("_mapping",)
    ):
        raise _static_failure(f"{location} native mapping storage needs an adapter")
    mapping = raw["_mapping"]
    if type(mapping) is not dict or any(type(key) not in {str, int} for key in mapping):
        raise _static_failure(f"{location} native mapping entries need an adapter")
    return tuple(mapping.items())


def _require_safe_variable_metadata(metadata: Any) -> None:
    _require_ordinary_variable_metadata(metadata)
    for value in metadata.values():
        _require_inert_static(value, "Variable metadata")


def _require_native_variable_type(owner_type: type[Any]) -> None:
    for name in (
        "__getattribute__",
        "__getattr__",
        "get_raw_value",
        "get_metadata",
        "var_type",
        "_new",
        "from_metadata",
        "_raw_value",
        "_var_metadata",
    ):
        if inspect.getattr_static(owner_type, name, None) is not inspect.getattr_static(
            nnx.Variable, name, None
        ):
            raise _static_failure("custom Variable storage/hooks need an adapter")


def _require_native_graph_type(owner_type: type[Any]) -> None:
    if inspect.getattr_static(
        owner_type, "__getattribute__"
    ) is not inspect.getattr_static(pytreelib.Pytree, "__getattribute__"):
        raise _static_failure("custom NNX attribute access needs an adapter")
    fallback = inspect.getattr_static(owner_type, "__getattr__", None)
    if fallback is not None:
        role = next((item for item in _NATIVE_FALLBACKS if owner_type is item[0]), None)
        if (
            role is None
            or fallback is not role[1]
            or type(fallback) is not FunctionType
            or fallback.__code__ is not role[2]
            or fallback.__closure__ is not role[3]
            or any(
                cell.cell_contents is not expected
                for cell, expected in zip(
                    fallback.__closure__ or (), role[4], strict=True
                )
            )
            or fallback.__defaults__ is not None
            or fallback.__kwdefaults__ is not None
            or type(fallback.__dict__) is not dict
            or fallback.__dict__
            or nnx.Rngs is not _NATIVE_RNGS
            or nnx_wrappers.ToNNX is not _NATIVE_TONNX
        ):
            raise _static_failure("custom NNX attribute fallback needs an adapter")
    if type(inspect.getattr_static(owner_type, "__dict__")) is not GetSetDescriptorType:
        raise _static_failure("custom NNX storage descriptor needs an adapter")
    implementation = nnx_graph.get_node_impl_for_type(owner_type)
    if type(implementation) is not nnx_graph.GraphNodeImpl:
        raise _static_failure("custom NNX graph registration needs an adapter")
    for callback, name in (
        ("flatten", "_graph_node_flatten"),
        ("create_empty", "_graph_node_create_empty"),
        ("init", "_graph_node_init"),
        ("set_key", "_graph_node_set_key"),
        ("pop_key", "_graph_node_pop_key"),
        ("clear", "_graph_node_clear"),
    ):
        native_type = (
            nnx.List
            if owner_type is nnx.List and callback == "set_key"
            else pytreelib.Pytree
        )
        if inspect.getattr_static(owner_type, name) is not inspect.getattr_static(
            native_type, name
        ) or getattr(implementation, callback) is not getattr(native_type, name):
            raise _static_failure("custom NNX graph callbacks need an adapter")


def _require_safe_tree(tree: jtu.PyTreeDef) -> None:
    """Validate aux data retained by the supported native reconstruction hooks."""
    data = tree.node_data()
    if data is not None:
        node_type, auxiliary = data
        if node_type is dict:
            # JAX's native dict node stores its keys in a list, which it does
            # not return to the reconstructed dict. Its entries remain static.
            if type(auxiliary) not in {list, tuple}:
                raise _static_failure("dict PyTree auxiliary schema needs an adapter")
            for key in auxiliary:
                _require_inert_static(key, "dict PyTree key")
        elif any(node_type is native for native in (list, tuple, type(None))):
            if auxiliary is not None:
                raise _static_failure(
                    "container PyTree auxiliary schema needs an adapter"
                )
        elif node_type is pytreelib.PytreeState:
            if not (
                type(auxiliary) is tuple
                and len(auxiliary) == 2
                and all(type(flag) is bool for flag in auxiliary)
            ):
                raise _static_failure("native PyTree state schema needs an adapter")
        elif isinstance(node_type, type) and issubclass(node_type, nnx.Variable):
            _require_native_variable_type(node_type)
            _require_inert_static(auxiliary, "Variable PyTree metadata")
        else:
            _require_inert_static(auxiliary, "PyTree auxiliary metadata")
            raise _static_failure("custom PyTree reconstruction hooks need an adapter")
    for child in tree.children():
        _require_safe_tree(child)


def _graph_attribute_owners(graphdef: Any) -> tuple[Any, ...]:
    """Bind native preorder attributes to their actual containing node types."""
    if type(graphdef) is not nnx.GraphDef or tuple(
        item.name for item in fields(graphdef)
    ) != ("nodes", "attributes", "num_leaves"):
        raise _static_failure("native GraphDef schema needs an exact adapter")
    if type(graphdef.nodes) is not list or type(graphdef.attributes) is not list:
        raise _static_failure("native GraphDef outer storage needs an adapter")
    for item in graphdef.attributes:
        if type(item) is not tuple or len(item) != 2:
            raise _static_failure("native GraphDef attribute schema needs an adapter")
        key, attribute = item
        if type(key) not in {str, int} or type(attribute) not in {
            nnx_graph.Static,
            nnx_graph.NodeAttr,
            nnx_graph.LeafAttr,
        }:
            raise _static_failure("native GraphDef attribute schema needs an adapter")
    owners = []
    node_index = attribute_index = 0

    def consume() -> None:
        nonlocal node_index, attribute_index
        if node_index >= len(graphdef.nodes):
            raise _static_failure("native GraphDef node layout changed")
        node = graphdef.nodes[node_index]
        node_index += 1
        if type(node) is nnx_graph.VariableDef or type(node) is nnx_graph.NodeRef:
            return
        if (
            type(node) is not nnx_graph.NodeDef
            or type(node.num_attributes) is not int
            or node.num_attributes < 0
        ):
            raise _static_failure("native GraphDef node layout needs an adapter")
        for _ in range(node.num_attributes):
            if attribute_index >= len(graphdef.attributes):
                raise _static_failure("native GraphDef attribute layout changed")
            attribute = graphdef.attributes[attribute_index][1]
            owners.append(node.type)
            attribute_index += 1
            if type(attribute) is nnx_graph.NodeAttr:
                consume()

    consume()
    if node_index != len(graphdef.nodes) or attribute_index != len(graphdef.attributes):
        raise _static_failure("native GraphDef layout has unconsumed entries")
    return tuple(owners)


def _require_graph_metadata(graphdef: Any) -> None:
    _graph_attribute_owners(graphdef)
    for node in graphdef.nodes:
        if type(node) is nnx_graph.VariableDef:
            _require_native_variable_type(node.type)
            metadata = dict(_native_mapping_items(node.metadata, "Variable metadata"))
            _require_safe_variable_metadata(metadata)
        elif type(node) is nnx_graph.NodeDef:
            if node.type is nnx_graph.GenericPytree:
                metadata = node.metadata
                if type(metadata) is not nnx_graph.IndexesPytreeDef:
                    raise _static_failure("native PyTree graph schema needs an adapter")
                for _, index in _native_mapping_items(
                    metadata.key_index, "PyTree key indices"
                ):
                    if type(index) is not int:
                        raise _static_failure("native PyTree indices need an adapter")
                _require_safe_tree(metadata.treedef)
            elif (
                isinstance(node.type, type)
                and issubclass(node.type, pytreelib.Pytree)
                and node.metadata is node.type
            ):
                # Native graph mode retains the class as construction metadata,
                # rather than treating it as a user-provided static payload.
                _require_native_graph_type(node.type)
            elif any(node.type is native for native in (list, tuple, dict, type(None))):
                _require_inert_static(node.metadata, "native container metadata")
            else:
                raise _static_failure("custom NNX graph metadata needs an adapter")
        elif type(node) is not nnx_graph.NodeRef:
            raise _static_failure("nonordinary NNX graph nodes need an adapter")


@dataclass(frozen=True)
class _NativeNodeMapping:
    """Inert snapshot of NNX's mutable field-classification envelope."""

    items: tuple[tuple[str, bool], ...]


def _node_mapping(value: Any) -> _NativeNodeMapping:
    items = _native_mapping_items(value, "NNX field classification")
    if any(type(key) is not str or type(flag) is not bool for key, flag in items):
        raise _static_failure("NNX field classification values need an adapter")
    return _NativeNodeMapping(items)


@dataclass(frozen=True)
class _BatchAxes:
    index: int
    items: tuple[tuple[int, int], ...]


def _require_batch_axes_role(owner_type: Any, name: str | None) -> None:
    if (
        owner_type is not _NATIVE_LINEAR_GENERAL
        or name != "batch_axis"
        or nnx.LinearGeneral is not _NATIVE_LINEAR_GENERAL
    ):
        raise _static_failure(
            "FrozenDict requires the native LinearGeneral batch_axis role"
        )


def _require_frozen_dict_schema() -> None:
    if (
        flax_frozen_dict.FrozenDict is not _NATIVE_FROZEN_DICT
        or _NATIVE_FROZEN_DICT.__init__ is not _FROZEN_DICT_INIT
        or _FROZEN_DICT_INIT.__code__ is not _FROZEN_DICT_INIT_CODE
        or inspect.getattr_static(_NATIVE_FROZEN_DICT, "__new__")
        is not _FROZEN_DICT_NEW
        or any(
            type(descriptor) is not MemberDescriptorType
            or inspect.getattr_static(_NATIVE_FROZEN_DICT, name) is not descriptor
            for name, descriptor in _FROZEN_DICT_SLOTS
        )
    ):
        raise _static_failure("native FrozenDict storage/constructor schema changed")


def _require_batch_axes_items(items: Any) -> None:
    if type(items) is not tuple or any(
        type(item) is not tuple
        or len(item) != 2
        or type(item[0]) is not int
        or type(item[1]) is not int
        for item in items
    ):
        raise _static_failure(
            "LinearGeneral batch_axis needs exact integer keys and sizes"
        )
    if len({key for key, _ in items}) != len(items):
        raise _static_failure("LinearGeneral batch_axis has duplicate keys")


def _snapshot_batch_axes(value: Any, index: int) -> _BatchAxes:
    _require_frozen_dict_schema()
    if type(value) is not _NATIVE_FROZEN_DICT:
        raise _static_failure(
            "LinearGeneral batch_axis needs an exact native FrozenDict"
        )
    mapping = object.__getattribute__(value, "_dict")
    cache = object.__getattribute__(value, "_hash")
    if type(mapping) is not dict or (cache is not None and type(cache) is not int):
        raise _static_failure(
            "LinearGeneral batch_axis native raw storage needs an adapter"
        )
    items = tuple(mapping.items())
    _require_batch_axes_items(items)
    return _BatchAxes(index, items)


def _preflight_native_storage(
    owner: Any, *, expected_callbacks: CallbackGraph | None = None
) -> None:
    """Inspect supported raw storage before native split can run flatten hooks."""
    visited: set[int] = set()
    active_containers: set[int] = set()
    callbacks = CallbackCapture(expected=expected_callbacks)

    def visit(
        value: Any,
        *,
        static: bool = False,
        containing_type: Any = None,
        field_name: str | None = None,
    ) -> None:
        owner_type = type(value)
        if containing_type is _NATIVE_LINEAR_GENERAL and field_name == "batch_axis":
            _require_batch_axes_role(containing_type, field_name)
            _snapshot_batch_axes(value, 0)
            return
        if any(
            owner_type is literal
            for literal in (type(None), bool, int, float, complex, str, bytes)
        ):
            return
        if _native_anchor(value, containing_type, field_name) is not None:
            return
        if owner_type is FunctionType:
            _capture_callback(callbacks, value, f"NNX field {field_name!r}")
            return
        # Audit ordinary dataclass storage before native isinstance/flatten
        # probes can consult a hostile instance's __class__ accessor.
        if (
            _carrier_class_attribute(
                owner_type, "__dataclass_fields__", _CARRIER_MISSING
            )
            is not _CARRIER_MISSING
            and not any(
                base is nnx.Module or base is linen_module.Module
                for base in _carrier_mro(owner_type)
            )
            and _carrier_fields(value) is not None
        ):
            return
        if isinstance(value, nnx.Variable):
            if id(value) in visited:
                return
            visited.add(id(value))
            _require_native_variable_type(owner_type)
            metadata = object.__getattribute__(value, "_var_metadata")
            _require_safe_variable_metadata(metadata)
            visit(object.__getattribute__(value, "_raw_value"))
            return
        if isinstance(value, pytreelib.Pytree):
            if id(value) in visited:
                return
            visited.add(id(value))
            _require_native_graph_type(owner_type)
            raw = object.__getattribute__(value, "__dict__")
            if type(raw) is not dict or any(type(key) is not str for key in raw):
                raise _static_failure("native NNX raw storage needs an adapter")
            if owner_type is _NATIVE_TONNX:
                if not is_static_linen_declaration(
                    raw.get("to_nnx__module"), failure=_static_failure
                ):
                    raise _static_failure(
                        "ToNNX needs a pristine unbound Linen declaration"
                    )
                if "to_nnx__rngs" not in raw or (
                    raw["to_nnx__rngs"] is not None
                    and type(raw["to_nnx__rngs"]) is not _NATIVE_RNGS
                ):
                    raise _static_failure(
                        "ToNNX needs None or exact native Rngs storage"
                    )
            classification = _node_mapping(raw.get("_pytree__nodes"))
            node_fields = dict(classification.items)
            pytree = inspect.getattr_static(owner_type, "_pytree__is_pytree")
            if type(pytree) is not bool:
                raise _static_failure(
                    "native NNX field classification needs an adapter"
                )
            for name, item in raw.items():
                if name == "_pytree__nodes":
                    continue
                visit(
                    item,
                    static=pytree and not node_fields.get(name, False),
                    containing_type=owner_type,
                    field_name=name,
                )
            return
        if owner_type is pytreelib.PytreeState:
            for name in ("_initializing", "_is_setup"):
                if type(object.__getattribute__(value, name)) is not bool:
                    raise _static_failure("native PyTree state schema needs an adapter")
            return
        if isinstance(value, np.ndarray):
            raise _static_failure(
                "mutable native numeric buffers need isolation/effect support"
            )
        if _numeric_carrier_field(value):
            if static:
                raise _static_failure(
                    "static numeric payloads need explicit state ports"
                )
            return
        if static:
            if _carrier_fields(value) is not None:
                return
            if is_static_linen_declaration(value, failure=_static_failure):
                return
            _require_inert_static(value, "GraphDef.Static payload")
            return
        if owner_type is dict or owner_type is list or owner_type is tuple:
            if id(value) in active_containers:
                raise _static_failure("cyclic state containers need an adapter")
            active_containers.add(id(value))
            if owner_type is dict:
                if any(type(key) not in {str, int} for key in value):
                    raise _static_failure("native container keys need an adapter")
                items = value.values()
            else:
                items = value
            for item in items:
                visit(item)
            active_containers.remove(id(value))
            return
        if nnx_graph.get_node_impl(value) is not None:
            raise _static_failure(
                "custom PyTree flatten/reconstruction hooks need an adapter"
            )
        if _carrier_fields(value) is not None:
            return
        if is_static_linen_declaration(value, failure=_static_failure):
            return
        _require_inert_static(value, "GraphDef.Static payload")

    visit(owner)


@dataclass(frozen=True)
class _CarrierToken:
    index: int


@dataclass(frozen=True)
class _StaticField:
    value_type: type[Any]
    fingerprint: bytes
    value: Any = field(compare=False, hash=False)


@dataclass(frozen=True)
class _LinenDeclaration:
    index: int
    declaration_type: type[Any]
    config: tuple[tuple[str, _StaticField], ...]
    identifier: int


def _snapshot_linen(value: Any, index: int) -> _LinenDeclaration:
    raw = object.__getattribute__(value, "__dict__")
    identifier = object.__getattribute__(raw["_id"], "__dict__")
    if (
        type(identifier) is not dict
        or tuple(identifier) != ("id",)
        or type(identifier["id"]) is not int
    ):
        raise _static_failure("native Linen identity storage needs an adapter")
    return _LinenDeclaration(
        index,
        type(value),
        tuple(
            (item.name, _static_carrier_field(raw[item.name]))
            for item in fields(value)
            if item.name != "parent"
        ),
        identifier["id"],
    )


def _reconstruct_linen(value: _LinenDeclaration) -> Any:
    # Direct native storage avoids user constructors, setup and copying hooks.
    if (
        type(inspect.getattr_static(value.declaration_type, "__dict__"))
        is not GetSetDescriptorType
    ):
        raise _static_failure("Linen declaration storage descriptor needs an adapter")
    declaration = object.__new__(value.declaration_type)
    raw = object.__getattribute__(declaration, "__dict__")
    raw.update({name: item.value for name, item in value.config})
    raw.update(
        _parent_ref=None,
        _id=FlaxId(value.identifier),
        _state=linen_module._ModuleInternalState(),
    )
    if not is_static_linen_declaration(declaration, failure=_static_failure):
        raise _static_failure("Linen declaration schema changed")
    return declaration


@dataclass(frozen=True)
class _CarrierField:
    name: str
    port: int | None
    static: _StaticField | None


@dataclass(frozen=True)
class _CarrierStorage:
    fields: tuple[tuple[str, MemberDescriptorType | GetSetDescriptorType], ...]
    slots: tuple[tuple[type[Any], tuple[str, ...]], ...]


@dataclass(frozen=True)
class _CarrierSpec:
    carrier_type: type[Any]
    fields: tuple[_CarrierField, ...]
    frozen: bool
    storage: _CarrierStorage


_CARRIER_MISSING: object = object()
_DATACLASS_PARAMS_TYPE: type[Any] = type(
    inspect.getattr_static(_CarrierSpec, "__dataclass_params__")
)


def _carrier_mro(carrier_type: type[Any]) -> tuple[type[Any], ...]:
    # The native type descriptor bypasses custom metaclass attribute/hash hooks.
    return cast(tuple[type[Any], ...], type.__dict__["__mro__"].__get__(carrier_type))


def _carrier_class_attribute(
    carrier_type: type[Any], name: str, default: Any = None
) -> Any:
    for base in _carrier_mro(carrier_type):
        namespace = type.__dict__["__dict__"].__get__(base)
        if name in namespace:
            return namespace[name]
    return default


def _numeric_carrier_field(value: Any) -> bool:
    # Never consult an unsupported instance's potentially custom __class__.
    return any(
        base is numeric
        for base in _carrier_mro(type(value))
        for numeric in (jax.Array, Tracer, jax.ShapeDtypeStruct, np.ndarray, np.generic)
    )


def _static_carrier_field(value: Any) -> _StaticField:
    if not any(
        type(value) is literal
        for literal in (type(None), bool, int, float, complex, str, bytes)
    ):
        raise _foreign_failure("nonliteral or nested carrier metadata needs an adapter")
    return _StaticField(type(value), marshal.dumps(value), value)


def _require_plain_carrier_descriptor(carrier_type: type[Any], name: str) -> None:
    descriptor = _carrier_class_attribute(carrier_type, name, None)
    if type(descriptor) is MemberDescriptorType:
        if descriptor.__name__ != name or not any(
            descriptor.__objclass__ is base for base in _carrier_mro(carrier_type)
        ):
            raise _foreign_failure("carrier field descriptor needs an adapter")
    elif any(
        _carrier_class_attribute(type(descriptor), hook, _CARRIER_MISSING)
        is not _CARRIER_MISSING
        for hook in ("__get__", "__set__", "__delete__")
    ):
        raise _foreign_failure("carrier field descriptor needs an adapter")


def _carrier_schema(
    carrier_type: type[Any],
) -> tuple[tuple[str, ...], bool, _CarrierStorage]:
    """Validate passive class storage before reading any carrier instance."""
    if type(carrier_type) is not type:
        raise _foreign_failure("nonordinary dataclass metaclass needs an adapter")
    frozen: bool | None = None
    for base in carrier_type.__mro__:
        if base is object:
            continue
        declared = _carrier_class_attribute(base, "__dataclass_fields__", None)
        if type(declared) is not dict:
            raise _foreign_failure("nonordinary dataclass inheritance needs an adapter")
        if any(name in vars(base) for name in ("__getattribute__", "__getattr__")):
            raise _foreign_failure("custom dataclass field access needs an adapter")
        if "__del__" in vars(base):
            raise _foreign_failure("custom dataclass lifetime hooks need an adapter")
        if any(
            type(name) is not str
            or type(item) is not Field
            or type(item.name) is not str
            or item.name != name
            for name, item in declared.items()
        ):
            raise _foreign_failure("dataclass field layout needs an adapter")
        parameters = _carrier_class_attribute(base, "__dataclass_params__", None)
        if (
            type(parameters) is not _DATACLASS_PARAMS_TYPE
            or type(parameters.frozen) is not bool
            or (frozen is not None and parameters.frozen is not frozen)
        ):
            raise _foreign_failure("dataclass frozen status needs an adapter")
        frozen = parameters.frozen
    # Class-only dataclasses.fields is safe after exact metadata/access checks.
    names = tuple(item.name for item in fields(carrier_type))
    for name in names:
        _require_plain_carrier_descriptor(carrier_type, name)
    dictionary_descriptor = _carrier_class_attribute(carrier_type, "__dict__", None)
    if (dictionary_descriptor is None and carrier_type.__dictoffset__ != 0) or (
        dictionary_descriptor is not None
        and (
            type(dictionary_descriptor) is not GetSetDescriptorType
            or dictionary_descriptor.__name__ != "__dict__"
            or not any(
                dictionary_descriptor.__objclass__ is base
                for base in carrier_type.__mro__
            )
        )
    ):
        raise _foreign_failure("custom dataclass storage descriptor needs an adapter")
    slot_descriptors: dict[str, MemberDescriptorType] = {}
    slot_layout: list[tuple[type[Any], tuple[str, ...]]] = []
    for base in carrier_type.__mro__:
        namespace = vars(base)
        slots = namespace.get("__slots__", ())
        if type(slots) is str:
            slots = (slots,)
        if type(slots) is not tuple or any(type(name) is not str for name in slots):
            raise _foreign_failure("dataclass slot layout needs an adapter")
        declared_slots = set()
        normalized_slots = []
        for slot in slots:
            name = slot
            if name.startswith("__") and not name.endswith("__"):
                owner_name = base.__name__.lstrip("_")
                if owner_name:
                    name = f"_{owner_name}{name}"
            descriptor = namespace.get(name)
            expected = (
                GetSetDescriptorType
                if name in {"__dict__", "__weakref__"}
                else MemberDescriptorType
            )
            if (
                type(descriptor) is not expected
                or descriptor.__name__ != name
                or descriptor.__objclass__ is not base
                or name in slot_descriptors
            ):
                raise _foreign_failure("dataclass slot descriptor or layout changed")
            if name not in {"__dict__", "__weakref__"}:
                if name not in names:
                    raise _foreign_failure(
                        "carrier has storage outside its declared fields"
                    )
                slot_descriptors[name] = cast(MemberDescriptorType, descriptor)
                declared_slots.add(name)
            normalized_slots.append(name)
        if "__slots__" in namespace:
            slot_layout.append((base, tuple(normalized_slots)))
        if any(
            type(descriptor) is MemberDescriptorType and name not in declared_slots
            for name, descriptor in namespace.items()
        ):
            raise _foreign_failure("dataclass slot descriptor or layout changed")
    if any(
        _carrier_class_attribute(carrier_type, name) is not descriptor
        for name, descriptor in slot_descriptors.items()
    ):
        raise _foreign_failure("inherited dataclass slot descriptor changed")
    if (
        any(name not in slot_descriptors for name in names)
        and dictionary_descriptor is None
    ):
        raise _foreign_failure("carrier declared dictionary storage is missing")
    storage = _CarrierStorage(
        tuple(
            (
                ("slot", slot_descriptors[name])
                if name in slot_descriptors
                else ("dict", cast(GetSetDescriptorType, dictionary_descriptor))
            )
            for name in names
        ),
        tuple(slot_layout),
    )
    assert frozen is not None
    return names, frozen, storage


def _carrier_fields(value: Any) -> tuple[tuple[str, Any], ...] | None:
    carrier_type = type(value)
    if any(
        base is native
        for base in _carrier_mro(carrier_type)
        for native in (type, nnx.Module, linen_module.Module)
    ) or (
        _carrier_class_attribute(carrier_type, "__dataclass_fields__", _CARRIER_MISSING)
        is _CARRIER_MISSING
    ):
        return None
    names, frozen, storage = _carrier_schema(carrier_type)
    dictionary_descriptor = _carrier_class_attribute(carrier_type, "__dict__", None)
    dictionary = (
        None
        if dictionary_descriptor is None
        else object.__getattribute__(value, "__dict__")
    )
    slot_names = {
        name
        for name, (kind, _) in zip(names, storage.fields, strict=True)
        if kind == "slot"
    }
    if dictionary is not None and (
        type(dictionary) is not dict
        or any(type(name) is not str for name in dictionary)
        or set(dictionary) != set(names) - slot_names
    ):
        raise _foreign_failure("carrier has state outside its declared fields")
    try:
        values = tuple((name, object.__getattribute__(value, name)) for name in names)
    except (AttributeError, TypeError) as exc:
        raise _foreign_failure("carrier declared storage is incomplete") from exc
    numeric = any(_numeric_carrier_field(item) for _, item in values)
    if not numeric and not frozen:
        return None
    if not numeric and any(
        not any(
            type(item) is literal
            for literal in (type(None), bool, int, float, complex, str, bytes)
        )
        for _, item in values
    ):
        raise _static_failure(
            "GraphDef.Static payload requires literal frozen dataclass fields"
        )
    for _, item in values:
        if _numeric_carrier_field(item):
            if isinstance(item, np.ndarray):
                raise _foreign_failure(
                    "mutable foreign-linked buffers need isolation/effect support"
                )
            _StateSpec.of(item)
        else:
            _static_carrier_field(item)
    return values


def _normalize_carriers(
    graphdef: Any,
    native: tuple[Any, ...],
    *,
    expected_callbacks: CallbackGraph | None = None,
) -> tuple[
    Any,
    tuple[_CarrierSpec, ...],
    tuple[int, ...],
    tuple[Any, ...],
    CallbackGraph | None,
]:
    _require_graph_metadata(graphdef)
    attributes = list(graphdef.attributes)
    owners = _graph_attribute_owners(graphdef)
    linen_indices: dict[int, int] = {}
    batch_axes_indices: dict[int, int] = {}
    raw_carriers: list[tuple[type[Any], tuple[tuple[str, Any], ...]]] = []
    identities: dict[int, int] = {}
    replacements: dict[int, int] = {}
    connected: set[int] = set()
    callback_capture = CallbackCapture(expected=expected_callbacks)
    for index, (key, attribute) in enumerate(graphdef.attributes):
        if type(attribute) is not nnx_graph.Static:
            continue
        value = attribute.value
        if key == "_pytree__nodes":
            attributes[index] = (key, replace(attribute, value=_node_mapping(value)))
            continue
        if owners[index] is _NATIVE_LINEAR_GENERAL and key == "batch_axis":
            _require_batch_axes_role(owners[index], key)
            identity = id(value)
            if identity not in batch_axes_indices:
                batch_axes_indices[identity] = len(batch_axes_indices)
            attributes[index] = (
                key,
                replace(
                    attribute,
                    value=_snapshot_batch_axes(value, batch_axes_indices[identity]),
                ),
            )
            continue
        mapped = _map_native_anchors(value, owners[index], key)
        if mapped is not value:
            if type(mapped) is not _NativeStaticAnchor:
                _require_inert_static(value, "GraphDef.Static payload")
            attributes[index] = (key, replace(attribute, value=mapped))
            continue
        if type(value) is FunctionType:
            attributes[index] = (
                key,
                replace(
                    attribute,
                    value=_capture_callback(
                        callback_capture, value, f"NNX GraphDef.Static[{index}:{key}]"
                    ),
                ),
            )
            continue
        carrier_values = _carrier_fields(value)
        if carrier_values is None:
            if is_static_linen_declaration(value, failure=_static_failure):
                identity = id(value)
                if identity not in linen_indices:
                    linen_indices[identity] = len(linen_indices)
                attributes[index] = (
                    key,
                    replace(
                        attribute, value=_snapshot_linen(value, linen_indices[identity])
                    ),
                )
            else:
                _require_inert_static(value, "GraphDef.Static payload")
            continue
        identity = id(value)
        if identity not in identities:
            identities[identity] = len(raw_carriers)
            raw_carriers.append((type(value), carrier_values))
            connected.update(
                id(item) for _, item in carrier_values if _numeric_carrier_field(item)
            )
        replacements[index] = identities[identity]
    callback_graph = callback_capture.finish()
    callbacks = None if callback_graph.is_empty else callback_graph
    if not raw_carriers:
        return replace(graphdef, attributes=attributes), (), (), native, callbacks
    # Only foreign-connected identity groups are canonicalized. Every unrelated
    # existing Variable/container port retains its own original position.
    state_values: list[Any] = []
    ports: dict[int, int] = {}
    aliases = []
    for item in native:
        identity = id(item)
        if identity not in connected or identity not in ports:
            port = len(state_values)
            state_values.append(item)
            if identity in connected:
                ports[identity] = port
        else:
            port = ports[identity]
        aliases.append(port)
    carriers = []
    for carrier_type, raw_fields in raw_carriers:
        carrier_fields = []
        for name, item in raw_fields:
            if _numeric_carrier_field(item):
                identity = id(item)
                if identity not in ports:
                    ports[identity] = len(state_values)
                    state_values.append(item)
                carrier_fields.append(_CarrierField(name, ports[identity], None))
            else:
                carrier_fields.append(
                    _CarrierField(name, None, _static_carrier_field(item))
                )
        _, frozen, storage = _carrier_schema(carrier_type)
        carriers.append(
            _CarrierSpec(carrier_type, tuple(carrier_fields), frozen, storage)
        )
    attributes = [
        (
            (key, replace(attribute, value=_CarrierToken(replacements[index])))
            if index in replacements
            else (key, attribute)
        )
        for index, (key, attribute) in enumerate(attributes)
    ]
    return (
        replace(graphdef, attributes=attributes),
        tuple(carriers),
        tuple(aliases),
        tuple(state_values),
        callbacks,
    )


def _split(
    owner: Any,
    *,
    expected_callbacks: CallbackGraph | None = None,
) -> tuple[
    Any,
    tuple[_StateEntry, ...],
    tuple[_CarrierSpec, ...],
    tuple[int, ...],
    tuple[Any, ...],
    CallbackGraph | None,
]:
    graphdef, state = _split_graph(owner, expected_callbacks=expected_callbacks)
    # Raw-storage preflight has limited split to audited native callbacks.
    # Check their envelopes before tracing, merging or entering the source body.
    _require_graph_metadata(graphdef)
    entries: list[_StateEntry] = []
    leaves: list[Any] = []
    for path, value in nnx.to_flat_state(state):
        values, tree = jtu.tree_flatten(value)
        _require_safe_tree(tree)
        start = len(leaves)
        leaves.extend(values)
        entries.append(_StateEntry(tuple(path), tree, start, len(leaves)))
    # Reconstruction must not retain a surrounding NNX update-context index.
    graphdef, carriers, aliases, state_values, callbacks = _normalize_carriers(
        graphdef.with_no_outer_index(),
        tuple(leaves),
        expected_callbacks=expected_callbacks,
    )
    return graphdef, tuple(entries), carriers, aliases, state_values, callbacks


@dataclass(frozen=True)
class NNXFunctionState:
    """Topology and an unbound call, with numeric values carried separately.

    Callback snapshots retain bounded watches for delayed validation, never a
    live NNX owner or its numeric state.
    """

    graphdef: Any
    entries: tuple[_StateEntry, ...]
    specs: tuple[_StateSpec, ...]
    method: Callable[..., Any]
    instance_key: int
    effects: tuple[int, ...] | None = None
    carriers: tuple[_CarrierSpec, ...] = ()
    native_aliases: tuple[int, ...] = ()
    callbacks: CallbackGraph | None = None

    @property
    def state_count(self) -> int:
        return len(self.specs)

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            "nnx-state",
            self.method,
            self.instance_key,
            self.graphdef,
            self.entries,
            self.specs,
            self.effects,
            self.carriers,
            self.native_aliases,
            self.callbacks,
        )

    def _foreign_ports(self) -> frozenset[int]:
        return frozenset(
            field.port
            for carrier in self.carriers
            for field in carrier.fields
            if field.port is not None
        )

    def with_effects(self, effects: tuple[int, ...]) -> NNXFunctionState:
        if tuple(sorted(set(effects))) != effects or any(
            index < 0 or index >= self.state_count for index in effects
        ):
            raise _failure("Malformed NNX Function state effect layout", defect=True)
        if self._foreign_ports().intersection(effects):
            raise _foreign_failure("foreign-linked effects need update support")
        return replace(self, effects=effects)

    def _check_values(self, state: tuple[Any, ...]) -> None:
        if len(state) != self.state_count:
            raise _failure(
                "NNX Function state changed shape, dtype, weakness or port count"
            )
        foreign = self._foreign_ports()
        if any(isinstance(state[index], np.ndarray) for index in foreign):
            raise _foreign_failure(
                "mutable foreign-linked buffers need isolation/effect support"
            )
        if any(isinstance(value, np.ndarray) for value in state):
            raise _static_failure(
                "mutable native numeric buffers need isolation/effect support"
            )
        actual = tuple(_StateSpec.of(value) for value in state)
        if actual != self.specs:
            if any(actual[index] != self.specs[index] for index in foreign):
                raise _foreign_failure(
                    "foreign-linked shape, dtype or weakness changed"
                )
            raise _failure(
                "NNX Function state changed shape, dtype, weakness or port count"
            )

    def _entry_ports(self, entry: _StateEntry) -> tuple[int, ...]:
        if self.native_aliases:
            return self.native_aliases[entry.start : entry.stop]
        return tuple(range(entry.start, entry.stop))

    def current_state(self, owner: object) -> tuple[Any, ...]:
        """Read numeric ports while checking exact topology, metadata and aliases."""
        return self._owner_state(owner)

    def paths_for_port(self, port: int) -> tuple[tuple[Any, ...], ...]:
        """Return every native entry path sharing this canonical numeric port."""
        if type(port) is not int or not 0 <= port < self.state_count:
            raise _failure("Malformed NNX state port selection", defect=True)
        return tuple(
            entry.path for entry in self.entries if port in self._entry_ports(entry)
        )

    def ports_for_filter(self, owner: object, mutable: Any) -> tuple[int, ...]:
        """Select from the same native graph protocol as this descriptor.

        Ambient tree mode can duplicate shared-owner/Variable paths. Filtering
        the exact graph-mode split state keeps selection on canonical entries
        without changing the caller's ambient graph configuration.
        """
        self._owner_state(owner)
        _, state = _split_graph(owner, expected_callbacks=self.callbacks)
        selected = nnx.filter_state(state, mutable)
        return self.ports_for_paths(
            tuple(tuple(path) for path, _ in nnx.to_flat_state(selected))
        )

    def ports_for_paths(self, paths: tuple[tuple[Any, ...], ...]) -> tuple[int, ...]:
        """Select canonical ports, closing the existing native alias relation."""
        selected = set(paths)
        known = {entry.path for entry in self.entries}
        if not selected <= known:
            raise _failure("Selected NNX state path is absent from the native schema")
        return tuple(
            sorted(
                {
                    port
                    for entry in self.entries
                    if entry.path in selected
                    for port in self._entry_ports(entry)
                }
            )
        )

    def _state(self, state: tuple[Any, ...], *, changed_only: bool = False) -> Any:
        self._check_values(state)
        changed = set(self.effects or ())
        return nnx.from_flat_state(
            (
                entry.path,
                jtu.tree_unflatten(
                    entry.tree, [state[index] for index in self._entry_ports(entry)]
                ),
            )
            for entry in self.entries
            if not changed_only or changed.intersection(self._entry_ports(entry))
        )

    def reconstruct(self, state: tuple[Any, ...]) -> Any:
        """Merge current carried values into fresh actual native owner/carriers."""
        # GraphDef's native outer lists and trusted native global anchors remain
        # externally accessible. Recheck delayed reconstruction before any
        # merge hook or source body can observe a newly introduced payload.
        _require_graph_metadata(self.graphdef)
        owners = _graph_attribute_owners(self.graphdef)
        for entry in self.entries:
            _require_safe_tree(entry.tree)
        callbacks = None if self.callbacks is None else self.callbacks.rebuild()
        resolved_statics: dict[int, Any] = {}
        linen: dict[int, tuple[_LinenDeclaration, Any]] = {}
        batch_axes: dict[int, tuple[_BatchAxes, Any]] = {}
        for index, (key, attribute) in enumerate(self.graphdef.attributes):
            if type(attribute) is not nnx_graph.Static:
                continue
            value = attribute.value
            if type(value) is _CarrierToken:
                if type(value.index) is not int or not 0 <= value.index < len(
                    self.carriers
                ):
                    raise _static_failure("malformed numeric carrier token")
            elif type(value) is CallbackToken:
                if callbacks is None:
                    raise _static_failure(
                        "callback token has no prepared callback graph"
                    )
                resolved_statics[index] = callbacks.resolve(value)
            elif type(value) is _NativeNodeMapping and key == "_pytree__nodes":
                _require_inert_static(value.items, "NNX field classification")
                if any(type(flag) is not bool for _, flag in value.items):
                    raise _static_failure(
                        "NNX field classification values need an adapter"
                    )
            elif type(value) is _LinenDeclaration:
                if value.index not in linen:
                    linen[value.index] = (value, _reconstruct_linen(value))
                elif linen[value.index][0] != value:
                    raise _static_failure("Linen declaration alias schema changed")
                resolved_statics[index] = linen[value.index][1]
            elif type(value) is _BatchAxes:
                _require_batch_axes_role(owners[index], key)
                _require_frozen_dict_schema()
                if type(value.index) is not int or value.index < 0:
                    raise _static_failure(
                        "malformed LinearGeneral batch_axis alias index"
                    )
                _require_batch_axes_items(value.items)
                if value.index not in batch_axes:
                    batch_axes[value.index] = (
                        value,
                        _NATIVE_FROZEN_DICT(dict(value.items)),
                    )
                elif batch_axes[value.index][0] != value:
                    raise _static_failure(
                        "LinearGeneral batch_axis alias schema changed"
                    )
                resolved_statics[index] = batch_axes[value.index][1]
            else:
                resolved = _map_native_anchors(value, owners[index], key, resolve=True)
                if _native_anchor(resolved, owners[index], key) is None:
                    _require_inert_static(resolved, "GraphDef.Static payload")
                resolved_statics[index] = resolved
        self._check_values(state)
        graphdef = self.graphdef
        carriers = []
        if self.carriers:
            for spec in self.carriers:
                names, frozen, storage = _carrier_schema(spec.carrier_type)
                if (
                    type(spec.frozen) is not bool
                    or frozen is not spec.frozen
                    or storage != spec.storage
                    or any(
                        type(item) is not _CarrierField or type(item.name) is not str
                        for item in spec.fields
                    )
                    or tuple(item.name for item in spec.fields) != names
                ):
                    raise _foreign_failure(
                        "carrier frozen status or field layout changed"
                    )
                for item in spec.fields:
                    _require_plain_carrier_descriptor(spec.carrier_type, item.name)
                    if item.port is None:
                        if type(item.static) is not _StaticField:
                            raise _foreign_failure("malformed carrier static metadata")
                        if _static_carrier_field(item.static.value) != item.static:
                            raise _foreign_failure("carrier static metadata changed")
                # Object allocation bypasses __new__/__init__/__post_init__.
                carrier = object.__new__(spec.carrier_type)
                for field in spec.fields:
                    if field.port is None:
                        assert field.static is not None
                        value = field.static.value
                    else:
                        value = state[field.port]
                    object.__setattr__(carrier, field.name, value)
                carriers.append(carrier)
        attributes = []
        for index, (key, attribute) in enumerate(graphdef.attributes):
            if type(attribute) is nnx_graph.Static:
                if type(attribute.value) is _CarrierToken:
                    attribute = replace(
                        attribute, value=carriers[attribute.value.index]
                    )
                elif type(attribute.value) is _NativeNodeMapping:
                    # copy=True preserves Static.value. Explicitly rebuild
                    # this audited native mutable envelope for every merge.
                    attribute = replace(
                        attribute,
                        value=HashableMapping(dict(attribute.value.items)),
                    )
                elif index in resolved_statics:
                    attribute = replace(attribute, value=resolved_statics[index])
            attributes.append((key, attribute))
        # This graph exists only for reconstruction, never descriptor/cache.
        graphdef = replace(graphdef, attributes=attributes)
        return nnx.merge(graphdef, self._state(state), copy=True)

    def _owner_state(self, owner: object) -> tuple[Any, ...]:
        graphdef, entries, carriers, aliases, state, callbacks = _split(
            owner, expected_callbacks=self.callbacks
        )
        if callbacks != self.callbacks:
            raise _static_failure("callback effects need explicit state/update support")
        if (
            graphdef != self.graphdef
            or entries != self.entries
            or carriers != self.carriers
            or aliases != self.native_aliases
        ):
            if self.carriers or carriers:
                raise _foreign_failure(
                    "carrier schema, metadata or alias topology changed"
                )
            raise _failure("NNX Function changed graph topology or state metadata")
        self._check_values(state)
        return state

    def _foreign_instances(self, owner: object) -> tuple[object, ...]:
        if not self.carriers:
            return ()
        graphdef, _ = _split_graph(owner, expected_callbacks=self.callbacks)
        instances = []
        for index, (_, attribute) in enumerate(self.graphdef.attributes):
            if (
                type(attribute) is nnx_graph.Static
                and type(attribute.value) is _CarrierToken
            ):
                if (
                    index >= len(graphdef.attributes)
                    or type(graphdef.attributes[index][1]) is not nnx_graph.Static
                ):
                    raise _foreign_failure("carrier holder layout changed")
                instances.append(graphdef.attributes[index][1].value)
        return tuple(instances)

    def invoke(
        self,
        arguments: tuple[Any, ...],
        keywords: Mapping[str, object],
        state: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...], tuple[int, ...]]:
        owner = self.reconstruct(state)
        foreign_before = self._foreign_instances(owner)
        result = self.method(owner, *arguments, **keywords)
        if any(
            before is not after
            for before, after in zip(
                foreign_before, self._foreign_instances(owner), strict=True
            )
        ):
            raise _foreign_failure(
                "carrier holder replacement needs effect/update support"
            )
        updated = self._owner_state(owner)
        if any(updated[index] is not state[index] for index in self._foreign_ports()):
            raise _foreign_failure(
                "foreign-linked numeric mutation needs effect/update support"
            )
        # This is an exact def-use alias test, not numerical tracer equality.
        effects = tuple(
            index
            for index, (before, after) in enumerate(zip(state, updated, strict=True))
            if after is not before
        )
        if self.effects is not None and effects != self.effects:
            raise _failure("NNX Function state alias effects changed while retracing")
        return result, updated, effects

    def validate_owner(self, owner: object) -> None:
        if self.effects is not None and self._foreign_ports().intersection(
            self.effects
        ):
            raise _foreign_failure("foreign-linked effects need update support")
        self._owner_state(owner)
        if self.effects is None:
            raise _failure(
                "NNX Function state has no traced effect layout", defect=True
            )
        if not self.effects:
            return
        try:
            # Public update checks every changed Variable/attribute at this trace
            # level. It writes the exact original objects, advancing no state.
            nnx.update(owner, self._original_state(owner))
        except errors.TraceContextError as exc:
            raise _failure(
                "Mutable NNX Function state requires a trace-local prepared owner"
            ) from exc

    def _original_state(self, owner: object) -> Any:
        """Transient envelopes retain original raw containers and numeric aliases.

        Flatten/unflatten is appropriate for the explicit numeric ports, but
        would rebuild a container-valued Variable's raw payload. A no-advance
        ownership check and rollback must instead retain that exact object.
        Nothing from this snapshot is stored in the descriptor or cache.
        """
        values = dict(_iter_graph(owner, expected_callbacks=self.callbacks))
        changed = set(self.effects or ())
        originals = []
        for entry in self.entries:
            if not changed.intersection(self._entry_ports(entry)):
                continue
            value = values[entry.path]
            if isinstance(value, nnx.Variable):
                metadata = value.get_metadata()
                _require_ordinary_variable_metadata(metadata)
                value = type(value).from_metadata(value.get_raw_value(), metadata)
            originals.append((entry.path, value))
        return nnx.from_flat_state(originals)

    def update_owner(self, owner: object, state: tuple[Any, ...]) -> None:
        self.validate_owner(owner)
        self._check_values(state)
        if not self.effects:
            return
        original = self._original_state(owner)
        try:
            nnx.update(owner, self._state(state, changed_only=True))
        except BaseException:
            # Restore existing Variables and aliases, including a partial update.
            nnx.update(owner, original)
            raise


def prepare_nnx_function_state(
    owner: object, method: Callable[..., Any], instance_key: int
) -> tuple[NNXFunctionState, tuple[Any, ...]] | None:
    """Describe an NNX owner with checked native storage and split/merge APIs.

    Finite official runtime anchors are trusted process globals; this is a
    model-state isolation boundary, not a sandbox for mutated framework code.
    """
    if not is_nnx_function_owner(owner):
        return None
    graphdef, entries, carriers, aliases, state, callbacks = _split(owner)
    adapter = NNXFunctionState(
        graphdef,
        entries,
        tuple(_StateSpec.of(value) for value in state),
        method,
        instance_key,
        carriers=carriers,
        native_aliases=aliases,
        callbacks=callbacks,
    )
    return adapter, state


def is_nnx_function_owner(owner: object) -> bool:
    return isinstance(owner, nnx.Module)
