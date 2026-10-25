# jax2onnx/converter/typing_support.py

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    TYPE_CHECKING,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    MutableSequence,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
import onnx_ir as ir

from jax2onnx._compat.jax import JaxprEqn
from .precision_policy import PrecisionPolicy
from .source_deployment import DeploymentTarget, MatrixPrecisionRequirement

if TYPE_CHECKING:
    from jax2onnx.utils.debug import RecordedPrimitiveCallLog
    from .function_scope import FunctionRegistry
    from .ir_context import IRContext
    from .ir_constants import ConstantFolder
    from .source_structure import SourceStructureLedger

ResolverFn = Callable[[object], object | None]


@dataclass(frozen=True)
class SymbolicDimOrigin:
    """Typed representation of where a symbolic dimension came from."""

    value: ir.Value
    axis: int

    def as_tuple(self) -> tuple[ir.Value, int]:
        return (self.value, self.axis)

    @classmethod
    def from_unknown(cls, origin: object | None) -> SymbolicDimOrigin | None:
        """Normalise `(value, axis)` tuples into ``SymbolicDimOrigin``."""
        if origin is None:
            return None
        if isinstance(origin, cls):
            return origin
        if (
            isinstance(origin, tuple)
            and len(origin) == 2
            and isinstance(origin[1], int)
            and isinstance(origin[0], ir.Value)
        ):
            return cls(value=origin[0], axis=int(origin[1]))
        return None

    @classmethod
    def resolve(
        cls,
        resolver: ResolverFn | None,
        dim: object,
    ) -> SymbolicDimOrigin | None:
        """Lookup ``dim`` via ``resolver`` while tolerating string fallbacks."""

        if resolver is None:
            return None
        origin = resolver(dim)
        if origin is None and not isinstance(dim, str):
            origin = resolver(str(dim))
        return cls.from_unknown(origin)


@runtime_checkable
class SymbolicDimTracker(Protocol):
    def get_symbolic_dim_origin(self, dim: object) -> SymbolicDimOrigin | None: ...

    def record_symbolic_dim_origin(
        self, dim: object, value: ir.Value, axis: int
    ) -> None: ...


@runtime_checkable
class IRBuilderAPI(Protocol):
    """Closed emission contract for native lowerings, backed by ONNX IR Tape."""

    @property
    def opset(self) -> int: ...

    @property
    def enable_double_precision(self) -> bool: ...

    @property
    def graph(self) -> ir.Graph: ...

    @property
    def inputs(self) -> MutableSequence[ir.Value]: ...

    @property
    def outputs(self) -> MutableSequence[ir.Value]: ...

    @outputs.setter
    def outputs(self, values: Iterable[ir.Value]) -> None: ...

    @property
    def initializers(self) -> Sequence[ir.Value]: ...

    @property
    def nodes(self) -> Sequence[ir.Node]: ...

    @property
    def stacktrace_metadata_enabled(self) -> bool: ...

    @property
    def current_jax_traceback(self) -> str | None: ...

    @property
    def current_plugin_identifier(self) -> str | None: ...

    @property
    def current_plugin_line(self) -> str | None: ...

    def set_current_jax_traceback(self, trace: str | None) -> None: ...

    def set_current_plugin_identifier(
        self, plugin_id: str | None, line: str | None = None
    ) -> None: ...

    def fresh_name(self, base: str) -> str: ...

    def op(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, ir.Attr | str | int | float] | None = None,
        *,
        domain: str = "",
        overload: str = "",
        version: int | None = None,
        graph: ir.Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
        output: ir.Value | None = None,
    ) -> ir.Value: ...

    def op_multi_out(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, ir.Attr | str | int | float] | None = None,
        *,
        num_outputs: int | None = None,
        outputs: Sequence[ir.Value] | None = None,
        domain: str = "",
        overload: str = "",
        version: int | None = None,
        graph: ir.Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> Sequence[ir.Value]: ...

    def add_initializer_from_scalar(self, name: str, value: object) -> ir.Value: ...

    def add_initializer_from_array(
        self,
        name: str,
        array: np.ndarray[Any, np.dtype[Any]],
    ) -> ir.Value: ...

    def const_i64(self, name: str, values: Sequence[int]) -> ir.Value: ...

    def add_node_obj(self, node: ir.Node) -> None: ...

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value],
        outputs: Sequence[ir.Value],
        attributes: Mapping[str, Any] | Sequence[ir.Attr] | None = None,
        name: str | None = None,
    ) -> ir.Node: ...

    def record_symbol_origin(self, sym: str, src_val: ir.Value, axis: int) -> None: ...

    def get_symbolic_dim_origin(self, sym: str) -> SymbolicDimOrigin | None: ...

    def to_ir_model(
        self,
        *,
        name: str,
        ir_version: int = 11,
        protective_clone: bool = True,
    ) -> ir.Model: ...


@runtime_checkable
class IRBuilderProtocol(IRBuilderAPI, Protocol):
    """Builder compatibility surface for existing dynamic operator shorthand."""

    def __getattr__(self, name: str) -> Any: ...


@runtime_checkable
class NativeLoweringContext(SymbolicDimTracker, Protocol):
    """Public operations for native plugins, with a closed typed IR builder."""

    @property
    def builder(self) -> IRBuilderAPI: ...

    @property
    def opset(self) -> int: ...

    @property
    def enable_double_precision(self) -> bool: ...

    @property
    def normalization_mode(self) -> str: ...

    @property
    def precision_policy(self) -> PrecisionPolicy: ...

    @property
    def deployment_target(self) -> DeploymentTarget | None: ...

    @property
    def source_scope(self) -> str: ...

    def record_matrix_precision_requirement(
        self,
        requirement: MatrixPrecisionRequirement,
        output: ir.Value,
        matrix_output: ir.Value,
    ) -> None: ...

    def fresh_name(self, base: str) -> str: ...

    def add_node(
        self,
        node: ir.Node,
        inputs: Sequence[ir.Value] | None = None,
        outputs: Sequence[ir.Value] | None = None,
    ) -> ir.Node: ...

    def require_value_for_var(
        self,
        var: object,
        *,
        prefer_np_dtype: np.dtype[Any] | None = None,
    ) -> ir.Value: ...

    def lookup_value_for_var(self, var: object) -> ir.Value | None: ...

    def allocate_value_for_var(
        self,
        var: object,
        *,
        name_hint: str | None = None,
        prefer_np_dtype: np.dtype[Any] | None = None,
    ) -> ir.Value: ...

    def bind_value_for_var(self, var: object, value: ir.Value) -> None: ...

    def bind_const_for_var(
        self,
        var: object,
        np_array: np.ndarray[Any, np.dtype[Any]],
    ) -> ir.Value: ...

    def try_evaluate_const(
        self, var: object
    ) -> np.ndarray[Any, np.dtype[Any]] | None: ...

    def cast_like(
        self,
        tensor: ir.Value,
        exemplar: ir.Value,
        *,
        name_hint: str | None = None,
    ) -> ir.Value: ...


@runtime_checkable
class ConverterContext(NativeLoweringContext, Protocol):
    """Converter bookkeeping and the historical dynamic plugin surface."""

    record_primitive_calls_file: str | None
    _current_eqn: JaxprEqn | None
    _primitive_call_records: list[RecordedPrimitiveCallLog]
    _lowering_record_depth: int
    _lowering_record_owner: ConverterContext
    _func_name_counters: dict[tuple[str, str, str], int]
    _jaxpr_validator: Callable[[Any, int], None] | None
    _source_structure: SourceStructureLedger | None
    _call_input_param_literals: dict[str, object]
    _call_param_value_by_name: dict[str, ir.Value]

    @property
    def builder(self) -> IRBuilderProtocol: ...

    @property
    def _const_folder(self) -> ConstantFolder: ...

    @property
    def _var2val(self) -> dict[Any, ir.Value]: ...

    @property
    def _inputs(self) -> Sequence[ir.Value]: ...

    @property
    def _initializers(self) -> Sequence[ir.Value]: ...

    @property
    def _nodes(self) -> Sequence[ir.Node]: ...

    @property
    def _inside_function_scope(self) -> bool: ...

    @property
    def _call_input_param_names(self) -> set[str]: ...

    @property
    def _promote_call_input_params(self) -> bool: ...

    @property
    def ir_functions(self) -> list[ir.Function]: ...

    def get_function_registry(self) -> FunctionRegistry | None: ...

    def set_function_registry(self, registry: FunctionRegistry) -> None: ...

    def create_child_context(
        self,
        *,
        scope: Literal["function", "control_flow"],
        prefix: str | None = None,
        implicit_captures: bool = False,
    ) -> IRContext: ...

    def get_value_for_var(
        self,
        var: object,
        *,
        name_hint: str | None = None,
        prefer_np_dtype: np.dtype[Any] | None = None,
    ) -> ir.Value: ...

    def ensure_external_flag(self, name: str, var: object | None) -> ir.Value: ...

    def ensure_training_mode(self, flag_name: str, var: Any) -> ir.Value: ...

    def set_external_flag_resolver(
        self, resolver: Callable[[str], ir.Value] | None
    ) -> None: ...

    def add_input_for_invar(self, var: object, index: int) -> ir.Value: ...


# Existing plugins retain their dynamic builder and converter conveniences.
LoweringContextProtocol: TypeAlias = ConverterContext


@dataclass(frozen=True)
class AxisOverrideInfo:
    """Structured axis-0 override metadata captured during lowering."""

    extent: int
    op_type: str | None = None

    def allows_restamp(self, allowed_ops: Collection[str] | None = None) -> bool:
        """Return True when the override may safely restamp ONNX metadata."""

        if self.extent <= 1:
            return False
        if not allowed_ops:
            return True
        return self.op_type in allowed_ops

    def as_tuple(self) -> tuple[int, str | None]:
        return (self.extent, self.op_type)


AxisOverrideMap = dict[str, AxisOverrideInfo]


@dataclass(frozen=True)
class RngTrace:
    """Tracks deterministic RNG helpers requested via construct_and_call()."""

    kind: str
    seed: int | None

    def describe(self) -> str:
        return f"{self.kind}(seed={self.seed})"


@runtime_checkable
class PrimitiveLowering(Protocol):
    def lower(
        self,
        ctx: ConverterContext,
        eqn: JaxprEqn,
    ) -> None: ...


# The dispatcher supplies the full context. Native plugins accept its smaller
# public supertype; parameter contravariance also preserves existing plugins.
CanonicalLowering: TypeAlias = Callable[[ConverterContext, JaxprEqn], None]


@runtime_checkable
class FunctionLowering(Protocol):
    def get_handler(
        self, converter: Any
    ) -> Callable[[Any, Any, Mapping[str, Any]], Any]: ...
