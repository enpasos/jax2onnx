# jax2onnx/_compat/_cast_lowering.py

"""Native IR/options from one existing original cast trace; never compile it."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
from pathlib import Path
from types import MemberDescriptorType
from typing import Any, cast

from ._cast_metadata import (
    CpuCastContext,
    _file,
    _pin_function,
    _require,
    project_cpu_cast_options,
)

_TEXT = 65536


@dataclass(frozen=True, slots=True)
class NativeCastModule:
    text: str
    text_sha256: str
    shape: tuple[int, ...]
    operations: tuple[str, ...]


def validate_original_cast_module(
    module: object, shape: tuple[int, ...]
) -> NativeCastModule:
    """Check the complete native module, not text substrings or emitted receipts."""
    from jaxlib.mlir import ir

    _require(
        type(shape) is tuple and all(type(dim) is int and dim >= 0 for dim in shape),
        "native cast expected shape differs",
    )
    _require(type(module) is ir.Module, "native cast module type differs")
    native_module: ir.Module = cast(ir.Module, module)
    root = native_module.operation
    _require(
        root.name == "builtin.module"
        and len(root.regions) == 1
        and len(root.regions[0].blocks) == 1
        and not root.operands
        and not root.results,
        "native cast module structure differs",
    )
    attrs = root.attributes
    _require(
        len(attrs) == 3
        and all(
            name in attrs
            for name in ("sym_name", "mhlo.num_replicas", "mhlo.num_partitions")
        ),
        "native cast module attributes differ",
    )
    symbol = ir.StringAttr(attrs["sym_name"]).value
    _require(
        0 < len(symbol) <= 4096 and "\0" not in symbol, "native module symbol differs"
    )
    for name in ("mhlo.num_replicas", "mhlo.num_partitions"):
        value = ir.IntegerAttr(attrs[name])
        integer = ir.IntegerType(value.type)
        _require(
            value.value == 1 and integer.width == 32 and integer.is_signless,
            "native cast partition/replica differs",
        )
    functions = list(itertools.islice(native_module.body.operations, 2))
    _require(len(functions) == 1, "native cast function count differs")
    function = functions[0].operation
    attrs = function.attributes
    required = ("sym_name", "sym_visibility", "function_type")
    allowed = (*required, "arg_attrs", "res_attrs")
    _require(
        function.name == "func.func"
        and len(function.regions) == 1
        and len(function.regions[0].blocks) == 1
        and not function.operands
        and not function.results
        and all(name in attrs for name in required)
        and sum(name in attrs for name in allowed) == len(attrs),
        "native cast function attributes/structure differ",
    )
    _require(
        ir.StringAttr(attrs["sym_name"]).value == "main"
        and ir.StringAttr(attrs["sym_visibility"]).value == "public",
        "native cast entry interface differs",
    )
    for key in ("arg_attrs", "res_attrs"):
        if key not in attrs:
            continue
        ports = ir.ArrayAttr(attrs[key])
        _require(len(ports) == 1, "native cast port attribute arity differs")
        values = ir.DictAttr(ports[0])
        if key == "arg_attrs":
            _require(len(values) == 0, "native cast argument has nondefault metadata")
        else:
            _require(
                len(values) == 0 or (len(values) == 1 and "jax.result_info" in values),
                "native cast result has semantic metadata",
            )
            if "jax.result_info" in values:
                text = ir.StringAttr(values["jax.result_info"]).value
                _require(
                    len(text) <= 4096 and "\0" not in text,
                    "native result name exceeds bound",
                )
    signature = ir.FunctionType(ir.TypeAttr(attrs["function_type"]).value)
    _require(
        len(signature.inputs) == len(signature.results) == 1,
        "native cast signature arity differs",
    )
    with native_module.context, native_module.operation.location:
        expected_input = ir.RankedTensorType.get(shape, ir.F64Type.get())
        expected_output = ir.RankedTensorType.get(shape, ir.F16Type.get())
        _require(
            signature.inputs[0] == expected_input
            and signature.results[0] == expected_output,
            "native cast signature types differ",
        )
        block = function.regions[0].blocks[0]
        operations = list(itertools.islice(block.operations, 3))
        _require(
            len(block.arguments) == 1 and len(operations) == 2,
            "native cast body extent differs",
        )
        convert, returned = (operation.operation for operation in operations)
        _require(
            convert.name == "stablehlo.convert"
            and returned.name == "func.return"
            and len(convert.operands)
            == len(convert.results)
            == len(returned.operands)
            == 1
            and not convert.attributes
            and not convert.regions
            and not returned.attributes
            and not returned.regions
            and not returned.results,
            "native cast operations differ",
        )
        _require(
            block.arguments[0].type == expected_input
            and convert.results[0].type == expected_output
            and convert.operands[0] == block.arguments[0]
            and returned.operands[0] == convert.results[0],
            "native cast SSA edges/types differ",
        )
    try:
        _require(native_module.operation.verify(), "native cast verification failed")
    except ir.MLIRError as error:
        raise ValueError("native cast verification failed") from error
    text = str(native_module)
    _require(len(text.encode()) <= _TEXT, "native cast text exceeds bound")
    return NativeCastModule(
        text,
        hashlib.sha256(text.encode()).hexdigest(),
        shape,
        ("stablehlo.convert", "func.return"),
    )


class CastInputDeclaration:
    """Original exact SDS slots, captured before converter normalization."""

    def __init__(self, inputs: object) -> None:
        import jax
        import numpy as np
        from jax._src import core

        if type(inputs) is not tuple and type(inputs) is not list:
            raise ValueError("original unary declaration unavailable")
        native_inputs = cast(tuple[object, ...] | list[object], inputs)
        _require(len(native_inputs) == 1, "original unary declaration unavailable")
        value = native_inputs[0]
        _require(
            jax.ShapeDtypeStruct is core.ShapeDtypeStruct
            and type(value) is core.ShapeDtypeStruct,
            "exact original native SDS required",
        )
        cls = core.ShapeDtypeStruct
        names: tuple[str, ...] = (
            "shape",
            "dtype",
            "_sharding",
            "_dll",
            "weak_type",
            "manual_axis_type",
            "is_ref",
        )
        if jax.__version__ == "0.12.0.1":
            names += ("_memory_space",)
        self._type: Any = cls
        self._value: Any = value
        slots = tuple((name, vars(cls).get(name)) for name in names)
        _require(
            all(type(descriptor) is MemberDescriptorType for _, descriptor in slots),
            "native SDS slot descriptors differ",
        )
        self._slots: tuple[tuple[str, MemberDescriptorType], ...] = tuple(
            (name, cast(MemberDescriptorType, descriptor)) for name, descriptor in slots
        )
        self._fields = tuple(
            (name, descriptor.__get__(value, cls)) for name, descriptor in self._slots
        )
        fields = dict(self._fields)
        shape = fields["shape"]
        _require(
            type(shape) is tuple
            and all(type(dim) is int and dim >= 0 for dim in shape),
            "original native SDS shape differs",
        )
        _require(
            type(fields["dtype"]) is type(np.dtype(np.float64))
            and fields["dtype"] == np.dtype(np.float64)
            and fields["weak_type"] is False
            and fields["is_ref"] is False
            and all(
                fields[name] is None
                for name in ("_sharding", "_dll", "manual_axis_type")
            )
            and fields.get("_memory_space") is None,
            "original native SDS overrides unavailable",
        )
        self.shape: tuple[int, ...] = shape

    @property
    def value(self) -> Any:
        self.assert_current()
        return self._value

    def assert_current(self) -> None:
        _require(
            self._value is not None and type(self._value) is self._type,
            "original input declaration lease closed",
        )
        for (name, descriptor), (_, expected) in zip(
            self._slots, self._fields, strict=True
        ):
            _require(
                vars(self._type).get(name) is descriptor
                and descriptor.__get__(self._value, self._type) is expected,
                "original declaration slot changed",
            )

    def close(self) -> None:
        self._value = self._type = None
        self._slots = ()
        self._fields = ()


@dataclass(frozen=True, slots=True)
class NativeCastLowering:
    module: NativeCastModule
    options: bytes
    options_sha256: str
    input_propagation: tuple[bool, ...]
    output_propagation: tuple[bool, ...]
    detailed_logging: bool
    client_identity: int
    device_identity: int
    preexisting_executable: bool
    resolver_requests: int = 1
    source_compile_requests: int = 0
    source_execute_requests: int = 0
    options_scope: str = (
        "current-native-original-trace-default-construction-not-historical-executable"
    )
    cache_scope: str = "ordinary-native-IR-caches-may-retain-entries"


_LOWERING_SOURCES = {
    "0.10.2": (
        "337a95777c1b079001319eefa299ce8accf290474c8cea8267286c6d9715d0f3",
        "19f522992c322bf12cab2baa5754b197779763e085ad3eb3b382a11ad1c028b1",
    ),
    "0.12.0.1": (
        "4cbac41ed693853ca871599ba3a6b0d9db9c21998f93facb4066442dc1ac8a62",
        "9961c2bd4891f695aed813b0440b2965877f6cbe20253c2bf7fec7d2c347a5a5",
    ),
}


class CastNativeLoweringLease:
    """Transient original declaration and native binding/registry ownership."""

    def __init__(self, inputs: object, name: str) -> None:
        import jax
        from jax._src import pjit, util, config, compiler
        from jax._src.interpreters import mlir, pxla
        from jax._src.lax import lax
        from jaxlib import _jax as native
        from .jax import _CAST_LOWERING_ANCHORS, _CAST_LOWERING_MLIR_ROLES

        self.declaration = CastInputDeclaration(inputs)
        _require(
            type(name) is str and 0 < len(name) <= 4096 and "\0" not in name,
            "native cast lowering name differs",
        )
        self._name = name
        self._used = False
        self._closed = False
        _require(
            jax.__version__ in _LOWERING_SOURCES,
            "native IR-lowering source epoch differs",
        )
        mlir_roles = _CAST_LOWERING_MLIR_ROLES[jax.__version__]
        expected_mlir_roles = (
            (
                "lower_jaxpr_to_module",
                "convert_hlo",
                "lower_with_sharding_in_types",
            )
            if jax.__version__ == "0.10.2"
            else (
                "lower_jaxpr_to_module",
                "convert_hlo",
                "lower_with_explicit_types",
                "lower_with_sharding_in_types",
                "lower_with_layout_in_types",
            )
        )
        expected_roles = tuple(
            (module, member)
            for module, names in (
                (
                    pjit,
                    (
                        "convert_to_metaty",
                        "create_meta_ty",
                        "_resolve_and_lower",
                        "_resolve_in_shardings",
                        "_resolve_in_layouts",
                        "_resolve_out_layouts",
                        "_pjit_lower",
                        "get_ctx_mesh",
                    ),
                ),
                (
                    pxla,
                    (
                        "lower_sharding_computation",
                        "_cached_lowering_to_hlo",
                        "create_compile_options",
                        "get_prop_to_input_output",
                        "maybe_concretize_mesh",
                    ),
                ),
                (mlir, expected_mlir_roles),
            )
            for member in names
        )
        self._anchors = _CAST_LOWERING_ANCHORS
        _require(
            mlir_roles == expected_mlir_roles
            and len(self._anchors) == len(expected_roles)
            and all(
                anchor[0] is module
                and type(anchor[1]) is str
                and anchor[1] == member
                and anchor[2] is not None
                for anchor, (module, member) in zip(self._anchors, expected_roles)
            ),
            "native lowering import anchors incomplete",
        )
        self._states = [
            _pin_function(module, member)
            for module, names in (
                (
                    pjit,
                    (
                        "convert_to_metaty",
                        "_resolve_and_lower",
                        "_resolve_in_shardings",
                        "_resolve_in_layouts",
                        "_resolve_out_layouts",
                        "get_ctx_mesh",
                    ),
                ),
                (
                    pxla,
                    (
                        "create_compile_options",
                        "get_prop_to_input_output",
                        "maybe_concretize_mesh",
                    ),
                ),
                (mlir, mlir_roles),
                (lax, ("_convert_element_type_lower",)),
                (util, ("test_event",)),
            )
            for member in names
        ]
        self._states.append(_pin_function(compiler, "use_detailed_logging"))
        self._files = {
            Path(pjit.__file__): _file(Path(pjit.__file__)),
            Path(mlir.__file__): _file(Path(mlir.__file__)),
        }
        _require(
            jax.__version__ in _LOWERING_SOURCES
            and tuple(value[1] for value in self._files.values())
            == _LOWERING_SOURCES[jax.__version__],
            "native IR-lowering source epoch differs",
        )
        self._primitive: Any = lax.convert_element_type_p
        self._registry: Any = mlir._lowerings
        self._platform_registry: Any = mlir._platform_specific_lowerings
        raw_entry = dict.get(self._registry, self._primitive)
        _require(
            type(self._registry) is dict and type(raw_entry) is mlir.LoweringRuleEntry,
            "native cast default lowering entry differs",
        )
        entry: mlir.LoweringRuleEntry = cast(mlir.LoweringRuleEntry, raw_entry)
        _require(
            entry.rule is lax._convert_element_type_lower and entry.inline is True,
            "native cast default lowering entry differs",
        )
        self._entry: Any = entry
        self._rule: Any = self._entry.rule
        self._serializer: Any = vars(native.CompileOptions).get("SerializeAsString")
        _require(
            type(self._serializer).__module__ == "nanobind"
            and type(self._serializer).__name__ in ("nb_method", "nb_func"),
            "native options serializer differs",
        )
        self._members: tuple[tuple[dict[str, Any], str, Any], ...] = (
            (vars(mlir), "_lowerings", self._registry),
            (vars(mlir), "_platform_specific_lowerings", self._platform_registry),
            (vars(lax), "_convert_element_type_lower", self._entry.rule),
            (vars(util), "test_event_listener", None),
            (vars(config), "jax_dump_ir_to", config.jax_dump_ir_to),
            (vars(native.CompileOptions), "SerializeAsString", self._serializer),
            (vars(compiler), "use_detailed_logging", compiler.use_detailed_logging),
        )
        self.assert_current()

    def assert_current(self) -> None:
        from jax._src import util, config
        from jax._src.interpreters import mlir

        _require(not self._closed, "native cast lowering lease closed")
        self.declaration.assert_current()
        for module, name, bound, code, cells, cls, call in self._anchors:
            _require(
                vars(module).get(name) is bound
                and type(bound) is cls
                and vars(cls).get("__call__") is call,
                "native lowering function/cache anchor changed",
            )
            if code is not None:
                _require(
                    bound.__code__ is code
                    and all(cell.cell_contents is value for cell, value in cells),
                    "native lowering function code/closure changed",
                )
        for state in self._states:
            state.assert_current()
        _require(
            all(
                namespace.get(name) is value for namespace, name, value in self._members
            ),
            "native lowering member/listener changed",
        )
        _require(
            util.test_event_listener is None
            and config.jax_dump_ir_to.value in (None, ""),
            "native lowering diagnostics must be inactive",
        )
        _require(
            dict.get(self._registry, self._primitive) is self._entry,
            "native cast lowering entry changed",
        )
        current_entry: mlir.LoweringRuleEntry = cast(
            mlir.LoweringRuleEntry, self._entry
        )
        _require(
            current_entry.rule is self._rule and current_entry.inline is True,
            "native cast lowering entry changed",
        )
        cpu = dict.get(self._platform_registry, "cpu")
        _require(
            cpu is None or (type(cpu) is dict and self._primitive not in cpu),
            "custom CPU cast lowering unavailable",
        )
        _require(
            all(_file(path) == identity for path, identity in self._files.items()),
            "native lowering source changed",
        )

    def lower(self, context: CpuCastContext, closed: object) -> NativeCastLowering:
        import jax
        import numpy as np
        from jax._src import pjit, api
        from jax._src.interpreters import mlir, pxla
        from jax._src.lax import lax
        from jax._src.layout import Format
        from jaxlib import _jax as native
        from .jax import (
            ClosedJaxpr,
            ShapedArray,
            _CAST_META_TYPE,
            _CAST_MESH_COMPUTATION_TYPE,
            _CAST_MESH_EXECUTABLE_TYPE,
            _CAST_LOWERING_PARAMETER_TYPE,
            _native_context_fields,
            current_native_eqn_context,
        )

        self.assert_current()
        context.assert_current()
        _require(
            type(context) is CpuCastContext and not self._used,
            "native cast lowering requires a fresh owned context",
        )
        self._used = True
        _require(type(closed) is ClosedJaxpr, "original native closed trace required")
        native_closed: ClosedJaxpr = cast(ClosedJaxpr, closed)
        body: Any = native_closed.jaxpr
        _require(
            not native_closed.consts
            and not body.constvars
            and not body.effects
            and len(body.invars) == len(body.outvars) == len(body.eqns) == 1,
            "original native cast trace extent differs",
        )
        equation: Any = body.eqns[0]
        _require(
            equation.primitive is lax.convert_element_type_p
            and len(equation.invars) == len(equation.outvars) == 1
            and equation.invars[0] is body.invars[0]
            and equation.outvars[0] is body.outvars[0]
            and not equation.effects
            and set(equation.params) == {"new_dtype", "weak_type", "sharding"}
            and type(equation.params["new_dtype"]) is type(np.dtype(np.float16))
            and equation.params["new_dtype"] == np.dtype(np.float16)
            and equation.params["weak_type"] is False
            and equation.params["sharding"] is None,
            "original native cast equation differs",
        )

        def aval_matches(aval: Any, dtype: Any) -> bool:
            return (
                type(aval) is ShapedArray
                and type(aval.shape) is tuple
                and all(type(dim) is int for dim in aval.shape)
                and aval.shape == self.declaration.shape
                and type(aval.dtype) is type(dtype)
                and aval.dtype == dtype
                and aval.weak_type is False
            )

        _require(
            aval_matches(body.invars[0].aval, np.dtype(np.float64))
            and aval_matches(body.outvars[0].aval, np.dtype(np.float16)),
            "original native cast ports differ",
        )
        recorded = _native_context_fields(equation.ctx)
        ambient = _native_context_fields(current_native_eqn_context())
        _require(
            recorded == ambient and recorded[0] is None and recorded[3] in (None, ()),
            "native cast recorded context override unavailable",
        )
        computation = module = meta = params = ctx_mesh = device_array = options = (
            backend
        ) = device_list = existing = None
        in_shardings: tuple[Any, ...] | None = None
        out_shardings: tuple[Any, ...] | None = None
        try:
            meta = pjit.convert_to_metaty(self.declaration.value)
            _require(
                type(meta) is _CAST_META_TYPE
                and aval_matches(meta.aval, np.dtype(np.float64))
                and meta.sharding is None
                and type(meta.format) is Format
                and meta.format.layout is None
                and meta.format.sharding is None
                and type(meta.committed) is bool
                and meta.is_np_array is False,
                "original native declaration MetaTy differs",
            )
            params = mlir.LoweringParameters()
            _require(
                type(params) is _CAST_LOWERING_PARAMETER_TYPE
                and params.override_lowering_rules is None
                and params.global_constant_computation is False
                and params.for_export is False
                and params.export_ignore_forward_compatibility is False
                and type(params.hoist_constants_as_args) is bool,
                "native default lowering parameters differ",
            )
            ctx_mesh = pjit.get_ctx_mesh(False)
            _require(ctx_mesh.empty, "native context mesh is not empty")
            inline = api.Inline.AUTO if jax.__version__ == "0.12.0.1" else False
            computation = pjit._resolve_and_lower(
                (meta,),
                jaxpr=native_closed,
                in_shardings=(pjit.UNSPECIFIED,),
                out_shardings=(pjit.UNSPECIFIED,),
                in_layouts=(None,),
                out_layouts=(None,),
                donated_invars=(False,),
                ctx_mesh=ctx_mesh,
                name=self._name,
                keep_unused=False,
                inline=inline,
                lowering_platforms=None,
                lowering_parameters=params,
                pgle_profiler=None,
                compiler_options_kvs=(),
            )
            _require(
                type(computation) is _CAST_MESH_COMPUTATION_TYPE,
                "native MeshComputation type differs",
            )
            storage = object.__getattribute__(computation, "__dict__")
            _require(
                type(storage) is dict
                and type(storage.get("_compiler_options_kvs")) is tuple
                and not storage["_compiler_options_kvs"]
                and type(storage.get("_donated_invars")) is tuple
                and len(storage["_donated_invars"]) == 1
                and storage["_donated_invars"][0] is False
                and type(storage.get("_platforms")) is tuple
                and storage["_platforms"] == ("cpu",)
                and type(storage.get("const_args")) in (list, tuple)
                and not storage["const_args"],
                "native computation default parameters differ",
            )
            existing = storage.get("_executable")
            _require(
                existing is None or type(existing) is _CAST_MESH_EXECUTABLE_TYPE,
                "native cached executable type differs",
            )
            module = storage.get("_hlo")
            checked = validate_original_cast_module(module, self.declaration.shape)
            args = storage.get("compile_args")
            keys = {
                "global_in_avals",
                "global_out_avals",
                "in_shardings",
                "out_shardings",
                "tuple_args",
                "unordered_effects",
                "ordered_effects",
                "host_callbacks",
                "keepalive",
                "kept_var_idx",
                "mut",
                "backend",
                "num_devices",
                "committed",
                "in_layouts",
                "out_layouts",
                "shape_poly_state",
                "all_args_info",
                "pgle_profiler",
                "intermediate_shardings",
                "context_mesh",
            }
            _require(
                type(args) is dict and set(args) == keys,
                "native compile-argument schema differs",
            )
            _require(
                len(args["global_in_avals"]) == len(args["global_out_avals"]) == 1
                and aval_matches(args["global_in_avals"][0], np.dtype(np.float64))
                and aval_matches(args["global_out_avals"][0], np.dtype(np.float16)),
                "native resolved avals differ",
            )
            _require(
                args["tuple_args"] is False
                and args["num_devices"] == 1
                and type(args["num_devices"]) is int
                and args["committed"] is False
                and args["mut"] is None
                and args["pgle_profiler"] is None
                and not any(
                    args[key]
                    for key in (
                        "unordered_effects",
                        "ordered_effects",
                        "host_callbacks",
                        "keepalive",
                        "intermediate_shardings",
                    )
                )
                and tuple(args["in_layouts"]) == tuple(args["out_layouts"]) == (None,),
                "native resolved callbacks/effects/layout/placement differ",
            )
            kept = args["kept_var_idx"]
            _require(
                type(kept) in (set, frozenset)
                and len(kept) == 1
                and all(type(index) is int and index == 0 for index in kept),
                "native original input was removed or replaced",
            )
            poly = args["shape_poly_state"]
            _require(
                poly is None
                or (
                    type(poly) is mlir.ShapePolyLoweringState
                    and poly.uses_dim_vars is False
                    and not poly.dim_vars
                    and poly.has_platform_index_argument is False
                ),
                "native polymorphic lowering unavailable",
            )
            _require(
                type(args["all_args_info"]) is pxla.AllArgsInfo
                and len(args["all_args_info"].in_avals) == 1
                and aval_matches(
                    args["all_args_info"].in_avals[0], np.dtype(np.float64)
                )
                and args["context_mesh"].empty,
                "native original argument/context map differs",
            )
            backend, device_list = args["backend"], storage.get("_device_list")
            _require(
                type(backend) is native.Client
                and id(backend) == context._acquired_client_identity
                and type(device_list) is native.DeviceList
                and len(device_list) == 1
                and type(device_list[0]) is native.Device
                and vars(native.Device)["client"].__get__(device_list[0], native.Device)
                is backend,
                "native lowering target/device differs from acquired client",
            )
            in_shardings = tuple(
                pxla.maybe_concretize_mesh(value, device_list)
                for value in args["in_shardings"]
            )
            out_shardings = tuple(
                pxla.maybe_concretize_mesh(value, device_list)
                for value in args["out_shardings"]
            )
            _require(
                len(in_shardings) == len(out_shardings) == 1,
                "native resolved sharding arity differs",
            )
            input_prop, output_prop = pxla.get_prop_to_input_output(
                in_shardings, out_shardings, len(args["ordered_effects"])
            )
            _require(
                type(input_prop) is tuple
                and type(output_prop) is tuple
                and len(input_prop) == len(output_prop) == 1
                and all(type(value) is bool for value in (*input_prop, *output_prop)),
                "native propagation result differs",
            )
            device_array = np.array([device_list[0]], dtype=object)
            self.assert_current()
            context.assert_current()
            options = pxla.create_compile_options(
                module,
                args["tuple_args"],
                input_prop,
                output_prop,
                backend,
                device_array,
                {},
            )
            _require(
                type(options) is native.CompileOptions,
                "native source options type differs",
            )
            serializer = self._serializer
            raw_options = serializer(options)
            project_cpu_cast_options(raw_options)
            _require(
                len(raw_options)
                + len(checked.text.encode())
                + 16 * len(checked.shape)
                + 4096
                <= 256 * 1024,
                "native source result metadata exceeds bound",
            )
            detailed = (
                options.executable_build_options.debug_options.xla_detailed_logging
            )
            _require(
                type(detailed) is bool
                and raw_options == serializer(options)
                and storage.get("_executable") is existing,
                "native source construction mutated or compiled",
            )
            _require(
                checked
                == validate_original_cast_module(module, self.declaration.shape),
                "native source module changed",
            )
            self.assert_current()
            context.assert_current()
            return NativeCastLowering(
                checked,
                raw_options,
                hashlib.sha256(raw_options).hexdigest(),
                input_prop,
                output_prop,
                detailed,
                id(backend),
                id(device_list[0]),
                existing is not None,
            )
        finally:
            # Normal native lowering caches may retain their own entries. Drop
            # this producer's references, never clear or inspect unrelated caches.
            computation = module = meta = params = ctx_mesh = device_array = options = (
                backend
            ) = device_list = existing = native_closed = body = equation = None
            storage = args = poly = kept = in_shardings = out_shardings = None
            context.require_profiles_current()

    def close(self) -> None:
        self.declaration.close()
        self._states.clear()
        self._files.clear()
        self._members = ()
        self._anchors = ()
        self._primitive = self._registry = self._platform_registry = self._entry = (
            self._rule
        ) = None
        self._serializer = None
        self._closed = True


def lower_original_cast(context: Any, closed: object, lease: Any) -> NativeCastLowering:
    _require(
        type(context) is CpuCastContext and type(lease) is CastNativeLoweringLease,
        "actual conversion-owned native lowering lease required",
    )
    native_lease: CastNativeLoweringLease = cast(CastNativeLoweringLease, lease)
    native_context: CpuCastContext = cast(CpuCastContext, context)
    return native_lease.lower(native_context, closed)
