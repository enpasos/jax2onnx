# file: jax2onnx/plugin_system.py
import pkgutil
import importlib
import os
import jax.numpy as jnp
from jax.extend.core import Primitive
from typing import Optional, Callable, Dict, Any, Tuple, Type, Union
import numpy as np
from onnx import TensorProto

PLUGIN_REGISTRY: Dict[str, Union["ExamplePlugin", "PrimitiveLeafPlugin"]] = {}

# Track ONNX-decorated modules and their plugins
ONNX_FUNCTION_REGISTRY: Dict[str, Any] = {}
ONNX_FUNCTION_PRIMITIVE_REGISTRY: Dict[str, Tuple[Primitive, Any]] = {}
ONNX_FUNCTION_PLUGIN_REGISTRY: Dict[str, "FunctionPlugin"] = {}


def _tensorproto_dtype_to_numpy(onnx_dtype):
    return {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
    }.get(onnx_dtype, np.float32)


#####################################
# Primitive Plugin System
#####################################


class PrimitivePlugin:

    def get_patch_params(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_handler(self, converter: Any) -> Callable:
        raise NotImplementedError("Subclasses should implement this method")


class PrimitiveLeafPlugin(PrimitivePlugin):
    primitive: str
    metadata: Dict[str, Any]
    patch_info: Optional[Callable[[], Dict[str, Any]]] = None

    def get_patch_params(self):
        patch_info = self.patch_info()
        target = patch_info["patch_targets"][0]
        patch_func = patch_info["patch_function"]
        attr = patch_info.get("target_attribute", "__call__")
        return target, attr, patch_func

    def get_handler(self, converter: Any) -> Callable:
        return lambda node_inputs, node_outputs, params: self.to_onnx(
            converter, node_inputs, node_outputs, params
        )

    def to_onnx(
        self, converter: Any, node_inputs: Any, node_outputs: Any, params: Any
    ) -> None:
        raise NotImplementedError


class FunctionPlugin(PrimitivePlugin):
    target: Any

    def __init__(self, name: str, target: Any):
        self.name = name
        self.target = target
        self.primitive = Primitive(name)
        self.primitive.def_abstract_eval(lambda *args: args[0])
        self._orig_fn = None
        self._nested_builder = None

    def get_handler(self, converter: Any) -> Callable:
        return lambda converter, eqn, params: self._function_handler(
            converter, eqn, params
        )

    def _function_handler(self, converter, eqn, params):
        if self._orig_fn is None:
            raise RuntimeError(f"Original function for {self.name} not recorded.")

        print(f"\n🚀 Tracing function: {self.name}")
        input_shapes = [var.aval.shape for var in eqn.invars]
        input_names = [converter.get_name(v) for v in eqn.invars]
        example_args = [
            jnp.ones(shape, dtype=var.aval.dtype)
            for shape, var in zip(input_shapes, eqn.invars)
        ]

        parent_builder = converter.builder
        sub_converter = converter.__class__(name_counter=parent_builder.name_counter)

        sub_converter.trace_jaxpr(
            self._orig_fn,
            example_args,
            preserve_graph=True,
        )

        # Transfer initializers as function parameters
        function_initializers = sub_converter.builder.initializers
        param_input_names = [init.name for init in function_initializers]

        existing_input_names = {vi.name for vi in parent_builder.inputs}
        for init in function_initializers:
            if init.name not in existing_input_names:
                parent_builder.add_input(
                    init.name,
                    list(init.dims),
                    _tensorproto_dtype_to_numpy(init.data_type),
                )
                parent_builder.initializers.append(init)

        parent_builder.add_function(self.name, sub_converter.builder, param_input_names)

        # ✅ Explicitly propagate nested ONNX functions upward
        for (
            nested_func_name,
            nested_func_proto,
        ) in sub_converter.builder.functions.items():
            if nested_func_name not in parent_builder.functions:
                parent_builder.functions[nested_func_name] = nested_func_proto
                print(f"🚀 Propagated nested ONNX function: {nested_func_name}")

        parent_builder.name_counter = sub_converter.builder.name_counter

        parent_builder.add_function_call_node(
            self.name,
            input_names + param_input_names,
            [converter.get_var_name(v) for v in eqn.outvars],
        )

        print(f"✅ Finished tracing function: {self.name}\n")

    def get_patch_fn(self, primitive):
        def patch(original_call):
            def wrapped(instance, *args):
                class_name = instance.__class__.__name__
                print(f"🧠 Calling ONNX-decorated function: {class_name}")
                if class_name in ONNX_FUNCTION_PLUGIN_REGISTRY:
                    # ✅ Capture the original callable directly bound to the instance
                    ONNX_FUNCTION_PLUGIN_REGISTRY[class_name]._orig_fn = (
                        original_call.__get__(instance, type(instance))
                    )
                return primitive.bind(*args)

            return wrapped

        return patch


########################################
# Decorators
########################################


def onnx_function(cls):
    name = cls.__name__
    primitive = Primitive(name)
    primitive.def_abstract_eval(lambda x: x)

    cls._onnx_primitive = primitive

    ONNX_FUNCTION_REGISTRY[name] = cls
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, cls)

    plugin = FunctionPlugin(name, cls)
    ONNX_FUNCTION_PLUGIN_REGISTRY[name] = plugin

    # PLUGIN_REGISTRY[name] = plugin

    return cls


class ExamplePlugin:
    metadata: Dict[str, Any]


def register_example(**metadata: Any) -> ExamplePlugin:
    instance = ExamplePlugin()
    instance.metadata = metadata
    component = metadata.get("component")
    if isinstance(component, str):
        PLUGIN_REGISTRY[component] = instance
    return instance


def register_primitive(
    **metadata: Any,
) -> Callable[[Type[PrimitiveLeafPlugin]], Type[PrimitiveLeafPlugin]]:
    primitive = metadata.get("jaxpr_primitive", "")

    def decorator(cls: Type[PrimitiveLeafPlugin]) -> Type[PrimitiveLeafPlugin]:
        if not issubclass(cls, PrimitiveLeafPlugin):
            raise TypeError("Plugin must subclass PrimitivePlugin")

        instance = cls()
        instance.primitive = primitive
        instance.metadata = metadata or {}

        if hasattr(cls, "patch_info"):
            instance.patch_info = getattr(cls, "patch_info")

        if isinstance(primitive, str):
            PLUGIN_REGISTRY[primitive] = instance
        return cls

    return decorator


_already_imported_plugins = False


def import_all_plugins() -> None:
    global _already_imported_plugins
    if _already_imported_plugins:
        return
    plugins_path = os.path.join(os.path.dirname(__file__), "plugins")
    for _, module_name, _ in pkgutil.walk_packages(
        [plugins_path], prefix="jax2onnx.plugins."
    ):
        importlib.import_module(module_name)
    _already_imported_plugins = True
