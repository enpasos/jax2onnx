# file: jax2onnx/plugin_system.py
import functools
import importlib
import inspect
import os
import pkgutil
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union

import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive

import logging

from jax2onnx.converter.name_generator import get_qualified_name
from jax2onnx.converter.function_handling import function_handler

logger = logging.getLogger("jax2onnx.plugin_system")

# A global registry to store plugins for extending functionality.
# Plugins can be of different types, such as FunctionPlugin, ExamplePlugin, or PrimitiveLeafPlugin.
PLUGIN_REGISTRY: dict[
    str, Union["FunctionPlugin", "ExamplePlugin", "PrimitiveLeafPlugin"]
] = {}

# Track ONNX-decorated modules and their plugins
ONNX_FUNCTION_REGISTRY: dict[str, Any] = {}
ONNX_FUNCTION_PRIMITIVE_REGISTRY: dict[str, tuple[Primitive, Any]] = {}
ONNX_FUNCTION_PLUGIN_REGISTRY: dict[str, "FunctionPlugin"] = {}

INSTANCE_MAP: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


#####################################
# Primitive Plugin System
#####################################


class PrimitivePlugin(ABC):

    @abstractmethod
    def get_patch_params(self):
        """Retrieve patch parameters for the plugin."""
        pass

    @abstractmethod
    def get_handler(self, converter: Any) -> Callable:
        """Retrieve the handler function for the plugin."""
        pass


class PrimitiveLeafPlugin(PrimitivePlugin):
    primitive: str
    metadata: dict[str, Any]
    patch_info: Callable[[], dict[str, Any]] | None = None

    def get_patch_params(self):
        if not self.patch_info:
            raise ValueError("patch_info is not defined for this plugin.")
        info = self.patch_info()
        targets = info["patch_targets"]
        patch_func = info["patch_function"]
        attr = info.get("target_attribute", "__call__")
        # Return a list entry per patch target
        return [(t, attr, patch_func) for t in targets]

    def get_handler(self, converter: Any) -> Callable:
        return lambda converter, eqn, params: self.to_onnx(
            converter, eqn.invars, eqn.outvars, params
        )

    @abstractmethod
    def to_onnx(
        self, converter: Any, node_inputs: Any, node_outputs: Any, params: Any
    ) -> None:
        """Convert the plugin to ONNX format."""
        pass


class FunctionPlugin(PrimitivePlugin):
    def __init__(self, name: str, target: Any):
        self.name = name
        self.target = target
        self.primitive = Primitive(name)
        self.primitive.def_abstract_eval(self.abstract_eval_with_kwargs)
        self.primitive.def_impl(self.primitive_impl)
        self._orig_fn = None

    def to_function_proto(self, context, builder, inputs, outputs):
        # Generate a unique name for this function instance
        function_name = context.next_function_name(self.target.__name__)

        # Start building the FunctionProto
        builder.start_function(function_name, inputs, outputs)

        # The actual conversion logic would go here...
        # e.g., trace self.target, emit intermediate nodes, etc.

        return builder.end_function()

    @staticmethod
    def _aval_to_shaped_array(aval):
        """Converts a ShapeDtypeStruct or other aval to ShapedArray."""
        if isinstance(aval, ShapedArray):
            # It's already the type we need
            return aval
        elif hasattr(aval, "shape") and hasattr(aval, "dtype"):
            # Covers ShapeDtypeStruct and other array-like abstract values
            return ShapedArray(aval.shape, aval.dtype)
        else:
            # Handle non-array abstract values if necessary, or raise error
            raise TypeError(
                f"Cannot convert abstract value of type {type(aval)} to ShapedArray."
            )

    def abstract_eval_with_kwargs(self, *args, **kwargs):
        """
        Correctly performs abstract evaluation using the original function's signature
        and preserves symbolic dimensions without trying to evaluate the function directly.

        Args:
            *args: Tuple of abstract values (e.g., ShapedArray) for positional inputs.
            **kwargs: Dictionary of keyword arguments (static parameters).

        Returns:
            A ShapedArray with the correct shape and dtype matching the output of the original function.
        """
        if self._orig_fn is None:
            raise ValueError(
                f"Original function (_orig_fn) not set for abstract evaluation of primitive {self.name}"
            )

        # If instance_key exists in kwargs, remove it
        if "instance_key" in kwargs:
            del kwargs["instance_key"]

        try:
            if self._orig_fn is not None:
                # Drop tracing helper kwarg that we added in the wrapper.
                kwargs = {k: v for k, v in kwargs.items() if k != "instance_key"}

                # convert all args from ShapeArray to jax.ShapeDtypeStruct
                specs = [
                    (
                        jax.ShapeDtypeStruct(arg.shape, arg.dtype)
                        if isinstance(arg, ShapedArray)
                        else arg
                    )
                    for arg in args
                ]

                out_aval = jax.eval_shape(self._orig_fn, *specs, **kwargs)

                # we need to convert the output back to ShapedArray
                if isinstance(out_aval, jax.ShapeDtypeStruct):
                    # Convert the output to ShapedArray
                    out_aval = self._aval_to_shaped_array(out_aval)
                elif isinstance(out_aval, tuple):
                    # If the output is a tuple, convert each element to ShapedArray
                    out_aval = tuple(
                        self._aval_to_shaped_array(aval) for aval in out_aval
                    )
                elif isinstance(out_aval, list):
                    # If the output is a list, convert each element to ShapedArray
                    out_aval = [self._aval_to_shaped_array(aval) for aval in out_aval]

                return out_aval

        except Exception as e:
            logger.error(
                f"[ERROR] Abstract evaluation failed for primitive {self.name} on function {self._orig_fn}: {e}"
            )
            raise e

    def primitive_impl(self, *args, **kwargs):
        if self._orig_fn is None:
            raise ValueError("Original function not set for primitive!")
        return self._orig_fn(*args, **kwargs)

    def get_patch_fn(self, primitive, is_class: bool) -> Callable:
        def patch(original_call):
            sig = inspect.signature(original_call)
            params = list(sig.parameters.keys())

            @functools.wraps(original_call)
            def wrapped(*args, **kwargs):
                expects_self = params and params[0] == "self"

                if expects_self:
                    instance = args[0]
                    instance_key = id(instance)
                    INSTANCE_MAP[instance_key] = instance
                    qualname = get_qualified_name(instance.__class__)
                    if qualname in ONNX_FUNCTION_PLUGIN_REGISTRY:
                        plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[qualname]
                        plugin._orig_fn = original_call.__get__(
                            instance, type(instance)
                        )
                    # Pass instance_key as a kwarg
                    return primitive.bind(
                        *args[1:], **{**kwargs, "instance_key": instance_key}
                    )
                else:
                    # Non-class function
                    qualname = self.name  # self.name is already qualified
                    if qualname in ONNX_FUNCTION_PLUGIN_REGISTRY:
                        plugin = ONNX_FUNCTION_PLUGIN_REGISTRY[qualname]
                        plugin._orig_fn = original_call
                    return primitive.bind(*args, **kwargs)

            return wrapped

        return patch

    def get_patch_params(self):
        # Determine if the target is a class or a function
        if inspect.isclass(self.target):
            # Patch the __call__ method of the class
            return (self.target, "__call__", self.get_patch_fn(self.primitive, True))
        elif callable(self.target):
            # Patch the function in its module by name
            module = inspect.getmodule(self.target)
            func_name = self.target.__name__
            return (module, func_name, self.get_patch_fn(self.primitive, False))
        else:
            raise TypeError(
                f"Unsupported target type for patching: {type(self.target)}"
            )

    # Add this implementation
    def get_handler(self, converter: Any) -> Callable:
        return lambda conv, eqn, params: self._function_handler(
            converter, conv, eqn, params
        )

    def _function_handler(self, plugin_converter, converter, eqn, params):
        orig_fn = self._orig_fn

        # if existing remove "instance_key" from params
        if "instance_key" in params:
            key = params["instance_key"]
            del params["instance_key"]
            instance = INSTANCE_MAP.get(key)
            orig_fn = instance

        from jax import numpy as jnp
        import numpy as np
        import copy

        # --- Separate ONNX params and JAX params ---
        # Avoid deepcopy on JAX Tracers: do a shallow copy and only deepcopy safe values
        from jax.core import Tracer

        def safe_copy_dict(d):
            result = {}
            for k, v in d.items():
                # Don't deepcopy JAX Tracers or arrays
                if isinstance(v, Tracer):
                    result[k] = v
                elif hasattr(v, "aval"):
                    result[k] = v
                else:
                    try:
                        result[k] = copy.deepcopy(v)
                    except Exception:
                        result[k] = v
            return result

        onnx_params = safe_copy_dict(params)
        jax_params = safe_copy_dict(params)

        # ── new: lift scalar kwargs to ONNX constants ─────────────────────
        for k, v in list(params.items()):
            # ── 1. Python scalar: lift to 0‑D initializer named *exactly* k ──
            if isinstance(v, (bool, int, float, np.generic)):
                const_name = k  # <── keep original name
                # avoid double‑registration if several eqns share that kw‑arg
                if const_name not in converter.builder.initializers:
                    converter.builder.add_initializer_from_scalar(const_name, v)
                onnx_params[k] = const_name
                # jax_params[k] stays as the original scalar

            # ── 2. traced scalar (shape == ()): choose a sane default value ──
            elif isinstance(v, Tracer) and getattr(v.aval, "shape", None) == ():
                aval_dtype = v.aval.dtype
                if aval_dtype == jnp.bool_:
                    default_value = True
                elif np.issubdtype(aval_dtype, np.integer):
                    default_value = 0
                else:
                    default_value = 0.0

                const_name = k  # <── same trick as above
                if const_name not in converter.builder.initializers:
                    converter.builder.add_initializer_from_scalar(
                        const_name, default_value
                    )
                onnx_params[k] = const_name
                # jax_params[k] stays as the original traced value

        # Call function_handler with ONNX params for graph, but pass jax_params for tracing
        # Pass the original params as well for use in function_handler
        function_handler(
            self.name, converter, eqn, orig_fn, onnx_params, jax_params, params
        )


########################################
# Decorators
########################################


def onnx_function(target):
    name = get_qualified_name(target)
    primitive = Primitive(name)
    primitive.def_abstract_eval(lambda x: x)

    target._onnx_primitive = primitive

    ONNX_FUNCTION_REGISTRY[name] = target
    ONNX_FUNCTION_PRIMITIVE_REGISTRY[name] = (primitive, target)

    plugin = FunctionPlugin(name, target)
    ONNX_FUNCTION_PLUGIN_REGISTRY[name] = plugin

    return target


class ExamplePlugin:
    metadata: dict[str, Any]


def register_example(**metadata: Any) -> ExamplePlugin:
    instance = ExamplePlugin()
    instance.metadata = metadata
    component = metadata.get("component")
    if isinstance(component, str):
        PLUGIN_REGISTRY[component] = instance
    return instance


def register_primitive(
    **metadata: Any,
) -> Callable[[type[PrimitiveLeafPlugin]], type[PrimitiveLeafPlugin]]:
    primitive = metadata.get("jaxpr_primitive", "")

    def decorator(cls: type[PrimitiveLeafPlugin]) -> type[PrimitiveLeafPlugin]:
        if not issubclass(cls, PrimitiveLeafPlugin):
            raise TypeError("Plugin must subclass PrimitivePlugin")

        instance = cls()
        instance.primitive = primitive
        instance.metadata = metadata or {}

        if hasattr(cls, "patch_info"):
            instance.patch_info = cls.patch_info

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
