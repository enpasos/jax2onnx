import jax
import jax.random
import numpy as np
from onnx import helper
from typing import TYPE_CHECKING, Dict, Callable

from jax.extend import core as extend_core

# Ensure JaxprEqn is accessible
from jax.core import JaxprEqn

if TYPE_CHECKING:
    from .converter import Jaxpr2OnnxConverter


class PrimitiveDispatcher:
    def __init__(self):
        self.builtin_handlers: Dict[str, Callable] = {}
        self._define_builtin_handlers()

    def _define_builtin_handlers(self):
        self.builtin_handlers[jax._src.prng.random_seed_p.name] = (
            self._handle_random_seed
        )
        self.builtin_handlers[jax._src.prng.random_wrap_p.name] = (
            self._handle_random_wrap
        )
        self.builtin_handlers[jax._src.prng.random_split_p.name] = (
            self._handle_random_split
        )
        self.builtin_handlers[jax._src.prng.random_unwrap_p.name] = (
            self._handle_random_unwrap
        )
        self.builtin_handlers[jax.lax.convert_element_type_p.name] = (
            self._handle_convert_element_type
        )
        self.builtin_handlers[jax.lax.device_put_p.name] = self._handle_device_put

    def dispatch_and_execute(self, converter: "Jaxpr2OnnxConverter", eqn: JaxprEqn):
        primitive = eqn.primitive
        handler_key = primitive.name
        handler = converter.primitive_handlers.get(
            handler_key
        ) or self.builtin_handlers.get(handler_key)

        if handler is None:
            raise NotImplementedError(
                f"No handler registered for primitive: {handler_key}"
            )

        try:
            # Try plugin-style first
            handler(converter, eqn, **eqn.params)
        except TypeError:
            # Fallback: legacy-style handlers
            handler(eqn.invars, eqn.outvars, eqn.params)

    def _create_identity_node(self, converter: "Jaxpr2OnnxConverter", eqn: JaxprEqn):
        input_names = [converter.get_name(inp) for inp in eqn.invars]
        output_names = [converter.get_name(out) for out in eqn.outvars]
        if not output_names:
            return

        node = converter.builder.create_node(
            "Identity",
            input_names,
            output_names,
            name=converter.get_unique_name(f"identity_{eqn.primitive.name}"),
        )
        converter.add_node(node)

    def _handle_random_seed(self, converter, eqn, **params):
        self._create_identity_node(converter, eqn)

    def _handle_random_wrap(self, converter, eqn, **params):
        self._create_identity_node(converter, eqn)

    def _handle_random_unwrap(self, converter, eqn, **params):
        self._create_identity_node(converter, eqn)

    def _handle_random_split(self, converter, eqn, **params):
        input_name = converter.get_name(eqn.invars[0])
        output_name = converter.get_name(eqn.outvars[0])
        intermediate = converter.get_unique_name("random_split:x")

        reshape_name = converter.get_constant_name(np.array([1, 2], dtype=np.int64))
        repeat_name = converter.get_constant_name(
            np.array([params["shape"][0], 1], dtype=np.int64)
        )

        node1 = helper.make_node(
            "Reshape",
            [input_name, reshape_name],
            [intermediate],
            name=converter.get_unique_name("random_split:reshape"),
        )
        node2 = helper.make_node(
            "Tile",
            [intermediate, repeat_name],
            [output_name],
            name=converter.get_unique_name("random_split:tile"),
        )

        converter.add_node(node1)
        converter.add_node(node2)

    def _handle_convert_element_type(self, converter, eqn, **params):
        input_names = [converter.get_name(inp) for inp in eqn.invars]
        output_name = converter.get_name(eqn.outvars[0])
        new_dtype = converter.builder.numpy_dtype_to_onnx(params["new_dtype"])

        node = converter.builder.create_node(
            "Cast",
            input_names,
            [output_name],
            name=converter.get_unique_name("convert_element_type"),
            to=new_dtype,
        )
        converter.add_node(node)

    def _handle_device_put(self, converter, eqn, **params):
        inp = eqn.invars[0]
        out = eqn.outvars[0]

        if isinstance(inp, extend_core.Literal):
            val = inp.val
            np_val = np.array(val)
            if np_val.dtype == np.int64:
                np_val = np_val.astype(np.int32)
            elif np_val.dtype == np.float64:
                np_val = np_val.astype(np.float32)

            tensor_name = converter.get_unique_name("const")
            tensor = converter.builder.create_tensor(
                name=tensor_name,
                data_type=converter.builder.numpy_dtype_to_onnx(np_val.dtype),
                dims=np_val.shape,
                vals=np_val.flatten().tolist(),
            )
            converter.builder.add_initializer(tensor)

            output_name = converter.get_name(out)
            node = converter.builder.create_node(
                "Identity",
                [tensor_name],
                [output_name],
                name=converter.get_unique_name("device_put"),
            )
            converter.add_node(node)
        else:
            self._create_identity_node(converter, eqn)


# Instantiate a singleton dispatcher (or manage instantiation differently)
primitive_dispatcher = PrimitiveDispatcher()
