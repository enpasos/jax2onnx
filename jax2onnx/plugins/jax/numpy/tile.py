from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import onnx
from jax import core
from jax import numpy as jnp
from jax.extend.core import Literal, Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive
import logging

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# --- Define custom primitive
jnp.tile_p = Primitive("jnp.tile")
jnp.tile_p.multiple_results = False


# input for testcases
def _my_tile(t):
    repeats_tensor = jnp.array([3, 1, 1], dtype=jnp.int32)
    return jnp.tile(t, repeats_tensor)


# --- Register the plugin
@register_primitive(
    jaxpr_primitive=jnp.tile_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html",
    onnx=[
        {"component": "Tile", "doc": "https://onnx.ai/onnx/operators/onnx__Tile.html"}
    ],
    since="v0.1.0",
    context="primitives.jnp",
    component="tile",
    testcases=[
        {
            "testcase": "tile_repeats_tensor",
            "callable": _my_tile,
            "input_shapes": [(1, 1, 8)],
        },
        {
            "testcase": "tile_a",
            "callable": lambda a: jnp.tile(a, (1, 2)),
            "input_shapes": [(2, 3)],
        },
        {
            "testcase": "tile_b",
            "callable": lambda a: jnp.tile(a, (1, 2, 1)),
            "input_shapes": [(1, 5, 5)],
        },
        {
            "testcase": "tile_c",
            "callable": lambda a: jnp.tile(a, (1, 4)),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "tile_d",
            "callable": lambda a: jnp.tile(a, 2),
            "input_shapes": [(3, 3)],
        },
        {
            "testcase": "tile_dynamic",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [("B", 3)],
        },
        {
            "testcase": "tile_pad",
            "callable": lambda a: jnp.tile(a, (2, 3, 4)),
            "input_shapes": [(4, 5)],
        },
    ],
)
class TilePlugin(PrimitiveLeafPlugin):

    @staticmethod
    def abstract_eval(x, repeats):  # Keep signature matching primitive def
        import numpy as np
        from jax import core

        # --- Input Validation and Info Extraction ---
        if not hasattr(x, "shape") or not hasattr(x, "dtype"):
            raise TypeError(
                f"Input 'x' to Tile abstract eval must be an abstract value, got {type(x)}"
            )
        x_shape = list(x.shape)
        x_rank = len(x_shape)
        try:
            x_dtype = np.dtype(x.dtype)
        except TypeError:
            x_dtype = np.dtype("float32")  # Fallback

        # --- Determine repeats_list (concrete ints or -1) and repeats_rank ---
        repeats_list = []
        repeats_rank = 0

        # Check concrete types FIRST
        if isinstance(repeats, (int, np.integer)):
            logging.debug("Repeats type: int/np.integer")  # DEBUG
            repeats_list = [int(repeats)] if int(repeats) >= 0 else [-1]
            repeats_rank = 1
            if repeats_list[0] == -1:
                pass
        elif isinstance(repeats, (tuple, list)):
            logging.debug("Repeats type: tuple/list")  # DEBUG
            repeats_rank = len(repeats)
            repeats_list = [-1] * repeats_rank
            for i, r in enumerate(repeats):
                if isinstance(r, (int, np.integer)) and r >= 0:
                    repeats_list[i] = int(r)
                else:
                    pass
        elif isinstance(repeats, np.ndarray):
            logging.debug("Repeats type: np.ndarray")  # DEBUG
            if repeats.ndim == 1:
                repeats_list_raw = repeats.tolist()
                repeats_rank = len(repeats_list_raw)
                repeats_list = [-1] * repeats_rank
                for i, r in enumerate(repeats_list_raw):
                    if isinstance(r, (int, np.integer)) and r >= 0:
                        repeats_list[i] = int(r)
                    else:
                        pass
            else:
                raise TypeError("Tile repeats array must be 1D")
        # Check Literal tracer SECOND (represents constants)
        elif isinstance(repeats, Literal):
            logging.debug("Repeats type: core.Literal")  # DEBUG
            val = repeats.val
            if isinstance(val, (int, np.integer)):
                repeats_list = [int(val)] if int(val) >= 0 else [-1]
                repeats_rank = 1
            elif isinstance(val, np.ndarray) and val.ndim == 1:
                repeats_list_raw = val.tolist()
                repeats_rank = len(repeats_list_raw)
                repeats_list = [-1] * repeats_rank
                for i, r in enumerate(repeats_list_raw):
                    if isinstance(r, (int, np.integer)) and r >= 0:
                        repeats_list[i] = int(r)
                    else:
                        pass
            elif isinstance(val, (tuple, list)):
                repeats_list_raw = list(val)
                repeats_rank = len(repeats_list_raw)
                repeats_list = [-1] * repeats_rank
                for i, r in enumerate(repeats_list_raw):
                    if isinstance(r, (int, np.integer)) and r >= 0:
                        repeats_list[i] = int(r)
                    else:
                        pass
            else:
                raise TypeError(
                    f"Unsupported Literal value type for repeats: {type(val)}"
                )
            if -1 in repeats_list:
                pass
        # Check ShapedArray tracer LAST (assume dynamic if not Literal)
        elif isinstance(repeats, core.ShapedArray):
            logging.debug("Repeats type: core.ShapedArray (Tracer)")  # DEBUG
            if repeats.ndim == 1:
                repeats_rank = repeats.shape[0]
                # Assume dynamic if it's a ShapedArray and not a Literal
                repeats_list = [-1] * repeats_rank
                logging.debug(f"  Marking repeats as abstract: {repeats_list}")  # DEBUG
            else:
                raise TypeError("Tile repeats tensor must be 1D")
        # Handle any other Tracer type
        elif isinstance(repeats, core.Tracer):
            logging.debug(f"Repeats type: Generic Tracer {type(repeats)}")  # DEBUG
            logging.debug(
                f"[WARN] Tile abstract eval got generic Tracer for repeats: {repeats}. Treating as fully abstract."
            )
            repeats_rank = x_rank  # Best guess
            repeats_list = [-1] * repeats_rank
        else:
            raise TypeError(
                f"Unsupported repeats type for abstract eval: {type(repeats)}"
            )

        logging.debug(
            f"Initial repeats_list: {repeats_list}, Rank: {repeats_rank}"
        )  # DEBUG

        # --- Padding logic ---
        if repeats_rank < x_rank:
            repeats_list = [1] * (x_rank - repeats_rank) + repeats_list
            repeats_rank = x_rank
            logging.debug(f"Padded repeats_list: {repeats_list}")  # DEBUG
        elif x_rank < repeats_rank:
            x_shape = [-1] * (repeats_rank - x_rank) + x_shape
            x_rank = repeats_rank
            logging.debug(f"Padded x_shape: {x_shape}")  # DEBUG

        output_rank = x_rank
        output_shape = []

        # --- Calculate output shape ---
        for i in range(output_rank):
            s_raw = x_shape[i]
            s = (
                int(s_raw)
                if isinstance(s_raw, (int, np.integer)) and s_raw >= 0
                else -1
            )
            r = repeats_list[i]  # Already int or -1
            if s == -1 or r == -1:
                output_shape.append(-1)
            else:
                output_shape.append(s * r)

        final_output_shape_tuple = tuple(output_shape)
        logging.debug(
            f"Final computed shape tuple: {final_output_shape_tuple}"
        )  # DEBUG

        return core.ShapedArray(final_output_shape_tuple, x_dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        input_name = s.get_name(node_inputs[0])
        output_name = s.get_name(node_outputs[0])
        input_aval = node_inputs[0].aval
        input_shape = input_aval.shape
        input_dtype = input_aval.dtype
        onnx_dtype_enum = helper.np_dtype_to_tensor_dtype(np.dtype(input_dtype))

        if len(node_inputs) > 1:
            repeats_input_name = s.get_name(node_inputs[1])
            repeats_dtype = node_inputs[1].aval.dtype
            if repeats_dtype != np.int64:
                casted_name = s.get_unique_name("repeats_casted")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[repeats_input_name],
                    outputs=[casted_name],
                    name=s.get_unique_name("cast_repeats"),
                    to=onnx.TensorProto.INT64,
                )
                s.add_node(cast_node)
                s.builder.add_value_info(
                    casted_name,
                    shape=node_inputs[1].aval.shape,
                    dtype=onnx.TensorProto.INT64,
                )
                s.builder.register_value_info_metadata(
                    casted_name,
                    shape=node_inputs[1].aval.shape,
                    dtype=onnx.TensorProto.INT64,
                )
                repeats_input_name = casted_name
            repeats_rank = node_inputs[1].aval.shape[0]
        elif "repeats" in params:
            repeats = params["repeats"]
            if isinstance(repeats, int):
                repeats = (repeats,)
            if any(isinstance(r, core.Tracer) for r in repeats):
                raise TypeError("Cannot export jnp.tile with tracer in static repeats.")
            repeats = tuple(int(r) for r in repeats)
            repeats_input_name = s.builder.get_constant_name(
                np.array(repeats, dtype=np.int64)
            )
            repeats_rank = len(repeats)
        else:
            raise ValueError("Missing repeats information for jnp.tile.")

        if repeats_rank > len(input_shape):
            padded_shape = (1,) * (repeats_rank - len(input_shape)) + input_shape
            shape_name = s.builder.get_constant_name(
                np.array(padded_shape, dtype=np.int64)
            )
            reshaped_name = s.get_unique_name("tile_reshape")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[input_name, shape_name],
                    outputs=[reshaped_name],
                    name=s.get_unique_name("reshape_before_tile"),
                )
            )
            s.builder.register_value_info_metadata(
                reshaped_name, shape=padded_shape, dtype=onnx_dtype_enum
            )
            s.builder.add_value_info(
                reshaped_name, shape=padded_shape, dtype=onnx_dtype_enum
            )
            actual_input = reshaped_name
        else:
            actual_input = input_name

        tile_node = helper.make_node(
            "Tile",
            inputs=[actual_input, repeats_input_name],
            outputs=[output_name],
            name=s.get_unique_name("tile"),
        )
        s.add_node(tile_node)

        # Register output shape and type explicitly
        out_aval = node_outputs[0].aval
        out_shape = out_aval.shape
        out_dtype = helper.np_dtype_to_tensor_dtype(np.dtype(out_aval.dtype))
        s.builder.register_value_info_metadata(
            output_name, shape=out_shape, dtype=out_dtype
        )
        s.builder.add_value_info(output_name, shape=out_shape, dtype=out_dtype)

    @staticmethod
    def _tile(a, reps: int | Sequence[int] | core.Tracer):
        if isinstance(reps, core.Tracer) or isinstance(reps, core.ShapedArray):
            return jnp.tile_p.bind(a, reps)
        else:
            reps_tuple = TilePlugin._determine_dimensions(reps, len(a.shape))
            return jnp.tile_p.bind(a, repeats=reps_tuple)

    @staticmethod
    def _determine_dimensions(
        reps: int | Sequence[int], operand_ndim: int
    ) -> tuple[int, ...]:
        reps_tuple = (reps,) if isinstance(reps, int) else tuple(reps)
        if len(reps_tuple) < operand_ndim:
            reps_tuple = (1,) * (operand_ndim - len(reps_tuple)) + reps_tuple
        return reps_tuple

    @staticmethod
    def get_monkey_patch():
        def patched_tile(a, reps):
            return TilePlugin._tile(a, reps)

        return patched_tile

    @staticmethod
    def patch_info():
        return {
            "patch_targets": [jnp],
            "patch_function": lambda _: TilePlugin._tile,
            "target_attribute": "tile",
        }


# Register abstract evaluation rule
jnp.tile_p.def_abstract_eval(TilePlugin.abstract_eval)
