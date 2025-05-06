# file: jax2onnx/plugins/jax/numpy/tile.py

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import logging
import numpy as np
import jax
from onnx import helper, TensorProto

from jax import core
from jax import numpy as jnp

from jax.extend.core import Literal, Primitive
from jax._src.export.shape_poly import _DimExpr as DimExpr

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

logger = logging.getLogger("jax2onnx.plugins.jax.numpy.tile")

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter
    from jax2onnx.converter.onnx_builder import OnnxBuilder


# ---------------------------------------------------------------------------
# Helper: do two shape "tokens" refer to the *same* symbolic dimension?
# ---------------------------------------------------------------------------
def _same_symbol(a, b) -> bool:
    if a is b:
        return True
    return str(a) == str(b)


# --- Define custom primitive ---
jnp.tile_p = Primitive("jnp.tile")
jnp.tile_p.multiple_results = False


# --- Input functions for testcases ---


def _tile_param(a):
    B = a.shape[0]
    param = jnp.zeros((1, 1, 4), dtype=a.dtype)
    return jnp.tile(param, (B, 1, 1))


def _my_dynamic_tile(x):
    B = x.shape[0]
    repeats = (B, 1, 1)
    return jnp.tile(x, repeats)


# --- Register the plugin ---
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
            "testcase": "tile_repeats",
            "callable": lambda a: jnp.tile(a, (3, 1, 1)),
            "input_shapes": [(1, 1, 8)],
            "expected_output_shapes": [(3, 1, 8)],
        },
        {
            "testcase": "tile_a",
            "callable": lambda a: jnp.tile(a, (1, 2)),
            "input_shapes": [(2, 3)],
            "expected_output_shapes": [(2, 6)],
        },
        {
            "testcase": "tile_b",
            "callable": lambda a: jnp.tile(a, (1, 2, 1)),
            "input_shapes": [(1, 5, 5)],
            "expected_output_shapes": [(1, 10, 5)],
        },
        {
            "testcase": "tile_c",
            "callable": lambda a: jnp.tile(a, (1, 4)),
            "input_shapes": [(3, 3)],
            "expected_output_shapes": [(3, 12)],
        },
        {
            "testcase": "tile_d",  # Tests scalar repeat
            "callable": lambda a: jnp.tile(a, 2),
            "input_shapes": [(3, 3)],
            "expected_output_shapes": [(3, 6)],
        },
        {
            "testcase": "tile_dynamic_input_static",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [(7, 3)],
            "expected_output_shapes": [(14, 3)],
        },
        {
            "testcase": "tile_dynamic_input",
            "callable": lambda a: jnp.tile(a, (2, 1)),
            "input_shapes": [("B", 3)],
            # TODO: "expected_output_shapes": [("2*B", 3)],
        },
        {
            "testcase": "tile_pad",  # Repeats rank > input rank
            "callable": lambda a: jnp.tile(a, (2, 3, 4)),
            "input_shapes": [(4, 5)],
            "expected_output_shapes": [(2, 12, 20)],
        },
        {
            "testcase": "tile_with_symbolic_repeats_static",
            "callable": _my_dynamic_tile,  # repeats=(B, 1, 1)
            "input_shapes": [(11, 1, 256)],  # Symbolic input
            "expected_output_shapes": [(121, 1, 256)],
        },
        {
            "testcase": "tile_with_symbolic_repeats",
            "callable": _my_dynamic_tile,  # repeats=(B, 1, 1)
            "input_shapes": [("B", 1, 256)],  # Symbolic input
            # TODO: "expected_output_shapes": [("B*B", 1, 256)],
        },
        {
            "testcase": "tile_param_symbolic",
            "callable": _tile_param,
            "input_shapes": [("B", 5)],
            "expected_output_shapes": [("B", 1, 4)],
        },
    ],
)
class TilePlugin(PrimitiveLeafPlugin):
    _orig_tile = None  # Cache original jnp.tile

    @staticmethod
    def abstract_eval(x_aval, *maybe_repeats_aval, **static):
        """Computes the output ShapedArray of `tile`."""
        repeats_static_or_aval = static.get("repeats")
        if repeats_static_or_aval is None and maybe_repeats_aval:
            repeats_static_or_aval = maybe_repeats_aval[0]

        if repeats_static_or_aval is None:
            raise ValueError("Could not determine repeats for abstract evaluation.")

        repeats_rank = 0
        repeats_li = []

        # Determine repeats rank and content
        if isinstance(repeats_static_or_aval, (int, np.integer, DimExpr)):
            repeats_rank = 1
            repeats_li = [repeats_static_or_aval]
        elif isinstance(repeats_static_or_aval, (tuple, list)):
            repeats_rank = len(repeats_static_or_aval)
            repeats_li = list(repeats_static_or_aval)
        elif isinstance(repeats_static_or_aval, Literal):
            val = np.asarray(repeats_static_or_aval.val)
            if val.ndim == 0:
                repeats_rank, repeats_li = 1, [int(val)]
            elif val.ndim == 1:
                repeats_rank, repeats_li = len(val), val.tolist()
            else:
                raise ValueError(
                    f"Tile repeats Literal must be scalar or 1D, got {val.ndim}D"
                )
        elif hasattr(repeats_static_or_aval, "shape"):  # Dynamic array (ShapedArray)
            rep_aval = repeats_static_or_aval
            if rep_aval.ndim != 1:
                raise ValueError(f"Tile repeats array must be 1D, got {rep_aval.ndim}D")
            if not isinstance(rep_aval.shape[0], int):
                raise ValueError(
                    f"Rank of dynamic repeats tensor must be concrete, got {rep_aval.shape[0]}"
                )
            repeats_rank = rep_aval.shape[0]
            repeats_li = [None] * repeats_rank  # Values are unknown
        else:
            raise TypeError(
                f"Unsupported repeats type for abstract eval: {type(repeats_static_or_aval)}"
            )

        # ---- Symbolic Multiplication Helper ----
        def _mul_dim(d, r):
            if r is None:
                return None  # Unknown repeat -> unknown dim
            if isinstance(r, DimExpr) or isinstance(d, DimExpr):
                if _same_symbol(r, 1):
                    return d
                if _same_symbol(d, 1):
                    return r
                if _same_symbol(d, r):
                    return d
                try:
                    return d * r  # Let JAX handle DimExpr * DimExpr or DimExpr * int
                except Exception:
                    return None  # Fallback if symbolic multiplication fails
            if isinstance(r, str) or isinstance(
                d, str
            ):  # Basic string handling (fallback)
                if str(r) == "1":
                    return d
                if str(d) == "1":
                    return r
                return None  # Treat complex string cases as unknown
            # Standard integer multiplication
            if r == 1:
                return d
            if d == 1:
                return r
            if d is None:
                return None  # Propagate unknown dimension
            # Ensure both are numeric before multiplying
            if isinstance(d, (int, np.integer)) and isinstance(r, (int, np.integer)):
                return d * r
            # If we reach here, types are incompatible for concrete multiplication
            logger.warning(
                f"Cannot multiply types {type(d)} and {type(r)}, returning None."
            )
            return None

        # ---- Align Ranks and Calculate Output Shape ----
        in_shape = list(x_aval.shape)
        in_rank = len(in_shape)

        if repeats_rank < in_rank:
            repeats_li = [1] * (in_rank - repeats_rank) + repeats_li
        elif in_rank < repeats_rank:
            in_shape = [1] * (repeats_rank - in_rank) + in_shape
        elif in_rank == 0:  # Handle scalar input
            if repeats_rank > 0:
                in_shape = [1] * repeats_rank
            else:  # both scalar -> output scalar
                return core.ShapedArray((), x_aval.dtype)

        out_shape_list = []
        for d, r in zip(in_shape, repeats_li):
            out_dim = _mul_dim(d, r)
            # Replace complex DimExpr results with None for ShapedArray compatibility
            if (
                isinstance(out_dim, DimExpr)
                and hasattr(out_dim, "_op")
                and out_dim._op is not None
            ):
                logger.debug(
                    f"Replacing complex DimExpr {out_dim} with None for ShapedArray."
                )
                out_shape_list.append(None)
            else:
                out_shape_list.append(out_dim)

        # Final check: ensure no None remains if not allowed (might depend on JAX version)
        # If ShapedArray requires concrete ints or simple DimExpr, this needs adjustment.
        # For now, assume None is okay if a dimension is truly dynamic/unknown.
        final_out_shape = tuple(out_shape_list)

        # Check if final_out_shape contains None before creating ShapedArray
        # If JAX fails with None, we might need to use symbolic zeros or other placeholders.
        # Let's try with None first as it reflects the abstract state.
        try:
            return core.ShapedArray(final_out_shape, x_aval.dtype)
        except TypeError as e:
            # If TypeError occurs (e.g., "Shapes must be 1D sequences of integer scalars..."),
            # it means None is not acceptable. Provide a more informative error or fallback.
            logger.error(
                f"Failed to create ShapedArray with shape {final_out_shape} containing None. Error: {e}"
            )
            # Fallback: return a shape with rank but unknown dims (all None), hoping downstream handles it.
            # This might still fail if the rank itself becomes problematic.
            output_rank = len(final_out_shape)
            # Using jax.core.DYN_DIM if available
            try:
                unknown_dim_rep = core.DYN_DIM  # JAX >= 0.4.14 ?
                logger.warning(
                    f"Falling back to dynamic shape ({output_rank}*[DYN_DIM]) due to None issue."
                )
                return core.ShapedArray((unknown_dim_rep,) * output_rank, x_aval.dtype)
            except AttributeError:
                logger.error(
                    "Cannot create ShapedArray with None dimensions and core.DYN_DIM not found. Abstract eval inaccurate."
                )
                # Last resort fallback, likely incorrect shape but avoids crash here.
                return core.ShapedArray((1,) * output_rank, x_aval.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[Any],
        node_outputs: Sequence[Any],
        params: dict[str, Any],
    ):
        """Converts jnp.tile primitive to ONNX Tile operator."""
        input_var = node_inputs[0]
        output_var = node_outputs[0]
        builder: OnnxBuilder = s.builder

        input_name = s.get_name(input_var)
        output_name = s.get_name(output_var)
        input_aval = input_var.aval
        input_rank = len(input_aval.shape)

        tile_input_name = input_name
        repeats_name = ""
        repeats_rank = 0

        # -------- Figure out "repeats" input tensor --------
        if len(node_inputs) == 2:
            # Case 1: repeats is a dynamic JAX value
            repeats_var = node_inputs[1]
            repeats_name = s.get_name(repeats_var)
            repeats_aval = repeats_var.aval

            if not hasattr(repeats_aval, "shape") or not hasattr(repeats_aval, "dtype"):
                raise TypeError(
                    f"Expected abstract value for dynamic repeats, got {type(repeats_aval)}"
                )
            if repeats_aval.ndim != 1:
                raise ValueError(
                    f"ONNX Tile requires repeats to be a 1D tensor, got shape {repeats_aval.shape}"
                )

            repeats_rank_dim = repeats_aval.shape[0]
            if not isinstance(repeats_rank_dim, int):
                raise ValueError(
                    f"Rank of dynamic 'repeats' tensor ('{repeats_name}') is symbolic ({repeats_rank_dim}), cannot determine padding."
                )
            repeats_rank = repeats_rank_dim

            if repeats_aval.dtype != np.int64:
                casted_repeats_name = s.get_unique_name(f"{repeats_name}_int64")
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[repeats_name],
                    outputs=[casted_repeats_name],
                    to=TensorProto.INT64,
                )
                s.add_node(cast_node)
                # Assuming builder._dim_to_symbol is fixed/works
                shape_symbolic = tuple(
                    builder._dim_to_symbol(d) for d in repeats_aval.shape
                )
                builder.register_value_info_metadata(
                    casted_repeats_name, shape_symbolic, TensorProto.INT64
                )
                builder.add_value_info(
                    casted_repeats_name, shape_symbolic, TensorProto.INT64
                )
                repeats_name = casted_repeats_name
        elif "repeats" in params:
            # Case 2: repeats is a static tuple/list in params
            repeats_tuple_raw = params["repeats"]
            repeats_tuple = (
                (repeats_tuple_raw,)
                if isinstance(repeats_tuple_raw, int)
                else tuple(repeats_tuple_raw)
            )
            repeats_rank = len(repeats_tuple)
            repeats_parts_names = []

            for i, r in enumerate(repeats_tuple):
                part_name = ""
                if isinstance(r, (int, np.integer)):
                    val = np.array([int(r)], dtype=np.int64)
                    part_name = builder.get_constant_name(val)
                elif isinstance(r, DimExpr):
                    origin = s.symbolic_dim_to_origin.get(
                        r
                    ) or s.symbolic_dim_to_origin.get(str(r))
                    if origin is None:
                        raise ValueError(
                            f"Symbolic dimension '{r}' has no registered input origin."
                        )

                    source_tensor_name, source_axis_index = origin

                    # ---- Shape node ---------------------------------------------------
                    shape_out = s.get_unique_name(f"shape_of_{source_tensor_name}")
                    s.add_node(
                        helper.make_node(
                            "Shape", inputs=[source_tensor_name], outputs=[shape_out]
                        )
                    )

                    # try to add value‑info with the real rank if we know it
                    try:
                        source_shape_meta, _ = builder.get_shape_dtype(
                            source_tensor_name
                        )
                        builder.register_value_info_metadata(
                            shape_out, (len(source_shape_meta),), TensorProto.INT64
                        )
                        builder.add_value_info(
                            shape_out, (len(source_shape_meta),), TensorProto.INT64
                        )
                    except ValueError:
                        # builder doesn't know the rank yet → register a minimal stub
                        builder.add_value_info(
                            shape_out, (None,), TensorProto.INT64  # 1‑D, unknown length
                        )

                    # ---- Gather the single dimension we need --------------------------
                    axis_const = builder.get_constant_name(
                        np.array([source_axis_index], dtype=np.int64)
                    )
                    gather_out = s.get_unique_name(f"gather_dim_{source_axis_index}")
                    s.add_node(
                        helper.make_node(
                            "Gather",
                            inputs=[shape_out, axis_const],
                            outputs=[gather_out],
                            axis=0,
                        )
                    )
                    builder.register_value_info_metadata(
                        gather_out, (1,), TensorProto.INT64
                    )
                    builder.add_value_info(gather_out, (1,), TensorProto.INT64)
                    part_name = gather_out
                else:
                    raise TypeError(
                        f"Unsupported type in static tile repeats: {type(r)} ({r})"
                    )
                repeats_parts_names.append(part_name)

            final_repeats_name = s.get_unique_name("repeats_concat")
            concat_node = helper.make_node(
                "Concat",
                inputs=repeats_parts_names,
                outputs=[final_repeats_name],
                axis=0,
            )
            s.add_node(concat_node)
            builder.register_value_info_metadata(
                final_repeats_name, (repeats_rank,), TensorProto.INT64
            )
            builder.add_value_info(
                final_repeats_name, (repeats_rank,), TensorProto.INT64
            )
            repeats_name = final_repeats_name
        else:
            raise ValueError("Missing 'repeats' information for jnp.tile operation.")

        # -------- Handle Rank Padding for Repeats tensor (if input_rank > repeats_rank) --------
        final_repeats_name_for_tile = repeats_name
        if input_rank > repeats_rank:
            logger.debug(
                f"Padding repeats tensor (rank {repeats_rank}) to match input rank ({input_rank})"
            )
            num_pads = input_rank - repeats_rank
            ones_const_name = builder.get_constant_name(
                np.ones(num_pads, dtype=np.int64)
            )
            padded_repeats_name = s.get_unique_name(f"{repeats_name}_rank_padded")
            s.add_node(
                helper.make_node(
                    "Concat",
                    inputs=[ones_const_name, repeats_name],
                    outputs=[padded_repeats_name],
                    axis=0,
                )
            )
            builder.register_value_info_metadata(
                padded_repeats_name, (input_rank,), TensorProto.INT64
            )
            builder.add_value_info(
                padded_repeats_name, (input_rank,), TensorProto.INT64
            )
            final_repeats_name_for_tile = padded_repeats_name

        # -------- Handle Rank Padding for Input tensor (if repeats_rank > input_rank) --------
        if repeats_rank > input_rank:
            num_new_dims = repeats_rank - input_rank
            target_shape_parts = []
            ones_const = builder.get_constant_name(
                np.array([1] * num_new_dims, dtype=np.int64)
            )
            target_shape_parts.append(ones_const)

            input_shape_name = s.get_unique_name(f"shape_{input_name}")
            s.add_node(
                helper.make_node(
                    "Shape", inputs=[input_name], outputs=[input_shape_name]
                )
            )
            builder.register_value_info_metadata(
                input_shape_name, (input_rank,), TensorProto.INT64
            )
            builder.add_value_info(input_shape_name, (input_rank,), TensorProto.INT64)
            target_shape_parts.append(input_shape_name)

            target_shape_name = s.get_unique_name("padded_shape_dynamic")
            s.add_node(
                helper.make_node(
                    "Concat",
                    inputs=target_shape_parts,
                    outputs=[target_shape_name],
                    axis=0,
                )
            )
            builder.register_value_info_metadata(
                target_shape_name, (repeats_rank,), TensorProto.INT64
            )
            builder.add_value_info(
                target_shape_name, (repeats_rank,), TensorProto.INT64
            )

            reshaped_input_name = s.get_unique_name(f"{input_name}_rank_padded")
            s.add_node(
                helper.make_node(
                    "Reshape",
                    inputs=[input_name, target_shape_name],
                    outputs=[reshaped_input_name],
                )
            )

            # Assuming builder._dim_to_symbol is fixed/works
            padded_symbolic_shape = tuple(
                builder._dim_to_symbol(d)
                for d in ([1] * num_new_dims + list(input_aval.shape))
            )
            onnx_dtype = helper.np_dtype_to_tensor_dtype(np.dtype(input_aval.dtype))
            builder.register_value_info_metadata(
                reshaped_input_name, padded_symbolic_shape, onnx_dtype
            )
            builder.add_value_info(
                reshaped_input_name, padded_symbolic_shape, onnx_dtype
            )
            tile_input_name = reshaped_input_name

        # -------- Create the final Tile node --------
        tile_node = helper.make_node(
            "Tile",
            inputs=[tile_input_name, final_repeats_name_for_tile],
            outputs=[output_name],
        )
        s.add_node(tile_node)

        # -------- Register output shape information --------
        out_aval = output_var.aval

        # Convert DimExprs to symbolic strings ('B') or None for complex expressions/unknowns
        def shape_elem_to_onnx(d):
            """Converts JAX shape element (int, DimExpr, str) to ONNX representation (int, str, None)."""
            if isinstance(d, (int, np.integer)):
                return int(d)
            if d is None:
                return None  # Propagate known unknown
            try:
                # Try resolving via builder (assumes builder typo _dimvar... is fixed!)
                resolved = builder._dim_to_symbol(d)
                # Check if resolution produced a complex string (like '2*B') -> represent as None
                # A simple symbol like 'B' is okay. Allow single letter/digit or starting with __sym
                if isinstance(resolved, str) and not (
                    len(resolved) == 1
                    or resolved.isalnum()
                    or resolved.startswith("__sym")
                ):
                    logger.warning(
                        f"Representing complex symbolic dim '{resolved}' as dynamic (None) in ONNX graph."
                    )
                    return None  # Represent complex/unmappable symbolic dims as dynamic in ONNX
                return resolved
            except AttributeError as e:
                if "_dimvar_to_name_by_str" in str(e):
                    logger.error(
                        "AttributeError in builder._dim_to_symbol still present! Fix onnx_builder.py."
                    )
                    return None
                else:
                    logger.warning(
                        f"AttributeError resolving {d}: {e}, marking as dynamic (None)."
                    )
                    return None
            except Exception as e:  # General fallback
                logger.warning(
                    f"Failed to resolve shape dim {d} via builder ({e}), marking as dynamic (None)."
                )
                return None

        out_shape_onnx = tuple(shape_elem_to_onnx(d) for d in out_aval.shape)
        out_dtype_enum = helper.np_dtype_to_tensor_dtype(np.dtype(out_aval.dtype))

        builder.register_value_info_metadata(
            output_name, out_shape_onnx, out_dtype_enum
        )
        builder.add_value_info(output_name, out_shape_onnx, out_dtype_enum)

    @staticmethod
    def _tile(a, reps: int | Sequence[int | DimExpr] | core.Tracer):
        """Helper to choose the correct JAX primitive binding."""
        # Check for jax.Array first, as it might also be a Tracer
        if isinstance(reps, jax.Array) and not isinstance(reps, core.Tracer):
            np_val = np.asarray(reps)
            if np_val.ndim == 0:
                return jnp.tile_p.bind(a, repeats=(int(np_val),))
            if np_val.ndim == 1:
                # Bind static arrays as tuples for keyword binding
                return jnp.tile_p.bind(a, repeats=tuple(int(x) for x in np_val))
            raise ValueError("tile: repeats array must be 0‑ or 1‑D")
        # Handle Tracers (dynamic arrays) -> positional binding
        elif isinstance(reps, core.Tracer):
            return jnp.tile_p.bind(a, reps)
        # Handle host-side constants (numpy arrays, lists, tuples, ints)
        elif isinstance(reps, np.ndarray):
            reps_jax = jnp.asarray(
                reps.astype(np.int64)
            )  # Convert to JAX array for binding
            return jnp.tile_p.bind(a, reps_jax)  # Bind JAX array positionally
        elif isinstance(reps, (int, np.integer)):
            return jnp.tile_p.bind(a, repeats=(int(reps),))  # Bind scalar via keyword
        elif isinstance(reps, (tuple, list)):
            # Bind tuple/list via keyword, preserving DimExpr objects if present
            return jnp.tile_p.bind(a, repeats=tuple(reps))
        else:
            raise TypeError(f"Unsupported 'reps' type for jnp.tile: {type(reps)}")

    @staticmethod
    def get_monkey_patch(orig_fn):
        """Returns the patched function."""
        if TilePlugin._orig_tile is None:
            TilePlugin._orig_tile = orig_fn

        def patched_tile(a, reps):
            return TilePlugin._tile(a, reps)

        return patched_tile

    @staticmethod
    def patch_info():
        """Provides monkey patching info."""
        return {
            "patch_targets": [jnp],
            "patch_function": TilePlugin.get_monkey_patch,
            "target_attribute": "tile",
        }


# Register abstract evaluation rule
jnp.tile_p.def_abstract_eval(TilePlugin.abstract_eval)
