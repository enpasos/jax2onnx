# file: jax2onnx/plugins/jax/numpy/take.py

from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax import core
from jax.extend.core import Primitive, Var
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

# 1. Define a custom JAX primitive to represent jnp.take.
#    This allows us to write a specific handler for it.
jnp.take_p = Primitive("jnp_take")


# Test function that reproduces the data-dependent arange->take issue
class ArangeTakeModule(nnx.Module):
    """A module to reproduce the gpt.py pattern."""

    def __init__(self, num_embeddings: int, features: int, *, rngs: nnx.Rngs):
        self.embedding = nnx.Param(
            jax.random.normal(rngs.params(), (num_embeddings, features))
        )

    def __call__(self, x: jax.Array):
        """
        1. Get shape `T` from input `x`.
        2. Create indices `[0, ..., T-1]` using arange.
        3. Use indices to gather from an embedding table via jnp.take.
        """
        T = x.shape[1]
        indices = jnp.arange(T)
        return jnp.take(self.embedding.value, indices, axis=0)


@register_primitive(
    jaxpr_primitive=jnp.take_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.take.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        }
    ],
    since="v0.7.0",
    context="primitives.jnp",
    component="take",
    testcases=[
        {
            "testcase": "take_data_dependent_indices",
            "callable": ArangeTakeModule(
                num_embeddings=10,  # Must be >= the sequence length
                features=16,
                rngs=nnx.Rngs(0),
            ),
            # Input `x` has shape (B, T) = (3, 10)
            "input_shapes": [(3, 10)],
            "input_dtypes": [jnp.float32],
            # The bug exists in f32, no need to run f64 as well.
            "run_only_f32_variant": True,
        },
    ],  # The `arange_gather_repro` will test this
)
class TakePlugin(PrimitiveLeafPlugin):
    """
    Plugin to convert jnp.take to ONNX Gather.

    This plugin intercepts calls to `jnp.take` and maps them to the
    ONNX Gather operator, which is the direct equivalent for the common
    use case (indexing along a single axis).
    """

    _ORIG_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(arr_aval, indices_aval, *, axis, **kwargs):
        """
        Abstract evaluation rule to determine the output shape of the take operation.
        """
        # The output shape is the concatenation of the array's shape before the axis,
        # the indices' shape, and the array's shape after the axis.
        output_shape = (
            arr_aval.shape[:axis] + indices_aval.shape + arr_aval.shape[axis + 1 :]
        )
        return core.ShapedArray(output_shape, arr_aval.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: list[Var],
        node_outputs: list[Var],
        params: dict[str, Any],
    ):
        """Converts the custom jnp_take primitive to an ONNX Gather node."""
        arr_var, indices_var = node_inputs
        (output_var,) = node_outputs
        axis = params["axis"]

        arr_name = s.get_name(arr_var)
        indices_name = s.get_name(indices_var)

        # ── ONNX Gather wants int64 indices ────────────────────────────────────
        # If our JAX indices are any integer type ≠ int64, insert a Cast→INT64.
        import numpy as _np
        from onnx import TensorProto

        if (
            _np.issubdtype(indices_var.aval.dtype, _np.integer)
            and indices_var.aval.dtype != _np.int64
        ):
            casted = s.get_unique_name(f"{indices_name}_cast_int64")
            cast_node = helper.make_node(
                "Cast",
                inputs=[indices_name],
                outputs=[casted],
                to=TensorProto.INT64,
                name=s.get_unique_name("Cast"),
            )
            s.add_node(cast_node)
            # preserve shape info (same shape, but now int64)
            s.add_shape_info(casted, indices_var.aval.shape, _np.int64)
            indices_name = casted
        # ─────────────────────────────────────────────────────────────────────

        output_name = s.get_name(output_var)

        # jnp.take with an axis corresponds directly to ONNX Gather.
        gather_node = helper.make_node(
            "Gather",
            inputs=[arr_name, indices_name],
            outputs=[output_name],
            axis=axis,
            name=s.get_unique_name("take_gather"),
        )
        s.add_node(gather_node)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable[..., Any]):
        """
        Returns a patched version of jnp.take that uses our custom primitive.
        """
        TakePlugin._ORIG_CALL = orig_fn

        def patched_take(arr, indices, *, axis=None, **kwargs):
            # For now, only intercept the simple case (axis is specified).
            # The GPT model uses `axis=0`.
            if axis is None or kwargs.get("mode") is not None:
                # Fallback to the original jnp.take for more complex cases
                # like 'clip' or 'wrap' modes, which require more logic.
                return TakePlugin._ORIG_CALL(arr, indices, axis=axis, **kwargs)

            # Bind our custom primitive, which will be handled by this plugin.
            return jnp.take_p.bind(arr, indices, axis=axis)

        return patched_take

    @staticmethod
    def patch_info() -> dict[str, Any]:
        """Provides the information needed to monkey-patch jnp.take."""
        return {
            "patch_targets": [jnp],
            "patch_function": TakePlugin.get_monkey_patch,
            "target_attribute": "take",
        }


# Register the abstract evaluation rule with our new primitive
jnp.take_p.def_abstract_eval(TakePlugin.abstract_eval)
