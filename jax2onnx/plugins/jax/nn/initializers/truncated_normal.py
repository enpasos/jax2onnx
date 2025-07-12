# file: jax2onnx/plugins/jax/nn/initializers/truncated_normal.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import jax
from jax import core
from jax.extend.core import Primitive
import jax.numpy as jnp
from jax.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
import numpy as np

# Import ONNX helpers
from onnx import helper as onnx_helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

logger = logging.getLogger("jax2onnx.plugins.jax.nn.initializers.truncated_normal")

# Define a custom primitive to intercept the call during tracing.
truncated_normal_p = Primitive("truncated_normal")
truncated_normal_p.multiple_results = False


def _test_truncated_normal_callable(key, lower, upper):
    """A simple test function that directly invokes the initializer."""
    return jax.nn.initializers.truncated_normal(
        key, lower, upper, shape=(4, 5), dtype=jnp.float32
    )


@register_primitive(
    jaxpr_primitive=truncated_normal_p.name,
    context="primitives.nn",
    component="truncated_normal",
    since="v0.7.1",
    testcases=[
        {
            "testcase": "initializer",
            "callable": _test_truncated_normal_callable,
            "input_values": [
                jax.random.PRNGKey(0),
                jnp.array(-2.0, dtype=jnp.float32),
                jnp.array(2.0, dtype=jnp.float32),
            ],
            "expected_output_shapes": [(4, 5)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            "testcase": "random_truncated_normal_positional",
            "callable": lambda key: jax.random.truncated_normal(
                key, -2.0, 2.0, (3, 3), jnp.float32
            ),
            "input_values": [jax.random.PRNGKey(0)],
            "expected_output_shapes": [(3, 3)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
        {
            # Mimics a flax.linen.Dense call-site where the first dim is dynamic
            "testcase": "flax_dense_like_init",
            "callable": lambda key, x: jax.random.truncated_normal(
                key, -2.0, 2.0, (x.shape[-1], 128), jnp.float32
            ),
            "input_values": [
                jax.random.PRNGKey(0),
                jnp.ones((1, 10), dtype=jnp.float32),
            ],
            "expected_output_shapes": [(10, 128)],
            "expected_output_dtypes": [jnp.float32],
            "run_only_f32_variant": True,
        },
    ],
)
class TruncatedNormalPlugin(PrimitiveLeafPlugin):
    """Plugin for converting JAX truncated_normal initializer to an ONNX Constant."""

    @staticmethod
    def abstract_eval(
        key_av: core.ShapedArray,
        lower_av: core.ShapedArray,
        upper_av: core.ShapedArray,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        **kwargs: Any,
    ) -> core.ShapedArray:
        # Replace un-hashable dims with None
        def _safe_dim(d):
            try:
                hash(d)
                return d
            except TypeError:
                return None

        safe_shape = tuple(_safe_dim(d) for d in shape)
        return core.ShapedArray(safe_shape, dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: Sequence[core.Var],
        node_outputs: Sequence[core.Var],
        params: dict[str, Any],
    ) -> None:
        output_var = node_outputs[0]
        output_name = s.get_name(output_var)
        output_aval = output_var.aval

        placeholder_values = np.zeros(output_aval.shape, dtype=output_aval.dtype)
        tensor_proto = onnx_helper.make_tensor(
            name=f"{output_name}_value",
            data_type=NP_TYPE_TO_TENSOR_TYPE[np.dtype(output_aval.dtype)],
            dims=output_aval.shape,
            vals=placeholder_values.flatten().tolist(),
        )
        const_node = onnx_helper.make_node(
            "Constant",
            inputs=[],
            outputs=[output_name],
            value=tensor_proto,
            name=s.builder.get_unique_name("Constant_TruncatedNormal"),
        )
        s.add_node(const_node)

    @staticmethod
    def patch_info():
        from jax import random
        from jax.nn import initializers as jax_init
        import jax.numpy as jnp

        def _to_int(d):
            try:
                return int(d)
            except Exception:
                aval = getattr(d, "aval", None)
                if aval is not None and getattr(aval, "val", None) is not None:
                    return int(aval.val)
                raise

        def _patched_truncated_normal(key, lower, upper, *pos, **kw):
            # resolve shape & dtype
            shape = kw.pop("shape", None)
            dtype = kw.pop("dtype", None)
            if len(pos) >= 1:
                shape = shape or pos[0]
            if len(pos) >= 2:
                dtype = dtype or pos[1]
            shape = shape or ()
            dtype = dtype or jnp.float_

            # sanitize shape dims to ints if possible
            try:
                shape_clean = tuple(_to_int(d) for d in shape)
                all_static = True
            except TypeError:
                shape_clean = shape
                all_static = False

            if all_static:
                return truncated_normal_p.bind(
                    key, lower, upper, shape=shape_clean, dtype=dtype
                )

            # dynamic fallback broadcasted zero
            zero_scalar = jnp.array(0, dtype)
            return jnp.broadcast_to(zero_scalar, shape_clean)

        return {
            "patch_targets": [random, jax_init],
            "target_attribute": "truncated_normal",
            "patch_function": lambda orig: _patched_truncated_normal,
        }


# Register abstract eval for the primitive
truncated_normal_p.def_abstract_eval(TruncatedNormalPlugin.abstract_eval)


def _truncated_normal_lowering_mlir(ctx, key, lower, upper, *, shape, dtype):
    aval_out = ctx.avals_out[0]
    tensor_type = ir.RankedTensorType.get(
        aval_out.shape, mlir.dtype_to_ir_type(aval_out.dtype)
    )
    zero = mlir.ir_constant(np.array(0, dtype=aval_out.dtype))
    return [hlo.BroadcastOp(tensor_type, zero, mlir.dense_int_elements([])).result]


# Register MLIR lowering
mlir.register_lowering(
    truncated_normal_p,
    _truncated_normal_lowering_mlir,
)
