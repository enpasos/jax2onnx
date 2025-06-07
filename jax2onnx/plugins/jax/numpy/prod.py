# file: jax2onnx/plugins/jax/numpy/prod.py

from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
from jax import core
from jax import numpy as jnp
from jax.extend.core import Primitive

from jax2onnx.plugins.jax.lax._reduce_utils import add_reduce_node
from jax2onnx.plugin_system import (
    PrimitiveLeafPlugin,
    register_primitive,
)

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter

jnp.prod_p = Primitive("jnp.prod")
jnp.prod_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=jnp.prod_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.prod.html#jax.numpy.prod",
    onnx=[
        {
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        }
    ],
    since="v0.6.2",
    context="primitives.jnp",
    component="prod",
    testcases=[
        {
            "testcase": "basic_prod",
            "callable": lambda x: jnp.prod(x),
            "input_shapes": [(3, 4)],
        },
        {
            "testcase": "prod_with_axis",
            "callable": lambda x: jnp.prod(x, axis=1),
            "input_shapes": [(3, 4, 5)],
        },
        {
            "testcase": "prod_with_keepdims",
            "callable": lambda x: jnp.prod(x, axis=0, keepdims=True),
            "input_shapes": [(3, 4)],
        },
    ],
)
class ProdPlugin(PrimitiveLeafPlugin):
    _ORIG_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(
        a: core.ShapedArray,
        *,
        axis: int | Sequence[int] | None = None,
        dtype: jnp.dtype | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> Sequence[core.ShapedArray]:
        """Computes the output shape and dtype for jnp.prod using jax.eval_shape."""
        orig_call = ProdPlugin._ORIG_CALL or jnp.prod

        def _helper(arr):
            return orig_call(arr, axis=axis, dtype=dtype, keepdims=keepdims)

        spec_a = jax.ShapeDtypeStruct(a.shape, a.dtype)
        out_spec = jax.eval_shape(_helper, spec_a)

        return [core.ShapedArray(out_spec.shape, out_spec.dtype)]

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        invars: Any,
        outvars: Any,
        params: dict[str, Any],
    ) -> None:  # Changed return type to match superclass
        inp_name = s.get_name(invars[0])
        out_name = s.get_name(outvars[0])
        axes = params.get("axes")
        in_rank = len(invars[0].aval.shape)
        out_rank = len(outvars[0].aval.shape)
        keepdims = in_rank == out_rank

        add_reduce_node(
            s.builder,
            op_type="ReduceProd",
            inp=inp_name,
            out=out_name,
            axes=axes,
            keepdims=int(keepdims),
        )
        # Make sure this function doesn't return anything
        return None  # explicit return None to match the superclass return type

    @staticmethod
    def _prod_binding(*args, **kwargs):
        """This is called as the monkey-patched jnp.prod."""
        out = ProdPlugin._ORIG_CALL(*args, **kwargs)
        # Always return as a tuple if single result
        return (out,) if not jnp.prod_p.multiple_results else out

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Returns the patched function that captures the original and binds the primitive."""
        ProdPlugin._ORIG_CALL = orig_fn
        return ProdPlugin._prod_binding

    @staticmethod
    def patch_info():
        """Provides patching information to the converter for jnp.prod."""
        return {
            "patch_targets": [jnp],
            "target_attribute": "prod",
            "patch_function": ProdPlugin.get_monkey_patch,
        }


jnp.prod_p.def_abstract_eval(ProdPlugin.abstract_eval)
