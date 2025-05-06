# file: jax2onnx/plugins/flax/nnx/einsum.py

from typing import TYPE_CHECKING, Callable, Any, Optional
from types import SimpleNamespace  # For dummy instance

import jax
from flax import nnx
from jax import core
from jax.extend.core import Primitive
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


# Define a primitive for nnx.Einsum (the module call)
# Use a distinct name to avoid clashing with jnp.einsum's primitive
einsum_module_p = Primitive("nnx_einsum_module")
einsum_module_p.multiple_results = False


@register_primitive(
    primitive_obj=einsum_module_p,
    # binding_factory = None, # Patching __call__ directly
    jaxpr_primitive=einsum_module_p.name,
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Einsum",
    onnx=[
        {
            "component": "Einsum",
            "doc": "https://onnx.ai/onnx/operators/onnx__Einsum.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="v0.4.2",
    context="primitives.nnx",
    component="einsum",
    testcases=[  # Keep existing testcases
        {
            "testcase": "einsum_module_with_bias",
            "callable": nnx.Einsum(
                "nta,hab->nthb", (8, 2, 4), (8, 4), rngs=nnx.Rngs(0)
            ),
            "input_shapes": [(16, 11, 2)],
        },
        {
            "testcase": "einsum_module_no_bias",
            "callable": nnx.Einsum("nta,hab->nthb", (8, 2, 4), None, rngs=nnx.Rngs(0)),
            "input_shapes": [(16, 11, 2)],
        },
    ],
)
class EinsumModulePlugin(PrimitiveLeafPlugin):
    """Plugin for flax.nnx.Einsum module using jax.eval_shape."""

    _ORIG_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        kernel: core.ShapedArray,
        # Bias is optional positional argument in bind based on _einsum_module_binding
        *maybe_bias: Optional[core.ShapedArray],
        # Primitive parameters passed via bind kwargs
        einsum_str: str,
        use_bias_bool: bool,  # Pass boolean indicating if bias was originally present
        precision: Any | None = None,
        dtype: Any | None = None,
        param_dtype: Any | None = None,
    ):
        """Abstract eval using jax.eval_shape on the original module call."""
        if EinsumModulePlugin._ORIG_CALL is None:
            raise RuntimeError("Original nnx.Einsum.__call__ not captured.")

        bias = maybe_bias[0] if maybe_bias else None

        def _helper(x_arg, kernel_arg, bias_arg=None):
            # Define the missing _infer_broadcasted_bias_shape method with correct signature
            def infer_bias_shape(
                self, x_shape, einsum_output_shape, input_for_bias=None
            ):
                # Just return the original bias shape without any complex broadcasting logic
                # This is only needed for shape inference during tracing
                # Match the signature with what's expected in flax.nnx.Einsum
                return bias_arg.shape if bias_arg is not None else None

            # Reconstruct a dummy instance with necessary attributes for the original call
            dummy_instance = SimpleNamespace(
                kernel=SimpleNamespace(value=kernel_arg),
                # Set bias correctly based on whether bias_arg was provided
                bias=SimpleNamespace(value=bias_arg) if bias_arg is not None else None,
                einsum_str=einsum_str,
                # Fix: Add required method implementations for internal flax.nnx.Einsum methods
                _einsum_str_check=lambda s: None,  # No-op function
                _infer_broadcasted_bias_shape=infer_bias_shape,  # Add missing method
                precision=precision,
                dtype=dtype,
                param_dtype=param_dtype,
                # Add other potentially accessed attributes
                bias_init=None,
                kernel_init=None,
                kernel_shape=kernel_arg.shape if hasattr(kernel_arg, "shape") else None,
                bias_shape=None if bias_arg is None else bias_arg.shape,
                promote_dtype=lambda a, **kw: a,
                einsum_op=jax.numpy.einsum,
                use_bias=bias_arg is not None,
            )

            # Call the original __call__ method captured from the instance
            return EinsumModulePlugin._ORIG_CALL(dummy_instance, x_arg)

        x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        kernel_spec = jax.ShapeDtypeStruct(kernel.shape, kernel.dtype)

        # Handle bias properly without triggering JAX tracer errors
        if bias is not None:
            bias_spec = jax.ShapeDtypeStruct(bias.shape, bias.dtype)
            # Use helper directly with all args
            out_spec = jax.eval_shape(_helper, x_spec, kernel_spec, bias_spec)
        else:
            # Call helper with None for bias_arg
            out_spec = jax.eval_shape(_helper, x_spec, kernel_spec)

        if not isinstance(out_spec, jax.ShapeDtypeStruct):
            leaves = jax.tree_util.tree_leaves(out_spec)
            if len(leaves) == 1 and isinstance(leaves[0], jax.ShapeDtypeStruct):
                out_spec = leaves[0]
            else:
                raise TypeError(
                    f"eval_shape for Einsum module returned {type(out_spec)}"
                )

        return core.ShapedArray(out_spec.shape, out_spec.dtype)

    def to_onnx(self, s: "Jaxpr2OnnxConverter", node_inputs, node_outputs, params):
        """Handles conversion of the Einsum module primitive."""
        # Input vars: x, kernel, [bias]
        input_var = node_inputs[0]
        kernel_var = node_inputs[1]
        bias_vars = node_inputs[2:]  # List will be empty if no bias

        output_var = node_outputs[0]

        einsum_str = params["einsum_str"]
        # use_bias_bool = params["use_bias_bool"] # Get from params

        input_name = s.get_name(input_var)
        kernel_name = s.get_name(kernel_var)
        output_name = s.get_name(output_var)
        output_aval = output_var.aval

        has_bias = bool(bias_vars)  # Check if bias input exists
        bias_name = s.get_name(bias_vars[0]) if has_bias else None

        einsum_out_name = s.get_unique_name("einsum_out") if has_bias else output_name

        einsum_node = helper.make_node(
            "Einsum",
            inputs=[input_name, kernel_name],
            outputs=[einsum_out_name],
            name=s.get_unique_name("einsum"),
            equation=einsum_str,
        )
        s.add_node(einsum_node)

        # Register shapes using output_aval determined by abstract_eval
        if has_bias:
            # Register intermediate shape (assuming Add preserves it)
            s.add_shape_info(einsum_out_name, output_aval.shape, output_aval.dtype)
        else:
            s.add_shape_info(output_name, output_aval.shape, output_aval.dtype)

        if has_bias:
            add_node = helper.make_node(
                "Add",
                inputs=[einsum_out_name, bias_name],
                outputs=[output_name],
                name=s.get_unique_name("einsum_add_bias"),
            )
            s.add_node(add_node)
            # Register final output shape
            s.add_shape_info(output_name, output_aval.shape, output_aval.dtype)

    @staticmethod
    def _einsum_module_binding(instance, x):
        """Binds inputs to the einsum_module primitive."""
        kernel = instance.kernel.value
        # --- Corrected bias check ---
        has_bias = instance.bias is not None
        bias = instance.bias.value if has_bias else None
        # --- End correction ---

        args = [x, kernel]
        if has_bias:
            args.append(bias)

        # Pass necessary parameters from instance to primitive
        kwargs = {
            "einsum_str": instance.einsum_str,
            "use_bias_bool": has_bias,  # Pass boolean flag
            "precision": getattr(instance, "precision", None),
            "dtype": getattr(instance, "dtype", None),
            "param_dtype": getattr(instance, "param_dtype", None),
        }

        return einsum_module_p.bind(*args, **kwargs)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable):
        """Capture original __call__ and return patched version."""
        EinsumModulePlugin._ORIG_CALL = orig_fn

        def patched_einsum_module_call(self, x):  # 'self' is the nnx.Einsum instance
            return EinsumModulePlugin._einsum_module_binding(self, x)

        return patched_einsum_module_call

    @staticmethod
    def patch_info():
        """Patch the __call__ method of nnx.Einsum."""
        return {
            "patch_targets": [nnx.Einsum],
            "patch_function": EinsumModulePlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


# Register abstract evaluation
einsum_module_p.def_abstract_eval(EinsumModulePlugin.abstract_eval)
