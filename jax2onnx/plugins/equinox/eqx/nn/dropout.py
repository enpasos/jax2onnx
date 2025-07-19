# jax2onnx/plugins/equinox/eqx/nn/dropout.py
"""
ONNX plugin for equinox.nn.Dropout.

This plugin provides a mapping from `equinox.nn.Dropout` to the ONNX `Dropout`
operator, handling both static and dynamic `inference` modes.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable

import equinox as eqx
import numpy as np
from jax import core
from jax.extend.core import Literal, Primitive, Var
from jax.interpreters import batching
from onnx import helper

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive

if TYPE_CHECKING:
    from jax2onnx.converter.jaxpr_converter import Jaxpr2OnnxConverter


eqx.nn.dropout_p = Primitive("eqx.nn.dropout")
eqx.nn.dropout_p.multiple_results = False


@register_primitive(
    jaxpr_primitive=eqx.nn.dropout_p.name,
    jax_doc="https://docs.kidger.site/equinox/api/nn/dropout/",
    onnx=[
        {
            "component": "Dropout",
            "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html",
        },
        {
            "component": "Not",
            "doc": "https://onnx.ai/onnx/operators/onnx__Not.html",
        },
    ],
    since="v0.7.2",
    context="primitives.eqx",
    component="dropout",
    testcases=[
        # TODO: enable testcases
        # {
        #     "testcase": "eqx_dropout_inference_mode",
        #     "callable": eqx.nn.Dropout(p=0.42, inference=True),
        #     "input_shapes": [(64,)],
        #     "post_check_onnx_graph": lambda m: (
        #         (
        #             dropout_node := next(
        #                 (n for n in m.graph.node if n.op_type == "Dropout"),
        #                 None,
        #             )
        #         )
        #         and (
        #             ratio_init := next(
        #                 (
        #                     i
        #                     for i in m.graph.initializer
        #                     if i.name == dropout_node.input[1]
        #                 ),
        #                 None,
        #             )
        #         )
        #         and np.isclose(numpy_helper.to_array(ratio_init), 0.42).all()
        #         and (
        #             training_mode_init := next(
        #                 (
        #                     i
        #                     for i in m.graph.initializer
        #                     if i.name == dropout_node.input[2]
        #                 ),
        #                 None,
        #             )
        #         )
        #         and not numpy_helper.to_array(training_mode_init)
        #     ),
        # },
        # {
        #     "testcase": "eqx_dropout_training_mode",
        #     "callable": lambda x,
        #     key,
        #     model=eqx.nn.Dropout(p=0.5, inference=False): model(x, key=key),
        #     "input_shapes": [("B", 64), ()],
        #     "post_check_onnx_graph": lambda m: (
        #         (
        #             dropout_node := next(
        #                 (n for n in m.graph.node if n.op_type == "Dropout"),
        #                 None,
        #             )
        #         )
        #         and (
        #             training_mode_init := next(
        #                 (
        #                     i
        #                     for i in m.graph.initializer
        #                     if i.name == dropout_node.input[2]
        #                 ),
        #                 None,
        #             )
        #         )
        #         and numpy_helper.to_array(training_mode_init)
        #     ),
        # },
        # {
        #     "testcase": "eqx_dropout_dynamic_inference",
        #     "callable": lambda x, key, inference, model=eqx.nn.Dropout(p=0.5): model(
        #         x, key=key, inference=inference
        #     ),
        #     "input_shapes": [(64,), ()],
        #     "input_params": {
        #         "inference": np.array(True, dtype=bool),
        #     },
        #     "input_names": ["x", "key", "inference"],
        #     "post_check_onnx_graph": lambda m: (
        #         (
        #             not_node := next(
        #                 (n for n in m.graph.node if n.op_type == "Not"), None
        #             )
        #         )
        #         and (
        #             dropout_node := next(
        #                 (n for n in m.graph.node if n.op_type == "Dropout"),
        #                 None,
        #             )
        #         )
        #         and dropout_node.input[2] == not_node.output[0]
        #         and not_node.input[0] == "inference"
        #         and (
        #             ratio_init := next(
        #                 (
        #                     i
        #                     for i in m.graph.initializer
        #                     if i.name == dropout_node.input[1]
        #                 ),
        #                 None,
        #             )
        #         )
        #         and np.isclose(numpy_helper.to_array(ratio_init), 0.5).all()
        #     ),
        # },
    ],
)
class EqxDropoutPlugin(PrimitiveLeafPlugin):
    """Convert **equinox.nn.Dropout** to an ONNX Dropout operator."""

    _ORIGINAL_DROPOUT_CALL: Callable[..., Any] | None = None

    @staticmethod
    def abstract_eval(
        x: core.ShapedArray, inference: core.ShapedArray, *, p: float
    ) -> core.ShapedArray:
        """The output shape is always identical to the input shape."""
        return core.ShapedArray(x.shape, x.dtype)

    def to_onnx(
        self,
        s: "Jaxpr2OnnxConverter",
        node_inputs: list,
        node_outputs: list,
        params: dict,
    ):
        """Maps the primitive to an ONNX Dropout node."""
        x_var, inference_var = node_inputs
        y_var = node_outputs[0]

        logging.debug("[DEBUG] Dropout to_onnx called")
        logging.debug(f"[DEBUG] Input tensor name: {x_var}")
        logging.debug(f"[DEBUG] Inference input: {inference_var}")
        logging.debug(f"[DEBUG] Output name: {y_var}")

        x_name = s.get_name(x_var)
        y_name = s.get_name(y_var)

        # The ONNX `ratio` is equivalent to equinox's `p`.
        # This is a static parameter for the primitive.
        p = params["p"]
        ratio_tensor = np.array(p, dtype=np.float32)
        ratio_name = s.builder.get_constant_name(ratio_tensor)
        logging.debug(f"[DEBUG] Dropout rate: {p}")

        # The ONNX `training_mode` input is the logical inverse of equinox's
        # `inference` flag. We must handle both static and dynamic cases.
        if isinstance(inference_var, Literal):
            # Static case: `inference` is a compile-time constant.
            is_training = not bool(inference_var.val)
            training_mode_tensor = np.array(is_training, dtype=bool)
            training_mode_name = s.builder.get_constant_name(training_mode_tensor)
        else:
            # Dynamic case: `inference` is a runtime input to the graph.
            # We need to add a `Not` operator to invert the boolean value.
            # inference_name = s.get_name(inference_var)
            # training_mode_name = s.get_unique_name("training_mode")
            # s.add_node(
            #     helper.make_node(
            #         "Not",
            #         inputs=[inference_name],
            #         outputs=[training_mode_name],
            #         name=s.get_unique_name("not_inference"),
            #     )
            # )
            inference_name = "inference"
            # FIX: Use Var from jax.extend.core for the type check
            if isinstance(inference_var, Var):
                s.change_var_name(inference_var, inference_name)

            # Add a `Not` operator to invert the boolean value.
            training_mode_name = s.get_unique_name("training_mode")
            s.add_node(
                helper.make_node(
                    "Not",
                    inputs=[inference_name],
                    outputs=[training_mode_name],
                    name=s.get_unique_name("not_inference"),
                )
            )
            s.add_shape_info(
                training_mode_name,
                inference_var.aval.shape,
                inference_var.aval.dtype,
            )

        # The ONNX Dropout operator has three inputs: data, ratio, training_mode.
        # The second output (mask) is optional and we do not produce it.
        s.add_node(
            helper.make_node(
                "Dropout",
                inputs=[x_name, ratio_name, training_mode_name],
                outputs=[y_name],
                name=s.get_unique_name("dropout"),
            )
        )
        s.add_shape_info(y_name, x_var.aval.shape, x_var.aval.dtype)

    @staticmethod
    def get_monkey_patch(orig_fn: Callable[..., Any]) -> Callable[..., Any]:
        """Return a patched version of __call__ that uses the primitive."""
        EqxDropoutPlugin._ORIGINAL_DROPOUT_CALL = orig_fn

        def patched_call(self, x, *, key=None, inference=None, deterministic=None):
            # This logic is copied directly from the original Equinox implementation
            # to ensure the correct `inference` value is determined before binding.
            if deterministic is not None:
                inference = deterministic

            if inference is None:
                inference = self.inference
            if isinstance(self.p, (int, float)) and self.p == 0:
                inference = True

            # The JAX random key is not needed for the ONNX graph definition.
            # Bind the inputs to our custom primitive.
            return eqx.nn.dropout_p.bind(x, inference, p=self.p)

        return patched_call

    @staticmethod
    def patch_info() -> dict:
        """Specifies the target for monkey-patching."""
        return {
            "patch_targets": [eqx.nn.Dropout],
            "patch_function": EqxDropoutPlugin.get_monkey_patch,
            "target_attribute": "__call__",
        }


eqx.nn.dropout_p.def_abstract_eval(EqxDropoutPlugin.abstract_eval)


def _eqx_dropout_batching_rule(batched_args, batch_dims, *, p: float):
    """Batching rule for `eqx.nn.dropout_p`."""
    x, inference = batched_args
    x_bdim, inference_bdim = batch_dims

    # Batching over the `inference` flag is not a standard use case.
    if inference_bdim is not None:
        raise NotImplementedError(
            "Batching over the `inference` parameter of `eqx.nn.Dropout` is not supported."
        )

    # The primitive is applied to the batched `x`. The `to_onnx` implementation
    # correctly handles inputs with leading batch dimensions.
    out = eqx.nn.dropout_p.bind(x, inference, p=p)

    # The output has a batch dimension at the same axis as the input.
    return out, x_bdim


batching.primitive_batchers[eqx.nn.dropout_p] = _eqx_dropout_batching_rule
