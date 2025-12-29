# jax2onnx/plugins/flax/linen/recurrent.py

from __future__ import annotations

from typing import ClassVar

from jax.core import ShapedArray
from jax.extend.core import Primitive
from flax import linen as nn

from jax2onnx.plugins.flax.test_utils import (
    linen_bidirectional_to_nnx,
    linen_rnn_cell_to_nnx,
    linen_rnn_to_nnx,
)
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_requested_dtype,
    with_rng_seed,
)


_RECURRENT_ONNX: list[dict[str, str]] = [
    {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
    {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    {
        "component": "Sigmoid",
        "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
    },
    {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"},
]


@register_primitive(
    jaxpr_primitive="linen.simple_cell",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.SimpleCell",
    onnx=_RECURRENT_ONNX,
    since="v0.11.0",
    context="primitives.linen",
    component="simple_cell",
    testcases=[
        {
            "testcase": "simple_cell_basic",
            "callable": construct_and_call(
                linen_rnn_cell_to_nnx,
                module_cls=nn.SimpleCell,
                input_shape=(1, 3),
                features=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                recurrent_kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4), ("B", 3)],
            "expected_output_shapes": [("B", 4), ("B", 4)],
        },
    ],
)
class SimpleCellPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.SimpleCell via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.simple_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "SimpleCell primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.gru_cell",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.GRUCell",
    onnx=_RECURRENT_ONNX,
    since="v0.11.0",
    context="primitives.linen",
    component="gru_cell",
    testcases=[
        {
            "testcase": "gru_cell_basic",
            "callable": construct_and_call(
                linen_rnn_cell_to_nnx,
                module_cls=nn.GRUCell,
                input_shape=(1, 3),
                features=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                recurrent_kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4), ("B", 3)],
            "expected_output_shapes": [("B", 4), ("B", 4)],
        },
    ],
)
class GRUCellPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.GRUCell via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.gru_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "GRUCell primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.mgu_cell",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.MGUCell",
    onnx=_RECURRENT_ONNX,
    since="v0.11.0",
    context="primitives.linen",
    component="mgu_cell",
    testcases=[
        {
            "testcase": "mgu_cell_basic",
            "callable": construct_and_call(
                linen_rnn_cell_to_nnx,
                module_cls=nn.MGUCell,
                input_shape=(1, 3),
                features=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                recurrent_kernel_init=nn.initializers.ones,
                forget_bias_init=nn.initializers.zeros,
                activation_bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4), ("B", 3)],
            "expected_output_shapes": [("B", 4), ("B", 4)],
        },
    ],
)
class MGUCellPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.MGUCell via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.mgu_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "MGUCell primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.lstm_cell",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.LSTMCell",
    onnx=_RECURRENT_ONNX,
    since="v0.11.0",
    context="primitives.linen",
    component="lstm_cell",
    testcases=[
        {
            "testcase": "lstm_cell_basic",
            "callable": construct_and_call(
                linen_rnn_cell_to_nnx,
                module_cls=nn.LSTMCell,
                input_shape=(1, 3),
                features=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                recurrent_kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4), ("B", 4), ("B", 3)],
            "expected_output_shapes": [("B", 4), ("B", 4), ("B", 4)],
        },
    ],
)
class LSTMCellPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.LSTMCell via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.lstm_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "LSTMCell primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.optimized_lstm_cell",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.OptimizedLSTMCell",
    onnx=_RECURRENT_ONNX,
    since="v0.11.0",
    context="primitives.linen",
    component="optimized_lstm_cell",
    testcases=[
        {
            "testcase": "optimized_lstm_cell_basic",
            "callable": construct_and_call(
                linen_rnn_cell_to_nnx,
                module_cls=nn.OptimizedLSTMCell,
                input_shape=(1, 3),
                features=4,
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                kernel_init=nn.initializers.ones,
                recurrent_kernel_init=nn.initializers.ones,
                bias_init=nn.initializers.zeros,
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 4), ("B", 4), ("B", 3)],
            "expected_output_shapes": [("B", 4), ("B", 4), ("B", 4)],
        },
    ],
)
class OptimizedLSTMCellPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.OptimizedLSTMCell via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.optimized_lstm_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "OptimizedLSTMCell primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.conv_lstm_cell",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.ConvLSTMCell",
    onnx=_RECURRENT_ONNX
    + [{"component": "Conv", "doc": "https://onnx.ai/onnx/operators/onnx__Conv.html"}],
    since="v0.11.0",
    context="primitives.linen",
    component="conv_lstm_cell",
    testcases=[
        {
            "testcase": "conv_lstm_cell_basic",
            "callable": construct_and_call(
                linen_rnn_cell_to_nnx,
                module_cls=nn.ConvLSTMCell,
                input_shape=(1, 4, 4, 3),
                features=2,
                kernel_size=(3, 3),
                dtype=with_requested_dtype(),
                param_dtype=with_requested_dtype(),
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [
                ("B", 4, 4, 2),
                ("B", 4, 4, 2),
                ("B", 4, 4, 3),
            ],
            "expected_output_shapes": [
                ("B", 4, 4, 2),
                ("B", 4, 4, 2),
                ("B", 4, 4, 2),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class ConvLSTMCellPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.ConvLSTMCell via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.conv_lstm_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "ConvLSTMCell primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.rnn",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.RNN",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"},
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="rnn",
    testcases=[
        {
            "testcase": "rnn_basic",
            "callable": construct_and_call(
                linen_rnn_to_nnx,
                cell_cls=nn.SimpleCell,
                input_shape=(1, 5, 3),
                dtype=with_requested_dtype(),
                cell_kwargs={
                    "features": 4,
                    "dtype": with_requested_dtype(),
                    "param_dtype": with_requested_dtype(),
                    "kernel_init": nn.initializers.ones,
                    "recurrent_kernel_init": nn.initializers.ones,
                    "bias_init": nn.initializers.zeros,
                },
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 5, 3)],
            "expected_output_shapes": [("B", 5, 4)],
        },
    ],
)
class RNNPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.RNN via scan."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.rnn")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "RNN primitive should not reach lowering; it is inlined."
        )


@register_primitive(
    jaxpr_primitive="linen.bidirectional",
    jax_doc="https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Bidirectional",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="v0.11.0",
    context="primitives.linen",
    component="bidirectional",
    testcases=[
        {
            "testcase": "bidirectional_basic",
            "callable": construct_and_call(
                linen_bidirectional_to_nnx,
                cell_cls=nn.GRUCell,
                input_shape=(1, 5, 3),
                dtype=with_requested_dtype(),
                cell_kwargs={
                    "features": 4,
                    "dtype": with_requested_dtype(),
                    "param_dtype": with_requested_dtype(),
                    "kernel_init": nn.initializers.ones,
                    "recurrent_kernel_init": nn.initializers.ones,
                    "bias_init": nn.initializers.zeros,
                },
                rngs=with_rng_seed(0),
            ),
            "input_shapes": [("B", 5, 3)],
            "expected_output_shapes": [("B", 5, 8)],
        },
    ],
)
class BidirectionalPlugin(PrimitiveLeafPlugin):
    """IR-only support for flax.linen.Bidirectional via scan."""

    _PRIM: ClassVar[Primitive] = Primitive("linen.bidirectional")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, *args, **kwargs):
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        raise NotImplementedError(
            "Bidirectional primitive should not reach lowering; it is inlined."
        )
