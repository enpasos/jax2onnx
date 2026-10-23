# jax2onnx/plugins/equinox/eqx/nn/recurrent.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import equinox as eqx
import jax
from jax.core import ShapedArray
from jax.extend.core import Primitive

from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    with_requested_dtype,
    with_prng_key,
    register_native_primitive,
)

_RECURRENT_ONNX: Final[list[dict[str, str]]] = [
    {"component": "Gemm", "doc": "https://onnx.ai/onnx/operators/onnx__Gemm.html"},
    {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    {
        "component": "Sigmoid",
        "doc": "https://onnx.ai/onnx/operators/onnx__Sigmoid.html",
    },
    {"component": "Tanh", "doc": "https://onnx.ai/onnx/operators/onnx__Tanh.html"},
]


def _make_eqx_gru_cell(dtype: Any, key: jax.Array) -> Callable[..., Any]:
    cell = eqx.nn.GRUCell(input_size=3, hidden_size=4, dtype=dtype, key=key)

    def call(h: jax.Array, x: jax.Array) -> jax.Array:
        return cell(x, h)

    return call


def _make_eqx_lstm_cell(dtype: Any, key: jax.Array) -> Callable[..., Any]:
    cell = eqx.nn.LSTMCell(input_size=3, hidden_size=4, dtype=dtype, key=key)

    def call(h: jax.Array, c: jax.Array, x: jax.Array) -> Any:
        return cell(x, (h, c))

    return call


_EQX_GRU_CELL: Final = construct_and_call(
    _make_eqx_gru_cell,
    dtype=with_requested_dtype(),
    key=with_prng_key(0),
)
_EQX_LSTM_CELL: Final = construct_and_call(
    _make_eqx_lstm_cell,
    dtype=with_requested_dtype(),
    key=with_prng_key(1),
)


@register_native_primitive(
    jaxpr_primitive="eqx.nn.gru_cell",
    jax_doc="https://docs.kidger.site/equinox/api/nn/rnn/#equinox.nn.GRUCell",
    onnx=_RECURRENT_ONNX,
    since="0.12.2",
    context="primitives.eqx",
    component="gru_cell",
    testcases=[
        {
            "testcase": "eqx_gru_cell_basic",
            "callable": _EQX_GRU_CELL,
            "input_shapes": [(4,), (3,)],
            "expected_output_shapes": [(4,)],
            "run_only_f32_variant": True,
        },
    ],
)
class GRUCellPlugin(PrimitiveLeafPlugin):
    """IR-only support marker for ``equinox.nn.GRUCell`` via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.gru_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: ShapedArray, *args: object, **kwargs: object) -> ShapedArray:
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: object, eqn: object) -> None:
        raise NotImplementedError(
            "GRUCell primitive should not reach lowering; it is inlined."
        )


@register_native_primitive(
    jaxpr_primitive="eqx.nn.lstm_cell",
    jax_doc="https://docs.kidger.site/equinox/api/nn/rnn/#equinox.nn.LSTMCell",
    onnx=_RECURRENT_ONNX,
    since="0.12.2",
    context="primitives.eqx",
    component="lstm_cell",
    testcases=[
        {
            "testcase": "eqx_lstm_cell_basic",
            "callable": _EQX_LSTM_CELL,
            "input_shapes": [(4,), (4,), (3,)],
            "expected_output_shapes": [(4,), (4,)],
            "run_only_f32_variant": True,
        },
    ],
)
class LSTMCellPlugin(PrimitiveLeafPlugin):
    """IR-only support marker for ``equinox.nn.LSTMCell`` via JAX ops."""

    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.lstm_cell")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x: ShapedArray, *args: object, **kwargs: object) -> ShapedArray:
        del args, kwargs
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: object, eqn: object) -> None:
        raise NotImplementedError(
            "LSTMCell primitive should not reach lowering; it is inlined."
        )
