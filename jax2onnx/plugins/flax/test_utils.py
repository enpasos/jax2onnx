# jax2onnx/plugins/flax/test_utils.py

from __future__ import annotations

import inspect
from contextlib import contextmanager

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from flax.nnx import bridge


class _LinenToNNXCallable:
    def __init__(self, model, rngs):
        self._model = model
        self._rngs = rngs

    def __call__(self, *args, **kwargs):
        return self._model(*args, rngs=self._rngs, **kwargs)


def _filter_kwargs_for_signature(target, kwargs):
    if not kwargs:
        return {}
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters)
    allowed.discard("self")
    return {key: val for key, val in kwargs.items() if key in allowed}


def _extract_raw_key(rngs):
    if rngs is None:
        return None
    if isinstance(rngs, nnx.Rngs):
        if "params" in rngs:
            return rngs["params"].key.value
        if "default" in rngs:
            return rngs["default"].key.value
        raise ValueError("NNX RNGs must define a 'params' or 'default' stream.")
    return rngs


def _split_keys(key, count):
    if key is None:
        return (None,) * count
    try:
        return tuple(jax.random.split(key, count))
    except Exception:
        return (key,) * count


def _call_with_rngs(model, rngs, *args, **kwargs):
    if rngs is None:
        return model(*args, **kwargs)
    return model(*args, rngs=rngs, **kwargs)


@contextmanager
def _use_original_jnp_shape():
    try:
        from jax2onnx.plugins.jax.numpy._common import get_orig_impl
        from jax2onnx.plugins.jax.numpy.shape import JnpShapePlugin
    except Exception:
        yield
        return

    try:
        orig_shape = get_orig_impl(JnpShapePlugin._PRIM, JnpShapePlugin._FUNC_NAME)
    except Exception:
        yield
        return

    current_shape = jnp.shape
    if current_shape is orig_shape:
        yield
        return

    jnp.shape = orig_shape
    try:
        yield
    finally:
        jnp.shape = current_shape


def _rnn_input_shape_without_time(inputs, cell, time_major: bool):
    time_axis = 0 if time_major else inputs.ndim - (cell.num_feature_axes + 1)
    if time_axis < 0:
        time_axis += inputs.ndim
    return inputs.shape[:time_axis] + inputs.shape[time_axis + 1 :]


def _rnn_carry_dtype(cell, inputs):
    dtype = getattr(cell, "param_dtype", None)
    if dtype is None:
        dtype = getattr(cell, "dtype", None)
    if dtype is None:
        dtype = inputs.dtype
    return dtype


def _make_rnn_initial_carry(cell, inputs, *, time_major: bool):
    input_shape = _rnn_input_shape_without_time(inputs, cell, time_major)
    dtype = _rnn_carry_dtype(cell, inputs)

    if isinstance(cell, (nn.LSTMCell, nn.OptimizedLSTMCell)):
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (cell.features,)
        c = jnp.zeros(mem_shape, dtype=dtype)
        h = jnp.zeros(mem_shape, dtype=dtype)
        return c, h

    if isinstance(cell, nn.ConvLSTMCell):
        num_feature_axes = cell.num_feature_axes
        signal_dims = input_shape[-num_feature_axes:-1]
        batch_dims = input_shape[:-num_feature_axes]
        mem_shape = batch_dims + signal_dims + (cell.features,)
        c = jnp.zeros(mem_shape, dtype=dtype)
        h = jnp.zeros(mem_shape, dtype=dtype)
        return c, h

    batch_dims = input_shape[:-1]
    mem_shape = batch_dims + (cell.features,)
    return jnp.zeros(mem_shape, dtype=dtype)


class _LinenToNNXRecurrentCallable:
    def __init__(self, model, rngs, *, carry_is_tuple):
        self._model = model
        self._rngs = rngs
        self._carry_is_tuple = carry_is_tuple

    def __call__(self, *args, **kwargs):
        if self._carry_is_tuple:
            if len(args) < 3:
                raise ValueError("Expected carry tuple (c, h) and inputs.")
            carry = (args[0], args[1])
            inputs = args[2]
            rest = args[3:]
            return _call_with_rngs(
                self._model, self._rngs, carry, inputs, *rest, **kwargs
            )
        if len(args) < 2:
            raise ValueError("Expected carry and inputs.")
        carry = args[0]
        inputs = args[1]
        rest = args[2:]
        return _call_with_rngs(self._model, self._rngs, carry, inputs, *rest, **kwargs)


class _LinenToNNXRNNCallable:
    def __init__(self, model, rngs, carry_init):
        self._model = model
        self._rngs = rngs
        self._carry_init = carry_init

    def __call__(self, inputs, **kwargs):
        if "initial_carry" in kwargs:
            initial_carry = kwargs.pop("initial_carry")
        else:
            initial_carry = self._carry_init(inputs)
        with _use_original_jnp_shape():
            return _call_with_rngs(
                self._model,
                self._rngs,
                inputs,
                initial_carry=initial_carry,
                **kwargs,
            )


def linen_to_nnx(
    module_cls,
    input_shape=(1, 32),
    dtype=jnp.float32,
    rngs=None,
    **kwargs,
):
    """Wrap a Linen module as NNX and initialize it with a dummy input."""
    module = module_cls(**kwargs)
    model = bridge.ToNNX(module, rngs=rngs)
    dummy_x = jnp.zeros(input_shape, dtype=dtype)
    if isinstance(rngs, nnx.Rngs):
        # Avoid mutating NNX RNG state during JAX tracing by using a raw key.
        if "params" in rngs:
            rngs = rngs["params"].key.value
        elif "default" in rngs:
            rngs = rngs["default"].key.value
        else:
            raise ValueError("NNX RNGs must define a 'params' or 'default' stream.")
    if rngs is None:
        model.lazy_init(dummy_x)
        return model
    model.lazy_init(dummy_x, rngs=rngs)
    return _LinenToNNXCallable(model, rngs)


def linen_rnn_cell_to_nnx(
    module_cls,
    input_shape=(1, 32),
    dtype=jnp.float32,
    rngs=None,
    **kwargs,
):
    """Wrap a Linen RNN cell as NNX and initialize it with a dummy carry+input."""
    module_kwargs = _filter_kwargs_for_signature(module_cls, kwargs)
    module = module_cls(**module_kwargs)
    model = bridge.ToNNX(module, rngs=rngs)
    dummy_x = jnp.zeros(input_shape, dtype=dtype)

    raw_key = _extract_raw_key(rngs)
    carry_key, init_key, call_key = _split_keys(raw_key, 3)
    if carry_key is None:
        carry_key = jax.random.PRNGKey(0)
    carry = module.initialize_carry(carry_key, dummy_x.shape)

    if init_key is None:
        model.lazy_init(carry, dummy_x)
    else:
        model.lazy_init(carry, dummy_x, rngs=init_key)

    carry_is_tuple = isinstance(carry, (tuple, list))
    return _LinenToNNXRecurrentCallable(model, call_key, carry_is_tuple=carry_is_tuple)


def linen_rnn_to_nnx(
    cell_cls,
    input_shape=(1, 4, 8),
    dtype=jnp.float32,
    rngs=None,
    *,
    cell_kwargs=None,
    time_major: bool = False,
    return_carry: bool = False,
):
    """Wrap a Linen RNN module as NNX and initialize it with a fixed carry."""
    cell_kwargs = cell_kwargs or {}
    cell = cell_cls(**_filter_kwargs_for_signature(cell_cls, cell_kwargs))
    rnn = nn.RNN(cell=cell, time_major=time_major, return_carry=return_carry)
    model = bridge.ToNNX(rnn, rngs=rngs)
    dummy_inputs = jnp.zeros(input_shape, dtype=dtype)

    raw_key = _extract_raw_key(rngs)
    init_key = raw_key
    call_key = raw_key

    def carry_init(xs):
        return _make_rnn_initial_carry(cell, xs, time_major=time_major)

    initial_carry = carry_init(dummy_inputs)

    if init_key is None:
        model.lazy_init(dummy_inputs, initial_carry=initial_carry)
    else:
        model.lazy_init(dummy_inputs, initial_carry=initial_carry, rngs=init_key)

    return _LinenToNNXRNNCallable(model, call_key, carry_init)


def linen_bidirectional_to_nnx(
    cell_cls,
    input_shape=(1, 4, 8),
    dtype=jnp.float32,
    rngs=None,
    *,
    cell_kwargs=None,
    time_major: bool = False,
    return_carry: bool = False,
):
    """Wrap a Linen Bidirectional module as NNX with fixed carries."""
    cell_kwargs = cell_kwargs or {}
    forward_cell = cell_cls(**_filter_kwargs_for_signature(cell_cls, cell_kwargs))
    backward_cell = cell_cls(**_filter_kwargs_for_signature(cell_cls, cell_kwargs))

    forward_rnn = nn.RNN(cell=forward_cell, time_major=time_major)
    backward_rnn = nn.RNN(cell=backward_cell, time_major=time_major)

    bidir = nn.Bidirectional(
        forward_rnn=forward_rnn,
        backward_rnn=backward_rnn,
        time_major=time_major,
        return_carry=return_carry,
    )
    model = bridge.ToNNX(bidir, rngs=rngs)
    dummy_inputs = jnp.zeros(input_shape, dtype=dtype)

    raw_key = _extract_raw_key(rngs)
    init_key = raw_key
    call_key = raw_key

    def carry_init(xs):
        return (
            _make_rnn_initial_carry(forward_cell, xs, time_major=time_major),
            _make_rnn_initial_carry(backward_cell, xs, time_major=time_major),
        )

    initial_carry = carry_init(dummy_inputs)

    if init_key is None:
        model.lazy_init(dummy_inputs, initial_carry=initial_carry)
    else:
        model.lazy_init(dummy_inputs, initial_carry=initial_carry, rngs=init_key)

    return _LinenToNNXRNNCallable(model, call_key, carry_init)
