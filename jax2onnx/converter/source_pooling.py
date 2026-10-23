# jax2onnx/converter/source_pooling.py

"""Native FP32 disjoint NHWC window sums; target reduction order is separate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site, _Tensor


def _positive_integer(value: Any) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) > 0
    )


def validate_native_window_sum(
    eqn: Any, inputs: Sequence[_Tensor], outputs: Sequence[_Tensor], site: _Site
) -> tuple[int, int]:
    from jax2onnx.converter.source_profile import _arity, _same_shape

    _arity(eqn, 1, 1, site)
    fields = {
        "window_dimensions",
        "window_strides",
        "padding",
        "base_dilation",
        "window_dilation",
    }
    site.require(
        eqn.primitive is jax.lax.reduce_window_sum_p and set(eqn.params) == fields,
        "window-sum carrier or parameters differ from the native schema",
        "the actual reduce_window_sum primitive and complete five-field native parameters",
        category="lowering-defect",
    )
    source, result = inputs[0], outputs[0]
    site.require(
        site.opset == 23 and source.dtype == result.dtype == "float32",
        "window-sum dtype or opset needs separate implementation",
        "unchanged FP32 input/output at original opset23",
    )
    site.require(
        len(source.shape) == len(result.shape) == 4,
        "window-sum layout needs separate implementation",
        "rank-four NHWC tensors with untouched batch/channel axes",
    )
    site.require(
        all(_positive_integer(value) for value in source.shape[1:])
        and (source.symbol is not None or _positive_integer(source.shape[0])),
        "window-sum extents need separate implementation",
        "positive static H/W/C and a positive static or actual shared leading batch",
    )
    params = eqn.params
    for name in (
        "window_dimensions",
        "window_strides",
        "base_dilation",
        "window_dilation",
    ):
        values = params[name]
        site.require(
            isinstance(values, tuple)
            and len(values) == 4
            and all(_positive_integer(value) for value in values),
            f"window-sum {name} differs from native positive integer parameters",
            "four positive non-Boolean native integers",
            category="lowering-defect",
        )
    padding = params["padding"]
    site.require(
        isinstance(padding, tuple)
        and len(padding) == 4
        and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(
                isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                for value in pair
            )
            for pair in padding
        ),
        "window-sum padding is not native resolved integer pairs",
        "four explicit integer padding pairs after upstream normalization",
        category="lowering-defect",
    )
    windows, strides = params["window_dimensions"], params["window_strides"]
    site.require(
        windows[0] == windows[3] == 1
        and strides == windows
        and params["base_dilation"] == params["window_dilation"] == (1, 1, 1, 1)
        and padding == ((0, 0),) * 4,
        "overlapping, padded, dilated or batch/channel windows need separate implementation",
        "nonoverlapping spatial window=stride, unit dilations and zero resolved padding with unit N/C windows",
    )
    kh, kw = int(windows[1]), int(windows[2])
    height, width, channels = map(int, source.shape[1:])
    site.require(
        height % kh == 0 and width % kw == 0,
        "partial spatial windows need separate implementation",
        "exact positive spatial divisibility by the actual native windows",
    )
    _same_shape(
        result.shape, (source.shape[0], height // kh, width // kw, channels), site
    )
    site.require(
        result.symbol == source.symbol
        and eqn.outvars[0].aval.weak_type == eqn.invars[0].aval.weak_type,
        "window-sum result scope or weakness differs from the native operand",
        "the unchanged native leading batch scope and input weak_type",
        category="lowering-defect",
    )
    return kh, kw
