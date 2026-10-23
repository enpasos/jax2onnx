# jax2onnx/converter/source_convolution.py

"""Finite native FP32 convolution contracts for ordinary SOURCE admission.

The shared existing Conv lowerer owns layouts and emission. These checks preserve
the actual traced convolution; target execution and numerical evidence remain
separate, and transposed/complex/other-precision convolutions remain excluded.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import jax
import numpy as np

if TYPE_CHECKING:
    from jax2onnx.converter.source_profile import _Site, _Tensor


def _integer(value: Any, *, positive: bool = True) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and (int(value) > 0 if positive else int(value) >= 0)
    )


def validate_native_conv(
    eqn: Any, inputs: Sequence[_Tensor], outputs: Sequence[_Tensor], site: _Site
) -> None:
    from jax2onnx.converter.source_profile import _arity, _same_shape

    _arity(eqn, 2, 1, site)
    fields = {
        "window_strides",
        "padding",
        "lhs_dilation",
        "rhs_dilation",
        "dimension_numbers",
        "feature_group_count",
        "batch_group_count",
        "precision",
        "preferred_element_type",
    }
    site.require(
        eqn.primitive is jax.lax.conv_general_dilated_p
        and fields <= set(eqn.params)
        and set(eqn.params) <= fields | {"out_sharding"},
        "convolution carrier identity or parameters are not the native schema",
        "the actual two-input native convolution and its complete finite parameter schema",
        category="lowering-defect",
    )
    site.require(
        site.opset == 23
        and all(value.dtype == "float32" for value in (*inputs, *outputs)),
        "convolution dtype or opset is outside the native FP32 route",
        "unchanged FP32 operands/result at the original opset23",
    )
    params = eqn.params
    precision, preferred = params["precision"], params["preferred_element_type"]
    site.require(
        (
            precision is None
            or isinstance(precision, tuple)
            and len(precision) == 2
            and all(value is jax.lax.Precision.DEFAULT for value in precision)
        )
        and (
            preferred is None
            or isinstance(preferred, np.dtype)
            and preferred == np.dtype(np.float32)
        )
        and params.get("out_sharding") is None,
        "convolution precision, preferred result or placement is outside the native route",
        "DEFAULT precision, absent/same FP32 preferred_element_type, and unspecified placement",
    )
    lhs, rhs = inputs
    result = outputs[0]
    rank = len(lhs.shape)
    site.require(
        rank in {3, 4} and len(rhs.shape) == len(result.shape) == rank,
        "convolution rank is outside the existing layout routes",
        "native 1D/2D tensors with matching operand/result ranks",
    )
    dimensions = params["dimension_numbers"]
    site.require(
        type(dimensions) is jax.lax.ConvDimensionNumbers
        and all(
            isinstance(spec, tuple)
            and len(spec) == rank
            and all(_integer(axis, positive=False) for axis in spec)
            and tuple(sorted(spec)) == tuple(range(rank))
            for spec in dimensions
        ),
        "convolution dimension_numbers are not native axis permutations",
        "actual ConvDimensionNumbers with distinct integral axes covering each rank",
        category="lowering-defect",
    )
    lhs_spec, rhs_spec, out_spec = dimensions
    data_layouts = {(0, 1, 2), (0, 2, 1)} if rank == 3 else {(0, 1, 2, 3), (0, 3, 1, 2)}
    kernel_layouts = (
        {(0, 1, 2), (2, 1, 0)} if rank == 3 else {(0, 1, 2, 3), (3, 2, 0, 1)}
    )
    site.require(
        lhs_spec in data_layouts
        and rhs_spec in kernel_layouts
        and out_spec in data_layouts,
        "convolution layouts are outside the existing native Conv lowering",
        "NCW/NWC with OIW/WIO, or NCHW/NHWC with OIHW/HWIO; leading batch remains axis zero",
    )
    site.require(
        rhs.symbol is None
        and all(_integer(size) for size in rhs.shape)
        and all(
            _integer(size)
            for size in (lhs.shape if lhs.symbol is None else lhs.shape[1:])
        ),
        "convolution spatial, kernel or batch extents are outside the static contract",
        "positive static spatial/kernel/features and a positive static or actual shared leading batch",
    )
    spatial = rank - 2
    for key in ("window_strides", "lhs_dilation", "rhs_dilation"):
        values = params[key]
        site.require(
            isinstance(values, tuple)
            and len(values) == spatial
            and all(_integer(value) for value in values),
            f"convolution {key} is not a native spatial tuple",
            "one positive non-Boolean integer per spatial axis",
            category="lowering-defect",
        )
    site.require(
        all(value == 1 for value in params["lhs_dilation"]),
        "input-dilated convolution requires separate ConvTranspose admission",
        "unit lhs_dilation for the existing ordinary Conv route",
    )
    padding = params["padding"]
    site.require(
        isinstance(padding, tuple)
        and len(padding) == spatial
        and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(_integer(value, positive=False) for value in pair)
            for pair in padding
        ),
        "convolution padding is outside the native nonnegative Conv pads contract",
        "actual explicit nonnegative integer pairs after native VALID/SAME normalization; cropping remains excluded",
    )
    groups, batch_groups = params["feature_group_count"], params["batch_group_count"]
    site.require(
        _integer(groups) and _integer(batch_groups),
        "convolution group counts are not native positive integers",
        "positive non-Boolean feature/batch group counts",
        category="lowering-defect",
    )
    site.require(
        batch_groups == 1,
        "batch-grouped convolution is not implemented by the shared Conv lowerer",
        "batch_group_count=1 with actual feature groups",
    )
    site.require(
        lhs.shape[lhs_spec[1]] == rhs.shape[rhs_spec[1]] * groups
        and rhs.shape[rhs_spec[0]] % groups == 0,
        "convolution feature-group channel dimensions disagree",
        "lhs channels equal kernel input channels times groups, and kernel output channels divisible by groups",
        category="lowering-defect",
    )
    expected: list[int | str] = [0] * rank
    expected[out_spec[0]] = lhs.shape[lhs_spec[0]]
    expected[out_spec[1]] = rhs.shape[rhs_spec[0]]
    for axis, (lo, hi), stride, dilation in zip(
        range(spatial),
        padding,
        params["window_strides"],
        params["rhs_dilation"],
        strict=True,
    ):
        size = lhs.shape[lhs_spec[axis + 2]]
        kernel = rhs.shape[rhs_spec[axis + 2]]
        extent = (
            int(size) + lo + hi - ((int(kernel) - 1) * dilation + 1)
        ) // stride + 1
        site.require(
            extent > 0,
            "empty convolution spatial outputs lack a native target contract",
            "positive static output extents under the actual padding/stride/dilation",
        )
        expected[out_spec[axis + 2]] = int(extent)
    _same_shape(result.shape, tuple(expected), site)
    site.require(
        result.symbol == lhs.symbol
        and eqn.outvars[0].aval.weak_type
        == all(var.aval.weak_type for var in eqn.invars),
        "convolution result scope or weakness differs from native operands",
        "the same leading batch scope and native all-input weak-type rule",
        category="lowering-defect",
    )
