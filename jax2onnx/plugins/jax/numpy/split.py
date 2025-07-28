# jax2onnx/plugins/jax/numpy/split.py

from __future__ import annotations

from typing import Sequence, Union, Tuple

import jax.numpy as jnp
import numpy as np
from onnx import helper
from jax import core
from jax.extend.core import Primitive

from jax2onnx.plugin_system import PrimitiveLeafPlugin, register_primitive


# ---------------------------------------------------------------------- #
# 1. A dedicated primitive for jnp.split                                 #
# ---------------------------------------------------------------------- #
split_p = Primitive("jnp.split")
split_p.multiple_results = True


# ---------------------------------------------------------------------- #
# 2. Helper function to compute split sizes                              #
# ---------------------------------------------------------------------- #
def _get_split_sizes(
    dim_size: int, indices_or_sections: Union[int, Sequence[int]]
) -> Tuple[int, ...]:
    """Calculate split sizes for jnp.split, mimicking its logic."""
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections
        if dim_size % sections != 0:
            # This logic is consistent with JAX's jnp.split behavior.
            raise ValueError(
                f"Dimension size {dim_size} must be evenly divisible by the number of"
                f" sections {sections}"
            )
        return (dim_size // sections,) * sections
    else:
        # Treats sequence of integers as split points.
        indices = [0] + list(indices_or_sections) + [dim_size]
        return tuple(np.diff(indices).astype(np.int64))


# ---------------------------------------------------------------------- #
# 3. Plugin registration                                                 #
# ---------------------------------------------------------------------- #
@register_primitive(
    jaxpr_primitive=split_p.name,
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.split.html",
    onnx=[
        {"component": "Split", "doc": "https://onnx.ai/onnx/operators/onnx__Split.html"}
    ],
    since="v0.7.2",
    context="primitives.jnp",
    component="split",
    testcases=[
        {
            "testcase": "split_by_sections",
            "callable": lambda x: jnp.split(x, 3, axis=1),
            "input_shapes": [(1, 9)],
        },
        {
            "testcase": "split_by_indices",
            "callable": lambda x: jnp.split(x, np.array([2, 5]), axis=1),
            "input_shapes": [(1, 9)],
        },
        {
            "testcase": "split_by_indices_symbolic",
            "callable": lambda x: jnp.split(x, [2, 5], axis=1),
            "input_shapes": [("B", 9)],
        },
    ],
)
class SplitPlugin(PrimitiveLeafPlugin):
    """ONNX plugin for jnp.split."""

    # ───────────────────────────────────────────────────────────────
    # ❶  Abstract‑eval: compute shapes of the pieces
    # ───────────────────────────────────────────────────────────────
    @staticmethod
    def abstract_eval(
        x: core.ShapedArray,
        *,
        axis: int,
        indices_or_sections: Union[int, Sequence[int]],
        **_,
    ):
        """Abstract evaluation for jnp.split."""
        axis = int(axis)
        # For abstract eval, if a dimension is symbolic, we can't know the size.
        # We rely on JAX's shape polymorphism for symbolic dimension handling.
        if isinstance(x.shape[axis], int):
            sizes = _get_split_sizes(x.shape[axis], indices_or_sections)
        else:
            # When the axis dimension is symbolic, we cannot compute concrete sizes.
            # We must create symbolic output shapes.
            if isinstance(indices_or_sections, int):
                # Cannot split a symbolic dim into equal sections without knowing its size.
                # Let JAX handle this, or it will raise a TypeError during tracing.
                raise TypeError(
                    f"Cannot split symbolic dimension {x.shape[axis]} into {indices_or_sections} sections."
                )
            indices = [0] + list(indices_or_sections)

            # First build a *list* of concrete lengths between split points …
            sizes_list: list[int] = [
                indices[i + 1] - indices[i] for i in range(len(indices) - 1)
            ]

            # … then turn it into the final *tuple* that may end with a symbolic size.
            sizes = tuple(sizes_list) + (
                core.DimExpr.make_sum(x.shape[axis], -indices[-1]),
            )

        out_specs = []
        for sz in sizes:
            shape = list(x.shape)
            shape[axis] = sz
            out_specs.append(core.ShapedArray(tuple(shape), x.dtype))
        return tuple(out_specs)

    # ───────────────────────────────────────────────────────────────
    # ❷  Lower to ONNX
    # ───────────────────────────────────────────────────────────────
    def to_onnx(self, s, node_inputs, node_outputs, params):
        """Lowering rule for jnp.split."""
        axis: int = int(params["axis"])
        indices_or_sections = params["indices_or_sections"]

        inp_name = s.get_name(node_inputs[0])
        in_shape, _ = s.builder.value_info_metadata[inp_name]

        # In ONNX, the `split` attribute must be concrete.
        if not isinstance(in_shape[axis], int):
            raise TypeError(
                f"ONNX 'Split' operator requires a concrete dimension size for the split axis. "
                f"Got symbolic dimension '{in_shape[axis]}' for axis {axis}."
            )

        sizes = _get_split_sizes(in_shape[axis], indices_or_sections)

        out_names = [s.get_name(v) for v in node_outputs]

        # ONNX's Split operator requires the `split` attribute to be a list of integers.
        split_node = helper.make_node(
            "Split",
            inputs=[inp_name],
            outputs=out_names,
            name=s.get_unique_name("split"),
            axis=axis,
            split=list(sizes),
        )
        s.add_node(split_node)


split_p.def_abstract_eval(SplitPlugin.abstract_eval)
