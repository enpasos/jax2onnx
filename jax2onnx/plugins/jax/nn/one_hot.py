# jax2onnx/plugins/jax/nn/one_hot.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax2onnx._compat.jax import (
    AbstractValue,
    JaxprEqn,
    Primitive,
    ShapedArray,
    batching,
)
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import NativeLoweringContext
from jax2onnx.ir_utils import numpy_dtype_to_ir
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._utils import cast_value_to_dtype
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    register_native_primitive,
)


_ONE_HOT_PRIM: Final[Primitive] = Primitive("jax.nn.one_hot")
_ONE_HOT_PRIM.multiple_results = False
_JAX_ONE_HOT_ORIG: Final = jax.nn.one_hot


def _comparison_dtype(index_dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Dtype in which the indices are compared with the class positions.

    ONNX Runtime has `Equal` kernels for int32, int64, float32 and float64.
    Every other index dtype is widened to the next of these, which is exact.
    """
    if jnp.issubdtype(index_dtype, jnp.integer):
        return np.dtype(np.int32 if index_dtype == np.int32 else np.int64)
    if jnp.issubdtype(index_dtype, jnp.floating):
        return np.dtype(np.float64 if index_dtype == np.float64 else np.float32)
    # jax.nn.one_hot builds the class positions with lax.iota, which rejects
    # other dtypes with the same error.
    raise TypeError(
        f"one_hot does not accept indices of dtype {index_dtype}; "
        "use an integer or floating dtype."
    )


@register_native_primitive(
    jaxpr_primitive=_ONE_HOT_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.nn.one_hot.html",
    onnx=[
        {
            "component": "Unsqueeze",
            "doc": "https://onnx.ai/onnx/operators/onnx__Unsqueeze.html",
        },
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
        {
            "component": "Cast",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cast.html",
        },
    ],
    since="0.12.1",
    context="primitives.nn",
    component="one_hot",
    testcases=[
        {
            "testcase": "jaxnn_one_hot_default",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=6),
            "input_values": [np.array([0, 2, 4, 5], dtype=np.int32)],
            "expected_output_shapes": [(4, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:4x1 -> Equal:4x6 -> Cast:4x6"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_one_hot_axis0",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=5, axis=0),
            "input_values": [np.array([1, 3], dtype=np.int32)],
            "expected_output_shapes": [(5, 2)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:1x2 -> Equal:5x2 -> Cast:5x2"],
                no_unused_inputs=True,
            ),
        },
        {
            # The axis refers to each vmapped sample, not to the batched array.
            "testcase": "jaxnn_one_hot_vmap_positive_axis",
            "callable": lambda x: jax.vmap(
                lambda i: jax.nn.one_hot(i, num_classes=4, axis=1)
            )(x),
            "input_values": [
                np.array(
                    [[[0, 3, 1, 2, 2], [1, 1, 0, 3, 0], [2, 0, 3, 1, 1]]] * 2,
                    dtype=np.int32,
                )
            ],
            "expected_output_shapes": [(2, 3, 4, 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:2x3x1x5 -> Equal:2x3x4x5 -> Cast:2x3x4x5"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_one_hot_vmap_trailing_batch_negative_axis",
            "callable": lambda x: jax.vmap(
                lambda i: jax.nn.one_hot(i, num_classes=4, axis=-2), in_axes=1
            )(x),
            "input_values": [np.array([[0, 3], [2, 1], [1, 1]], dtype=np.int32)],
            "expected_output_shapes": [(2, 4, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Transpose:2x3 -> Unsqueeze:2x1x3 -> Equal:2x4x3 -> Cast:2x4x3"],
                no_unused_inputs=True,
            ),
        },
        {
            # JAX returns a row of zeros for every index outside [0, depth).
            "testcase": "jaxnn_one_hot_out_of_range",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=6),
            "input_values": [np.array([-1, 0, 5, 6, -6], dtype=np.int32)],
            "expected_output_shapes": [(5, 6)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:5x1 -> Equal:5x6 -> Cast:5x6"],
                no_unused_inputs=True,
            ),
        },
        {
            # ... and for every index that is not a whole number.
            "testcase": "jaxnn_one_hot_float_indices",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=3),
            "input_values": [np.array([0.0, 1.0, 1.5, -0.5, 2.0], dtype=np.float32)],
            "expected_output_shapes": [(5, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:5x1 -> Equal:5x3 -> Cast:5x3"],
                no_unused_inputs=True,
            ),
        },
        {
            # JAX builds the class positions in the dtype of the indices, so
            # they wrap at 256 here: index 43 also selects class 299.
            "testcase": "jaxnn_one_hot_uint8_indices",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=300),
            "input_values": [np.array([0, 43, 255], dtype=np.uint8)],
            "expected_output_shapes": [(3, 300)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Cast:3 -> Unsqueeze:3x1 -> Equal:3x300 -> Cast:3x300"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_one_hot_bool",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=4, dtype=jnp.bool_),
            "input_values": [np.array([3, -1, 0], dtype=np.int32)],
            "expected_output_shapes": [(3, 4)],
            "expected_output_dtypes": [np.bool_],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:3x1 -> Equal:3x4"],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "jaxnn_one_hot_float16",
            "callable": lambda x: jax.nn.one_hot(x, num_classes=4, dtype=jnp.float16),
            "input_values": [np.array([3, -1, 0], dtype=np.int32)],
            "expected_output_shapes": [(3, 4)],
            "expected_output_dtypes": [np.float16],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": EG(
                ["Unsqueeze:3x1 -> Equal:3x4 -> Cast:3x4"],
                no_unused_inputs=True,
            ),
        },
    ],
)
class OneHotPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.nn.one_hot`` to the comparison that defines it in JAX."""

    _PRIM: ClassVar[Primitive] = _ONE_HOT_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: AbstractValue,
        *,
        num_classes: int,
        dtype: Any = jnp.float32,
        axis: int = -1,
    ) -> ShapedArray:
        x_shape = tuple(getattr(x, "shape", ()))
        out_rank = len(x_shape) + 1

        axis_int = int(axis)
        if axis_int < 0:
            axis_int += out_rank
        if axis_int < 0 or axis_int > len(x_shape):
            raise ValueError(
                f"one_hot axis={axis} is out of bounds for input rank {len(x_shape)}"
            )

        out_shape = list(x_shape)
        out_shape.insert(axis_int, int(num_classes))
        return ShapedArray(tuple(out_shape), np.dtype(dtype))

    def lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None:
        # jax.nn.one_hot is `(expand_dims(x, axis) == iota(x.dtype)).astype(dtype)`.
        # The export writes that comparison. ONNX OneHot differs from it: an
        # index in [-depth, -1] selects a class counted from the end, and a
        # non-integer index is truncated, where JAX returns a row of zeros.
        x_var = eqn.invars[0]
        out_var = eqn.outvars[0]
        num_classes = int(eqn.params["num_classes"])
        x_dtype: np.dtype[Any] = np.dtype(x_var.aval.dtype)
        out_type = numpy_dtype_to_ir(np.dtype(out_var.aval.dtype))
        rank = len(x_var.aval.shape)
        axis = int(eqn.params.get("axis", -1))
        position = axis + rank + 1 if axis < 0 else axis

        x_val = ctx.require_value_for_var(x_var)
        output = ctx.allocate_value_for_var(
            out_var, name_hint=ctx.fresh_name("one_hot_out")
        )

        work_dtype = _comparison_dtype(x_dtype)
        work_type = numpy_dtype_to_ir(work_dtype)
        indices = cast_value_to_dtype(
            ctx, x_val, work_type, name_hint="one_hot_indices"
        )
        axes = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("one_hot_axis"),
            array=np.asarray([position], dtype=np.int64),
        )
        expanded_shape = None
        if indices.shape is not None:
            dims = list(indices.shape.dims)
            expanded_shape = ir.Shape([*dims[:position], 1, *dims[position:]])
        expanded = ctx.builder.op(
            "Unsqueeze",
            [indices, axes],
            output=ir.Value(
                name=ctx.fresh_name("one_hot_expanded"),
                type=ir.TensorType(work_type),
                shape=expanded_shape,
            ),
        )

        # The class positions as JAX's iota yields them in the dtype of x, which
        # wraps for a narrow integer dtype and rounds for float16. Widening them
        # to the comparison dtype afterwards is exact.
        classes_shape = [1] * rank
        classes_shape.insert(position, num_classes)
        classes = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("one_hot_classes"),
            array=np.arange(num_classes)
            .astype(x_dtype)
            .astype(work_dtype)
            .reshape(classes_shape),
        )

        if out_type == ir.DataType.BOOL:
            result = ctx.builder.op("Equal", [expanded, classes], output=output)
        else:
            matches = ctx.builder.op(
                "Equal",
                [expanded, classes],
                output=ir.Value(
                    name=ctx.fresh_name("one_hot_matches"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=output.shape,
                ),
            )
            result = ctx.builder.op(
                "Cast", [matches], {"to": int(out_type.value)}, output=output
            )
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.nn.one_hot not found")

            def _patched(
                x: ArrayLike,
                num_classes: int,
                *,
                dtype: Any = None,
                axis: int = -1,
            ) -> ArrayLike:
                # Like jax.nn.one_hot: the default is the default float dtype,
                # and a requested dtype follows the caller's x64 setting.
                resolved = jax.dtypes.canonicalize_dtype(
                    jnp.float_ if dtype is None else dtype
                )
                return cls._PRIM.bind(
                    x,
                    num_classes=int(num_classes),
                    dtype=np.dtype(resolved),
                    axis=int(axis),
                )

            return _patched

        return [
            AssignSpec("jax.nn", "one_hot_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.nn",
                attr="one_hot",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen.activation",
                attr="one_hot",
                make_value=_make_value,
                delete_if_missing=False,
            ),
            MonkeyPatchSpec(
                target="flax.linen",
                attr="one_hot",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]


@OneHotPlugin._PRIM.def_impl
def _one_hot_impl(
    x: ArrayLike,
    *,
    num_classes: int,
    dtype: Any = jnp.float32,
    axis: int = -1,
) -> ArrayLike:
    return _JAX_ONE_HOT_ORIG(x, num_classes=num_classes, dtype=dtype, axis=axis)


def _one_hot_batch_rule(
    batched_args: tuple[jax.Array, ...],
    batch_dims: tuple[int | None, ...],
    *,
    num_classes: int,
    dtype: Any = jnp.float32,
    axis: int = -1,
) -> tuple[jax.Array, int | None]:
    (x,) = batched_args
    (bd,) = batch_dims

    if bd is None:
        out = OneHotPlugin._PRIM.bind(
            x, num_classes=num_classes, dtype=dtype, axis=axis
        )
        return out, None

    # `axis` indexes one sample's output, whose rank equals the batched input's.
    x_front = batching.bdim_at_front(x, bd, x.shape[bd])
    sample_axis = axis + x_front.ndim if axis < 0 else axis
    out = OneHotPlugin._PRIM.bind(
        x_front, num_classes=num_classes, dtype=dtype, axis=sample_axis + 1
    )
    return out, 0


batching.primitive_batchers[OneHotPlugin._PRIM] = _one_hot_batch_rule
