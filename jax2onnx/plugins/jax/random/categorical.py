# jax2onnx/plugins/jax/random/categorical.py

from __future__ import annotations

from typing import Callable, ClassVar, Final

import jax
import jax.numpy as jnp
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins.jax.lax._index_utils import _const_i64


_CATEGORICAL_PRIM: Final[Primitive] = Primitive("jax.random.categorical")
_CATEGORICAL_PRIM.multiple_results = False


@register_primitive(
    jaxpr_primitive=_CATEGORICAL_PRIM.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html",
    onnx=[
        {
            "component": "Multinomial",
            "doc": "https://onnx.ai/onnx/operators/onnx__Multinomial.html",
        }
    ],
    since="0.12.1",
    context="primitives.random",
    component="random_categorical",
    testcases=[
        {
            "testcase": "random_categorical_logits_batch",
            "callable": lambda logits: jax.random.categorical(
                jax.random.PRNGKey(0), logits
            ),
            "input_values": [
                np.asarray(
                    [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.dtype(np.int32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["Multinomial"],
                no_unused_inputs=True,
            ),
            "opset_version": 21,
        },
        {
            "testcase": "random_categorical_logits_batch_opset23",
            "callable": lambda logits: jax.random.categorical(
                jax.random.PRNGKey(0), logits
            ),
            "input_values": [
                np.asarray(
                    [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1]],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2,)],
            "expected_output_dtypes": [np.dtype(np.int32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                ["RandomUniformLike -> Log -> Neg -> Log -> Neg -> Add -> ArgMax"],
                no_unused_inputs=True,
            ),
            "opset_version": 23,
        },
        {
            "testcase": "random_categorical_logits_rank3",
            "callable": lambda logits: jax.random.categorical(
                jax.random.PRNGKey(0), logits
            ),
            "input_values": [
                np.asarray(
                    [
                        [
                            [0.1, 0.2, 0.7],
                            [0.2, 0.7, 0.1],
                            [0.6, 0.2, 0.2],
                            [0.3, 0.3, 0.4],
                        ],
                        [
                            [0.7, 0.2, 0.1],
                            [0.1, 0.3, 0.6],
                            [0.4, 0.5, 0.1],
                            [0.2, 0.2, 0.6],
                        ],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2, 4)],
            "expected_output_dtypes": [np.dtype(np.int32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    "Reshape:8x3 -> Softmax:8x3 -> Multinomial:8x1 -> Squeeze:8 -> Cast:8 -> Reshape:2x4"
                ],
                no_unused_inputs=True,
            ),
            "opset_version": 21,
        },
        {
            "testcase": "random_categorical_logits_rank3_opset23",
            "callable": lambda logits: jax.random.categorical(
                jax.random.PRNGKey(0), logits
            ),
            "input_values": [
                np.asarray(
                    [
                        [
                            [0.1, 0.2, 0.7],
                            [0.2, 0.7, 0.1],
                            [0.6, 0.2, 0.2],
                            [0.3, 0.3, 0.4],
                        ],
                        [
                            [0.7, 0.2, 0.1],
                            [0.1, 0.3, 0.6],
                            [0.4, 0.5, 0.1],
                            [0.2, 0.2, 0.6],
                        ],
                    ],
                    dtype=np.float32,
                )
            ],
            "expected_output_shapes": [(2, 4)],
            "expected_output_dtypes": [np.dtype(np.int32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    "Reshape:8x3 -> RandomUniformLike:8x3 -> Log:8x3 -> Neg:8x3 -> Log:8x3 -> Neg:8x3 -> Add:8x3 -> ArgMax:8 -> Cast:8 -> Reshape:2x4"
                ],
                must_absent=["Multinomial"],
                no_unused_inputs=True,
            ),
            "opset_version": 23,
        },
        {
            "testcase": "random_categorical_logits_symbolic_batch",
            "callable": lambda logits: jax.random.categorical(
                jax.random.PRNGKey(0), logits
            ),
            "input_shapes": [("B", 3)],
            "run_only_dynamic": True,
            "expected_output_shapes": [("B",)],
            "expected_output_dtypes": [np.dtype(np.int32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    "Softmax -> Multinomial -> Squeeze -> Cast -> Reshape:B",
                    "Shape:2 -> Slice:1 -> Reshape:B",
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
            "opset_version": 21,
        },
        {
            "testcase": "random_categorical_logits_symbolic_batch_opset23",
            "callable": lambda logits: jax.random.categorical(
                jax.random.PRNGKey(0), logits
            ),
            "input_shapes": [("B", 3)],
            "run_only_dynamic": True,
            "expected_output_shapes": [("B",)],
            "expected_output_dtypes": [np.dtype(np.int32)],
            "run_only_f32_variant": True,
            "skip_numeric_validation": True,
            "post_check_onnx_graph": EG(
                [
                    "RandomUniformLike -> Log -> Neg -> Log -> Neg -> Add -> ArgMax -> Cast -> Reshape:B",
                    "Shape:2 -> Slice:1 -> Reshape:B",
                ],
                must_absent=["Multinomial"],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
            "opset_version": 23,
        },
    ],
)
class RandomCategoricalPlugin(PrimitiveLeafPlugin):
    """Lower ``jax.random.categorical`` to ONNX ``Multinomial``."""

    _PRIM: ClassVar[Primitive] = _CATEGORICAL_PRIM
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        key: jax.core.AbstractValue,
        logits: jax.core.AbstractValue,
        *,
        axis: int = -1,
        shape: tuple[int, ...] | None = None,
        replace: bool = True,
        mode: str | None = None,
    ) -> jax.core.ShapedArray:
        del key, replace, mode
        logits_shape = tuple(getattr(logits, "shape", ()))
        logits_rank = len(logits_shape)
        axis_norm = axis if axis >= 0 else logits_rank + axis
        if shape is None:
            out_shape = logits_shape[:axis_norm] + logits_shape[axis_norm + 1 :]
        else:
            out_shape = tuple(int(d) for d in shape)
        return jax.core.ShapedArray(out_shape, np.dtype(np.int32))

    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        key_var, logits_var = eqn.invars
        (out_var,) = eqn.outvars

        axis = int(eqn.params.get("axis", -1))
        shape_param = eqn.params.get("shape", None)
        replace = bool(eqn.params.get("replace", True))

        if shape_param is not None:
            raise NotImplementedError(
                "jax.random.categorical lowering currently supports shape=None only"
            )
        if not replace:
            raise NotImplementedError(
                "jax.random.categorical lowering currently supports replace=True only"
            )

        logits_shape = tuple(getattr(getattr(logits_var, "aval", None), "shape", ()))
        if not logits_shape:
            raise NotImplementedError(
                "jax.random.categorical lowering requires rank>=1 logits"
            )
        logits_rank = len(logits_shape)
        axis_norm = axis if axis >= 0 else logits_rank + axis
        if axis_norm != logits_rank - 1:
            raise NotImplementedError(
                "jax.random.categorical lowering currently supports axis=-1 only"
            )

        class_count = logits_shape[-1]
        if not isinstance(class_count, (int, np.integer)):
            raise NotImplementedError(
                "jax.random.categorical lowering requires static class dimension"
            )
        batch_dims_raw = logits_shape[:-1]
        static_batch_shape = all(
            isinstance(dim, (int, np.integer)) for dim in batch_dims_raw
        )
        batch_shape = (
            tuple(int(dim) for dim in batch_dims_raw) if static_batch_shape else None
        )
        num_distributions = (
            int(np.prod(batch_shape)) if batch_shape is not None and batch_shape else 1
        )

        # Materialize key so RNG plumbing remains explicit in traced graphs.
        ctx.get_value_for_var(key_var, name_hint=ctx.fresh_name("categorical_key"))

        logits_val = ctx.get_value_for_var(
            logits_var, name_hint=ctx.fresh_name("categorical_logits")
        )
        out_spec = ctx.get_value_for_var(
            out_var, name_hint=ctx.fresh_name("categorical_out")
        )

        reshape_2d_target = (
            np.asarray([num_distributions, int(class_count)], dtype=np.int64)
            if batch_shape is not None
            else np.asarray([-1, int(class_count)], dtype=np.int64)
        )
        reshape_2d_shape = ctx.builder.add_initializer_from_array(
            name=ctx.fresh_name("categorical_shape_2d"),
            array=reshape_2d_target,
        )
        logits_2d = ctx.builder.Reshape(
            logits_val,
            reshape_2d_shape,
            _outputs=[ctx.fresh_name("categorical_logits_2d")],
        )
        logits_2d.type = logits_val.type
        if batch_shape is not None:
            logits_2d.shape = ir.Shape((num_distributions, int(class_count)))

        probs = ctx.builder.Softmax(
            logits_2d,
            axis=1,
            _outputs=[ctx.fresh_name("categorical_probs")],
        )
        probs.type = logits_2d.type
        if getattr(logits_2d, "shape", None) is not None:
            probs.shape = logits_2d.shape

        opset = int(getattr(ctx.builder, "opset", 21))
        if opset <= 21:
            sampled = ctx.builder.Multinomial(
                probs,
                _outputs=[ctx.fresh_name("categorical_sampled")],
                sample_size=1,
                dtype=int(ir.DataType.INT64.value),
                seed=0.0,
            )
            sampled.type = ir.TensorType(ir.DataType.INT64)
            if batch_shape is not None:
                sampled.shape = ir.Shape((num_distributions, 1))

            squeeze_axes = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("categorical_squeeze_axes"),
                array=np.asarray([1], dtype=np.int64),
            )
            sampled_1d = ctx.builder.Squeeze(
                sampled,
                squeeze_axes,
                _outputs=[ctx.fresh_name("categorical_sampled_1d")],
            )
            sampled_1d.type = sampled.type
            if batch_shape is not None:
                sampled_1d.shape = ir.Shape((num_distributions,))
        else:
            # ORT does not implement Multinomial(22) yet; use Gumbel-max fallback.
            rng_dtype = getattr(
                getattr(logits_2d, "type", None), "dtype", ir.DataType.FLOAT
            )
            if rng_dtype not in {
                ir.DataType.FLOAT16,
                ir.DataType.BFLOAT16,
                ir.DataType.FLOAT,
                ir.DataType.DOUBLE,
            }:
                rng_dtype = ir.DataType.FLOAT
            uniform = ctx.builder.RandomUniformLike(
                logits_2d,
                _outputs=[ctx.fresh_name("categorical_uniform")],
                low=float(np.finfo(np.float32).tiny),
                high=1.0,
                dtype=int(rng_dtype.value),
                seed=0.0,
            )
            uniform.type = ir.TensorType(rng_dtype)
            uniform.shape = logits_2d.shape

            log_u = ctx.builder.Log(
                uniform,
                _outputs=[ctx.fresh_name("categorical_log_u")],
            )
            log_u.type = uniform.type
            log_u.shape = uniform.shape

            neg_log_u = ctx.builder.Neg(
                log_u,
                _outputs=[ctx.fresh_name("categorical_neg_log_u")],
            )
            neg_log_u.type = uniform.type
            neg_log_u.shape = uniform.shape

            log_neg_log_u = ctx.builder.Log(
                neg_log_u,
                _outputs=[ctx.fresh_name("categorical_log_neg_log_u")],
            )
            log_neg_log_u.type = uniform.type
            log_neg_log_u.shape = uniform.shape

            gumbel = ctx.builder.Neg(
                log_neg_log_u,
                _outputs=[ctx.fresh_name("categorical_gumbel")],
            )
            gumbel.type = uniform.type
            gumbel.shape = uniform.shape

            noisy_logits = ctx.builder.Add(
                logits_2d,
                gumbel,
                _outputs=[ctx.fresh_name("categorical_noisy_logits")],
            )
            noisy_logits.type = logits_2d.type
            noisy_logits.shape = logits_2d.shape

            sampled_1d = ctx.builder.ArgMax(
                noisy_logits,
                _outputs=[ctx.fresh_name("categorical_sampled_1d")],
                axis=1,
                keepdims=0,
                select_last_index=0,
            )
            sampled_1d.type = ir.TensorType(ir.DataType.INT64)
            if batch_shape is not None:
                sampled_1d.shape = ir.Shape((num_distributions,))

        out_dtype = getattr(getattr(out_spec, "type", None), "dtype", ir.DataType.INT32)
        sampled_cast = ctx.builder.Cast(
            sampled_1d,
            _outputs=[ctx.fresh_name("categorical_sampled_cast")],
            to=int(out_dtype.value),
        )
        sampled_cast.type = ir.TensorType(out_dtype)
        if getattr(sampled_1d, "shape", None) is not None:
            sampled_cast.shape = sampled_1d.shape

        if batch_shape is not None:
            out_shape_source = ctx.builder.add_initializer_from_array(
                name=ctx.fresh_name("categorical_out_shape"),
                array=np.asarray(batch_shape, dtype=np.int64),
            )
        else:
            logits_shape_value = ctx.builder.Shape(
                logits_val,
                _outputs=[ctx.fresh_name("categorical_logits_shape")],
            )
            logits_shape_value.type = ir.TensorType(ir.DataType.INT64)
            logits_shape_value.shape = ir.Shape((logits_rank,))
            if logits_rank <= 1:
                out_shape_source = ctx.builder.add_initializer_from_array(
                    name=ctx.fresh_name("categorical_out_shape_empty"),
                    array=np.asarray([], dtype=np.int64),
                )
            else:
                starts = _const_i64(
                    ctx,
                    np.asarray([0], dtype=np.int64),
                    "categorical_out_shape_slice_start",
                )
                ends = _const_i64(
                    ctx,
                    np.asarray([logits_rank - 1], dtype=np.int64),
                    "categorical_out_shape_slice_end",
                )
                axes = _const_i64(
                    ctx,
                    np.asarray([0], dtype=np.int64),
                    "categorical_out_shape_slice_axes",
                )
                steps = _const_i64(
                    ctx,
                    np.asarray([1], dtype=np.int64),
                    "categorical_out_shape_slice_steps",
                )
                out_shape_source = ctx.builder.Slice(
                    logits_shape_value,
                    starts,
                    ends,
                    axes,
                    steps,
                    _outputs=[ctx.fresh_name("categorical_out_shape_dynamic")],
                )
                out_shape_source.type = ir.TensorType(ir.DataType.INT64)
                out_shape_source.shape = ir.Shape((logits_rank - 1,))

        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name(
            "categorical_out"
        )
        result = ctx.builder.Reshape(
            sampled_cast,
            out_shape_source,
            _outputs=[desired_name],
        )
        if getattr(out_spec, "type", None) is not None:
            result.type = out_spec.type
        else:
            result.type = ir.TensorType(out_dtype)
        if getattr(out_spec, "shape", None) is not None:
            result.shape = out_spec.shape
        elif batch_shape is not None:
            result.shape = ir.Shape(batch_shape)
        ctx.bind_value_for_var(out_var, result)

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(cls.abstract_eval)
            cls._ABSTRACT_EVAL_BOUND = True

    @classmethod
    def binding_specs(cls) -> list[MonkeyPatchSpec]:
        storage_slot = "__orig_impl__categorical"

        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            if orig is None:
                raise RuntimeError("Original jax.random.categorical not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(
                key: ArrayLike,
                logits: ArrayLike,
                axis: int = -1,
                shape: tuple[int, ...] | list[int] | None = None,
                replace: bool = True,
                mode: str | None = None,
            ) -> ArrayLike:
                shape_param = (
                    None if shape is None else tuple(int(dim) for dim in shape)
                )
                return cls._PRIM.bind(
                    key,
                    jnp.asarray(logits),
                    axis=int(axis),
                    shape=shape_param,
                    replace=bool(replace),
                    mode=mode,
                )

            return _patched

        return [
            MonkeyPatchSpec(
                target="jax.random",
                attr="categorical",
                make_value=_make_value,
                delete_if_missing=False,
            )
        ]


@RandomCategoricalPlugin._PRIM.def_impl
def _random_categorical_impl(
    key: ArrayLike,
    logits: ArrayLike,
    *,
    axis: int = -1,
    shape: tuple[int, ...] | None = None,
    replace: bool = True,
    mode: str | None = None,
) -> ArrayLike:
    orig = getattr(
        RandomCategoricalPlugin._PRIM,
        "__orig_impl__categorical",
        jax.random.categorical,
    )
    return orig(
        key,
        logits,
        axis=axis,
        shape=shape,
        replace=replace,
        mode=mode,
    )
