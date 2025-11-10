# jax2onnx/plugins/equinox/eqx/nn/dropout.py

from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional

import equinox as eqx
import jax
import jax.core as jax_core
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.core import ShapedArray
from jax.interpreters import batching

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _const_tensor(ctx: LoweringContextProtocol, array: Any, *, name: str) -> ir.Value:
    arr = np.asarray(array)
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder for constant tensor")
    init_name = ctx.fresh_name(name)
    if arr.ndim == 0:
        value = builder.add_initializer_from_scalar(name=init_name, value=arr)
    else:
        value = builder.add_initializer_from_array(name=init_name, array=arr)
    _stamp_type_and_shape(value, arr.shape if arr.shape else ())
    _ensure_value_metadata(ctx, value)
    return value


def _ensure_scalar_bool_input(ctx: LoweringContextProtocol, name: str) -> ir.Value:
    builder = getattr(ctx, "builder", None)
    if builder is None:
        raise AttributeError("IR build context missing builder for dropout input")
    for vi in getattr(builder, "inputs", []):
        if getattr(vi, "name", "") == name:
            return vi
    value = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(()),
    )
    builder.inputs.append(value)
    _ensure_value_metadata(ctx, value)
    return value


try:
    from jax.extend.core import Literal as _JaxLiteral
except ImportError:  # pragma: no cover - older jax
    from jax.core import Literal as _JaxLiteral


def _extract_python_bool(var: Any) -> Optional[bool]:
    if isinstance(var, _JaxLiteral):
        val = getattr(var, "val", None)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
    return None


@register_primitive(
    jaxpr_primitive="eqx.nn.dropout",
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
    since="v0.8.0",
    context="primitives.eqx",
    component="dropout",
    testcases=[
        {
            "testcase": "eqx_dropout_inference_mode",
            "callable": eqx.nn.Dropout(p=0.42, inference=True),
            "input_shapes": [(64,)],
            "post_check_onnx_graph": expect_graph(
                [
                    {
                        "path": "Dropout:64",
                        "inputs": {1: {"const": 0.42}, 2: {"const_bool": False}},
                        "must_absent": ["Not"],
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_dropout_training_mode",
            "callable": lambda x, model=eqx.nn.Dropout(p=0.5, inference=False): model(
                x, key=jax.random.PRNGKey(0)
            ),
            "input_shapes": [(64,)],
            "skip_numeric_validation": True,
            "post_check_onnx_graph": expect_graph(
                [
                    {
                        "path": "Dropout:64",
                        "inputs": {1: {"const": 0.5}, 2: {"const_bool": True}},
                        "must_absent": ["Not"],
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_dropout_dynamic_inference",
            "callable": lambda x, inference, key=None, model=eqx.nn.Dropout(
                p=0.5
            ): model(x, key=key, inference=inference),
            "input_shapes": [(64,)],
            "input_params": {"inference": np.array(True, dtype=bool)},
            "post_check_onnx_graph": expect_graph(
                [
                    {
                        "path": "Not -> Dropout:64",
                        "inputs": {1: {"const": 0.5}},
                    }
                ],
                no_unused_inputs=True,
            ),
        },
        {
            "testcase": "eqx_dropout_batched_inference",
            "callable": lambda xs, _mod=eqx.nn.Dropout(p=0.3, inference=True): jax.vmap(
                _mod
            )(xs),
            "input_shapes": [("B", 64)],
            "post_check_onnx_graph": expect_graph(
                [
                    {
                        "path": "Dropout:Bx64",
                        "inputs": {1: {"const": 0.3}, 2: {"const_bool": False}},
                        "must_absent": ["Not"],
                    }
                ],
                symbols={"B": None},
                no_unused_inputs=True,
            ),
        },
    ],
)
class DropoutPlugin(PrimitiveLeafPlugin):
    _PRIM: ClassVar[Primitive] = Primitive("eqx.nn.dropout")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(
        x: jax_core.AbstractValue,
        inference: jax_core.AbstractValue,
        *,
        p: float,
        call_time: bool = False,
    ) -> ShapedArray:
        del inference, p, call_time
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: LoweringContextProtocol, eqn: jax_core.JaxprEqn) -> None:
        builder = getattr(ctx, "builder", None)
        if builder is None:
            raise AttributeError(
                "IR build context missing builder for Eqx Dropout lowering"
            )

        x_var, inference_var = eqn.invars
        out_var = eqn.outvars[0]
        rate = float(eqn.params.get("p", 0.5))
        call_time = bool(eqn.params.get("call_time", False))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("drop_x"))
        ratio_val = _const_tensor(ctx, np.asarray(rate, dtype=np.float32), name="ratio")

        inference_bool = _extract_python_bool(inference_var)
        call_params = set(getattr(ctx, "_call_input_param_names", set()))
        param_name = "inference"

        def _materialize_not(value: ir.Value) -> ir.Value:
            not_val = builder.Not(
                value,
                _outputs=[ctx.fresh_name("training_not")],
            )
            if getattr(not_val, "type", None) is None:
                not_val.type = ir.TensorType(ir.DataType.BOOL)
            _stamp_type_and_shape(not_val, ())
            _ensure_value_metadata(ctx, not_val)
            return not_val

        if call_time:
            inside_fn = bool(getattr(ctx, "_inside_function_scope", False))
            if param_name not in call_params and inference_bool is not None:
                det_val = _const_tensor(
                    ctx,
                    np.asarray(inference_bool, dtype=np.bool_),
                    name="inference_const",
                )
            else:
                if inside_fn and inference_var is not None:
                    det_val = ctx.get_value_for_var(
                        inference_var, name_hint=ctx.fresh_name("inference")
                    )
                else:
                    det_val = _ensure_scalar_bool_input(ctx, param_name)
            training_val = _materialize_not(det_val)
        else:
            if inference_bool is not None:
                training_val = _const_tensor(
                    ctx, np.asarray(not inference_bool, dtype=np.bool_), name="training"
                )
            else:
                det_val = ctx.get_value_for_var(
                    inference_var, name_hint=ctx.fresh_name("inference")
                )
                training_val = _materialize_not(det_val)

        out_spec = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("drop_out"))
        desired_name = getattr(out_spec, "name", None) or ctx.fresh_name("drop_out")
        producer = getattr(out_spec, "producer", None)
        if callable(producer) and producer() is not None:
            desired_name = ctx.fresh_name("drop_out")

        dropout_val = builder.Dropout(
            x_val,
            ratio_val,
            training_val,
            _outputs=[desired_name],
        )

        if getattr(out_spec, "type", None) is not None:
            dropout_val.type = out_spec.type
        elif getattr(x_val, "type", None) is not None:
            dropout_val.type = x_val.type

        if getattr(out_spec, "shape", None) is not None:
            dropout_val.shape = out_spec.shape
        else:
            x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
            if x_shape:
                _stamp_type_and_shape(dropout_val, x_shape)
        _ensure_value_metadata(ctx, dropout_val)
        ctx.bind_value_for_var(out_var, dropout_val)

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("equinox.nn", "dropout_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="equinox.nn.Dropout",
                attr="__call__",
                make_value=lambda orig: cls._patch_call(orig),
                delete_if_missing=False,
            ),
        ]

    @staticmethod
    def _patch_call(
        orig: Callable[..., jax.Array] | None,
    ) -> Callable[..., jax.Array]:
        del orig

        def wrapped(
            self: eqx.nn.Dropout,
            x: jax.Array,
            *,
            key: jax.Array | None = None,
            inference: bool | np.bool_ | jax.Array | None = None,
            deterministic: bool | None = None,
        ) -> jax.Array:
            del key
            call_time = deterministic is not None or inference is not None
            if deterministic is not None:
                inference_arg = deterministic
            elif inference is not None:
                inference_arg = inference
            else:
                inference_arg = self.inference
            if isinstance(self.p, (int, float)) and self.p == 0:
                inference_arg = True
                call_time = False
            return DropoutPlugin._PRIM.bind(
                x, inference_arg, p=float(self.p), call_time=call_time
            )

        return wrapped

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, inference, *, p, call_time=False: cls.abstract_eval(
                    x, inference, p=p, call_time=call_time
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@DropoutPlugin._PRIM.def_impl
def _dropout_impl(
    x: jax.Array,
    inference: bool | np.bool_ | jax.Array,
    *,
    p: float,
    call_time: bool = False,
) -> jax.Array:
    del p, call_time
    inference_bool = bool(inference)
    if inference_bool:
        return x
    return x


def _dropout_batch_rule(
    batched_args: tuple[jax.Array, jax.Array],
    batch_dims: tuple[int | None, int | None],
    *,
    p: float,
    call_time: bool = False,
) -> tuple[jax.Array, int | None]:
    x, inference = batched_args
    x_bdim, inf_bdim = batch_dims
    if inf_bdim is not None:
        raise NotImplementedError(
            "Batching over the `inference` flag is not supported."
        )
    out = DropoutPlugin._PRIM.bind(x, inference, p=p, call_time=call_time)
    return out, x_bdim


batching.primitive_batchers[DropoutPlugin._PRIM] = _dropout_batch_rule
