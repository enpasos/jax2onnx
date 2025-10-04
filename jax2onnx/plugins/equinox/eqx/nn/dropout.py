# jax2onnx/plugins/equinox/eqx/nn/dropout.py

from __future__ import annotations

from typing import ClassVar, Optional

import equinox as eqx
import jax
import numpy as np
import onnx_ir as ir
from jax.extend.core import Primitive
from jax.core import ShapedArray
from jax.interpreters import batching

from jax2onnx.plugins._ir_shapes import _ensure_value_info, _stamp_type_and_shape
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


def _ir_dtype_from_numpy(dtype: np.dtype) -> ir.DataType:
    dt = np.dtype(dtype)
    if dt == np.float64:
        return getattr(ir.DataType, "DOUBLE", ir.DataType.FLOAT)
    if dt == np.float32:
        return ir.DataType.FLOAT
    if dt == np.bool_ or dt == np.dtype(bool):
        return ir.DataType.BOOL
    if dt == np.int64:
        return ir.DataType.INT64
    if dt == np.int32:
        return ir.DataType.INT32
    return ir.DataType.FLOAT


def _const_tensor(ctx, array, *, name: str) -> ir.Value:
    arr = np.asarray(array)
    dtype = _ir_dtype_from_numpy(arr.dtype)
    shape = arr.shape if arr.shape else ()
    if hasattr(ir, "tensor"):
        tensor_obj = ir.tensor(arr)
    else:
        tensor_obj = arr
    value = ir.Value(
        name=ctx.fresh_name(name),
        type=ir.TensorType(dtype),
        shape=ir.Shape(shape),
        const_value=tensor_obj if hasattr(ir.Value, "const_value") else None,
    )
    try:
        ctx._initializers.append(value)
    except Exception:
        attrs = []
        Attr = getattr(ir, "Attr", None)
        AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
        if Attr is not None:
            try:
                if hasattr(Attr, "t"):
                    attrs = [Attr.t("value", tensor_obj)]
                elif AttrType is not None:
                    attrs = [Attr("value", AttrType.TENSOR, tensor_obj)]
                else:
                    attrs = [Attr("value", tensor_obj)]
            except Exception:
                attrs = []
        node = ir.Node(
            op_type="Constant",
            domain="",
            inputs=[],
            outputs=[value],
            name=ctx.fresh_name("Constant"),
            attributes=attrs,
            num_outputs=1,
        )
        ctx.add_node(node)
    return value


try:
    from jax.extend.core import Literal as _JaxLiteral
except ImportError:  # pragma: no cover - older jax
    from jax.core import Literal as _JaxLiteral


def _extract_python_bool(var) -> Optional[bool]:
    if isinstance(var, _JaxLiteral):
        val = getattr(var, "val", None)
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
    return None


def _ensure_scalar_bool_input(ctx, name: str) -> ir.Value:
    inputs = getattr(ctx.builder, "inputs", None)
    if inputs is None:
        inputs = []
        try:
            ctx.builder.inputs = inputs
        except Exception:
            pass
    for vi in inputs:
        if getattr(vi, "name", "") == name:
            return vi
    value = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(()),
    )
    inputs.append(value)
    if getattr(ctx.builder, "inputs", None) is not inputs:
        try:
            ctx.builder.inputs = inputs
        except Exception:
            pass
    return value


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
                        "path": "Dropout",
                        "inputs": {
                            1: {"const": 0.42},
                            2: {"const_bool": False},
                        },
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
                        "path": "Dropout",
                        "inputs": {
                            1: {"const": 0.5},
                            2: {"const_bool": True},
                        },
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
                        "path": "Not -> Dropout",
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
                        "path": "Dropout",
                        "inputs": {
                            1: {"const": 0.3},
                            2: {"const_bool": False},
                        },
                        "must_absent": ["Not"],
                    }
                ],
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
    def abstract_eval(x, inference, *, p, call_time=False):
        del inference, p, call_time
        return ShapedArray(x.shape, x.dtype)

    def lower(self, ctx, eqn):
        x_var, inference_var = eqn.invars
        out_var = eqn.outvars[0]
        rate = float(eqn.params.get("p", 0.5))
        call_time = bool(eqn.params.get("call_time", False))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("drop_x"))
        ratio_val = _const_tensor(ctx, np.asarray(rate, dtype=np.float32), name="ratio")

        inference_bool = _extract_python_bool(inference_var)
        call_params = getattr(ctx, "_call_input_param_names", set())
        param_name = "inference"
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
            training_val = ir.Value(
                name=ctx.fresh_name("training"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=ir.Shape(()),
            )
            ctx.add_node(
                ir.Node(
                    op_type="Not",
                    domain="",
                    inputs=[det_val],
                    outputs=[training_val],
                    name=ctx.fresh_name("Not"),
                    num_outputs=1,
                )
            )
        else:
            if inference_bool is not None:
                training_val = _const_tensor(
                    ctx, np.asarray(not inference_bool, dtype=np.bool_), name="training"
                )
            else:
                det_val = ctx.get_value_for_var(
                    inference_var, name_hint=ctx.fresh_name("inference")
                )
                training_val = ir.Value(
                    name=ctx.fresh_name("training"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Not",
                        domain="",
                        inputs=[det_val],
                        outputs=[training_val],
                        name=ctx.fresh_name("Not"),
                        num_outputs=1,
                    )
                )

        out_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("drop_out"))
        ctx.add_node(
            ir.Node(
                op_type="Dropout",
                domain="",
                inputs=[x_val, ratio_val, training_val],
                outputs=[out_val],
                name=ctx.fresh_name("Dropout"),
                num_outputs=1,
            )
        )
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if x_shape:
            _stamp_type_and_shape(out_val, x_shape)
        _ensure_value_info(ctx, out_val)

    @classmethod
    def binding_specs(cls):
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
    def _patch_call(orig):
        del orig

        def wrapped(self, x, *, key=None, inference=None, deterministic=None):
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
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, inference, *, p, call_time=False: cls.abstract_eval(
                    x, inference, p=p, call_time=call_time
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@DropoutPlugin._PRIM.def_impl
def _dropout_impl(x, inference, *, p, call_time=False):
    del p, call_time
    inference_bool = bool(inference)
    if inference_bool:
        return x
    return x


def _dropout_batch_rule(batched_args, batch_dims, *, p, call_time=False):
    x, inference = batched_args
    x_bdim, inf_bdim = batch_dims
    if inf_bdim is not None:
        raise NotImplementedError(
            "Batching over the `inference` flag is not supported."
        )
    out = DropoutPlugin._PRIM.bind(x, inference, p=p, call_time=call_time)
    return out, x_bdim


batching.primitive_batchers[DropoutPlugin._PRIM] = _dropout_batch_rule
