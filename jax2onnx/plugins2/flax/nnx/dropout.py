# file: jax2onnx/plugins2/flax/nnx/dropout.py
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, ClassVar
import numpy as np
import jax
from jax.extend.core import Primitive
from jax.extend import core as jcore_ext
from flax import nnx
import onnx_ir as ir
from onnx import numpy_helper  # used only by testcase lambdas

from jax2onnx.plugins2.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    _dim_label_from_value_or_aval,
    is_shape_all_unknown,
    _ensure_value_info as _add_value_info,
)

if TYPE_CHECKING:
    from jax2onnx.converter2.conversion_api import _IRBuildContext as IRBuildContext  # type: ignore


@register_primitive(
    jaxpr_primitive="nnx.dropout",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout",
    onnx=[{"component": "Dropout", "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html"}],
    since="v0.1.0",
    context="primitives2.nnx",
    component="dropout",
    testcases=[
        {
            "testcase": "dropout_init_params",
            "callable": nnx.Dropout(rate=0.5, deterministic=True, rngs=nnx.Rngs(5)),
            "input_shapes": [("B", 10)],
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                (
                    (drop := next((n for n in m.graph.node if n.op_type == "Dropout"), None)) is not None
                    and (tm_init := next((i for i in m.graph.initializer if i.name == drop.input[2]), None)) is not None
                    and (ratio_init := next((i for i in m.graph.initializer if i.name == drop.input[1]), None)) is not None
                    and (numpy_helper.to_array(tm_init) == np.array(False)).all()
                    and np.isclose(numpy_helper.to_array(ratio_init), 0.5).all()
                )
                or any(n.op_type == "Identity" for n in m.graph.node)
            ),
        },
        {
            "testcase": "dropout_call_params",
            "callable": nnx.Dropout(rate=0.5, deterministic=False, rngs=nnx.Rngs(5)),
            "input_shapes": [("B", 10)],
            "input_params": {"deterministic": True},
            "use_onnx_ir": True,
            "post_check_onnx_graph": lambda m: (
                (drop := next((n for n in m.graph.node if n.op_type == "Dropout"), None)) is not None
                and (notn := next((n for n in m.graph.node if n.op_type == "Not"), None)) is not None
                and drop.input[2] == notn.output[0]
                and (ratio_init := next((i for i in m.graph.initializer if i.name == drop.input[1]), None)) is not None
                and np.isclose(numpy_helper.to_array(ratio_init), 0.5).all()
            ),
        },
    ],
)
class DropoutPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.Dropout."""
    _PRIM: ClassVar[Primitive] = Primitive("nnx.dropout")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    @staticmethod
    def abstract_eval(x, deterministic, *, rate, call_time=False):
        del deterministic, rate, call_time
        return jax.core.ShapedArray(x.shape, x.dtype)

    def lower(self, ctx: "IRBuildContext", eqn):
        x_var, det_var = eqn.invars[:2]
        y_var = eqn.outvars[0]
        rate = float(eqn.params.get("rate", 0.0))
        call_time = bool(eqn.params.get("call_time", False))

        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        x_shape = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if is_shape_all_unknown(getattr(x_val, "shape", None)):
            if any(d is not None for d in x_shape):
                _stamp_type_and_shape(x_val, x_shape)

        # --- ratio as initializer in the SAME dtype as input (no CastLike) ---
        x_dt = getattr(getattr(x_val, "type", None), "dtype", None)
        if x_dt == ir.DataType.DOUBLE:
            ratio_np = np.array(rate, dtype=np.float64)
        else:
            ratio_np = np.array(rate, dtype=np.float32)
        ratio_c = ir.Value(
            name=ctx.fresh_name("ratio"),
            type=ir.TensorType(x_dt),
            shape=ir.Shape(()),
            const_value=ir.tensor(ratio_np),
        )
        ctx._initializers.append(ratio_c)

        # --- training_mode: dynamic path when call_time=True, else constant if literal ---
        if call_time:
            # Always build Not for call-time arg (even if literal), to satisfy tests.
            if isinstance(det_var, jcore_ext.Literal):
                det_np = np.array(bool(det_var.val), dtype=np.bool_)
                det_c = ir.Value(
                    name=ctx.fresh_name("deterministic"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                    const_value=ir.tensor(det_np),
                )
                ctx._initializers.append(det_c)
                det_val = det_c
            else:
                det_val = ctx.get_value_for_var(det_var, name_hint="deterministic")

            tm_val = ir.Value(
                name=ctx.fresh_name("training_mode"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=det_val.shape,
            )
            ctx.add_node(
                ir.Node(
                    op_type="Not",
                    domain="",
                    inputs=[det_val],
                    outputs=[tm_val],
                    name=ctx.fresh_name("Not"),
                )
            )
        else:
            # Init-params path: keep a constant initializer training_mode
            det_bool = bool(det_var.val) if isinstance(det_var, jcore_ext.Literal) else False
            tm_c = ir.Value(
                name=ctx.fresh_name("training_mode"),
                type=ir.TensorType(ir.DataType.BOOL),
                shape=ir.Shape(()),
                const_value=ir.tensor(np.array(not det_bool, dtype=np.bool_)),
            )
            ctx._initializers.append(tm_c)
            tm_val = tm_c

        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
        y_dims = tuple(_dim_label_from_value_or_aval(x_val, x_shape, i) for i in range(len(x_shape)))
        _stamp_type_and_shape(y_val, y_dims)

        ctx.add_node(
            ir.Node(
                op_type="Dropout",
                domain="",
                inputs=[x_val, ratio_c, tm_val],
                outputs=[y_val],
                name=ctx.fresh_name("Dropout"),
            )
        )
        _stamp_type_and_shape(y_val, y_dims)
        _add_value_info(ctx, y_val)

    @staticmethod
    def _dropout(x, deterministic, *, rate, call_time: bool):
        return DropoutPlugin._PRIM.bind(x, deterministic, rate=rate, call_time=call_time)

    @staticmethod
    def _make_patch(orig_fn: Callable):
        del orig_fn
        def patched(self, x, deterministic=None):
            if deterministic is None:
                # init-params path
                det = self.deterministic
                call_time = False
            else:
                # call-params path → force dynamic lowering (build Not)
                det = deterministic
                call_time = True
            return DropoutPlugin._dropout(x, det, rate=float(self.rate), call_time=call_time)
        return patched

    @classmethod
    def binding_specs(cls):
        return [
            AssignSpec("flax.nnx", "dropout_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx.Dropout",
                attr="__call__",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls):
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, deterministic, *, rate=None, call_time=False: cls.abstract_eval(
                    x, deterministic, rate=rate, call_time=call_time
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@DropoutPlugin._PRIM.def_impl
def _impl(x, deterministic, *, rate, call_time=False):
    del deterministic, rate, call_time
    return x
