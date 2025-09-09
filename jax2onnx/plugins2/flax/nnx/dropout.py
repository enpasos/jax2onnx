# file: jax2onnx/plugins2/flax/nnx/dropout.py
from __future__ import annotations
from typing import Callable, ClassVar, Any
import numpy as np
import jax
from jax.extend.core import Primitive
from jax.extend import core as jcore_ext
import logging
from jax2onnx.plugins2.plugin_system import register_primitive, PrimitiveLeafPlugin
import onnx_ir as ir

from jax2onnx.plugins2._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins2._ir_shapes import (
    _stamp_type_and_shape,
    is_shape_all_unknown,
    _ensure_value_info as _add_value_info,
)
from jax2onnx.plugins2._post_check_onnx_graph import expect_graph

from flax import nnx

# mypy/ruff-only import (avoid runtime cycles)
# from jax2onnx.converter2.conversion_api import _IRBuildContext
from jax2onnx.converter2.ir_context import IRContext

logger = logging.getLogger(__name__)


def _in_function_scope(ctx) -> bool:
    """Heuristic: in function bodies inputs are named like 'f_in_0'."""
    ins = getattr(ctx.builder, "inputs", [])
    return any(getattr(vi, "name", "").startswith("f_in_") for vi in ins)


def _get_or_make_bool_input(ctx, name: str) -> ir.Value:
    """Return existing graph input `name` or create a scalar BOOL input with that name."""
    for vi in getattr(ctx.builder, "inputs", []):
        if getattr(vi, "name", "") == name:
            return vi
    v = ir.Value(
        name=name,
        type=ir.TensorType(ir.DataType.BOOL),
        shape=ir.Shape(()),
    )
    ctx.builder.inputs.append(v)
    return v


def _tp_to_numpy(tp) -> np.ndarray:
    """Convert an ONNX TensorProto-like object to a NumPy array without importing onnx."""
    # Prefer raw_data when present
    if getattr(tp, "raw_data", None):
        # Minimal dtype map for the scalars we use here (BOOL/FP32/FP64)
        dt_map = {1: np.float32, 11: np.float64, 9: np.bool_}
        dtype = dt_map.get(getattr(tp, "data_type", 0), np.uint8)
        arr = np.frombuffer(tp.raw_data, dtype=dtype)
        shape = tuple(getattr(tp, "dims", ()))
        return arr.reshape(shape) if shape else (arr[0] if arr.size == 1 else arr)
    # Fallback to typed *_data fields
    dt = getattr(tp, "data_type", 0)
    if dt == 1 and getattr(tp, "float_data", None):
        arr = np.array(tp.float_data, dtype=np.float32)
    elif dt == 11 and getattr(tp, "double_data", None):
        arr = np.array(tp.double_data, dtype=np.float64)
    elif dt == 9 and getattr(tp, "int32_data", None):
        arr = np.array(tp.int32_data, dtype=np.bool_)
    else:
        arr = np.array([], dtype=np.float32)
    shape = tuple(getattr(tp, "dims", ()))
    return arr.reshape(shape) if shape else (arr[0] if arr.size == 1 else arr)


EXPECT_DROPOUT_WITH_FLAG = expect_graph(
    ["^Not->Dropout$"], match="contains", mode="any"
)


def post_check_onnx_graph(model):
    # Must have Not->Dropout and Dropout must have 3 inputs
    if not EXPECT_DROPOUT_WITH_FLAG(model):
        return False
    drop = next((n for n in model.graph.node if n.op_type == "Dropout"), None)
    if drop is None or len([i for i in drop.input if i]) != 3:
        return False
    notn = next((n for n in model.graph.node if n.op_type == "Not"), None)
    if notn is None or not notn.input:
        return False
    src = notn.input[0]
    if any(t.name == src for t in model.graph.initializer):
        return False
    # Require the exact input name: "deterministic"
    if (
        any(vi.name == "deterministic" for vi in model.graph.input)
        and src == "deterministic"
    ):
        return True
    producer = next((n for n in model.graph.node if src in n.output), None)
    return producer is not None


@register_primitive(
    jaxpr_primitive="nnx.dropout",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout",
    onnx=[
        {
            "component": "Dropout",
            "doc": "https://onnx.ai/onnx/operators/onnx__Dropout.html",
        }
    ],
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
                    (
                        drop := next(
                            (n for n in m.graph.node if n.op_type == "Dropout"), None
                        )
                    )
                    is not None
                    and (
                        tm_init := next(
                            (i for i in m.graph.initializer if i.name == drop.input[2]),
                            None,
                        )
                    )
                    is not None
                    and (
                        ratio_init := next(
                            (i for i in m.graph.initializer if i.name == drop.input[1]),
                            None,
                        )
                    )
                    is not None
                    and (_tp_to_numpy(tm_init) == np.array(False)).all()
                    and np.isclose(_tp_to_numpy(ratio_init), 0.5).all()
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
            # Structural check: Not->Dropout and 3 inputs on Dropout
            "post_check_onnx_graph": post_check_onnx_graph,
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

    def lower(self, ctx: "IRContext", eqn: Any, params: dict[str, Any]):
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

        # Build Dropout's training_mode:
        # - call_time=True  → keep dynamic: training_mode = Not(deterministic)
        # - call_time=False → if 'deterministic' is a literal, fold to initializer:
        #                     training_mode = const(not deterministic)
        training_mode: ir.Value
        if call_time:
            # Scope-agnostic: let the context synthesize training_mode.
            # - Top-level: deterministic (graph input) → Not → training_mode
            # - Function + Literal: training_mode const (no Not)
            # - Function + Tensor: f_in → Not → training_mode
            training_mode = ctx.ensure_training_mode("deterministic", det_var)
        else:
            # Init-params path: try constant-fold.
            det_is_lit = isinstance(det_var, jcore_ext.Literal) or (
                type(det_var).__name__ == "Literal" and hasattr(det_var, "val")
            )
            if det_is_lit:
                tm_np = np.array(not bool(det_var.val), dtype=np.bool_)
                training_mode = ir.Value(
                    name=ctx.fresh_name("training_mode"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                    const_value=ir.tensor(tm_np),
                )
                ctx._initializers.append(training_mode)
            else:
                # Fallback: keep dynamic via Not
                det_val = ctx.get_value_for_var(
                    det_var, name_hint=ctx.fresh_name("deterministic")
                )
                training_mode = ir.Value(
                    name=ctx.fresh_name("training_mode"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Not",
                        domain="",
                        inputs=[det_val],
                        outputs=[training_mode],
                        name=ctx.fresh_name("Not"),
                    )
                )

        # Dropout(data, ratio, training_mode)
        drop_inputs = [x_val, ratio_c, training_mode]
        y_val = ctx.get_value_for_var(y_var, name_hint=ctx.fresh_name("out"))
        ctx.add_node(
            ir.Node(
                op_type="Dropout",
                domain="",
                inputs=drop_inputs,
                outputs=[y_val],
                name=ctx.fresh_name("Dropout"),
            )
        )
        _stamp_type_and_shape(y_val, x_shape)
        _add_value_info(ctx, y_val)

    @staticmethod
    def _dropout(x, deterministic, *, rate, call_time: bool):
        return DropoutPlugin._PRIM.bind(
            x, deterministic, rate=rate, call_time=call_time
        )

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
            return DropoutPlugin._dropout(
                x, det, rate=float(self.rate), call_time=call_time
            )

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
