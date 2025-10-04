# jax2onnx/plugins/flax/nnx/dropout.py

from __future__ import annotations
from typing import Callable, ClassVar, Any, Optional, Set
import numpy as np
import jax
from jax.extend.core import Primitive
import logging
from jax2onnx.plugins.plugin_system import (
    PrimitiveLeafPlugin,
    construct_and_call,
    register_primitive,
    with_rng_seed,
)
import onnx_ir as ir

from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._ir_shapes import (
    _stamp_type_and_shape,
    _ensure_value_info as _add_value_info,
)

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG

from flax import nnx

# mypy/ruff-only import (avoid runtime cycles)
# from jax2onnx.converter.conversion_api import _IRBuildContext
from jax2onnx.converter.ir_context import IRContext

logger = logging.getLogger(__name__)


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


# Structural sanity check for call-params path:
#  - ensure a Not feeds Dropout with expected shapes (primitive vs MLP)
_CALL_CHECK = EG(
    ["Not -> Dropout:Bx10", "Not -> Dropout:Bx20"],
    symbols={"B": None},
    mode="any",
    search_functions=False,
)


def _find_initializer(model, name: str):
    g = getattr(model, "graph", None)
    if g is None:
        return None
    return next(
        (i for i in getattr(g, "initializer", []) if getattr(i, "name", "") == name),
        None,
    )


def post_check_onnx_graph_init(model) -> bool:
    """
    Init-params path checker:
      • path: input -> Dropout -> output with shape Bx10 (edge after Dropout)
      • no Not nodes
      • both ratio and training_mode are initializers, with values 0.5 and False
    """
    # 1) Structural + shape check on the top graph
    ok_path = EG(
        ["Dropout:Bx10"],
        symbols={"B": None},
        must_absent=["Not"],
        no_unused_inputs=True,
        mode="all",
        search_functions=False,
    )(model)
    if not ok_path:
        return False

    # 2) Initializer value checks
    g = getattr(model, "graph", None)
    if g is None:
        return False
    drop = next(
        (n for n in getattr(g, "node", []) if getattr(n, "op_type", "") == "Dropout"),
        None,
    )
    if drop is None:
        return False
    d_in = getattr(drop, "input", [])
    if len(d_in) < 3:
        return False
    ratio_name = d_in[1]
    tm_name = d_in[2]
    ratio_init = _find_initializer(model, ratio_name)
    tm_init = _find_initializer(model, tm_name)
    if ratio_init is None or tm_init is None:
        return False
    r_np = _tp_to_numpy(ratio_init)
    t_np = _tp_to_numpy(tm_init)
    try:
        ratio_ok = np.isclose(np.asarray(r_np).astype(np.float64), 0.5).all()
        tm_ok = (np.asarray(t_np).astype(np.bool_) == np.array(False)).all()
    except Exception:
        return False
    return bool(ratio_ok and tm_ok)


def post_check_onnx_graph(model) -> bool:
    if not _CALL_CHECK(model):
        return False
    g = getattr(model, "graph", None)
    if g is None:
        return False
    nodes = list(getattr(g, "node", []))
    prod_by_output = {}
    for n in nodes:
        for o in getattr(n, "output", []):
            if o:
                prod_by_output[o] = n

    d = next((n for n in nodes if getattr(n, "op_type", "") == "Dropout"), None)
    if d is None:
        return False
    d_in = list(getattr(d, "input", []))
    if len(d_in) < 3:
        return False

    ratio_name = d_in[1]
    tm_name = d_in[2]
    if not ratio_name:
        return False
    ratio_init = _find_initializer(model, ratio_name)
    if ratio_init is None:
        return False

    if not tm_name:
        return False
    if _find_initializer(model, tm_name) is not None:
        return False

    # Dynamic path: expect a Not producer and a non-initializer source
    not_node = prod_by_output.get(tm_name)
    if not_node is None or getattr(not_node, "op_type", "") != "Not":
        return False
    nin = list(getattr(not_node, "input", []))
    if not nin:
        return False
    src_name = nin[0]
    # Source must not be an initializer constant
    if _find_initializer(model, src_name) is not None:
        return False
    return True


# ---- helpers ---------------------------------------------------------------
def _ir_dtype_from_numpy(dt) -> "ir.DataType":
    dt = np.dtype(dt)
    if dt == np.float32:
        return ir.DataType.FLOAT
    if dt == np.float64:
        return getattr(ir.DataType, "DOUBLE", ir.DataType.FLOAT)
    if dt == np.int64:
        return ir.DataType.INT64
    if dt == np.int32:
        return ir.DataType.INT32
    if dt == np.bool_:
        return ir.DataType.BOOL
    return ir.DataType.FLOAT


def _ensure_scalar_bool_input(ctx: IRContext, name: str) -> ir.Value:
    # Ensure the builder carries an input list we can append to
    if not hasattr(ctx.builder, "inputs") or getattr(ctx.builder, "inputs") is None:
        try:
            ctx.builder.inputs = []
        except Exception:
            pass
    inputs = getattr(ctx.builder, "inputs", None) or []
    for vi in inputs:
        if getattr(vi, "name", "") == name:
            return vi
    v = ir.Value(name=name, type=ir.TensorType(ir.DataType.BOOL), shape=ir.Shape(()))
    try:
        inputs.append(v)
        # persist if inputs is a detached copy
        if getattr(ctx.builder, "inputs", None) is not inputs:
            ctx.builder.inputs = inputs
    except Exception:
        pass
    return v


def _const_tensor(ctx: IRContext, value: Any, *, name: str) -> ir.Value:
    """
    Create a scalar/nd tensor constant robustly:
      • inside a Function body  → Constant node with a tensor-valued attribute
      • at top level            → prefer initializer (to satisfy tests), fall back to Constant
    Returns the produced ir.Value (always pre-typed/shaped).
    """
    arr = np.asarray(value)
    val = ir.Value(
        name=ctx.fresh_name(name),
        type=ir.TensorType(_ir_dtype_from_numpy(arr.dtype)),
        shape=ir.Shape(arr.shape if arr.shape else ()),
        const_value=ir.tensor(arr),
    )

    # Helper: build a tensor-valued attribute robustly across onnx_ir variants
    def _tensor_attr(key: str, np_arr: np.ndarray):
        Attr = getattr(ir, "Attr", None)
        AttrType = getattr(ir, "AttributeType", getattr(ir, "AttrType", None))
        tens = ir.tensor(np_arr)
        if Attr is None:
            raise RuntimeError("onnx_ir.Attr not available")
        if hasattr(Attr, "t"):
            return Attr.t(key, tens)
        if AttrType is not None and hasattr(AttrType, "TENSOR"):
            return Attr(key, AttrType.TENSOR, tens)
        if hasattr(Attr, "tensor"):
            return Attr.tensor(key, tens)
        return Attr(key, tens)

    inside_fn = bool(getattr(ctx, "_inside_function_scope", False))
    if inside_fn:
        # Function body: must use Constant node (initializers don’t serialize into FunctionProto)
        try:
            cattr = _tensor_attr("value", arr)
            node = ir.Node(
                op_type="Constant",
                domain="",
                inputs=[],
                outputs=[val],
                name=ctx.fresh_name("Constant"),
                attributes=[cattr],
                num_outputs=1,
            )
            ctx.add_node(node)
        except Exception:
            # Best-effort fallback
            node = ir.Node(
                op_type="Constant",
                domain="",
                inputs=[],
                outputs=[val],
                name=ctx.fresh_name("Constant"),
                attributes=[],
                num_outputs=1,
            )
            ctx.add_node(node)
    else:
        # Top-level: prefer initializer to match tests/expectations
        appended = False
        # Ensure both context and builder keep initializer containers
        for cont_owner, attr in (
            (ctx, "_initializers"),
            (getattr(ctx, "builder", None), "initializers"),
        ):
            if cont_owner is None:
                continue
            try:
                cont = getattr(cont_owner, attr, None)
                if cont is None:
                    setattr(cont_owner, attr, [])
                    cont = getattr(cont_owner, attr)
                if isinstance(cont, list) and not any(
                    getattr(v, "name", None) == val.name for v in cont
                ):
                    cont.append(val)
                    appended = True
            except Exception:
                continue
        if not appended:
            # Fallback: Constant node if the builder doesn’t keep initializers
            try:
                cattr = _tensor_attr("value", arr)
                node = ir.Node(
                    op_type="Constant",
                    domain="",
                    inputs=[],
                    outputs=[val],
                    name=ctx.fresh_name("Constant"),
                    attributes=[cattr],
                    num_outputs=1,
                )
                ctx.add_node(node)
            except Exception:
                pass
    return val


def _extract_python_bool(var) -> Optional[bool]:
    """Best-effort extraction of a Python bool from a traced JAX variable."""
    if var is None:
        return None
    for attr in ("val", "value"):
        if hasattr(var, attr):
            try:
                return bool(getattr(var, attr))
            except Exception:
                pass
    aval = getattr(var, "aval", None)
    if aval is not None and hasattr(aval, "val"):
        try:
            return bool(aval.val)
        except Exception:
            pass
    return None


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
    context="primitives.nnx",
    component="dropout",
    testcases=[
        {
            "testcase": "dropout_init_params",
            "callable": construct_and_call(
                nnx.Dropout,
                rate=0.5,
                deterministic=True,
                rngs=with_rng_seed(5),
            ),
            "input_shapes": [("B", 10)],
            # Modern check: shape/path via expect_graph and strict initializer values
            "post_check_onnx_graph": post_check_onnx_graph_init,
        },
        {
            "testcase": "dropout_call_params",
            "callable": construct_and_call(
                nnx.Dropout,
                rate=0.5,
                deterministic=False,
                rngs=with_rng_seed(5),
            ),
            "input_shapes": [("B", 10)],
            "input_params": {"deterministic": True},
            # Structural check: Dropout retains shape while inlining training_mode=False
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

    def lower(self, ctx: IRContext, eqn):
        # JAXPR carries two invars: x, deterministic ; rate is a static param.
        invars = list(eqn.invars)
        x_var = invars[0]
        det_var = invars[1] if len(invars) > 1 else None
        out_var = eqn.outvars[0]

        # Params
        call_time = bool(eqn.params.get("call_time", False))
        call_params: Set[str] = getattr(ctx, "_call_input_param_names", set())
        rate = float(eqn.params.get("rate", 0.5))

        if "deterministic" in call_params:
            try:
                ctx.ensure_external_flag("deterministic", det_var)
            except Exception:
                pass

        # Inputs
        x_val = ctx.get_value_for_var(x_var, name_hint=ctx.fresh_name("x"))
        # ratio is always scalar float
        ratio_v = _const_tensor(ctx, np.asarray(rate, dtype=np.float32), name="ratio")

        # Build training flag (Dropout's training_mode == NOT deterministic):
        # - call_time=True:
        #     * top graph        → create/consume scalar BOOL graph input "deterministic"
        #                           and feed Not(deterministic) (matches structural test)
        #     * inside function  → DO NOT add new function inputs; use det_var directly:
        #                           literal → constant; non-literal → Not(det_var)
        # - call_time=False:
        #     literal → constant; otherwise Not(det_var)
        train_v: ir.Value
        if call_time:
            inside_fn = bool(getattr(ctx, "_inside_function_scope", False))
            det_lit = _extract_python_bool(det_var)
            if "deterministic" not in call_params and det_lit is not None:
                train_v = _const_tensor(
                    ctx, np.asarray(not det_lit, dtype=np.bool_), name="training"
                )
            else:
                if inside_fn and det_var is not None:
                    det_in = ctx.get_value_for_var(
                        det_var, name_hint=ctx.fresh_name("det")
                    )
                else:
                    det_in = _ensure_scalar_bool_input(ctx, "deterministic")
                not_out = ir.Value(
                    name=ctx.fresh_name("not_det"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Not",
                        domain="",
                        inputs=[det_in],
                        outputs=[not_out],
                        name=ctx.fresh_name("Not"),
                        num_outputs=1,
                    )
                )
                train_v = not_out
        else:
            # Try to read deterministic as a Python literal.
            # JAXPR may place the literal on the var itself (det_var.val)
            # or on its aval (det_var.aval.val). Handle both.
            det_py = _extract_python_bool(det_var)
            if det_py is not None and "deterministic" not in call_params:
                train_v = _const_tensor(
                    ctx, np.asarray(not det_py, dtype=np.bool_), name="training"
                )
            else:
                # Dynamic path via the actual value of det_var (no heuristics)
                if det_var is not None:
                    det_in = ctx.get_value_for_var(
                        det_var, name_hint=ctx.fresh_name("det")
                    )
                else:
                    det_in = _ensure_scalar_bool_input(ctx, "deterministic")
                not_out = ir.Value(
                    name=ctx.fresh_name("not_det"),
                    type=ir.TensorType(ir.DataType.BOOL),
                    shape=ir.Shape(()),
                )
                ctx.add_node(
                    ir.Node(
                        op_type="Not",
                        domain="",
                        inputs=[det_in],
                        outputs=[not_out],
                        name=ctx.fresh_name("Not"),
                        num_outputs=1,
                    )
                )
                train_v = not_out
        # Dropout has optional 2nd/3rd outputs; we only wire the first (y)
        y_val = ctx.get_value_for_var(out_var, name_hint=ctx.fresh_name("out"))
        ctx.add_node(
            ir.Node(
                op_type="Dropout",
                domain="",
                inputs=[x_val, ratio_v, train_v],
                outputs=[y_val],
                name=ctx.fresh_name("Dropout"),
                num_outputs=1,
            )
        )
        # annotate output: mirror input shape/dtype (preserve batch symbols)
        x_dims_meta = None
        x_shape_val = getattr(x_val, "shape", None)
        if x_shape_val is not None:
            x_dims_meta = getattr(x_shape_val, "dims", None)
            if x_dims_meta is None:
                try:
                    x_dims_meta = tuple(x_shape_val)
                except Exception:
                    x_dims_meta = None
        if not x_dims_meta:
            x_dims_meta = tuple(getattr(getattr(x_var, "aval", None), "shape", ()))
        if x_dims_meta:
            _stamp_type_and_shape(y_val, tuple(x_dims_meta))
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
