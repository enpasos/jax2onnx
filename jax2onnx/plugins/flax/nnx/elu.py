# jax2onnx/plugins/flax/nnx/elu.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, cast

import numpy as np
import jax
from jax.extend.core import Primitive
from flax import nnx
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
)


def _alpha_attr_equals(model: Any, expected: float) -> bool:
    node = next(
        (n for n in getattr(model.graph, "node", []) if n.op_type == "Elu"), None
    )
    if node is None:
        return False
    for attr in getattr(node, "attribute", []):
        if getattr(attr, "name", "") != "alpha":
            continue
        val = None
        if hasattr(attr, "f"):
            val = attr.f
        elif hasattr(attr, "FLOAT"):
            val = attr.FLOAT
        elif getattr(attr, "floats", None):
            arr = attr.floats
            if len(arr):
                val = arr[0]
        if val is None:
            continue
        return abs(float(val) - float(expected)) < 1e-6
    # Attribute not present => default alpha==1.0
    return abs(float(expected) - 1.0) < 1e-6


def _make_checker(
    specs: list[str], *, alpha: float, **kwargs: Any
) -> Callable[[Any], bool]:
    checker = expect_graph(specs, **kwargs)

    def _run(model: Any) -> bool:
        return checker(model) and _alpha_attr_equals(model, alpha)

    return _run


@register_primitive(
    jaxpr_primitive="nnx.elu",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.elu.html",
    onnx=[{"component": "Elu", "doc": "https://onnx.ai/onnx/operators/onnx__Elu.html"}],
    since="0.2.0",
    context="primitives.nnx",
    component="elu",
    testcases=[
        {
            "testcase": "elu",
            "callable": lambda x: nnx.elu(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_checker(
                ["Elu:1"], alpha=1.0, no_unused_inputs=True
            ),
        },
        {
            "testcase": "elu_default",
            "callable": lambda x: nnx.elu(x),
            "input_shapes": [("B", 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_checker(
                ["Elu:Bx3"], symbols={"B": None}, alpha=1.0, no_unused_inputs=True
            ),
        },
        {
            "testcase": "elu_alpha",
            "callable": lambda x: nnx.elu(x, alpha=0.5),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_checker(
                ["Elu:2x3"], alpha=0.5, no_unused_inputs=True
            ),
        },
    ],
)
class EluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.elu → ONNX Elu."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.elu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, alpha: float = 1.0
    ) -> jax.core.ShapedArray:
        del alpha
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        alpha = float(eqn.params.get("alpha", 1.0))
        attrs: dict[str, float] = {}
        if not np.isclose(alpha, 1.0):
            attrs["alpha"] = float(alpha)
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Elu",
            input_hint="elu_in",
            output_hint="elu_out",
            attrs=attrs,
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[..., ArrayLike]:
        del orig_fn
        prim = EluPlugin._PRIM

        def patched_elu(x: ArrayLike, alpha: float = 1.0) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x, alpha=alpha))

        return patched_elu

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.nnx", "elu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="elu",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, alpha=1.0: cls.abstract_eval(x, alpha=alpha)
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@EluPlugin._PRIM.def_impl
def _impl(x: ArrayLike, *, alpha: float) -> ArrayLike:
    return jax.nn.elu(x, alpha=alpha)
