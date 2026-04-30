# jax2onnx/plugins/flax/nnx/leaky_relu.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, Optional, cast

import jax
from jax.extend.core import Primitive
from flax import nnx
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins._post_check_onnx_graph import expect_graph
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
)


def _alpha_attr_equals(model: Any, expected: float) -> bool:
    node = next(
        (n for n in getattr(model.graph, "node", []) if n.op_type == "LeakyRelu"), None
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
    return abs(float(expected) - 0.01) < 1e-6


def _make_leaky_relu_checker(
    path: str,
    *,
    alpha: float,
    symbols: Optional[dict[str, Optional[int]]] = None,
) -> Callable[[Any], bool]:
    graph_check = expect_graph([path], symbols=symbols, no_unused_inputs=True)

    def _run(model: Any) -> bool:
        return graph_check(model) and _alpha_attr_equals(model, alpha)

    return _run


@register_primitive(
    jaxpr_primitive="nnx.leaky_relu",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.leaky_relu.html",
    onnx=[
        {
            "component": "LeakyRelu",
            "doc": "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html",
        }
    ],
    since="0.2.0",
    context="primitives.nnx",
    component="leaky_relu",
    testcases=[
        {
            "testcase": "leaky_relu",
            "callable": lambda x: nnx.leaky_relu(x),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_leaky_relu_checker(
                "LeakyRelu:1", alpha=0.01
            ),
        },
        {
            "testcase": "leaky_relu_default",
            "callable": lambda x: nnx.leaky_relu(x),
            "input_shapes": [("B", 5)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_leaky_relu_checker(
                "LeakyRelu:Bx5", symbols={"B": None}, alpha=0.01
            ),
        },
        {
            "testcase": "leaky_relu_custom",
            "callable": lambda x: nnx.leaky_relu(x, negative_slope=0.2),
            "input_shapes": [(2, 3)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_leaky_relu_checker(
                "LeakyRelu:2x3", alpha=0.2
            ),
        },
    ],
)
class LeakyReluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.leaky_relu → ONNX LeakyRelu."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.leaky_relu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, negative_slope: float = 0.01
    ) -> jax.core.ShapedArray:
        del negative_slope
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        negative_slope = float(eqn.params.get("negative_slope", 0.01))
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="LeakyRelu",
            input_hint="leaky_relu_in",
            output_hint="leaky_relu_out",
            attrs={"alpha": negative_slope},
        )

    # ---------- monkey-patch & binding specs ----------
    @staticmethod
    def _make_patch(
        orig_fn: Callable[..., ArrayLike] | None,
    ) -> Callable[..., ArrayLike]:
        del orig_fn
        prim = LeakyReluPlugin._PRIM

        def patched_leaky_relu(x: ArrayLike, negative_slope: float = 0.01) -> ArrayLike:
            return cast(ArrayLike, prim.bind(x, negative_slope=negative_slope))

        return patched_leaky_relu

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("flax.nnx", "leaky_relu_p", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="leaky_relu",
                make_value=lambda orig: cls._make_patch(orig),
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, negative_slope=0.01: cls.abstract_eval(
                    x, negative_slope=negative_slope
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


# ---------- concrete impl for eager execution ----------
@LeakyReluPlugin._PRIM.def_impl
def _impl(x: ArrayLike, *, negative_slope: float) -> ArrayLike:
    return jax.nn.leaky_relu(x, negative_slope=negative_slope)
