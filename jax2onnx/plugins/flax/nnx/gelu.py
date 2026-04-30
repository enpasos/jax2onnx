# jax2onnx/plugins/flax/nnx/gelu.py

from __future__ import annotations
from typing import Any, Callable, ClassVar, Optional, cast

import jax
from jax.extend.core import Primitive
from flax import nnx
from numpy.typing import ArrayLike

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.jax.nn._builder_utils import (
    lower_unary_elementwise,
)
from jax2onnx.plugins._post_check_onnx_graph import expect_graph


def _make_gelu_checker(
    path: str,
    *,
    approx: str,
    symbols: Optional[dict[str, Optional[int]]] = None,
) -> Callable[[Any], bool]:
    graph_check = expect_graph([path], symbols=symbols, no_unused_inputs=True)

    def _run(model: Any) -> bool:
        return graph_check(model) and _approx_attr_equals(model, approx)

    return _run


@register_primitive(
    jaxpr_primitive="nnx.gelu",
    jax_doc="https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/activations.html#flax.nnx.gelu",
    onnx=[
        {"component": "Gelu", "doc": "https://onnx.ai/onnx/operators/onnx__Gelu.html"}
    ],
    since="0.1.0",
    context="primitives.nnx",
    component="gelu",
    testcases=[
        {
            "testcase": "gelu",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1,)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_gelu_checker("Gelu:1", approx="none"),
        },
        {
            "testcase": "gelu_1",
            "callable": lambda x: nnx.gelu(x, approximate=False),
            "input_shapes": [(1, 10)],
            "run_only_f32_variant": True,
            "post_check_onnx_graph": _make_gelu_checker("Gelu:1x10", approx="none"),
        },
        {
            "testcase": "gelu_2",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [(1,)],
            "post_check_onnx_graph": _make_gelu_checker("Gelu:1", approx="tanh"),
        },
        {
            "testcase": "gelu_3",
            "callable": lambda x: nnx.gelu(x, approximate=True),
            "input_shapes": [("B", 10)],
            "post_check_onnx_graph": _make_gelu_checker(
                "Gelu:Bx10", symbols={"B": None}, approx="tanh"
            ),
        },
        {
            "testcase": "gelu_4",
            "callable": lambda x: nnx.gelu(x),
            "input_shapes": [(1,)],
            # default path (no flag) must be approximate=True => "tanh"
            "post_check_onnx_graph": _make_gelu_checker("Gelu:1", approx="tanh"),
        },
        {
            "testcase": "gelu_5",
            "callable": lambda x: nnx.gelu(x),
            "input_shapes": [("B", 10)],
            "post_check_onnx_graph": _make_gelu_checker(
                "Gelu:Bx10", symbols={"B": None}, approx="tanh"
            ),
        },
    ],
)
class GeluPlugin(PrimitiveLeafPlugin):
    """IR-only plugin for flax.nnx.gelu → ONNX Gelu."""

    _PRIM: ClassVar[Primitive] = Primitive("nnx.gelu")
    _PRIM.multiple_results = False
    _ABSTRACT_EVAL_BOUND: ClassVar[bool] = False

    # ---------- abstract eval ----------
    @staticmethod
    def abstract_eval(
        x: jax.core.AbstractValue, *, approximate: bool = True
    ) -> jax.core.ShapedArray:
        del approximate
        return jax.core.ShapedArray(x.shape, x.dtype)

    # ---------- lowering (IR) ----------
    def lower(self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn) -> None:
        approximate = bool(eqn.params.get("approximate", True))
        approx_str = "tanh" if approximate else "none"
        lower_unary_elementwise(
            ctx,
            eqn,
            op_name="Gelu",
            input_hint="gelu_in",
            output_hint="gelu_out",
            attrs={"approximate": approx_str},
        )

    # ---------- runtime impl (eager) ----------
    @staticmethod
    def _gelu(x: ArrayLike, *, approximate: bool = True) -> ArrayLike:
        return cast(ArrayLike, GeluPlugin._PRIM.bind(x, approximate=bool(approximate)))

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        def _make_value(
            orig: Callable[..., ArrayLike] | None,
        ) -> Callable[..., ArrayLike]:
            del orig

            def _patched(x: ArrayLike, approximate: bool = True) -> ArrayLike:
                return cls._gelu(x, approximate=approximate)

            return _patched

        return [
            # Export our private primitive under flax.nnx.gelu_p for compatibility
            AssignSpec("flax.nnx", "gelu_p", cls._PRIM, delete_if_missing=True),
            # Monkey-patch flax.nnx.gelu to bind our primitive while tracing
            MonkeyPatchSpec(
                target="flax.nnx",
                attr="gelu",
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]

    @classmethod
    def ensure_abstract_eval_bound(cls) -> None:
        if not cls._ABSTRACT_EVAL_BOUND:
            cls._PRIM.def_abstract_eval(
                lambda x, *, approximate=True: cls.abstract_eval(
                    x, approximate=approximate
                )
            )
            cls._ABSTRACT_EVAL_BOUND = True


@GeluPlugin._PRIM.def_impl
def _impl(x: ArrayLike, *, approximate: bool = True) -> ArrayLike:
    # Eager fallback using JAX's gelu
    return jax.nn.gelu(x, approximate=bool(approximate))


def _decode_attr_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8")
    return str(value)


def _approx_attr_equals(model: Any, expected: str) -> bool:
    """Post-check helper: ensure there is a Gelu node with approximate == expected."""
    # Find first Gelu node
    n = next((n for n in model.graph.node if n.op_type == "Gelu"), None)
    if n is None:
        return False
    # Attribute may be encoded differently; try common layouts
    for a in getattr(n, "attribute", []):
        nm = getattr(a, "name", "")
        if nm != "approximate":
            continue
        # ONNX NodeProto: a.s is bytes in many builds
        s = getattr(a, "s", None)
        if s:
            try:
                return _decode_attr_text(s) == expected
            except Exception:
                pass
        # Some tools put string in .strings[0]
        strs = getattr(a, "strings", None)
        if strs and len(strs) > 0:
            try:
                return _decode_attr_text(strs[0]) == expected
            except Exception:
                pass
    # If attribute list is missing, fail
    return False
