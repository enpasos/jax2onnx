# jax2onnx/plugins/jax/numpy/composite_metadata_batch3.py

from __future__ import annotations

from typing import Final

from jax import core
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.composite3"


class _JnpCompositeMetadataBatch3Plugin(PrimitiveLeafPlugin):
    """Metadata-only wrappers for jnp APIs that already lower compositionally."""

    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        specs: list[AssignSpec | MonkeyPatchSpec] = []
        return specs

    def lower(self, ctx: LoweringContextProtocol, eqn: core.JaxprEqn) -> None:
        del ctx, eqn
        raise NotImplementedError(
            "Metadata-only jnp composite plugin should not appear in jaxpr."
        )


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanargmax",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanargmax.html",
    onnx=[
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "ArgMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMax.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanargmax",
    testcases=[
        {
            "testcase": "jnp_nanargmax_basic",
            "callable": lambda x: jnp.nanargmax(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanArgMaxMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanargmin",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanargmin.html",
    onnx=[
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "ArgMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ArgMin.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanargmin",
    testcases=[
        {
            "testcase": "jnp_nanargmin_basic",
            "callable": lambda x: jnp.nanargmin(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanArgMinMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nancumsum",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nancumsum.html",
    onnx=[
        {
            "component": "IsNaN",
            "doc": "https://onnx.ai/onnx/operators/onnx__IsNaN.html",
        },
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
        {
            "component": "CumSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__CumSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nancumsum",
    testcases=[
        {
            "testcase": "jnp_nancumsum_basic",
            "callable": lambda x: jnp.nancumsum(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanCumSumMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.trapezoid",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trapezoid.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="trapezoid",
    testcases=[
        {
            "testcase": "jnp_trapezoid_basic",
            "callable": lambda x: jnp.trapezoid(x),
            "input_values": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpTrapezoidMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.ravel_multi_index",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ravel_multi_index.html",
    onnx=[
        {"component": "Mod", "doc": "https://onnx.ai/onnx/operators/onnx__Mod.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="ravel_multi_index",
    testcases=[
        {
            "testcase": "jnp_ravel_multi_index_wrap",
            "callable": lambda x, y: jnp.ravel_multi_index((x, y), (3, 4), mode="wrap"),
            "input_values": [
                np.array([0, 1, 2], dtype=np.int32),
                np.array([1, 2, 3], dtype=np.int32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpRavelMultiIndexMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.choose",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.choose.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="choose",
    testcases=[
        {
            "testcase": "jnp_choose_wrap",
            "callable": lambda a, b, c: jnp.choose(a, (b, c), mode="wrap"),
            "input_values": [
                np.array([0, 1, 0], dtype=np.int32),
                np.array([10, 20, 30], dtype=np.int32),
                np.array([1, 2, 3], dtype=np.int32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpChooseMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_trace",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.trace.html",
    onnx=[
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_trace",
    testcases=[
        {
            "testcase": "jnp_linalg_trace_basic",
            "callable": lambda x: jnp.linalg.trace(x),
            "input_values": [np.arange(9, dtype=np.float32).reshape(3, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgTraceMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_vecdot",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.vecdot.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_vecdot",
    testcases=[
        {
            "testcase": "jnp_linalg_vecdot_basic",
            "callable": lambda a, b: jnp.linalg.vecdot(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgVecDotMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_vector_norm",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.vector_norm.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_vector_norm",
    testcases=[
        {
            "testcase": "jnp_linalg_vector_norm_basic",
            "callable": lambda x: jnp.linalg.vector_norm(x),
            "input_values": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgVectorNormMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_matrix_norm",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.matrix_norm.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_matrix_norm",
    testcases=[
        {
            "testcase": "jnp_linalg_matrix_norm_basic",
            "callable": lambda x: jnp.linalg.matrix_norm(x),
            "input_values": [np.arange(9, dtype=np.float32).reshape(3, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgMatrixNormMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_tensordot",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.tensordot.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_tensordot",
    testcases=[
        {
            "testcase": "jnp_linalg_tensordot_basic",
            "callable": lambda a, b: jnp.linalg.tensordot(a, b, axes=1),
            "input_values": [
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.arange(12, dtype=np.float32).reshape(3, 4),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgTensorDotMetadata(_JnpCompositeMetadataBatch3Plugin):
    pass
