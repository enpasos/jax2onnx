# jax2onnx/plugins/jax/numpy/composite_metadata_batch2.py

from __future__ import annotations

from typing import Final

from jax import core
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.composite2"


class _JnpCompositeMetadataBatch2Plugin(PrimitiveLeafPlugin):
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
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.logspace",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.logspace.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
        {"component": "Pow", "doc": "https://onnx.ai/onnx/operators/onnx__Pow.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="logspace",
    testcases=[
        {
            "testcase": "jnp_logspace_basic",
            "callable": lambda: jnp.logspace(0.0, 2.0, num=5),
            "input_values": [],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLogspaceMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.matvec",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matvec.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="matvec",
    testcases=[
        {
            "testcase": "jnp_matvec_basic",
            "callable": lambda a, b: jnp.matvec(a, b),
            "input_values": [
                np.arange(12, dtype=np.float32).reshape(3, 4),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpMatVecMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.median",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="median",
    testcases=[
        {
            "testcase": "jnp_median_basic",
            "callable": lambda x: jnp.median(x),
            "input_values": [np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpMedianMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanmax",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmax.html",
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
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanmax",
    testcases=[
        {
            "testcase": "jnp_nanmax_basic",
            "callable": lambda x: jnp.nanmax(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanMaxMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanmean",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmean.html",
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
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanmean",
    testcases=[
        {
            "testcase": "jnp_nanmean_basic",
            "callable": lambda x: jnp.nanmean(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanMeanMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanmin",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmin.html",
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
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanmin",
    testcases=[
        {
            "testcase": "jnp_nanmin_basic",
            "callable": lambda x: jnp.nanmin(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanMinMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nansum",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nansum.html",
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
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nansum",
    testcases=[
        {
            "testcase": "jnp_nansum_basic",
            "callable": lambda x: jnp.nansum(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanSumMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanprod",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanprod.html",
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
            "component": "ReduceProd",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceProd.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanprod",
    testcases=[
        {
            "testcase": "jnp_nanprod_basic",
            "callable": lambda x: jnp.nanprod(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanProdMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanstd",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanstd.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanstd",
    testcases=[
        {
            "testcase": "jnp_nanstd_basic",
            "callable": lambda x: jnp.nanstd(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanStdMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanvar",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanvar.html",
    onnx=[
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanvar",
    testcases=[
        {
            "testcase": "jnp_nanvar_basic",
            "callable": lambda x: jnp.nanvar(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanVarMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanmedian",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanmedian.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanmedian",
    testcases=[
        {
            "testcase": "jnp_nanmedian_basic",
            "callable": lambda x: jnp.nanmedian(x),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanMedianMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanpercentile",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanpercentile.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanpercentile",
    testcases=[
        {
            "testcase": "jnp_nanpercentile_basic",
            "callable": lambda x: jnp.nanpercentile(x, 50.0),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanPercentileMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.nanquantile",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nanquantile.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "Where",
            "doc": "https://onnx.ai/onnx/operators/onnx__Where.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="nanquantile",
    testcases=[
        {
            "testcase": "jnp_nanquantile_basic",
            "callable": lambda x: jnp.nanquantile(x, 0.5),
            "input_values": [np.array([1.0, np.nan, 3.0, -2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNanQuantileMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.partition",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.partition.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="partition",
    testcases=[
        {
            "testcase": "jnp_partition_basic",
            "callable": lambda x: jnp.partition(x, 2),
            "input_values": [np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPartitionMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.percentile",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.percentile.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="percentile",
    testcases=[
        {
            "testcase": "jnp_percentile_basic",
            "callable": lambda x: jnp.percentile(x, 50.0),
            "input_values": [np.array([1.0, 3.0, -2.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPercentileMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.quantile",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.quantile.html",
    onnx=[
        {"component": "TopK", "doc": "https://onnx.ai/onnx/operators/onnx__TopK.html"},
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="quantile",
    testcases=[
        {
            "testcase": "jnp_quantile_basic",
            "callable": lambda x: jnp.quantile(x, 0.5),
            "input_values": [np.array([1.0, 3.0, -2.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpQuantileMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.roll",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.roll.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="roll",
    testcases=[
        {
            "testcase": "jnp_roll_basic",
            "callable": lambda x: jnp.roll(x, 2),
            "input_values": [np.arange(8, dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpRollMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.rot90",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.rot90.html",
    onnx=[
        {
            "component": "Transpose",
            "doc": "https://onnx.ai/onnx/operators/onnx__Transpose.html",
        },
        {
            "component": "Gather",
            "doc": "https://onnx.ai/onnx/operators/onnx__Gather.html",
        },
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="rot90",
    testcases=[
        {
            "testcase": "jnp_rot90_basic",
            "callable": lambda x: jnp.rot90(x),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpRot90Metadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.sinc",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sinc.html",
    onnx=[
        {"component": "Sin", "doc": "https://onnx.ai/onnx/operators/onnx__Sin.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="sinc",
    testcases=[
        {
            "testcase": "jnp_sinc_basic",
            "callable": lambda x: jnp.sinc(x),
            "input_values": [np.array([-1.0, 0.0, 1.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpSincMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.std",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.std.html",
    onnx=[
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
        {"component": "Sqrt", "doc": "https://onnx.ai/onnx/operators/onnx__Sqrt.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="std",
    testcases=[
        {
            "testcase": "jnp_std_basic",
            "callable": lambda x: jnp.std(x),
            "input_values": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpStdMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.var",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.var.html",
    onnx=[
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="var",
    testcases=[
        {
            "testcase": "jnp_var_basic",
            "callable": lambda x: jnp.var(x),
            "input_values": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpVarMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.tensordot",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tensordot.html",
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
    component="tensordot",
    testcases=[
        {
            "testcase": "jnp_tensordot_basic",
            "callable": lambda a, b: jnp.tensordot(a, b, axes=1),
            "input_values": [
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.arange(12, dtype=np.float32).reshape(3, 4),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpTensorDotMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.trace",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.trace.html",
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
    component="trace",
    testcases=[
        {
            "testcase": "jnp_trace_basic",
            "callable": lambda x: jnp.trace(x),
            "input_values": [np.arange(9, dtype=np.float32).reshape(3, 3)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpTraceMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.vdot",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vdot.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="vdot",
    testcases=[
        {
            "testcase": "jnp_vdot_basic",
            "callable": lambda a, b: jnp.vdot(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpVDotMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.vecdot",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vecdot.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="vecdot",
    testcases=[
        {
            "testcase": "jnp_vecdot_basic",
            "callable": lambda a, b: jnp.vecdot(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([4.0, 5.0, 6.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpVecDotMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.vecmat",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vecmat.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="vecmat",
    testcases=[
        {
            "testcase": "jnp_vecmat_basic",
            "callable": lambda a, b: jnp.vecmat(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.arange(12, dtype=np.float32).reshape(3, 4),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpVecMatMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.tri",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.tri.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
        {
            "component": "LessOrEqual",
            "doc": "https://onnx.ai/onnx/operators/onnx__LessOrEqual.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="tri",
    testcases=[
        {
            "testcase": "jnp_tri_basic",
            "callable": lambda: jnp.tri(3, 4, k=0, dtype=jnp.float32),
            "input_values": [],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpTriMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.unravel_index",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unravel_index.html",
    onnx=[
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Mod", "doc": "https://onnx.ai/onnx/operators/onnx__Mod.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="unravel_index",
    testcases=[
        {
            "testcase": "jnp_unravel_index_basic",
            "callable": lambda x: jnp.unravel_index(x, (3, 4)),
            "input_values": [np.array([0, 5, 11], dtype=np.int32)],
        },
    ],
)
class JnpUnravelIndexMetadata(_JnpCompositeMetadataBatch2Plugin):
    pass
