# jax2onnx/plugins/jax/numpy/composite_metadata.py

from __future__ import annotations

from typing import Final

import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.composite"


class _JnpCompositeMetadataPlugin(PrimitiveLeafPlugin):
    """Metadata-only wrappers for jnp APIs that already lower compositionally."""

    @classmethod
    def binding_specs(cls):
        return []

    def lower(self, ctx: LoweringContextProtocol, eqn) -> None:
        del ctx, eqn
        raise NotImplementedError(
            "Metadata-only jnp composite plugin should not appear in jaxpr."
        )


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.cov",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cov.html",
    onnx=[
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="cov",
    testcases=[
        {
            "testcase": "jnp_cov_basic",
            "callable": lambda x: jnp.cov(x),
            "input_values": [
                np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
            ],
        },
    ],
)
class JnpCovMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.cross",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.cross.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="cross",
    testcases=[
        {
            "testcase": "jnp_cross_vectors",
            "callable": lambda a, b: jnp.cross(a, b),
            "input_values": [
                np.array([[1.0, 2.0, 3.0], [3.0, -1.0, 0.5]], dtype=np.float32),
                np.array([[4.0, 5.0, 6.0], [0.0, 2.0, -2.0]], dtype=np.float32),
            ],
        },
    ],
)
class JnpCrossMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.delete",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.delete.html",
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
    component="delete",
    testcases=[
        {
            "testcase": "jnp_delete_axis0_index1",
            "callable": lambda x: jnp.delete(x, 1, axis=0),
            "input_values": [np.arange(12, dtype=np.float32).reshape(3, 4)],
        },
    ],
)
class JnpDeleteMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.fft_fftfreq",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.fftfreq.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="fft_fftfreq",
    testcases=[
        {
            "testcase": "jnp_fft_fftfreq_n8",
            "callable": lambda: jnp.fft.fftfreq(8),
            "input_values": [],
        },
    ],
)
class JnpFFTFreqMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.fft_rfftfreq",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.rfftfreq.html",
    onnx=[
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="fft_rfftfreq",
    testcases=[
        {
            "testcase": "jnp_fft_rfftfreq_n8",
            "callable": lambda: jnp.fft.rfftfreq(8),
            "input_values": [],
        },
    ],
)
class JnpRFFTFreqMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.fft_fftshift",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.fftshift.html",
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
    component="fft_fftshift",
    testcases=[
        {
            "testcase": "jnp_fft_fftshift_1d",
            "callable": lambda x: jnp.fft.fftshift(x),
            "input_values": [np.arange(8, dtype=np.float32)],
        },
    ],
)
class JnpFFTShiftMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.fft_ifftshift",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fft.ifftshift.html",
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
    component="fft_ifftshift",
    testcases=[
        {
            "testcase": "jnp_fft_ifftshift_1d",
            "callable": lambda x: jnp.fft.ifftshift(x),
            "input_values": [np.arange(8, dtype=np.float32)],
        },
    ],
)
class JnpIFFTShiftMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.geomspace",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.geomspace.html",
    onnx=[
        {"component": "Log", "doc": "https://onnx.ai/onnx/operators/onnx__Log.html"},
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="geomspace",
    testcases=[
        {
            "testcase": "jnp_geomspace_scalar",
            "callable": lambda: jnp.geomspace(1.0, 16.0, num=5),
            "input_values": [],
        },
    ],
)
class JnpGeomspaceMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.gradient",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.gradient.html",
    onnx=[
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="gradient",
    testcases=[
        {
            "testcase": "jnp_gradient_1d",
            "callable": lambda x: jnp.gradient(x),
            "input_values": [np.array([1.0, 2.0, 4.0, 7.0], dtype=np.float32)],
        },
    ],
)
class JnpGradientMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.i0",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.i0.html",
    onnx=[
        {"component": "Exp", "doc": "https://onnx.ai/onnx/operators/onnx__Exp.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="i0",
    testcases=[
        {
            "testcase": "jnp_i0_basic",
            "callable": lambda x: jnp.i0(x),
            "input_values": [np.array([0.0, 1.0, 2.0], dtype=np.float32)],
        },
    ],
)
class JnpI0Metadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.fill_diagonal",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.fill_diagonal.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="fill_diagonal",
    testcases=[
        {
            "testcase": "jnp_fill_diagonal_basic",
            "callable": lambda x: jnp.fill_diagonal(x, 1.0, inplace=False),
            "input_values": [np.zeros((3, 3), dtype=np.float32)],
        },
    ],
)
class JnpFillDiagonalMetadata(_JnpCompositeMetadataPlugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.histogram_bin_edges",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.histogram_bin_edges.html",
    onnx=[
        {
            "component": "ReduceMin",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMin.html",
        },
        {
            "component": "ReduceMax",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMax.html",
        },
        {
            "component": "Range",
            "doc": "https://onnx.ai/onnx/operators/onnx__Range.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="histogram_bin_edges",
    testcases=[
        {
            "testcase": "jnp_histogram_bin_edges_basic",
            "callable": lambda x: jnp.histogram_bin_edges(x, bins=4),
            "input_values": [np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)],
        },
    ],
)
class JnpHistogramBinEdgesMetadata(_JnpCompositeMetadataPlugin):
    pass
