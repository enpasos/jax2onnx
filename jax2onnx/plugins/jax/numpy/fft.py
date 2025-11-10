# jax2onnx/plugins/jax/numpy/fft.py

from __future__ import annotations

from typing import Final

import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.jax.lax.fft import FFTPlugin as _LaxFFTPlugin
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive
from jax2onnx.converter.typing_support import LoweringContextProtocol


_LAX_FFT_PLUGIN: _LaxFFTPlugin = _LaxFFTPlugin()
_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.fft"


class _JnpFFTMetadata(PrimitiveLeafPlugin):
    """Metadata-only wrapper that reuses lax.fft lowering for jnp.fft calls."""

    @classmethod
    def binding_specs(cls):
        # jnp.fft already lowers through lax.fft, so no monkey patches required.
        return []

    def lower(
        self, ctx: LoweringContextProtocol, eqn: jax.core.JaxprEqn
    ) -> None:  # pragma: no cover - delegated lowering
        _LAX_FFT_PLUGIN.lower(ctx, eqn)


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.fft",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.fft.html",
    onnx=[{"component": "DFT", "doc": "https://onnx.ai/onnx/operators/onnx__DFT.html"}],
    since="v0.10.1",
    context="primitives.jnp",
    component="fft",
    testcases=[
        {
            "testcase": "jnp_fft_complex64",
            "callable": lambda x: jnp.fft.fft(x),
            "input_values": [
                np.array(
                    [1.0 + 0.0j, -2.0 + 2.0j, 0.5 - 1.5j, 3.0 + 0.25j],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
        {
            "testcase": "jnp_fft_complex128",
            "callable": lambda x: jnp.fft.fft(x),
            "input_values": [
                np.array(
                    [0.5 - 0.25j, 1.0 + 0.5j, -1.5 + 1.75j, 0.25 - 2.0j],
                    dtype=np.complex128,
                )
            ],
            "expected_output_dtypes": [np.float64],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f64_variant": True,
        },
    ],
)
class JnpFFTMetadata(_JnpFFTMetadata):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.ifft",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.ifft.html",
    onnx=[{"component": "DFT", "doc": "https://onnx.ai/onnx/operators/onnx__DFT.html"}],
    since="v0.10.1",
    context="primitives.jnp",
    component="ifft",
    testcases=[
        {
            "testcase": "jnp_ifft_complex64",
            "callable": lambda x: jnp.fft.ifft(x),
            "input_values": [
                np.array(
                    [2.0 - 1.0j, -1.5 + 0.75j, 0.5 + 2.0j, -0.25 - 0.5j],
                    dtype=np.complex64,
                )
            ],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class JnpIFFTMetadata(_JnpFFTMetadata):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.rfft",
    jax_doc="https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.fft.rfft.html",
    onnx=[{"component": "DFT", "doc": "https://onnx.ai/onnx/operators/onnx__DFT.html"}],
    since="v0.10.1",
    context="primitives.jnp",
    component="rfft",
    testcases=[
        {
            "testcase": "jnp_rfft_float32",
            "callable": lambda x: jnp.fft.rfft(x),
            "input_values": [np.array([0.5, -1.0, 2.0, 0.75], dtype=np.float32)],
            "expected_output_dtypes": [np.float32],
            "post_check_onnx_graph": EG(
                [
                    {"path": "DFT", "counts": {"DFT": 1}},
                ],
                no_unused_inputs=True,
            ),
            "run_only_f32_variant": True,
        },
    ],
)
class JnpRFFTMetadata(_JnpFFTMetadata):
    pass
