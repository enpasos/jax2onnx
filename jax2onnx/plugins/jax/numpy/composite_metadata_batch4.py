# jax2onnx/plugins/jax/numpy/composite_metadata_batch4.py

from __future__ import annotations

from typing import Final

import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.composite4"


class _JnpCompositeMetadataBatch4Plugin(PrimitiveLeafPlugin):
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
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.block",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.block.html",
    onnx=[
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="block",
    testcases=[
        {
            "testcase": "jnp_block_basic",
            "callable": lambda x, y: jnp.block([[x, y], [y, x]]),
            "input_values": [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpBlockMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_matrix_rank",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.matrix_rank.html",
    onnx=[
        {"component": "SVD", "doc": "https://onnx.ai/onnx/operators/onnx__SVD.html"},
        {
            "component": "ReduceSum",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceSum.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_matrix_rank",
    testcases=[
        {
            "testcase": "jnp_linalg_matrix_rank_basic",
            "callable": lambda x: jnp.linalg.matrix_rank(x),
            "input_values": [np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgMatrixRankMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_cond",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.cond.html",
    onnx=[
        {"component": "SVD", "doc": "https://onnx.ai/onnx/operators/onnx__SVD.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_cond",
    testcases=[
        {
            "testcase": "jnp_linalg_cond_basic",
            "callable": lambda x: jnp.linalg.cond(x),
            "input_values": [np.array([[2.0, 0.5], [1.0, 1.5]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgCondMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_qr",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.qr.html",
    onnx=[{"component": "QR", "doc": "https://onnx.ai/onnx/operators/onnx__QR.html"}],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_qr",
    testcases=[
        {
            "testcase": "jnp_linalg_qr_basic",
            "callable": lambda x: jnp.linalg.qr(x),
            "input_values": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgQrMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_svdvals",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.svdvals.html",
    onnx=[{"component": "SVD", "doc": "https://onnx.ai/onnx/operators/onnx__SVD.html"}],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_svdvals",
    testcases=[
        {
            "testcase": "jnp_linalg_svdvals_basic",
            "callable": lambda x: jnp.linalg.svdvals(x),
            "input_values": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgSvdValsMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_cholesky",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.cholesky.html",
    onnx=[
        {
            "component": "Cholesky",
            "doc": "https://onnx.ai/onnx/operators/onnx__Cholesky.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_cholesky",
    testcases=[
        {
            "testcase": "jnp_linalg_cholesky_basic",
            "callable": lambda x: jnp.linalg.cholesky(x),
            "input_values": [np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgCholeskyMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_matrix_power",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.matrix_power.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_matrix_power",
    testcases=[
        {
            "testcase": "jnp_linalg_matrix_power_basic",
            "callable": lambda x: jnp.linalg.matrix_power(x, 3),
            "input_values": [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgMatrixPowerMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.polyadd",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyadd.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
        {
            "component": "Pad",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pad.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="polyadd",
    testcases=[
        {
            "testcase": "jnp_polyadd_basic",
            "callable": lambda a, b: jnp.polyadd(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPolyAddMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.polyder",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyder.html",
    onnx=[
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {
            "component": "Slice",
            "doc": "https://onnx.ai/onnx/operators/onnx__Slice.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="polyder",
    testcases=[
        {
            "testcase": "jnp_polyder_basic",
            "callable": lambda p: jnp.polyder(p),
            "input_values": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPolyDerMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.polydiv",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polydiv.html",
    onnx=[
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="polydiv",
    testcases=[
        {
            "testcase": "jnp_polydiv_basic",
            "callable": lambda a, b: jnp.polydiv(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPolyDivMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.polyint",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polyint.html",
    onnx=[
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
        {
            "component": "Concat",
            "doc": "https://onnx.ai/onnx/operators/onnx__Concat.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="polyint",
    testcases=[
        {
            "testcase": "jnp_polyint_basic",
            "callable": lambda p: jnp.polyint(p),
            "input_values": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPolyIntMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.polysub",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.polysub.html",
    onnx=[
        {"component": "Sub", "doc": "https://onnx.ai/onnx/operators/onnx__Sub.html"},
        {
            "component": "Pad",
            "doc": "https://onnx.ai/onnx/operators/onnx__Pad.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="polysub",
    testcases=[
        {
            "testcase": "jnp_polysub_basic",
            "callable": lambda a, b: jnp.polysub(a, b),
            "input_values": [
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                np.array([3.0, 4.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPolySubMetadata(_JnpCompositeMetadataBatch4Plugin):
    pass
