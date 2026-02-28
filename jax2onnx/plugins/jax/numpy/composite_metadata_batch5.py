# jax2onnx/plugins/jax/numpy/composite_metadata_batch5.py

from __future__ import annotations

from typing import Final

import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.composite5"


class _JnpCompositeMetadataBatch5Plugin(PrimitiveLeafPlugin):
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
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.corrcoef",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.corrcoef.html",
    onnx=[
        {
            "component": "ReduceMean",
            "doc": "https://onnx.ai/onnx/operators/onnx__ReduceMean.html",
        },
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="corrcoef",
    testcases=[
        {
            "testcase": "jnp_corrcoef_basic",
            "callable": lambda x: jnp.corrcoef(x),
            "input_values": [np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpCorrcoefMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_lstsq",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.lstsq.html",
    onnx=[
        {"component": "SVD", "doc": "https://onnx.ai/onnx/operators/onnx__SVD.html"},
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_lstsq",
    testcases=[
        {
            "testcase": "jnp_linalg_lstsq_basic",
            "callable": lambda a, b: jnp.linalg.lstsq(a, b)[0],
            "input_values": [
                np.array([[2.0]], dtype=np.float32),
                np.array([4.0], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgLstsqMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_pinv",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html",
    onnx=[
        {"component": "SVD", "doc": "https://onnx.ai/onnx/operators/onnx__SVD.html"},
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_pinv",
    testcases=[
        {
            "testcase": "jnp_linalg_pinv_basic",
            "callable": lambda x: jnp.linalg.pinv(x),
            "input_values": [np.array([[2.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgPinvMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_svd",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.svd.html",
    onnx=[{"component": "SVD", "doc": "https://onnx.ai/onnx/operators/onnx__SVD.html"}],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_svd",
    testcases=[
        {
            "testcase": "jnp_linalg_svd_basic",
            "callable": lambda x: jnp.linalg.svd(x),
            "input_values": [np.array([[2.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgSvdMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_eig",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eig.html",
    onnx=[{"component": "Eig", "doc": "https://onnx.ai/onnx/operators/onnx__Eig.html"}],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_eig",
    testcases=[
        {
            "testcase": "jnp_linalg_eig_basic",
            "callable": lambda x: jnp.linalg.eig(x),
            "input_values": [np.array([[2.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgEigMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_eigh",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigh.html",
    onnx=[
        {"component": "Eigh", "doc": "https://onnx.ai/onnx/operators/onnx__Eigh.html"}
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_eigh",
    testcases=[
        {
            "testcase": "jnp_linalg_eigh_basic",
            "callable": lambda x: jnp.linalg.eigh(x),
            "input_values": [np.array([[2.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgEighMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_eigvals",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigvals.html",
    onnx=[{"component": "Eig", "doc": "https://onnx.ai/onnx/operators/onnx__Eig.html"}],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_eigvals",
    testcases=[
        {
            "testcase": "jnp_linalg_eigvals_basic",
            "callable": lambda x: jnp.linalg.eigvals(x),
            "input_values": [np.array([[2.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgEigvalsMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_eigvalsh",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eigvalsh.html",
    onnx=[
        {"component": "Eigh", "doc": "https://onnx.ai/onnx/operators/onnx__Eigh.html"}
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_eigvalsh",
    testcases=[
        {
            "testcase": "jnp_linalg_eigvalsh_basic",
            "callable": lambda x: jnp.linalg.eigvalsh(x),
            "input_values": [np.array([[2.0]], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgEigvalshMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.linalg_multi_dot",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.multi_dot.html",
    onnx=[
        {
            "component": "MatMul",
            "doc": "https://onnx.ai/onnx/operators/onnx__MatMul.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="linalg_multi_dot",
    testcases=[
        {
            "testcase": "jnp_linalg_multi_dot_basic",
            "callable": lambda a, b: jnp.linalg.multi_dot([a, b]),
            "input_values": [
                np.arange(6, dtype=np.float32).reshape(2, 3),
                np.arange(12, dtype=np.float32).reshape(3, 4),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpLinalgMultiDotMetadata(_JnpCompositeMetadataBatch5Plugin):
    pass
