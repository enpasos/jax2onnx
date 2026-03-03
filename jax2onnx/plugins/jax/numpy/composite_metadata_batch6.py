# jax2onnx/plugins/jax/numpy/composite_metadata_batch6.py

from __future__ import annotations

from typing import Final

from jax import core
import jax.numpy as jnp
import numpy as np

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive


_METADATA_PRIMITIVE_PREFIX: Final[str] = "metadata::jnp.composite6"


class _JnpCompositeMetadataBatch6Plugin(PrimitiveLeafPlugin):
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
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.gcd",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.gcd.html",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"},
        {
            "component": "Equal",
            "doc": "https://onnx.ai/onnx/operators/onnx__Equal.html",
        },
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="gcd",
    testcases=[
        {
            "testcase": "jnp_gcd_basic",
            "callable": lambda a, b: jnp.gcd(a, b),
            "input_values": [
                np.array(12, dtype=np.int32),
                np.array(8, dtype=np.int32),
            ],
        },
    ],
)
class JnpGcdMetadata(_JnpCompositeMetadataBatch6Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.lcm",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.lcm.html",
    onnx=[
        {"component": "Loop", "doc": "https://onnx.ai/onnx/operators/onnx__Loop.html"},
        {"component": "Mul", "doc": "https://onnx.ai/onnx/operators/onnx__Mul.html"},
        {"component": "Div", "doc": "https://onnx.ai/onnx/operators/onnx__Div.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="lcm",
    testcases=[
        {
            "testcase": "jnp_lcm_basic",
            "callable": lambda a, b: jnp.lcm(a, b),
            "input_values": [
                np.array(12, dtype=np.int32),
                np.array(8, dtype=np.int32),
            ],
        },
    ],
)
class JnpLcmMetadata(_JnpCompositeMetadataBatch6Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.put",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.put.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="put",
    testcases=[
        {
            "testcase": "jnp_put_basic",
            "callable": lambda x: jnp.put(
                x,
                np.array([1, 3], dtype=np.int32),
                np.array([9.0, 8.0], dtype=np.float32),
                inplace=False,
            ),
            "input_values": [np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPutMetadata(_JnpCompositeMetadataBatch6Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.put_along_axis",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.put_along_axis.html",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="put_along_axis",
    testcases=[
        {
            "testcase": "jnp_put_along_axis_basic",
            "callable": lambda x: jnp.put_along_axis(
                x,
                np.array([[1], [0]], dtype=np.int32),
                np.array([[9.0], [8.0]], dtype=np.float32),
                axis=1,
                inplace=False,
            ),
            "input_values": [
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            ],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpPutAlongAxisMetadata(_JnpCompositeMetadataBatch6Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.ndarray_at",
    jax_doc="https://docs.jax.dev/en/latest/jax.numpy.ndarray.at",
    onnx=[
        {
            "component": "ScatterND",
            "doc": "https://onnx.ai/onnx/operators/onnx__ScatterND.html",
        }
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="ndarray_at",
    testcases=[
        {
            "testcase": "jnp_ndarray_at_set_basic",
            "callable": lambda x: x.at[1].set(9.0),
            "input_values": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpNdarrayAtMetadata(_JnpCompositeMetadataBatch6Plugin):
    pass


@register_primitive(
    jaxpr_primitive=f"{_METADATA_PRIMITIVE_PREFIX}.vectorize",
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.numpy.vectorize.html",
    onnx=[
        {"component": "Add", "doc": "https://onnx.ai/onnx/operators/onnx__Add.html"},
    ],
    since="0.12.1",
    context="primitives.jnp",
    component="vectorize",
    testcases=[
        {
            "testcase": "jnp_vectorize_basic",
            "callable": lambda x: jnp.vectorize(lambda t: t + 1.0)(x),
            "input_values": [np.array([1.0, 2.0, 3.0], dtype=np.float32)],
            "run_only_f32_variant": True,
        },
    ],
)
class JnpVectorizeMetadata(_JnpCompositeMetadataBatch6Plugin):
    pass
