# tests/extra_tests/converter/source_nnx_linear_fixtures.py

"""Fixed native hashes for the two unchanged seed-zero Linear recipes.

JAX/JAXLIB 0.8.1 with Flax 0.12.1 independently reproduces a different native
FP32 initializer. This narrowly qualified fixture observation does not change
model parameters, ONNX payload checks, or numerical acceptance criteria.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from typing import Any, TYPE_CHECKING

from flax import nnx
from flax.nnx.nn.dtypes import promote_dtype
import jax
import jax.numpy as jnp
import numpy as np

from jax2onnx.plugins.plugin_system import _FactoryValue

if TYPE_CHECKING:
    from tests.extra_tests.converter.source_precision_support import SourceSettings

MODERN_KERNEL_SHA = "d443c758235aff0c56e9af2d5862ae9807f6ca8e96d9aaf5289450663e4d3ef9"
MINIMUM_KERNEL_SHA = "3ec7968ee4c38dcbe6e3afc92afe01d046724c818169868d96a1909c3fd5ecfd"
BIAS_SHA = "5341e6b2646979a70e57653007a1f310169421ec9bdd9f1a5648f75ade005af1"


def _native_versions() -> tuple[str, str, str]:
    return version("jax"), version("jaxlib"), version("flax")


def expected_original_linear_kernel(
    recipe: Any, ordinal: int, *, settings: SourceSettings | None = None
) -> str:
    """Select a fixed expectation from the original recipe and measured binding."""
    assert ordinal in (0, 3), "Only the original Linear rows 0/3 were measured"
    assert recipe._ctor is nnx.Linear and recipe._args == (128, 64)
    assert recipe._dtype is None and recipe._instance is None
    assert set(recipe._kwargs) == (
        {"dtype", "rngs"} if ordinal == 0 else {"dtype", "rngs", "use_bias"}
    )
    if ordinal == 3:
        assert recipe._kwargs["use_bias"] is False
    dtype = recipe._kwargs["dtype"]
    assert type(dtype) is _FactoryValue
    assert dtype.resolve(jnp.float32) is jnp.float32
    assert dtype.resolve(jnp.float64) is jnp.float64
    rng = recipe._kwargs["rngs"]
    assert type(rng) is _FactoryValue
    trace = rng.metadata["rng_trace"]
    assert trace.kind == "nnx_rng" and trace.seed == 0
    if _native_versions() != ("0.8.1", "0.8.1", "0.12.1"):
        return MODERN_KERNEL_SHA
    observed = (
        (
            jax.config.jax_default_matmul_precision,
            jax.config.jax_platforms,
            jax.config.jax_numpy_dtype_promotion,
            jax.config.jax_numpy_rank_promotion,
        )
        if settings is None
        else (
            settings.default_matmul_precision,
            settings.jax_platforms,
            settings.numpy_dtype_promotion,
            settings.numpy_rank_promotion,
        )
    )
    # Both caller-x64 values were independently measured on CPU. An implicit
    # CPU selection is equivalent to setting jax_platforms="cpu" explicitly.
    assert observed[1] in (None, "cpu") and jax.default_backend() == "cpu"
    assert (observed[0], observed[2], observed[3]) == (
        None,
        "standard",
        "allow",
    ), f"Unmeasured minimum settings: {observed}"
    return MINIMUM_KERNEL_SHA


def assert_original_linear(
    model: Any,
    recipe: Any,
    ordinal: int,
    *,
    settings: SourceSettings | None = None,
) -> str:
    """Check actual native bytes against the fixed qualified fixture expectation."""
    expected = expected_original_linear_kernel(recipe, ordinal, settings=settings)
    assert type(model) is nnx.Linear
    assert model.dtype == model.param_dtype == jnp.float32
    assert model.precision is None and model.preferred_element_type is None
    assert (
        model.promote_dtype is promote_dtype
        and model.dot_general is jax.lax.dot_general
    )
    kernel = np.asarray(model.kernel[...])
    assert kernel.dtype == np.float32 and kernel.shape == (128, 64)
    assert (
        hashlib.sha256(kernel.tobytes(order="C")).hexdigest() == expected
    ), "Native kernel differs from qualified fixture"
    if ordinal == 0:
        bias = np.asarray(model.bias[...])
        assert bias.dtype == np.float32 and bias.shape == (64,)
        assert hashlib.sha256(bias.tobytes(order="C")).hexdigest() == BIAS_SHA
    else:
        assert model.bias is None
    return expected
