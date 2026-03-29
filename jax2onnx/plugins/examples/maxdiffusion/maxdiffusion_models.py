# jax2onnx/plugins/examples/maxdiffusion/maxdiffusion_models.py

from __future__ import annotations

import importlib
import importlib.machinery
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
)

logger: logging.Logger = logging.getLogger(__name__)

MAXDIFFUSION_SRC_ENV: str = os.environ.get("JAX2ONNX_MAXDIFFUSION_SRC", "").strip()
MODEL_FILTER_ENV: str = "JAX2ONNX_MAXDIFFUSION_MODELS"

# Pin to a known-good config set that does not require Wan/LTX transformer
# models (which have additional heavy deps).  SDXL-based configs are the
# safest to trace on CPU.
DEFAULT_MODELS: tuple[str, ...] = ("base_xl.yml",)


def _resolve_maxdiffusion_src() -> Path | None:
    """Resolve JAX2ONNX_MAXDIFFUSION_SRC to a valid MaxDiffusion checkout."""
    if not MAXDIFFUSION_SRC_ENV:
        return None
    p: Path = Path(MAXDIFFUSION_SRC_ENV).expanduser().resolve()
    if p.exists() and (p / "src" / "maxdiffusion").is_dir():
        return p
    return None


MAXDIFFUSION_SRC_PATH: Path | None = _resolve_maxdiffusion_src()
MAXDIFFUSION_PKG_PATH: Path | None = None
CONFIGS_DIR: Path | None = None
_MAXDIFFUSION_PATH_ERROR: Exception | None = None

if MAXDIFFUSION_SRC_PATH is not None:
    _src_dir: str = str(MAXDIFFUSION_SRC_PATH / "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    MAXDIFFUSION_PKG_PATH: Path | None = MAXDIFFUSION_SRC_PATH / "src" / "maxdiffusion"
    _cfg_candidate: Path = MAXDIFFUSION_PKG_PATH / "configs"
    if _cfg_candidate.is_dir():
        CONFIGS_DIR: Path | None = _cfg_candidate
elif MAXDIFFUSION_SRC_ENV:
    logger.warning(
        "JAX2ONNX_MAXDIFFUSION_SRC is set to '%s' but does not point to "
        "a valid MaxDiffusion checkout (expected <root>/src/maxdiffusion/).",
        MAXDIFFUSION_SRC_ENV,
    )
    _MAXDIFFUSION_PATH_ERROR: Exception | None = FileNotFoundError(
        f"JAX2ONNX_MAXDIFFUSION_SRC does not point to a MaxDiffusion checkout: "
        f"{MAXDIFFUSION_SRC_ENV}"
    )
else:
    logger.info(
        "JAX2ONNX_MAXDIFFUSION_SRC not set; skipping optional "
        "examples.maxdiffusion registration."
    )


# ---------------------------------------------------------------------------
# Dependency stubs
# ---------------------------------------------------------------------------


def _ensure_stubs() -> None:
    """Install module stubs for heavy optional deps that MaxDiffusion imports
    at module level (aqt, qwix, tokamax, google.cloud.storage, etc.).

    The MaxText plugin already carries comprehensive stubs for all of these.
    When the maxtext plugin is installed alongside maxdiffusion we simply
    delegate to its stub installers.  Otherwise we provide minimal
    self-contained stubs.
    """
    # --- Try the comprehensive MaxText stubs first. -------------------------
    try:
        from jax2onnx.plugins.examples.maxtext import maxtext_models as _mt

        _stub_fns: list[str] = [
            "_ensure_aqt_stub",
            "_ensure_qwix_stub",
            "_ensure_tokamax_stub",
            "_ensure_google_cloud_storage_stub",
            "_ensure_tensorflow_stub",
            "_ensure_tensorboardx_stub",
            "_ensure_omegaconf_stub",
            "_ensure_pil_stub",
            "_ensure_grain_stub",
            "_ensure_datasets_stub",
        ]
        for fn_name in _stub_fns:
            fn: Any = getattr(_mt, fn_name, None)
            if callable(fn):
                fn()
        logger.debug("MaxDiffusion stubs: delegated to maxtext_models.")
        _ensure_extra_stubs()
        return
    except Exception as exc:
        logger.debug("MaxText stubs unavailable (%s); using self-contained stubs.", exc)

    # --- Self-contained fallback stubs --------------------------------------
    _ensure_fallback_stubs()
    _ensure_extra_stubs()


def _mod(name: str) -> types.ModuleType:
    """Create a stub module registered in sys.modules."""
    m: types.ModuleType = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    m.__path__ = []
    return m


def _ensure_extra_stubs() -> None:
    """Stubs that are MaxDiffusion-specific (not covered by MaxText)."""
    # google.api_core.exceptions (used by generate_ltx2.py)
    if "google.api_core" not in sys.modules:
        google_mod: types.ModuleType = sys.modules.get("google") or _mod("google")
        if "google" not in sys.modules:
            sys.modules["google"] = google_mod

        api_core: types.ModuleType = _mod("google.api_core")
        api_core_exc: types.ModuleType = _mod("google.api_core.exceptions")

        class GoogleAPIError(Exception):
            """Stub."""

        api_core_exc.GoogleAPIError = GoogleAPIError
        api_core.exceptions = api_core_exc
        google_mod.api_core = api_core
        sys.modules["google.api_core"] = api_core
        sys.modules["google.api_core.exceptions"] = api_core_exc

    _ensure_modeling_flax_pytorch_utils_stub()


def _ensure_modeling_flax_pytorch_utils_stub() -> None:
    """Avoid hard torch imports for Flax-only MaxDiffusion tracing.

    Upstream ``modeling_flax_utils`` imports
    ``maxdiffusion.models.modeling_flax_pytorch_utils`` unconditionally even
    though our example path only instantiates random Flax weights and never
    converts PyTorch checkpoints.
    """
    if importlib.util.find_spec("torch") is not None:
        return

    module_name = "maxdiffusion.models.modeling_flax_pytorch_utils"
    if module_name in sys.modules:
        return

    stub: types.ModuleType = _mod(module_name)

    def _torch_required(*args: object, **kwargs: object) -> Any:
        raise ImportError(
            "torch is required for MaxDiffusion PyTorch checkpoint conversion."
        )

    stub.convert_pytorch_state_dict_to_flax = _torch_required
    stub.rename_key = _torch_required
    stub.rename_key_and_reshape_tensor = _torch_required
    stub.torch2jax = _torch_required
    stub.validate_flax_state_dict = _torch_required
    sys.modules[module_name] = stub


def _ensure_fallback_stubs() -> None:
    """Minimal self-contained stubs when MaxText is not available."""
    # AQT ------------------------------------------------------------------
    if "aqt" not in sys.modules:
        aqt_mod: types.ModuleType = _mod("aqt")
        aqt_jax: types.ModuleType = _mod("aqt.jax")
        aqt_v2: types.ModuleType = _mod("aqt.jax.v2")
        aqt_tensor: types.ModuleType = _mod("aqt.jax.v2.aqt_tensor")
        aqt_config: types.ModuleType = _mod("aqt.jax.v2.config")
        aqt_flax_pkg: types.ModuleType = _mod("aqt.jax.v2.flax")
        aqt_flax: types.ModuleType = _mod("aqt.jax.v2.flax.aqt_flax")

        class QTensor:
            """Stub."""

        class DotGeneral:
            """Stub."""

        class LocalAqt:
            """Stub."""

        class DequantMode:
            THIS_INPUT: str = "this_input"
            OTHER_INPUT: str = "other_input"

        class CalibrationMode:
            REMAINING_AXIS: str = "remaining_axis"

        def config_v3(*a: object, **kw: object) -> DotGeneral:
            return DotGeneral()

        def config_v4(*a: object, **kw: object) -> DotGeneral:
            return DotGeneral()

        aqt_tensor.QTensor = QTensor
        aqt_tensor.partition_spec = lambda *a, **kw: None

        aqt_config.DotGeneral = DotGeneral
        aqt_config.LocalAqt = LocalAqt
        aqt_config.DequantMode = DequantMode
        aqt_config.CalibrationMode = CalibrationMode
        aqt_config.dot_general_make = config_v3
        aqt_config.config_v3 = config_v3
        aqt_config.config_v4 = config_v4
        aqt_config.set_stochastic_rounding = lambda *a, **kw: None
        aqt_config.set_fwd_dequant_mode = lambda *a, **kw: None
        aqt_config.set_fwd_calibration_mode = lambda *a, **kw: None

        class QuantMode:
            TRAIN: str = "train"
            SERVE: str = "serve"
            CONVERT: str = "convert"

        class FreezerMode:
            NONE: str = "none"
            CALIBRATION: str = "calibration"
            CALIBRATION_AND_VALUE: str = "calibration_and_value"

        class AqtDotGeneral:
            """Stub."""

        class AqtEinsum:
            """Stub."""

        aqt_flax.QuantMode = QuantMode
        aqt_flax.FreezerMode = FreezerMode
        aqt_flax.AqtDotGeneral = AqtDotGeneral
        aqt_flax.AqtEinsum = AqtEinsum
        aqt_flax_pkg.aqt_flax = aqt_flax

        aqt_tiled: types.ModuleType = _mod("aqt.jax.v2.tiled_dot_general")
        aqt_calib: types.ModuleType = _mod("aqt.jax.v2.calibration")

        class AbsMaxCalibration:
            """Stub."""

        aqt_calib.AbsMaxCalibration = AbsMaxCalibration

        sys.modules.update(
            {
                "aqt": aqt_mod,
                "aqt.jax": aqt_jax,
                "aqt.jax.v2": aqt_v2,
                "aqt.jax.v2.aqt_tensor": aqt_tensor,
                "aqt.jax.v2.config": aqt_config,
                "aqt.jax.v2.flax": aqt_flax_pkg,
                "aqt.jax.v2.flax.aqt_flax": aqt_flax,
                "aqt.jax.v2.tiled_dot_general": aqt_tiled,
                "aqt.jax.v2.calibration": aqt_calib,
            }
        )
        aqt_mod.jax = aqt_jax
        aqt_jax.v2 = aqt_v2
        aqt_v2.aqt_tensor = aqt_tensor
        aqt_v2.config = aqt_config
        aqt_v2.flax = aqt_flax_pkg
        aqt_v2.tiled_dot_general = aqt_tiled
        aqt_v2.calibration = aqt_calib

    # QWIX -----------------------------------------------------------------
    if "qwix" not in sys.modules:
        qwix_mod: types.ModuleType = _mod("qwix")
        qwix_pallas: types.ModuleType = _mod("qwix.pallas")

        class QtProvider:
            """Stub."""

            def __init__(self, *a: object, **kw: object) -> None:
                pass

        class QtRule:
            """Stub."""

        qwix_mod.QtProvider = QtProvider
        qwix_mod.QtRule = QtRule
        qwix_mod.quantize_model = lambda m, *a, **kw: m
        qwix_mod.pallas = qwix_pallas
        sys.modules["qwix"] = qwix_mod
        sys.modules["qwix.pallas"] = qwix_pallas

    # TOKAMAX --------------------------------------------------------------
    if "tokamax" not in sys.modules:
        tokamax_mod: types.ModuleType = _mod("tokamax")
        tokamax_src: types.ModuleType = _mod("tokamax._src")
        tokamax_ops: types.ModuleType = _mod("tokamax._src.ops")
        tokamax_ragged: types.ModuleType = _mod("tokamax._src.ops.ragged_dot")
        tokamax_backend: types.ModuleType = _mod(
            "tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_kernel"
        )
        tokamax_exp: types.ModuleType = _mod("tokamax._src.ops.experimental")
        tokamax_tpu: types.ModuleType = _mod("tokamax._src.ops.experimental.tpu")
        tokamax_splash_pkg: types.ModuleType = _mod(
            "tokamax._src.ops.experimental.tpu.splash_attention"
        )
        tokamax_splash_kernel: types.ModuleType = _mod(
            "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel"
        )
        tokamax_splash_mask: types.ModuleType = _mod(
            "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask"
        )

        class SplashConfig:
            """Stub."""

        class _QKVLayout(dict):
            HEAD_DIM_MINOR: str = "head_dim_minor"

            def __getitem__(self, key: object) -> object:
                return key

        def _missing_splash(*a: object, **kw: object) -> None:
            raise ImportError("tokamax is required for splash attention.")

        class FullMask:
            """Stub."""

            def __init__(self, *a: object, **kw: object) -> None:
                pass

        tokamax_splash_kernel.SplashConfig = SplashConfig
        tokamax_splash_kernel.QKVLayout = _QKVLayout()
        tokamax_splash_kernel.make_splash_mha = _missing_splash
        tokamax_splash_mask.FullMask = FullMask
        tokamax_splash_mask.make_causal_mask = _missing_splash

        sys.modules.update(
            {
                "tokamax": tokamax_mod,
                "tokamax._src": tokamax_src,
                "tokamax._src.ops": tokamax_ops,
                "tokamax._src.ops.ragged_dot": tokamax_ragged,
                "tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_kernel": tokamax_backend,
                "tokamax._src.ops.experimental": tokamax_exp,
                "tokamax._src.ops.experimental.tpu": tokamax_tpu,
                "tokamax._src.ops.experimental.tpu.splash_attention": tokamax_splash_pkg,
                "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel": tokamax_splash_kernel,
                "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask": tokamax_splash_mask,
            }
        )
        tokamax_mod._src = tokamax_src
        tokamax_src.ops = tokamax_ops
        tokamax_ops.ragged_dot = tokamax_ragged
        tokamax_ragged.pallas_mosaic_tpu_kernel = tokamax_backend
        tokamax_ops.experimental = tokamax_exp
        tokamax_exp.tpu = tokamax_tpu
        tokamax_tpu.splash_attention = tokamax_splash_pkg
        tokamax_splash_pkg.splash_attention_kernel = tokamax_splash_kernel
        tokamax_splash_pkg.splash_attention_mask = tokamax_splash_mask

    # google.cloud.storage -------------------------------------------------
    if "google.cloud.storage" not in sys.modules:
        google_mod: types.ModuleType = sys.modules.get("google") or _mod("google")
        if "google" not in sys.modules:
            sys.modules["google"] = google_mod
        cloud_mod: types.ModuleType = sys.modules.get("google.cloud") or _mod(
            "google.cloud"
        )
        if "google.cloud" not in sys.modules:
            sys.modules["google.cloud"] = cloud_mod
            google_mod.cloud = cloud_mod

        storage_mod: types.ModuleType = _mod("google.cloud.storage")

        class _MissingClient:
            def __init__(self, *a: object, **kw: object) -> None:
                raise ImportError("google-cloud-storage is required.")

        storage_mod.Client = _MissingClient
        sys.modules["google.cloud.storage"] = storage_mod
        cloud_mod.storage = storage_mod


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MAXDIFFUSION_AVAILABLE: bool = False
_MAXDIFFUSION_IMPORT_ERROR: Exception | None = _MAXDIFFUSION_PATH_ERROR
FlaxUNet2DConditionModel: type | None = None


def _maybe_unbox_axis_metadata(tree: Any) -> Any:
    """Strip Flax AxisMetadata wrappers such as LogicallyPartitioned."""
    try:
        from flax.core import meta as flax_meta

        return flax_meta.unbox(tree)
    except Exception:
        return tree


class MaxDiffusionUNetWrapper:
    """Eagerly instantiatable wrapper around ``FlaxUNet2DConditionModel``.

    The wrapper is fed to ``construct_and_call`` so that the model is
    created (and random weights initialised) lazily at test time, not at
    plugin-import time.
    """

    def __init__(
        self,
        *,
        sample_size: int = 8,
        in_channels: int = 4,
        out_channels: int = 4,
        cross_attention_dim: int = 32,
        batch_size: int = 1,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self._error: Exception | None = None
        try:
            if FlaxUNet2DConditionModel is None:
                raise ImportError("FlaxUNet2DConditionModel not available")

            model: Any = FlaxUNet2DConditionModel(
                sample_size=sample_size,
                in_channels=in_channels,
                out_channels=out_channels,
                cross_attention_dim=cross_attention_dim,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                block_out_channels=(32, 64),
                layers_per_block=1,
                attention_head_dim=8,
                norm_num_groups=8,
                dtype=dtype,
                weights_dtype=dtype,
            )
            rng: jax.Array = jax.random.PRNGKey(0)

            orig_with_sharding_constraint = getattr(jax.lax, "with_sharding_constraint")
            try:
                jax.lax.with_sharding_constraint = lambda x, *args, **kwargs: x
                # MaxDiffusion wraps many params in Flax AxisMetadata
                # (e.g. LogicallyPartitioned). Our Linen export shims expect raw
                # arrays, so unwrap once after initialization.
                self._params: Any = _maybe_unbox_axis_metadata(model.init_weights(rng))
            finally:
                jax.lax.with_sharding_constraint = orig_with_sharding_constraint

            self._model: Any = model
            self._batch_size: int = batch_size
        except Exception as e:
            self._error = e
            self._model = None
            self._params = None
            self._batch_size = batch_size

    def __call__(
        self,
        sample: jax.Array,
        timesteps: jax.Array,
        encoder_hidden_states: jax.Array,
    ) -> jax.Array:
        if self._error:
            raise self._error

        # JAX requires an active Mesh context for with_sharding_constraint.
        # MaxDiffusion calls it in `_reshape_heads_to_batch_dim` etc.
        # Since we only trace to ONNX, we disable it locally.
        orig_with_sharding_constraint = getattr(jax.lax, "with_sharding_constraint")
        try:
            jax.lax.with_sharding_constraint = lambda x, *args, **kwargs: x
            out: Any = self._model.apply(
                {"params": self._params},
                sample,
                timesteps,
                encoder_hidden_states,
            )
        finally:
            jax.lax.with_sharding_constraint = orig_with_sharding_constraint

        # FlaxUNet2DConditionOutput.sample
        return out.sample


def get_maxdiffusion_unet(
    *,
    sample_size: int = 8,
    in_channels: int = 4,
    out_channels: int = 4,
    cross_attention_dim: int = 32,
    batch_size: int = 1,
    dtype: jnp.dtype = jnp.float32,
) -> MaxDiffusionUNetWrapper:
    """Factory function compatible with ``construct_and_call``."""
    return MaxDiffusionUNetWrapper(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        cross_attention_dim=cross_attention_dim,
        batch_size=batch_size,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _selected_model_names() -> set[str] | None:
    raw: str = os.environ.get(MODEL_FILTER_ENV, "").strip()
    if not raw:
        return set(DEFAULT_MODELS)
    if raw.lower() == "all":
        return None
    names: set[str] = {item.strip() for item in raw.split(",") if item.strip()}
    return {name if name.endswith(".yml") else f"{name}.yml" for name in names}


def iter_configs() -> list[Path]:
    if CONFIGS_DIR is None or not CONFIGS_DIR.exists():
        return []
    configs: list[Path] = sorted(CONFIGS_DIR.glob("*.yml"))
    allow: set[str] | None = _selected_model_names()
    if allow is None:
        return configs
    return [cfg for cfg in configs if cfg.name in allow]


# ---------------------------------------------------------------------------
# Monkey-patches for upstream MaxDiffusion issues during tracing
# ---------------------------------------------------------------------------


def _patch_maxdiffusion_bugs() -> None:
    """Monkey-patch bugs in MaxDiffusion that break JAX tracer evaluation."""
    try:
        import math

        embeddings_flax = importlib.import_module("maxdiffusion.models.embeddings_flax")

        # Original uses `jnp.shape(timesteps)[0]` which evaluates to a JitTracer
        # instead of a static integer during maxdiffusion ONNX export.
        def _get_sinusoidal_embeddings_patched(
            timesteps: jnp.ndarray,
            embedding_dim: int,
            freq_shift: float = 1,
            min_timescale: float = 1,
            max_timescale: float = 1.0e4,
            flip_sin_to_cos: bool = False,
            scale: float = 1.0,
        ) -> jnp.ndarray:
            assert timesteps.ndim == 1, "Timesteps should be a 1d-array"
            assert (
                embedding_dim % 2 == 0
            ), f"Embedding dimension {embedding_dim} should be even"
            num_timescales = float(embedding_dim // 2)
            log_timescale_increment = math.log(max_timescale / min_timescale) / (
                num_timescales - freq_shift
            )
            inv_timescales = min_timescale * jnp.exp(
                jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
            )
            emb = jnp.expand_dims(timesteps, 1) * jnp.expand_dims(inv_timescales, 0)
            scaled_time = scale * emb
            if flip_sin_to_cos:
                signal = jnp.concatenate(
                    [jnp.cos(scaled_time), jnp.sin(scaled_time)], axis=1
                )
            else:
                signal = jnp.concatenate(
                    [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1
                )

            # THE FIX: natively read the shape tuple instead of tracing `jnp.shape()`
            batch_size = (
                timesteps.shape[0] if hasattr(timesteps, "shape") else len(timesteps)
            )
            signal = jnp.reshape(signal, [int(batch_size), embedding_dim])
            return signal

        embeddings_flax.get_sinusoidal_embeddings = _get_sinusoidal_embeddings_patched
        logger.debug(
            "Patched maxdiffusion.models.embeddings_flax.get_sinusoidal_embeddings"
        )
    except Exception as e:
        logger.warning(f"Failed to patch MaxDiffusion bugs: {e}")


# ---------------------------------------------------------------------------
# Import MaxDiffusion modules (guarded)
# ---------------------------------------------------------------------------

if MAXDIFFUSION_PKG_PATH is not None and MAXDIFFUSION_PKG_PATH.exists():
    try:
        _ensure_stubs()
        _patch_maxdiffusion_bugs()
        _unet_module: types.ModuleType = importlib.import_module(
            "maxdiffusion.models.unet_2d_condition_flax"
        )
        _UNet = _unet_module.FlaxUNet2DConditionModel

        FlaxUNet2DConditionModel: type | None = _UNet
        MAXDIFFUSION_AVAILABLE: bool = True
    except Exception as exc:
        _MAXDIFFUSION_IMPORT_ERROR: Exception | None = exc
        if MAXDIFFUSION_SRC_ENV:
            logger.warning(
                "MaxDiffusion import failed (%s). "
                "JAX2ONNX_MAXDIFFUSION_SRC is set, so registering an "
                "explicit environment error example.",
                exc,
            )

            def _raise_import_error(*args: Any, **kwargs: Any) -> None:
                raise ImportError(
                    f"MaxDiffusion import failed previously: "
                    f"{_MAXDIFFUSION_IMPORT_ERROR}"
                )

            register_example(
                component="MaxDiffusion_Environment_Error",
                description="Placeholder reporting environment errors",
                context="examples.maxdiffusion",
                testcases=[
                    {
                        "testcase": "environment_check_fails",
                        "callable": _raise_import_error,
                        "input_shapes": [],
                        "run_only_f32_variant": True,
                        "skip_numeric_validation": True,
                    }
                ],
            )
        else:
            logger.info(
                "MaxDiffusion import failed during optional auto-discovery "
                "(%s); skipping examples.maxdiffusion registration.",
                exc,
            )


# ---------------------------------------------------------------------------
# Example registration
# ---------------------------------------------------------------------------

# SDXL-family configs that use a simple UNet forward pass without extra
# Wan/LTX transformer dependencies.
_SKIP_PATTERNS: list[str] = [
    "wan",
    "ltx",
    "flux",
]


def _register_examples(configs: list[Path]) -> None:
    batch_size: int = 1
    sample_size: int = 8
    in_channels: int = 4
    out_channels: int = 4
    cross_attention_dim: int = 32

    for config_path in configs:
        model_name: str = config_path.stem
        if any(p in model_name.lower() for p in _SKIP_PATTERNS):
            continue

        component_name: str = f"MaxDiffusion_{model_name.replace('-', '_')}"

        register_example(
            component=component_name,
            description=f"MaxDiffusion UNet: {model_name}",
            source="https://github.com/AI-Hypercomputer/maxdiffusion",
            since="0.12.4",
            context="examples.maxdiffusion",
            children=[],
            testcases=[
                {
                    "testcase": f"maxdiffusion_{model_name}",
                    "callable": construct_and_call(
                        get_maxdiffusion_unet,
                        sample_size=sample_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        cross_attention_dim=cross_attention_dim,
                        batch_size=batch_size,
                    ),
                    "input_shapes": [
                        (batch_size, in_channels, sample_size, sample_size),
                        (batch_size,),
                        (batch_size, 1, cross_attention_dim),
                    ],
                    "input_dtypes": [jnp.float32, jnp.int32, jnp.float32],
                    "run_only_f32_variant": True,
                },
            ],
        )
        logger.info("Registered MaxDiffusion example: %s", component_name)


if MAXDIFFUSION_AVAILABLE:
    _register_examples(iter_configs())
