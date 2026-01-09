from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
from flax import nnx

from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)

logger = logging.getLogger(__name__)

MODEL_FILTER_ENV = "JAX2ONNX_MAXTEXT_MODELS"
DEFAULT_MODELS = (
    "llama2-7b.yml",
    "gemma-2b.yml",
)

MAXTEXT_SRC_ENV = os.environ.get("JAX2ONNX_MAXTEXT_SRC", "").strip()


def _resolve_maxtext_paths(path: Path) -> tuple[Path | None, Path | None]:
    if (path / "MaxText").is_dir():
        return path, path / "MaxText"
    src_pkg = path / "src" / "MaxText"
    if src_pkg.is_dir():
        return path / "src", src_pkg
    if (path / "configs").is_dir():
        return path.parent, path
    return None, None


MAXTEXT_SRC_PATH: Path | None = None
MAXTEXT_PKG_PATH: Path | None = None
MAXTEXT_CONFIG_DIR: Path | None = None
_MAXTEXT_PATH_ERROR: Exception | None = None

if MAXTEXT_SRC_ENV:
    env_path = Path(MAXTEXT_SRC_ENV).expanduser().resolve()
    MAXTEXT_SRC_PATH, MAXTEXT_PKG_PATH = _resolve_maxtext_paths(env_path)
    if MAXTEXT_PKG_PATH is None:
        _MAXTEXT_PATH_ERROR = FileNotFoundError(
            f"JAX2ONNX_MAXTEXT_SRC does not point to a MaxText package: {env_path}"
        )
else:
    spec = importlib.util.find_spec("MaxText")
    if spec is not None:
        search_locations = list(spec.submodule_search_locations or [])
        if search_locations:
            MAXTEXT_PKG_PATH = Path(search_locations[0]).resolve()
        elif spec.origin:
            MAXTEXT_PKG_PATH = Path(spec.origin).resolve().parent

if MAXTEXT_PKG_PATH is not None:
    MAXTEXT_CONFIG_DIR = MAXTEXT_PKG_PATH / "configs"

MODELS_DIR = MAXTEXT_CONFIG_DIR / "models" if MAXTEXT_CONFIG_DIR else None

MODEL_OVERRIDES = {
    "override_model_config": True,
    "base_emb_dim": 64,
    "base_num_query_heads": 4,
    "base_num_kv_heads": 4,
    "base_mlp_dim": 128,
    "base_moe_mlp_dim": 128,
    "base_num_decoder_layers": 2,
    "head_dim": 16,
    "first_num_dense_layers": 0,
    "num_experts": 1,
    "num_experts_per_tok": 1,
    "vocab_size": 256,
    "enable_dropout": False,
    "scan_layers": False,
    "dtype": "float32",
    "weight_dtype": "float32",
    "grad_dtype": "float32",
    "log_config": False,
}

MODEL_MODE_TRAIN = "train"
MAXTEXT_AVAILABLE = False
_MAXTEXT_IMPORT_ERROR: Exception | None = _MAXTEXT_PATH_ERROR
pyconfig = None
model_creation_utils = None

if MAXTEXT_PKG_PATH is not None and MAXTEXT_PKG_PATH.exists():
    def _ensure_google_cloud_storage_stub() -> None:
        try:
            import google.cloud.storage  # noqa: F401
            return
        except Exception:
            pass

        google_mod = sys.modules.get("google")
        if google_mod is None:
            google_mod = types.ModuleType("google")
            google_mod.__path__ = []
            google_mod.__spec__ = importlib.machinery.ModuleSpec(
                "google", loader=None
            )
            sys.modules["google"] = google_mod

        cloud_mod = sys.modules.get("google.cloud")
        if cloud_mod is None:
            cloud_mod = types.ModuleType("google.cloud")
            cloud_mod.__path__ = []
            cloud_mod.__spec__ = importlib.machinery.ModuleSpec(
                "google.cloud", loader=None
            )
            sys.modules["google.cloud"] = cloud_mod

        storage_mod = types.ModuleType("google.cloud.storage")
        storage_mod.__spec__ = importlib.machinery.ModuleSpec(
            "google.cloud.storage", loader=None
        )

        class _MissingStorageClient:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError(
                    "google-cloud-storage is required for MaxText GCS features."
                )

        storage_mod.Client = _MissingStorageClient
        sys.modules["google.cloud.storage"] = storage_mod
        cloud_mod.storage = storage_mod
        google_mod.cloud = cloud_mod

    def _ensure_grain_stub() -> None:
        try:
            import grain  # noqa: F401
            return
        except Exception:
            pass

        grain_mod = types.ModuleType("grain")
        grain_mod.__path__ = []
        grain_mod.__spec__ = importlib.machinery.ModuleSpec("grain", loader=None)

        class DatasetIterator:  # noqa: D401 - stub only
            """Stub iterator for MaxText optional dependency."""

        grain_mod.DatasetIterator = DatasetIterator

        grain_python_mod = types.ModuleType("grain.python")
        grain_python_mod.__spec__ = importlib.machinery.ModuleSpec(
            "grain.python", loader=None
        )

        class PyGrainCheckpointHandler:  # noqa: D401 - stub only
            """Stub checkpoint handler for MaxText optional dependency."""

        grain_python_mod.PyGrainCheckpointHandler = PyGrainCheckpointHandler

        grain_experimental_mod = types.ModuleType("grain.experimental")
        grain_experimental_mod.__spec__ = importlib.machinery.ModuleSpec(
            "grain.experimental", loader=None
        )

        def pick_performance_config(*args: object, **kwargs: object) -> None:
            return None

        grain_experimental_mod.pick_performance_config = pick_performance_config

        sys.modules["grain"] = grain_mod
        sys.modules["grain.python"] = grain_python_mod
        sys.modules["grain.experimental"] = grain_experimental_mod
        grain_mod.python = grain_python_mod
        grain_mod.experimental = grain_experimental_mod

    def _ensure_datasets_stub() -> None:
        try:
            import datasets  # noqa: F401
            return
        except Exception:
            pass

        datasets_mod = types.ModuleType("datasets")
        datasets_mod.__path__ = []
        datasets_mod.__spec__ = importlib.machinery.ModuleSpec(
            "datasets", loader=None
        )

        def _unavailable(*args: object, **kwargs: object) -> None:
            raise ImportError("datasets is required for MaxText dataset pipelines.")

        datasets_mod.load_dataset = _unavailable

        datasets_distributed_mod = types.ModuleType("datasets.distributed")
        datasets_distributed_mod.__spec__ = importlib.machinery.ModuleSpec(
            "datasets.distributed", loader=None
        )

        def split_dataset_by_node(dataset, *args: object, **kwargs: object):
            return dataset

        datasets_distributed_mod.split_dataset_by_node = split_dataset_by_node

        sys.modules["datasets"] = datasets_mod
        sys.modules["datasets.distributed"] = datasets_distributed_mod
        datasets_mod.distributed = datasets_distributed_mod

    def _ensure_tensorflow_text_stub() -> None:
        try:
            import tensorflow_text  # noqa: F401
            return
        except Exception:
            pass

        tf_text_mod = types.ModuleType("tensorflow_text")
        tf_text_mod.__spec__ = importlib.machinery.ModuleSpec(
            "tensorflow_text", loader=None
        )

        class SentencepieceTokenizer:  # noqa: D401 - stub only
            """Stub tokenizer for missing tensorflow_text dependency."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError(
                    "tensorflow-text is required for MaxText tokenizers."
                )

        tf_text_mod.SentencepieceTokenizer = SentencepieceTokenizer
        sys.modules["tensorflow_text"] = tf_text_mod

    def _ensure_tiktoken_stub() -> None:
        try:
            import tiktoken  # noqa: F401
            return
        except Exception:
            pass

        tiktoken_mod = types.ModuleType("tiktoken")
        tiktoken_mod.__path__ = []
        tiktoken_mod.__spec__ = importlib.machinery.ModuleSpec(
            "tiktoken", loader=None
        )

        class Encoding:  # noqa: D401 - stub only
            """Stub encoding for missing tiktoken dependency."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError("tiktoken is required for MaxText tokenizers.")

        tiktoken_mod.Encoding = Encoding

        tiktoken_load_mod = types.ModuleType("tiktoken.load")
        tiktoken_load_mod.__spec__ = importlib.machinery.ModuleSpec(
            "tiktoken.load", loader=None
        )

        def load_tiktoken_bpe(*args: object, **kwargs: object) -> None:
            raise ImportError("tiktoken is required for MaxText tokenizers.")

        tiktoken_load_mod.load_tiktoken_bpe = load_tiktoken_bpe

        sys.modules["tiktoken"] = tiktoken_mod
        sys.modules["tiktoken.load"] = tiktoken_load_mod
        tiktoken_mod.load = tiktoken_load_mod

    def _ensure_input_pipeline_stub() -> None:
        if "MaxText.input_pipeline.input_pipeline_interface" in sys.modules:
            return

        input_pipeline_mod = types.ModuleType("MaxText.input_pipeline")
        input_pipeline_mod.__path__ = []
        input_pipeline_mod.__spec__ = importlib.machinery.ModuleSpec(
            "MaxText.input_pipeline", loader=None
        )

        interface_mod = types.ModuleType(
            "MaxText.input_pipeline.input_pipeline_interface"
        )
        interface_mod.__spec__ = importlib.machinery.ModuleSpec(
            "MaxText.input_pipeline.input_pipeline_interface", loader=None
        )

        class PlaceHolderDataIterator:  # noqa: D401 - stub only
            """Stub iterator for MaxText input pipeline."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                self._args = args
                self._kwargs = kwargs

        interface_mod.PlaceHolderDataIterator = PlaceHolderDataIterator

        sys.modules["MaxText.input_pipeline"] = input_pipeline_mod
        sys.modules[
            "MaxText.input_pipeline.input_pipeline_interface"
        ] = interface_mod
        input_pipeline_mod.input_pipeline_interface = interface_mod

    def _ensure_qwix_stub() -> None:
        try:
            import qwix  # noqa: F401
            return
        except Exception:
            pass

        qwix_mod = types.ModuleType("qwix")
        qwix_mod.__spec__ = importlib.machinery.ModuleSpec("qwix", loader=None)
        qwix_mod.__path__ = []

        class QtProvider:  # noqa: D401 - stub only
            """Stub Qwix provider for MaxText quantization paths."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                self._rules = args[0] if args else None

            def _get_current_rule_and_op_id(self, *args: object, **kwargs: object):
                return None, "op"

            @staticmethod
            def process_model_inputs(*args: object, **kwargs: object) -> None:
                return None

        class QtRule:  # noqa: D401 - stub only
            """Stub Qwix rule."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                self.args = args
                self.kwargs = kwargs

        def quantize_model(model, *args: object, **kwargs: object):
            return model

        qwix_pallas_mod = types.ModuleType("qwix.pallas")
        qwix_pallas_mod.__spec__ = importlib.machinery.ModuleSpec(
            "qwix.pallas", loader=None
        )

        class QArray:  # noqa: D401 - stub only
            """Stub quantized array."""

        def get_current_rule(*args: object, **kwargs: object):
            return None

        def quantize(x, *args: object, **kwargs: object):
            return x

        def _missing(*args: object, **kwargs: object) -> None:
            raise ImportError("qwix.pallas is required for quantized kernels.")

        def pallas_call(*args: object, **kwargs: object):
            def _call(*c_args: object, **c_kwargs: object) -> None:
                return _missing()

            return _call

        qwix_pallas_mod.QArray = QArray
        qwix_pallas_mod.get_current_rule = get_current_rule
        qwix_pallas_mod.quantize = quantize
        qwix_pallas_mod.dot_general = _missing
        qwix_pallas_mod.dot = _missing
        qwix_pallas_mod.pallas_call = pallas_call

        qwix_mod.QtProvider = QtProvider
        qwix_mod.QtRule = QtRule
        qwix_mod.quantize_model = quantize_model
        qwix_mod.pallas = qwix_pallas_mod
        sys.modules["qwix"] = qwix_mod
        sys.modules["qwix.pallas"] = qwix_pallas_mod

    def _ensure_aqt_stub() -> None:
        try:
            import aqt  # noqa: F401
            return
        except Exception:
            pass

        def _mod(name: str) -> types.ModuleType:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            return module

        aqt_mod = _mod("aqt")
        aqt_mod.__path__ = []
        aqt_jax_mod = _mod("aqt.jax")
        aqt_jax_mod.__path__ = []
        aqt_jax_v2_mod = _mod("aqt.jax.v2")
        aqt_jax_v2_mod.__path__ = []

        aqt_tensor_mod = _mod("aqt.jax.v2.aqt_tensor")

        class QTensor:  # noqa: D401 - stub only
            """Stub quantized tensor."""

        def partition_spec(*args: object, **kwargs: object):
            return None

        aqt_tensor_mod.QTensor = QTensor
        aqt_tensor_mod.partition_spec = partition_spec

        aqt_config_mod = _mod("aqt.jax.v2.config")

        class DotGeneral:  # noqa: D401 - stub only
            """Stub dot_general config."""

        class LocalAqt:  # noqa: D401 - stub only
            """Stub LocalAqt config."""

        class DequantMode:
            THIS_INPUT = "this_input"
            OTHER_INPUT = "other_input"

        class CalibrationMode:
            REMAINING_AXIS = "remaining_axis"

        def dot_general_make(*args: object, **kwargs: object) -> DotGeneral:
            return DotGeneral()

        def config_v3(*args: object, **kwargs: object) -> DotGeneral:
            return DotGeneral()

        def config_v4(*args: object, **kwargs: object) -> DotGeneral:
            return DotGeneral()

        def set_stochastic_rounding(*args: object, **kwargs: object) -> None:
            return None

        def set_fwd_dequant_mode(*args: object, **kwargs: object) -> None:
            return None

        def set_fwd_calibration_mode(*args: object, **kwargs: object) -> None:
            return None

        aqt_config_mod.DotGeneral = DotGeneral
        aqt_config_mod.LocalAqt = LocalAqt
        aqt_config_mod.DequantMode = DequantMode
        aqt_config_mod.CalibrationMode = CalibrationMode
        aqt_config_mod.dot_general_make = dot_general_make
        aqt_config_mod.config_v3 = config_v3
        aqt_config_mod.config_v4 = config_v4
        aqt_config_mod.set_stochastic_rounding = set_stochastic_rounding
        aqt_config_mod.set_fwd_dequant_mode = set_fwd_dequant_mode
        aqt_config_mod.set_fwd_calibration_mode = set_fwd_calibration_mode

        aqt_flax_pkg = _mod("aqt.jax.v2.flax")
        aqt_flax_pkg.__path__ = []
        aqt_flax_mod = _mod("aqt.jax.v2.flax.aqt_flax")

        class QuantMode:
            TRAIN = "train"
            SERVE = "serve"
            CONVERT = "convert"

        class FreezerMode:
            NONE = "none"
            CALIBRATION_AND_VALUE = "calibration_and_value"

        class AqtDotGeneral:  # noqa: D401 - stub only
            """Stub AqtDotGeneral."""

        class AqtEinsum:  # noqa: D401 - stub only
            """Stub AqtEinsum."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError("aqt is required for quantized MaxText runs.")

        aqt_flax_mod.QuantMode = QuantMode
        aqt_flax_mod.FreezerMode = FreezerMode
        aqt_flax_mod.AqtDotGeneral = AqtDotGeneral
        aqt_flax_mod.AqtEinsum = AqtEinsum
        aqt_flax_pkg.aqt_flax = aqt_flax_mod

        tiled_dot_general_mod = _mod("aqt.jax.v2.tiled_dot_general")

        class AxisTiling:
            def __init__(self, axis=None, tile_size=None, tile_count=None):
                self.axis = axis
                self.tile_size = tile_size
                self.tile_count = tile_count

        class TensorTiling:
            def __init__(self, contraction_axes=None, remaining_axes=None):
                self.contraction_axes = contraction_axes or []
                self.remaining_axes = remaining_axes or []

        class Cfg:
            def __init__(self, lhs=None, rhs=None):
                self.lhs = lhs
                self.rhs = rhs

        tiled_dot_general_mod.AxisTiling = AxisTiling
        tiled_dot_general_mod.TensorTiling = TensorTiling
        tiled_dot_general_mod.Cfg = Cfg

        calibration_mod = _mod("aqt.jax.v2.calibration")

        class AbsMaxCalibration:  # noqa: D401 - stub only
            """Stub calibration."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

        calibration_mod.AbsMaxCalibration = AbsMaxCalibration

        sys.modules["aqt"] = aqt_mod
        sys.modules["aqt.jax"] = aqt_jax_mod
        sys.modules["aqt.jax.v2"] = aqt_jax_v2_mod
        sys.modules["aqt.jax.v2.aqt_tensor"] = aqt_tensor_mod
        sys.modules["aqt.jax.v2.config"] = aqt_config_mod
        sys.modules["aqt.jax.v2.flax"] = aqt_flax_pkg
        sys.modules["aqt.jax.v2.flax.aqt_flax"] = aqt_flax_mod
        sys.modules["aqt.jax.v2.tiled_dot_general"] = tiled_dot_general_mod
        sys.modules["aqt.jax.v2.calibration"] = calibration_mod

        aqt_mod.jax = aqt_jax_mod
        aqt_jax_mod.v2 = aqt_jax_v2_mod
        aqt_jax_v2_mod.aqt_tensor = aqt_tensor_mod
        aqt_jax_v2_mod.config = aqt_config_mod
        aqt_jax_v2_mod.flax = aqt_flax_pkg
        aqt_jax_v2_mod.tiled_dot_general = tiled_dot_general_mod
        aqt_jax_v2_mod.calibration = calibration_mod

    def _ensure_tokamax_stub() -> None:
        try:
            import tokamax  # noqa: F401
            return
        except Exception:
            pass

        def _mod(name: str) -> types.ModuleType:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            return module

        tokamax_mod = _mod("tokamax")
        tokamax_mod.__path__ = []

        def ragged_dot(*args: object, **kwargs: object) -> None:
            raise ImportError("tokamax is required for MaxText ragged dot kernels.")

        tokamax_mod.ragged_dot = ragged_dot

        tokamax_src_mod = _mod("tokamax._src")
        tokamax_src_mod.__path__ = []
        tokamax_ops_mod = _mod("tokamax._src.ops")
        tokamax_ops_mod.__path__ = []
        tokamax_ragged_mod = _mod("tokamax._src.ops.ragged_dot")
        tokamax_ragged_mod.__path__ = []
        tokamax_backend_mod = _mod(
            "tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_kernel"
        )

        def _missing_backend(*args: object, **kwargs: object) -> None:
            raise ImportError("tokamax backend is required for this kernel.")

        tokamax_backend_mod.gmm = _missing_backend
        tokamax_backend_mod.tgmm = _missing_backend

        tokamax_exp_mod = _mod("tokamax._src.ops.experimental")
        tokamax_exp_mod.__path__ = []
        tokamax_exp_tpu_mod = _mod("tokamax._src.ops.experimental.tpu")
        tokamax_exp_tpu_mod.__path__ = []
        tokamax_splash_pkg = _mod("tokamax._src.ops.experimental.tpu.splash_attention")
        tokamax_splash_pkg.__path__ = []

        tokamax_splash_kernel = _mod(
            "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel"
        )
        tokamax_splash_mask = _mod(
            "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask"
        )

        class SplashConfig:  # noqa: D401 - stub only
            """Stub splash config."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                raise ImportError("tokamax is required for splash attention.")

        class _QKVLayout(dict):
            def __getitem__(self, key: object) -> object:
                return key

        def make_splash_mha(*args: object, **kwargs: object) -> None:
            raise ImportError("tokamax is required for splash attention.")

        tokamax_splash_kernel.SplashConfig = SplashConfig
        tokamax_splash_kernel.QKVLayout = _QKVLayout()
        tokamax_splash_kernel.make_splash_mha = make_splash_mha
        tokamax_splash_mask.make_causal_mask = make_splash_mha

        sys.modules["tokamax"] = tokamax_mod
        sys.modules["tokamax._src"] = tokamax_src_mod
        sys.modules["tokamax._src.ops"] = tokamax_ops_mod
        sys.modules["tokamax._src.ops.ragged_dot"] = tokamax_ragged_mod
        sys.modules[
            "tokamax._src.ops.ragged_dot.pallas_mosaic_tpu_kernel"
        ] = tokamax_backend_mod
        sys.modules["tokamax._src.ops.experimental"] = tokamax_exp_mod
        sys.modules["tokamax._src.ops.experimental.tpu"] = tokamax_exp_tpu_mod
        sys.modules[
            "tokamax._src.ops.experimental.tpu.splash_attention"
        ] = tokamax_splash_pkg
        sys.modules[
            "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_kernel"
        ] = tokamax_splash_kernel
        sys.modules[
            "tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_mask"
        ] = tokamax_splash_mask

        tokamax_mod._src = tokamax_src_mod
        tokamax_src_mod.ops = tokamax_ops_mod
        tokamax_ops_mod.ragged_dot = tokamax_ragged_mod
        tokamax_ragged_mod.pallas_mosaic_tpu_kernel = tokamax_backend_mod
        tokamax_ops_mod.experimental = tokamax_exp_mod
        tokamax_exp_mod.tpu = tokamax_exp_tpu_mod
        tokamax_exp_tpu_mod.splash_attention = tokamax_splash_pkg
        tokamax_splash_pkg.splash_attention_kernel = tokamax_splash_kernel
        tokamax_splash_pkg.splash_attention_mask = tokamax_splash_mask

    if MAXTEXT_SRC_PATH is not None and str(MAXTEXT_SRC_PATH) not in sys.path:
        sys.path.append(str(MAXTEXT_SRC_PATH))
    try:
        _ensure_google_cloud_storage_stub()
        _ensure_grain_stub()
        _ensure_datasets_stub()
        _ensure_tensorflow_text_stub()
        _ensure_tiktoken_stub()
        _ensure_input_pipeline_stub()
        _ensure_qwix_stub()
        _ensure_aqt_stub()
        _ensure_tokamax_stub()
        from MaxText import pyconfig as _pyconfig
        from MaxText import model_creation_utils as _model_creation_utils
        from MaxText.common_types import MODEL_MODE_TRAIN as _MODEL_MODE_TRAIN
        from MaxText.layers import embeddings as _embeddings

        pyconfig = _pyconfig
        model_creation_utils = _model_creation_utils
        MODEL_MODE_TRAIN = _MODEL_MODE_TRAIN
        if not getattr(_embeddings, "_JAX2ONNX_ROTARY_PATCHED", False):
            _embeddings._JAX2ONNX_ROTARY_PATCHED = True

            def _concrete_dim(value: object, name: str) -> int:
                return int(
                    jax.core.concrete_or_error(
                        int, value, f"{name} must be static for export"
                    )
                )

            def _rotary_timescale(self) -> jax.Array:
                embedding_dims = _concrete_dim(self.embedding_dims, "embedding_dims")
                half_embedding_dim = embedding_dims // 2
                iota = jax.lax.iota(jnp.int32, half_embedding_dim)
                fraction = 2 * iota / embedding_dims
                timescale = self.min_timescale * (
                    self.max_timescale / self.min_timescale
                ) ** fraction
                if self.rope_linear_scaling_factor != 1.0:
                    timescale = timescale * self.rope_linear_scaling_factor
                return timescale

            def _llama_timescale(self) -> jax.Array:
                embedding_dims = _concrete_dim(self.embedding_dims, "embedding_dims")
                half_embedding_dim = embedding_dims // 2
                iota = jax.lax.iota(jnp.int32, half_embedding_dim)
                fraction = 2 * iota / embedding_dims
                fraction = jnp.repeat(fraction, 2)
                timescale = self.min_timescale * (
                    self.max_timescale / self.min_timescale
                ) ** fraction
                if self.use_scale:
                    timescale = 1.0 / jax.vmap(self._apply_scaling_factor)(
                        1.0 / timescale
                    )
                return timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

            _embeddings.RotaryEmbedding.timescale = property(_rotary_timescale)
            _embeddings.LLaMARotaryEmbedding.timescale = property(_llama_timescale)
        MAXTEXT_AVAILABLE = True
    except Exception as exc:
        _MAXTEXT_IMPORT_ERROR = exc
        logger.warning(
            "MaxText import failed (%s). MaxText examples will be skipped.",
            exc,
        )


def _format_override(key: str, value: object) -> str:
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    return f"{key}={value}"


def _selected_model_names() -> set[str] | None:
    raw = os.environ.get(MODEL_FILTER_ENV, "").strip()
    if not raw:
        return set(DEFAULT_MODELS)
    if raw.lower() == "all":
        return None
    names = {item.strip() for item in raw.split(",") if item.strip()}
    return {name if name.endswith(".yml") else f"{name}.yml" for name in names}


def iter_model_configs() -> list[Path]:
    if MODELS_DIR is None or not MODELS_DIR.exists():
        return []
    configs = sorted(MODELS_DIR.glob("*.yml"))
    allow = _selected_model_names()
    if allow is None:
        return configs
    return [cfg for cfg in configs if cfg.name in allow]


def list_maxtext_components() -> list[str]:
    return [
        f"MaxText_{cfg.stem.replace('-', '_')}" for cfg in iter_model_configs()
    ]


def _strip_nnx_rngs(obj: object, seen: set[int] | None = None) -> None:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if hasattr(obj, "to_nnx__rngs"):
        try:
            setattr(obj, "to_nnx__rngs", None)
        except Exception:
            pass

    if isinstance(obj, nnx.Rngs):
        return
    if isinstance(obj, nnx.Variable):
        return
    if isinstance(obj, (jax.Array, jnp.ndarray)):
        return

    if isinstance(obj, nnx.Module):
        try:
            values = vars(obj).values()
        except TypeError:
            values = ()
        for val in values:
            _strip_nnx_rngs(val, seen)
    elif isinstance(obj, dict):
        for val in obj.values():
            _strip_nnx_rngs(val, seen)
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            _strip_nnx_rngs(val, seen)


def _patch_random_normal_to_zero() -> callable:
    import jax._src.random as _random  # noqa: PLC0415

    def _zero_normal(  # noqa: ARG001
        key,
        shape=(),
        dtype=jnp.float32,
        *,
        out_sharding=None,
        **kwargs,
    ):
        return jnp.zeros(shape, dtype=dtype)

    def _zero_normal_real(  # noqa: ARG001
        key,
        shape,
        dtype,
        *,
        out_sharding=None,
        **kwargs,
    ):
        return jnp.zeros(shape, dtype=dtype)

    def _zero_truncated_normal(  # noqa: ARG001
        key,
        lower,
        upper,
        shape=None,
        dtype=jnp.float32,
        *,
        out_sharding=None,
        **kwargs,
    ):
        if shape is None:
            shape = jnp.broadcast_shapes(jnp.shape(lower), jnp.shape(upper))
        return jnp.zeros(shape, dtype=dtype)

    orig_normal = jax.random.normal
    orig_truncated = jax.random.truncated_normal
    orig_module_normal = _random.normal
    orig_module_truncated = _random.truncated_normal
    orig_internal_normal = _random._normal
    orig_internal_normal_real = _random._normal_real
    orig_internal_truncated = _random._truncated_normal

    jax.random.normal = _zero_normal
    jax.random.truncated_normal = _zero_truncated_normal
    _random.normal = _zero_normal
    _random.truncated_normal = _zero_truncated_normal
    _random._normal = _zero_normal
    _random._normal_real = _zero_normal_real
    _random._truncated_normal = _zero_truncated_normal

    def _restore() -> None:
        jax.random.normal = orig_normal
        jax.random.truncated_normal = orig_truncated
        _random.normal = orig_module_normal
        _random.truncated_normal = orig_module_truncated
        _random._normal = orig_internal_normal
        _random._normal_real = orig_internal_normal_real
        _random._truncated_normal = orig_internal_truncated

    return _restore


class MaxTextTextOnlyWrapper(nnx.Module):
    def __init__(self, model: nnx.Module, *, seq_len: int, model_mode: str):
        self.model = model
        self.seq_len = seq_len
        self.model_mode = model_mode

    def __call__(self, input_tokens: jax.Array) -> jax.Array:
        positions = jax.lax.broadcasted_iota(
            jnp.int32,
            input_tokens.shape,
            1,
        )
        restore_random = _patch_random_normal_to_zero()
        try:
            return self.model(
                decoder_input_tokens=input_tokens,
                decoder_positions=positions,
                enable_dropout=False,
                model_mode=self.model_mode,
            )
        finally:
            restore_random()


def get_maxtext_model(
    config_path: str,
    *,
    batch_size: int = 1,
    seq_len: int = 128,
    rngs: nnx.Rngs | None = None,
) -> nnx.Module:
    """Instantiates a MaxText model suitable for ONNX export."""
    if not MAXTEXT_AVAILABLE or pyconfig is None or model_creation_utils is None:
        detail = f": {_MAXTEXT_IMPORT_ERROR}" if _MAXTEXT_IMPORT_ERROR else ""
        raise ImportError(f"MaxText not available{detail}")

    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute() and MODELS_DIR is not None:
        config_path_obj = MODELS_DIR / config_path_obj
    config_path_obj = config_path_obj.resolve()

    model_name = config_path_obj.stem
    base_config_path = config_path_obj.parent.parent / "base.yml"
    if not base_config_path.exists() and MAXTEXT_CONFIG_DIR is not None:
        base_config_path = MAXTEXT_CONFIG_DIR / "base.yml"

    argv = [
        "train.py",
        str(base_config_path),
        f"model_name={model_name}",
        "run_name=jax2onnx_export",
        "base_output_directory=/tmp/maxtext_export",
        "dataset_path=/tmp/dataset",
        f"per_device_batch_size={batch_size}",
        f"max_target_length={seq_len}",
        f"max_prefill_predict_length={seq_len}",
        "enable_checkpointing=false",
        "use_jax_splash=false",
    ]
    argv.extend(_format_override(k, v) for k, v in MODEL_OVERRIDES.items())

    config = pyconfig.initialize(argv)
    if rngs is None:
        rngs = nnx.Rngs(0)

    # Avoid tracing random normal initializers (erf_inv) during export.
    restore_random = _patch_random_normal_to_zero()
    try:
        model = model_creation_utils.from_config(
            config,
            model_mode=MODEL_MODE_TRAIN,
            rngs=rngs,
        )
    finally:
        restore_random()
    _strip_nnx_rngs(model)
    return MaxTextTextOnlyWrapper(model, seq_len=seq_len, model_mode=MODEL_MODE_TRAIN)


def _register_examples(configs: Iterable[Path]) -> None:
    batch_size = 1
    seq_len = 32

    for config_path in configs:
        model_name = config_path.stem
        component_name = f"MaxText_{model_name.replace('-', '_')}"

        register_example(
            component=component_name,
            description=f"MaxText model: {model_name}",
            source="https://github.com/AI-Hypercomputer/maxtext",
            since="v0.0.1",
            context="examples.maxtext",
            children=[],
            testcases=[
                {
                    "testcase": f"maxtext_{model_name}",
                    "callable": construct_and_call(
                        get_maxtext_model,
                        config_path=str(config_path),
                        batch_size=batch_size,
                        seq_len=seq_len,
                        rngs=with_rng_seed(42),
                    ),
                    "input_shapes": [(batch_size, seq_len)],
                    "input_dtypes": [jnp.int32],
                    "run_only_f32_variant": True,
                    "skip_numeric_validation": True,
                }
            ],
        )


if MAXTEXT_AVAILABLE:
    _register_examples(iter_model_configs())
