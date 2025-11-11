# tests/extra_tests/test_eqx_gpt_oss_parity.py

"""Parity tests between the torch GPT-OSS reference and the Equinox example."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax2onnx.plugins.examples.eqx.gpt_oss import (
    Transformer as EqxTransformer,
    _config_from_torch_transformer,
    _populate_eqx_from_torch,
)

pytestmark = pytest.mark.skip(
    reason=(
        "Equinox GPT-OSS parity is still being realigned with the torch "
        "reference; see tracking issue for re-enabling."
    )
)

try:
    import torch
    from gpt_oss.torch.model import ModelConfig, Transformer as TorchTransformer
except ImportError as exc:  # pragma: no cover - optional dependency guard
    pytest.skip(
        f"GPT-OSS parity tests require optional dependencies: {exc}",
        allow_module_level=True,
    )


def _make_tiny_config() -> ModelConfig:
    return ModelConfig(
        num_hidden_layers=2,
        num_experts=8,
        experts_per_token=2,
        vocab_size=64,
        hidden_size=128,
        intermediate_size=128,
        swiglu_limit=7.0,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=0,
        initial_context_length=64,
        rope_theta=10_000.0,
        rope_scaling_factor=1.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=8.0,
    )


def _randomise_parameters(
    module: torch.nn.Module, *, generator: torch.Generator
) -> None:
    for param in module.parameters():
        if param.data.dtype.is_floating_point:
            noise = torch.randn(
                param.data.shape,
                generator=generator,
                dtype=torch.float32,
            )
            param.data.copy_(noise.to(param.data.dtype))
        else:
            param.data.zero_()


def _forward_logits(
    torch_model: TorchTransformer,
    eqx_model: EqxTransformer,
    tokens: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    torch_out = torch_model(tokens).detach().to(torch.float32).cpu().numpy()
    eqx_tokens = jnp.asarray(tokens.cpu().numpy(), dtype=jnp.int32)
    eqx_out = np.asarray(eqx_model(eqx_tokens), dtype=np.float32)
    return torch_out, eqx_out


def test_eqx_matches_torch_with_float32_weights() -> None:
    generator = torch.Generator().manual_seed(0)
    config = _make_tiny_config()

    torch_model = TorchTransformer(config=config, device=torch.device("cpu")).to(
        torch.float32
    )
    _randomise_parameters(torch_model, generator=generator)
    torch_model.eval()

    config_eqx = _config_from_torch_transformer(torch_model)
    eqx_model = EqxTransformer(
        config_eqx, key=jax.random.PRNGKey(0), param_dtype=jnp.float32
    )
    eqx_model = _populate_eqx_from_torch(
        torch_model, eqx_model, param_dtype=jnp.float32
    )

    tokens = torch.randint(
        0, config.vocab_size, (config.initial_context_length // 2,), dtype=torch.int64
    )
    torch_logits, eqx_logits = _forward_logits(torch_model, eqx_model, tokens)
    np.testing.assert_allclose(torch_logits, eqx_logits, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("param_dtype", [jnp.bfloat16, jnp.float32])
def test_eqx_matches_torch_with_bfloat16_weights(param_dtype: jnp.dtype) -> None:
    generator = torch.Generator().manual_seed(1)
    config = _make_tiny_config()

    torch_model = TorchTransformer(config=config, device=torch.device("cpu"))
    _randomise_parameters(torch_model, generator=generator)
    torch_model.eval()

    config_eqx = _config_from_torch_transformer(torch_model)
    eqx_model = EqxTransformer(
        config_eqx, key=jax.random.PRNGKey(1), param_dtype=param_dtype
    )
    eqx_model = _populate_eqx_from_torch(
        torch_model, eqx_model, param_dtype=param_dtype
    )

    tokens = torch.randint(
        0, config.vocab_size, (config.initial_context_length // 2,), dtype=torch.int64
    )
    torch_logits, eqx_logits = _forward_logits(torch_model, eqx_model, tokens)
    np.testing.assert_allclose(torch_logits, eqx_logits, rtol=5e-2, atol=2.0)
