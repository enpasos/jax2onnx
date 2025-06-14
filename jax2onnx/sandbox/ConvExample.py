from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax2onnx import to_onnx
from einops import rearrange, reduce


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        strides: Sequence[int] = (1, 1),
        rngs: nnx.Rngs,
    ) -> None:
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=strides,
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=strides,
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.leaky_relu(x, 0.1)

        x = self.conv2(x)
        x = nnx.leaky_relu(x, 0.1)

        return x


class SimpleCNN(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.contraction_factor = 0.5
        self.implicit = True
        self.deterministic = False
        self.lat_layers = 2

        self.conv_d1 = ConvBlock(1, 16, strides=(1, 1), rngs=rngs)
        self.conv_d2 = ConvBlock(16, 32, strides=(1, 1), rngs=rngs)
        self.bn1 = nnx.GroupNorm(16, 4, rngs=rngs)
        self.bn2 = nnx.GroupNorm(32, 8, rngs=rngs)

        self.fpi_blocks = [
            ConvBlock(32, 32, strides=(1, 1), rngs=rngs) for _ in range(self.lat_layers)
        ]
        self.norms = [nnx.GroupNorm(32, 8, rngs=rngs) for _ in range(self.lat_layers)]

        self.classifier = nnx.Linear(32, num_classes, rngs=rngs)

    def data_space_forward(self, d: jax.Array) -> jax.Array:
        Qd = nnx.max_pool(
            nnx.leaky_relu(self.bn1(self.conv_d1(d)), 0.1),
            window_shape=(3, 3),
            strides=(3, 3),
            padding="VALID",
        )
        Qd = nnx.leaky_relu(self.bn2(self.conv_d2(Qd)), 0.1)

        return Qd

    def latent_space_forward(self, u: jax.Array, v: jax.Array) -> jax.Array:
        uv = u + v

        for blk, norm in zip(self.fpi_blocks, self.norms):
            res = nnx.leaky_relu(blk(uv), 0.1)
            uv = norm(uv + res)

        R_uv = self.contraction_factor * uv

        return R_uv

    def map_latent_to_inference(self, u: jax.Array) -> jax.Array:
        u = reduce(u, "b h w c -> b c", "mean")
        y = self.classifier(u)

        return y

    def forward_explicit(
        self, d: jax.Array, *, key: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:  # type: ignore
        Qd = self.data_space_forward(d)
        u = jnp.zeros_like(Qd)
        Ru = self.latent_space_forward(u, Qd)
        y = self.map_latent_to_inference(Ru)
        return y, jnp.array(0), jnp.array(1.0)

    def _calculate_lip_norm_factor(self, u: jax.Array, v: jax.Array, *, key: jax.Array):
        key_u, key_v = jr.split(key)

        noise_u = jr.normal(key_u, u.shape)
        noise_v = jr.normal(key_v, v.shape)

        w = u + noise_u
        v_noisy = v + noise_v

        Rwv = self.latent_space_forward(w, v_noisy)
        Ruv = self.latent_space_forward(u, v_noisy)

        w_flat = rearrange(w, "b h w c -> b (h w c)")
        u_flat = rearrange(u, "b h w c -> b (h w c)")
        Rwv_flat = rearrange(Rwv, "b h w c -> b (h w c)")
        Ruv_flat = rearrange(Ruv, "b h w c -> b (h w c)")

        R_diff_norm = jnp.mean(jnp.linalg.norm(Rwv_flat - Ruv_flat, axis=1))
        u_diff_norm = jnp.mean(jnp.linalg.norm(w_flat - u_flat, axis=1))

        R_is_gamma_lip = R_diff_norm <= self.contraction_factor * u_diff_norm

        def calculate_factor():
            violation_ratio = (
                self.contraction_factor * u_diff_norm / (R_diff_norm + 1e-6)
            )
            return violation_ratio ** (1.0 / self.lat_layers)

        def identity_factor():
            return jnp.array(1.0)

        normalize_factor = jax.lax.cond(
            R_is_gamma_lip, identity_factor, calculate_factor
        )
        return normalize_factor

    def forward_implicit(
        self,
        d: jax.Array,
        *,
        eps: float = 1e-3,
        max_depth: int = 100,
        key: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        Qd = self.data_space_forward(d)

        def cond_fun(val):
            u, u_prev, depth = val
            u_flat = rearrange(u, "b h w c -> b (h w c)")
            u_prev_flat = rearrange(u_prev, "b h w c -> b (h w c)")
            res_norm = jnp.max(jnp.linalg.norm(u_flat - u_prev_flat, axis=1))
            return (res_norm > eps) & (depth < max_depth)

        def body_fun(val):
            u, _, depth = val
            u_prev = u
            u_new = self.latent_space_forward(u, Qd)
            return u_new, u_prev, depth + 1

        u_init = jnp.zeros_like(Qd)
        u_prev_init = jnp.full_like(Qd, jnp.inf)
        init_val = (u_init, u_prev_init, 0)

        u_star, u_prev, depth = jax.lax.stop_gradient(
            jax.lax.while_loop(cond_fun, body_fun, init_val)
        )

        def train_phase(operands):
            u_prev_op, qd_detached_op = operands
            return self._calculate_lip_norm_factor(u_prev_op, qd_detached_op, key=key)

        def eval_phase(operands):
            # In eval mode, no normalization is needed. Return neutral factor.
            return jnp.array(1.0)

        normalize_factor = jax.lax.cond(
            self.deterministic, eval_phase, train_phase, (u_prev, Qd)
        )

        Ru = self.latent_space_forward(u_star, Qd)
        y = self.map_latent_to_inference(Ru)

        return y, depth, normalize_factor

    def __call__(self, d: jax.Array, key: jax.Array):
        def implicit_fwd(d_op):
            return self.forward_implicit(d_op, key=key)

        def explicit_fwd(d_op):
            return self.forward_explicit(d_op, key=key)

        return jax.lax.cond(self.implicit, implicit_fwd, explicit_fwd, d)


# To expose a simplified interface to export the model
class SimpleCNNInference(nnx.Module):
    def __init__(self, model):
        self.model = model

    def __call__(
        self,
        d: jax.Array,
        *,
        eps: float = 1e-3,
        max_depth: int = 100,
    ) -> Tuple[jax.Array, jax.Array]:
        Qd = self.model.data_space_forward(d)

        def cond_fun(val):
            u, u_prev, depth = val
            u_flat = rearrange(u, "b h w c -> b (h w c)")
            u_prev_flat = rearrange(u_prev, "b h w c -> b (h w c)")
            res_norm = jnp.max(jnp.linalg.norm(u_flat - u_prev_flat, axis=1))
            return (res_norm > eps) & (depth < max_depth)

        def body_fun(val):
            u, _, depth = val
            u_prev = u
            u_new = self.model.latent_space_forward(u, Qd)
            return u_new, u_prev, depth + 1

        u_init = jnp.zeros_like(Qd)
        u_prev_init = jnp.full_like(Qd, jnp.inf)
        init_val = (u_init, u_prev_init, 0)

        u_star, u_prev, depth = jax.lax.stop_gradient(
            jax.lax.while_loop(cond_fun, body_fun, init_val)
        )

        Ru = self.model.latent_space_forward(u_star, Qd)
        y = self.model.map_latent_to_inference(Ru)

        return y, depth


key = jr.PRNGKey(0)
rngs = nnx.Rngs(0)
x = jr.normal(key, (1, 28, 28, 1))
model = SimpleCNN(10, rngs=rngs)
inference_model = SimpleCNNInference(model=model)
inference_model.eval()
onnx_model = to_onnx(inference_model, inputs=[("B", 28, 28, 1)])
