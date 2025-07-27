# file: jax2onnx/plugins/examples/nnx/cnn2.py

import jax
import jax.numpy as jnp
from flax import nnx
from jax2onnx.plugin_system import register_example
from einops import reduce


class ConvBlock(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        strides: tuple[int, int] = (1, 1),
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


class CNN2(nnx.Module):
    def __init__(self, num_classes: int, *, rngs: nnx.Rngs):
        self.contraction_factor = 0.5
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
        return self.contraction_factor * uv

    def map_latent_to_inference(self, u: jax.Array) -> jax.Array:
        u_reduced = reduce(u, "b h w c -> b c", "mean")
        return self.classifier(u_reduced)

    def __call__(
        self, d: jax.Array, *, eps: float = 1e-3, max_depth: int = 100
    ) -> tuple[jax.Array, jax.Array]:
        Qd = self.data_space_forward(d)

        def cond_fun(val):
            u, u_prev, depth = val
            res_norm = jnp.max(jnp.linalg.norm(u - u_prev, axis=1))
            return (res_norm > eps) & (depth < max_depth)

        def body_fun(val):
            u, _, depth = val
            u_prev = u
            u_new = self.latent_space_forward(u, Qd)
            return u_new, u_prev, depth + 1

        u_init = jnp.zeros_like(Qd)
        u_prev_init = jnp.full_like(Qd, jnp.inf)
        init_val = (u_init, u_prev_init, 0)
        u_star, _, depth = jax.lax.while_loop(cond_fun, body_fun, init_val)
        Ru = self.latent_space_forward(u_star, Qd)
        y = self.map_latent_to_inference(Ru)
        return y, depth


register_example(
    component="CNN2",
    description="A CNN with a while_loop.",
    since="v0.6.5",
    context="examples.nnx",
    testcases=[
        # TODO: enable testcases
        # {
        #     "testcase": "simple_cnn_inference",
        #     "callable": CNN2(num_classes=10, rngs=nnx.Rngs(0)),
        #     "input_shapes": [("B", 28, 28, 1)],
        #     "run_only_f32_variant": True,
        # },
    ],
)
