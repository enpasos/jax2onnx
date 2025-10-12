# jax2onnx/plugins/examples/jaxfluids/reconstructed_patterns.py

from __future__ import annotations

import jax.numpy as jnp

from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins.plugin_system import register_example


# ------------------------------------------------------------------------------
# PATTERN 1: WENO (Weighted Essentially Non-Oszillatory) Reconstruction
# ------------------------------------------------------------------------------


def weno_reconstruction_f64(
    u_im2: jnp.ndarray,
    u_im1: jnp.ndarray,
    u_i: jnp.ndarray,
    u_ip1: jnp.ndarray,
    u_ip2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    beta_0 = (13.0 / 12.0) * (u_im2 - 2 * u_im1 + u_i) ** 2 + 0.25 * (
        u_im2 - 4 * u_im1 + 3 * u_i
    ) ** 2
    beta_1 = (13.0 / 12.0) * (u_im1 - 2 * u_i + u_ip1) ** 2 + 0.25 * (
        u_im1 - u_ip1
    ) ** 2
    beta_2 = (13.0 / 12.0) * (u_i - 2 * u_ip1 + u_ip2) ** 2 + 0.25 * (
        3 * u_i - 4 * u_ip1 + u_ip2
    ) ** 2

    epsilon = 1e-30
    alpha_0_num = 0.1 / (beta_0 + epsilon) ** 2
    alpha_1_num = 0.6 / (beta_1 + epsilon) ** 2
    alpha_2_num = 0.3 / (beta_2 + epsilon) ** 2
    alpha_sum = alpha_0_num + alpha_1_num + alpha_2_num

    alpha_0 = alpha_0_num / alpha_sum
    alpha_1 = alpha_1_num / alpha_sum
    alpha_2 = alpha_2_num / alpha_sum

    q_r_0 = (1.0 / 3.0) * u_im2 - (7.0 / 6.0) * u_im1 + (11.0 / 6.0) * u_i
    q_r_1 = (-1.0 / 6.0) * u_im1 + (5.0 / 6.0) * u_i + (1.0 / 3.0) * u_ip1
    q_r_2 = (1.0 / 3.0) * u_i + (5.0 / 6.0) * u_ip1 - (1.0 / 6.0) * u_ip2

    final_value = alpha_0 * q_r_0 + alpha_1 * q_r_1 + alpha_2 * q_r_2
    return final_value, alpha_0, alpha_1, alpha_2


register_example(
    component="weno_reconstruction",
    description="Tests the complex arithmetic pattern found in WENO schemes.",
    since="v0.6.5",
    context="examples.jaxfluids",
    testcases=[
        {
            "testcase": "weno_reconstruction_f64",
            "callable": weno_reconstruction_f64,
            "input_shapes": [
                (5, 201, 1, 1),
                (5, 201, 1, 1),
                (5, 201, 1, 1),
                (5, 201, 1, 1),
                (5, 201, 1, 1),
            ],
            "input_dtypes": [jnp.float64] * 5,
            "expected_output_shapes": [
                (5, 201, 1, 1),
                (5, 201, 1, 1),
                (5, 201, 1, 1),
                (5, 201, 1, 1),
            ],
            "expected_output_dtypes": [jnp.float64] * 4,
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    "Sub:5x201x1x1 -> Add:5x201x1x1 -> Pow:5x201x1x1 -> "
                    "Mul:5x201x1x1 -> Add:5x201x1x1 -> Add:5x201x1x1 -> "
                    "Pow:5x201x1x1 -> Div:5x201x1x1 -> Div:5x201x1x1",
                    "Sub:5x201x1x1 -> Add:5x201x1x1 -> Pow:5x201x1x1 -> "
                    "Mul:5x201x1x1 -> Add:5x201x1x1 -> Add:5x201x1x1 -> "
                    "Pow:5x201x1x1 -> Div:5x201x1x1 -> Div:5x201x1x1 -> "
                    "Mul:5x201x1x1 -> Add:5x201x1x1 -> Add:5x201x1x1",
                ],
                no_unused_inputs=True,
            ),
        }
    ],
)


# ------------------------------------------------------------------------------
# PATTERN 3: CFL-based Timestep Calculation
# ------------------------------------------------------------------------------


def cfl_timestep_f64(dt_previous: jnp.ndarray, wave_speeds: jnp.ndarray) -> jnp.ndarray:
    cfl_number = 0.9
    max_wave_speed = jnp.max(jnp.abs(wave_speeds)) + 2.22e-16
    new_dt = dt_previous / max_wave_speed
    return new_dt * cfl_number


register_example(
    component="cfl_timestep",
    description="Tests the CFL condition timestep calculation.",
    since="v0.6.5",
    context="examples.jaxfluids",
    testcases=[
        {
            "testcase": "cfl_timestep_f64",
            "callable": cfl_timestep_f64,
            "input_shapes": [(), (200,)],
            "input_dtypes": [jnp.float64, jnp.float64],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
            "post_check_onnx_graph": EG(
                [
                    {
                        "inputs": {1: {"const": 0.9}},
                        "path": "Div -> Mul",
                    }
                ],
                no_unused_inputs=True,
            ),
        }
    ],
)
