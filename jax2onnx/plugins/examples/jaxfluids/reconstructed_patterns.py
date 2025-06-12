import jax
import jax.numpy as jnp

from jax2onnx.plugin_system import register_example

# This file contains self-contained test cases for computational patterns
# extracted from the JAXPR of a larger jaxfluids simulation. The goal is
# to create minimal, reproducible failing tests that isolate bugs in the
# conversion of specific JAX primitives.

# ------------------------------------------------------------------------------
# PATTERN 1: WENO (Weighted Essentially Non-Oszillatory) Reconstruction
# ------------------------------------------------------------------------------
# This function represents the complex arithmetic for a high-order numerical
# scheme, identified as a potential source of numerical discrepancies.


def weno_reconstruction_f64(u_im2, u_im1, u_i, u_ip1, u_ip2):
    """
    Reconstructs the WENO smoothness indicators and final high-order values.
    This corresponds to the jaxpr section from vars `bq` through `ft`.
    """
    # --- Smoothness Indicators (beta factors) ---
    beta_0 = (13.0 / 12.0) * (u_im2 - 2 * u_im1 + u_i) ** 2 + 0.25 * (
        u_im2 - 4 * u_im1 + 3 * u_i
    ) ** 2
    beta_1 = (13.0 / 12.0) * (u_im1 - 2 * u_i + u_ip1) ** 2 + 0.25 * (
        u_im1 - u_ip1
    ) ** 2
    beta_2 = (13.0 / 12.0) * (u_i - 2 * u_ip1 + u_ip2) ** 2 + 0.25 * (
        3 * u_i - 4 * u_ip1 + u_ip2
    ) ** 2

    # --- WENO Weights (alpha values) ---
    epsilon = 1e-30
    alpha_0_num = 0.1 / (beta_0 + epsilon) ** 2
    alpha_1_num = 0.6 / (beta_1 + epsilon) ** 2
    alpha_2_num = 0.3 / (beta_2 + epsilon) ** 2
    alpha_sum = alpha_0_num + alpha_1_num + alpha_2_num
    alpha_0 = alpha_0_num / alpha_sum
    alpha_1 = alpha_1_num / alpha_sum
    alpha_2 = alpha_2_num / alpha_sum

    # --- Reconstructed Values on Sub-stencils ---
    q_r_0 = (1.0 / 3.0) * u_im2 - (7.0 / 6.0) * u_im1 + (11.0 / 6.0) * u_i
    q_r_1 = (-1.0 / 6.0) * u_im1 + (5.0 / 6.0) * u_i + (1.0 / 3.0) * u_ip1
    q_r_2 = (1.0 / 3.0) * u_i + (5.0 / 6.0) * u_ip1 - (1.0 / 6.0) * u_ip2

    # --- Final High-Order Reconstructed Value ---
    final_value = alpha_0 * q_r_0 + alpha_1 * q_r_1 + alpha_2 * q_r_2

    # Return multiple values to test PyTree output handling
    return (final_value, alpha_0, alpha_1, alpha_2)


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
                (5, 201, 1, 1),  # u_im2
                (5, 201, 1, 1),  # u_im1
                (5, 201, 1, 1),  # u_i
                (5, 201, 1, 1),  # u_ip1
                (5, 201, 1, 1),  # u_ip2
            ],
            "input_dtypes": [jnp.float64] * 5,
            "expected_output_shapes": [
                (5, 201, 1, 1),  # final_value
                (5, 201, 1, 1),  # alpha_0
                (5, 201, 1, 1),  # alpha_1
                (5, 201, 1, 1),  # alpha_2
            ],
            "expected_output_dtypes": [jnp.float64] * 4,
            "run_only_f64_variant": True,
        },
    ],
)


# ------------------------------------------------------------------------------
# PATTERN 2: Periodic Boundary Update using Scatter
# ------------------------------------------------------------------------------
# This function represents the logic for applying periodic boundary conditions,
# which involves wrapping indices and performing a scatter-set operation.


def periodic_boundary_scatter_f64(data, indices, updates):
    """
    Applies a periodic boundary condition and then uses lax.scatter to
    update the data. This directly mimics the `lt`, `add`, `select_n`, and
    `scatter` sequence from the jaxpr.
    """
    axis_size = data.shape[1]
    boundary_width = 5  # Value taken from the jaxpr's `add bl, 5` operation

    wrapped_indices = jnp.where(indices < 0, indices + boundary_width, indices)
    wrapped_indices = jnp.mod(wrapped_indices, axis_size)

    if wrapped_indices.ndim == 1:
        wrapped_indices = jnp.expand_dims(wrapped_indices, axis=-1)

    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    # Use lax.scatter directly, as this is what's in the jaxpr.
    return jax.lax.scatter(data, wrapped_indices, updates, dimension_numbers)


# register_example(
#     component="periodic_boundary_scatter",
#     description="Tests index wrapping and scatter for periodic boundary conditions.",
#     since="v0.6.5",
#     context="examples.jaxfluids",
#     testcases=[
#         {
#             "testcase": "periodic_boundary_scatter_f64",
#             "callable": periodic_boundary_scatter_f64,
#             "input_shapes": [
#                 (5, 10, 1, 1),  # data
#                 (2,),  # indices
#                 (2, 10, 1, 1),  # updates
#             ],
#             "input_dtypes": [jnp.float64, jnp.int64, jnp.float64],
#             "expected_output_shapes": [(5, 10, 1, 1)],
#             "expected_output_dtypes": [jnp.float64],
#             "run_only_f64_variant": True,

#         },
#     ],
# )

# ------------------------------------------------------------------------------
# PATTERN 3: CFL-based Timestep Calculation
# ------------------------------------------------------------------------------
# This function represents the calculation of a stable timestep based on the
# maximum wave speed in the simulation domain.


def cfl_timestep_f64(dt_previous, wave_speeds):
    """
    Calculates a new stable time step based on the CFL condition.
    Corresponds to jaxpr vars `egd` through `egm`.
    """
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
            "input_shapes": [
                (),  # dt_previous
                (200,),  # wave_speeds
            ],
            "input_dtypes": [jnp.float64, jnp.float64],
            "expected_output_shapes": [()],
            "expected_output_dtypes": [jnp.float64],
            "run_only_f64_variant": True,
        },
    ],
)
