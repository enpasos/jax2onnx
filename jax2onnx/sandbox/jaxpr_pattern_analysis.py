import jax
import jax.numpy as jnp

"""
This file contains reconstructed, self-contained JAX functions based on an
analysis of a large JAXPR. Each function represents a distinct computational
pattern identified within the original computation graph. This aids in
understanding, debugging, and verifying the high-level logic of the model.
"""


def weno_reconstruction_and_smoothness(
    u_im2: jax.Array,
    u_im1: jax.Array,
    u_i: jax.Array,
    u_ip1: jax.Array,
    u_ip2: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Reconstructs the WENO (Weighted Essentially Non-Oszillatory) smoothness
    indicators and the final high-order reconstructed values at the cell interface.

    This function corresponds to the jaxpr section that calculates intermediate
    values for a high-order numerical scheme, from variables `bq` through `ft`.

    Args:
        u_im2: State variable at stencil index i-2 (e.g., jaxpr var `bq`)
        u_im1: State variable at stencil index i-1 (e.g., jaxpr var `br`)
        u_i:   State variable at stencil index i   (e.g., jaxpr var `bs`)
        u_ip1: State variable at stencil index i+1 (e.g., jaxpr var `bt`)
        u_ip2: State variable at stencil index i+2 (e.g., jaxpr var `bu`)

    Returns:
        A tuple containing:
        - alpha0: WENO weights for the first substencil (e.g., jaxpr var `eo`).
        - alpha1: WENO weights for the second substencil (e.g., jaxpr var `ep`).
        - alpha2: WENO weights for the third substencil (e.g., jaxpr var `eq`).
        - final_flux: The high-order reconstructed flux (e.g., jaxpr var `ft`).
    """
    # --- Smoothness Indicators (beta factors) ---
    # These correspond to the jaxpr variables `cl`, `cx`, and `do`.
    # They measure the smoothness of the solution on three different sub-stencils.

    # beta_0, corresponds to jaxpr `cl`
    beta_0 = (13.0 / 12.0) * (u_im2 - 2 * u_im1 + u_i) ** 2 + 0.25 * (
        u_im2 - 4 * u_im1 + 3 * u_i
    ) ** 2

    # beta_1, corresponds to jaxpr `cx`
    beta_1 = (13.0 / 12.0) * (u_im1 - 2 * u_i + u_ip1) ** 2 + 0.25 * (
        u_im1 - u_ip1
    ) ** 2

    # beta_2, corresponds to jaxpr `do`
    beta_2 = (13.0 / 12.0) * (u_i - 2 * u_ip1 + u_ip2) ** 2 + 0.25 * (
        3 * u_i - 4 * u_ip1 + u_ip2
    ) ** 2

    # --- WENO Weights (alpha values) ---
    # These correspond to jaxpr variables `eo`, `ep`, and `eq`.
    # The weights are based on the smoothness indicators. A small epsilon (1e-30)
    # is added to avoid division by zero, which is also seen in the jaxpr.
    epsilon = 1e-30

    (
        jnp.abs(beta_0 - beta_2)
    ) ** 2  # Not explicitly named in jaxpr, but part of the tau calculation
    # which is a more advanced WENO-Z scheme. For simplicity, we
    # stick to the direct alpha calculation seen in the jaxpr.

    alpha_0_num = 0.1 / (beta_0 + epsilon) ** 2
    alpha_1_num = 0.6 / (beta_1 + epsilon) ** 2
    alpha_2_num = 0.3 / (beta_2 + epsilon) ** 2

    alpha_sum = alpha_0_num + alpha_1_num + alpha_2_num

    alpha_0 = alpha_0_num / alpha_sum
    alpha_1 = alpha_1_num / alpha_sum
    alpha_2 = alpha_2_num / alpha_sum

    # --- Reconstructed Right-Side Values on Sub-stencils ---
    # These correspond to jaxpr variables `ey`, `fg`, and `fo`.

    q_r_0 = (1.0 / 3.0) * u_im2 - (7.0 / 6.0) * u_im1 + (11.0 / 6.0) * u_i
    q_r_1 = (-1.0 / 6.0) * u_im1 + (5.0 / 6.0) * u_i + (1.0 / 3.0) * u_ip1
    q_r_2 = (1.0 / 3.0) * u_i + (5.0 / 6.0) * u_ip1 - (1.0 / 6.0) * u_ip2

    # --- Final High-Order Reconstructed Flux ---
    # This corresponds to jaxpr var `ft`.
    final_flux = alpha_0 * q_r_0 + alpha_1 * q_r_1 + alpha_2 * q_r_2

    return alpha_0, alpha_1, alpha_2, final_flux


def apply_periodic_boundary_and_update(
    data: jax.Array, indices: jax.Array, updates: jax.Array, boundary_width: int
) -> jax.Array:
    """
    Applies a periodic boundary condition to indices and then performs a
    scatter update.

    This pattern corresponds to the repeated sequence of `lt`, `add`, `select_n`,
    `convert_element_type`, `broadcast_in_dim`, `gather`, and `scatter`
    seen throughout the jaxpr (e.g., vars `na` through `oc`).

    Args:
        data: The tensor to be updated.
        indices: The original indices, which might be out-of-bounds.
        updates: The values to scatter into the data tensor.
        boundary_width: The width of the boundary region used for wrapping.

    Returns:
        The updated data tensor.
    """
    # JAX handles negative indices by wrapping them around, so `idx % shape`
    # is implicitly done. The `select_n` logic seen in the jaxpr, such as:
    #   na = lt(bl, 0)
    #   nb = add(bl, 5)
    #   nc = select_n(na, bl, nb)
    # is a manual implementation of this periodic wrapping.
    # In high-level JAX, we can simply rely on the default indexing behavior
    # or use `mode='wrap'` in `at[...].set()`. For clarity, we'll write it
    # out to match the jaxpr's intent.

    # Get the size of the axis being indexed
    axis_size = data.shape[1]

    # Manually wrap indices to be positive, similar to the jaxpr's logic
    wrapped_indices = jnp.where(indices < 0, indices + boundary_width, indices)
    wrapped_indices = jnp.mod(wrapped_indices, axis_size)

    # The jaxpr shows a `scatter` operation, which is an overwrite.
    # The high-level equivalent is `.at[...].set(...)`
    # We need to ensure the indices have the correct shape for updating slices.
    # The jaxpr uses `broadcast_in_dim` to shape `[2]` indices to `[2, 1]`.
    if wrapped_indices.ndim == 1:
        wrapped_indices = jnp.expand_dims(wrapped_indices, axis=-1)

    # The `scatter` op in the jaxpr has update_window_dims=(1, 2, 3) and
    # inserted_window_dims=(0,), which means it updates slices along axis 0.
    return data.at[wrapped_indices].set(updates)


def compute_cfl_timestep(
    dt_previous: jax.Array, wave_speeds: jax.Array, cfl_number: float = 0.9
) -> jax.Array:
    """
    Calculates a new stable time step based on the CFL condition.

    This corresponds to the jaxpr section that involves `reduce_max` on
    wave speeds and division of the previous time step.
    (e.g., vars `egd` through `egm`).

    Args:
        dt_previous: The time step from the previous iteration (e.g., jaxpr var `bn`).
        wave_speeds: A tensor of wave speeds across the grid.
        cfl_number: The Courant-Friedrichs-Lewy number (a safety factor).

    Returns:
        The new, stable time step.
    """
    # Add a small epsilon to avoid division by zero, as seen in the jaxpr (`egk` + `2.22e-16`)
    max_wave_speed = jnp.max(wave_speeds) + 2.22e-16

    # The jaxpr computes `dt / max_wave_speed`.
    new_dt = dt_previous / max_wave_speed

    return new_dt * cfl_number


if __name__ == "__main__":
    # --- Demonstrate usage of the reconstructed functions ---

    # 1. WENO Reconstruction Demonstration
    print("--- 1. Demonstrating WENO-like Reconstruction ---")
    key = jax.random.PRNGKey(0)
    # Create 5 stencil slices, similar to what the jaxpr does
    # The shape matches the variables like `bq` after the initial slice.
    stencil_shape = (5, 201, 1, 1)

    u_im2, u_im1, u_i, u_ip1, u_ip2 = [
        jax.random.normal(key, shape=stencil_shape) for _ in range(5)
    ]

    alpha0, alpha1, alpha2, final_flux = weno_reconstruction_and_smoothness(
        u_im2, u_im1, u_i, u_ip1, u_ip2
    )
    print(f"Shape of final_flux: {final_flux.shape}")
    print(f"Shape of alpha0 weights: {alpha0.shape}\n")

    # 2. Boundary Condition Update Demonstration
    print("--- 2. Demonstrating Boundary Condition Update ---")
    data_tensor = jnp.zeros((5, 10, 1, 1))  # A smaller grid for demonstration
    # Indices that might point into a "ghost cell" region
    # The jaxpr has `bl = [0, -1]` which is used for this.
    indices_to_update = jnp.array([0, -1])
    # Updates to be placed at the wrapped indices
    update_values = jnp.ones((2, 10, 1, 1)) * 55.0

    updated_tensor = apply_periodic_boundary_and_update(
        data=data_tensor,
        indices=indices_to_update,
        updates=update_values,
        boundary_width=5,  # The width of the periodic domain
    )
    print("Original data tensor (slice):")
    print(data_tensor[:, :5, ...].squeeze())
    print("\nUpdated data tensor (slice):")
    # After wrapping, index -1 becomes index 4.
    print(updated_tensor[:, :5, ...].squeeze())
    print("\n")

    # 3. Time Step Calculation Demonstration
    print("--- 3. Demonstrating Time Step (CFL) Calculation ---")
    # Corresponds to jaxpr var `bn`
    dt_prev = jnp.array(0.001, dtype=jnp.float64)
    # Corresponds to a grid of computed wave speeds, e.g., jaxpr var `egh`
    speeds = jnp.array([100.0, 250.0, 300.0, 150.0])

    new_dt = compute_cfl_timestep(dt_prev, speeds, cfl_number=0.9)
    print(f"Previous dt: {dt_prev}")
    print(f"Max wave speed: {jnp.max(speeds)}")
    print(f"New stable dt: {new_dt}")
