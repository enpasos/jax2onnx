# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
* Support for **[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)**.


## Current Version

### **jax2onnx 0.12.3**

* **Autodiff rule completeness + safety:** Added centralized AD registration/backfill utilities in `jax2onnx/plugins/jax/_autodiff_utils.py`, including explicit allowlists/blocklists for original-rule forwarding and conversion-time transpose backfill via `jax.linear_transpose` for allowlisted primitives.
* **JVP/transpose policy enforcement:** Enforced the JVP-to-transpose coverage invariant while preserving explicit/custom transpose overrides, with operational controls via `JAX2ONNX_DISABLE_AD_BACKFILL` and `JAX2ONNX_AD_DEBUG`.
* **Framework policy tests:** Added framework guard tests for AD completeness, no-override forwarding policy, forwarding correctness, and backfill idempotence.
* **Linear-transpose regression hardening (`#203`):** Added regression coverage for `linear_transpose` conversion paths (`add`, `transpose`, and allowlisted forwarding ops) to keep transpose-based traces stable.
* **Strict typing completion across JAX plugins:** Finished strict-mypy cleanup and typed registration migration across core/lax/nn/numpy/random plugin modules, including remaining `lax.eig`/`lax.eigh`/`lax.scan`/`lax.while_loop` paths and the `jnp.arange` hook fix.
* **Structural tests + docs refresh:** Updated `while_loop` structural `expect_graph` checks to match current lowering, documented AD registration helpers in the plugin-system guide, and removed obsolete coverage work-note files.
* **Dependency docs refresh:** Updated documented JAX version to `0.9.1`.

## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
