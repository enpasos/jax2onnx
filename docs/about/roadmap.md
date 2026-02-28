# Roadmap

## Planned

* Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
* Expand SotA example support for vision and language models.
* Improve support for **physics-based simulations**.
* Support for **[MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)**.


## Current Version

### **jax2onnx 0.12.2**

* **Performance + CI pipeline hardening (`#196`):** Introduced a unified `scripts/run_all_checks.sh` flow (pre-commit, test generation, pytest), enabled `pytest-xdist`, reduced full-check runtime (from ~9:38 to ~3:30 in the tracked benchmark), and switched CI to run full checks with MaxText on Python 3.12.
* **MaxText integration robustness (`#196`):** Strengthened MaxText source discovery and setup behavior (`JAX2ONNX_MAXTEXT_SRC`, model selection, pinned checkout handling) to make optional MaxText registrations and test runs more deterministic.
* **Major primitive coverage expansion (`#197`, `#198`, `#199`, `#200`):** Added **141 new plugin modules** overall: `jax.lax` (+42), `jax.nn` (+12), `jax.numpy` (+68), Flax NNX (+6), and Equinox NN (+13).
* **LAX/NN converter hardening (`#197`, `#198`, `#200`):** Added broad new lowerings (linear algebra/special functions/reductions/RNG/scatter families) and fixed edge cases around `reduce`/`reduce_window` reducer lambdas, `scan` graph checks, and reduction-pattern lowering behavior.
* **Coverage transparency wave (`#197`–`#201`):** Added/generated new coverage guides and scripts for JAX LAX, JAX NumPy, Flax API, and Equinox NN; refreshed ONNX operator coverage outputs; and integrated the new pages into MkDocs navigation.
* **Flax NNX + Equinox feature expansion (`#199`, `#200`, `#201`):** Added new Flax NNX plugin coverage (`combine_masks`, `flip_sequences`, `glu`, `hard_tanh`, `linear`, `log_sigmoid`, `lora`) and substantial Equinox NN coverage (pooling, normalization, embedding, recurrent, sequential, prelu, spectral/weight norm), with corresponding docs/examples tables updates.
* **Dependency/tooling refresh (`#196`):** Updated lockfile/runtime docs to `onnx-ir 0.2.0`, added `pytest-xdist`, and aligned contributor/test automation workflows with the new full-check pipeline.


## Past Versions

See [Past Versions](past_versions.md) for the full release archive.
