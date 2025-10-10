# AGENTS.md

This page is the quick briefing for automated agents (or humans) dropping into the repo. It highlights what matters immediately and points to the longer-form docs when you need depth.

---

## Project Snapshot

- **Pipeline:** Everything runs through the IR-only converter. `converter/` builds `onnx_ir` graphs; `plugins/` land primitive-specific lowering; `tests/` cover both policy and regression cases.
- **Docs map:** Start with `docs/design.md` for the architecture overview. Technical playbooks live under `docs/dev_guides/` (builder etiquette, expect_graph usage, control-flow wiring, IR optimizer, etc.). Release assets, coverage tables, and images are under `docs/readme/`.
- **Tooling:** Python 3.11+, Poetry, Ruff (lint + format), mypy, pytest. Supported runtime stack is **JAX ≥0.7.2** and **Flax/NNX ≥0.12.0**.
- **Recent heads-up (2025-10-02):**
  - NNX examples construct RNGs via `with_rng_seed(...)`; avoid inline lambdas so callables stay hashable under JAX 0.7.
  - Attention plugins normalise masked weights to avoid float32 NaNs; tests rely on that behaviour.

---

## Where to Read Next

- Architecture: `docs/design.md`
- IR builder rules: `docs/dev_guides/onnx_ir_builder.md`
- Structural tests (`expect_graph`): `docs/dev_guides/expect_graph_reference.md`
- Control-flow body wiring: `docs/dev_guides/subgraph_input_handling.md`
- IR optimizer passes: `docs/dev_guides/ir_optimizer.md`
- Regression matrices & history: `docs/readme/coverage_tables.md`, `docs/readme/past_versions.md`

Keep these handy—most deep-dives live there instead of here.

---

## Toolchain & Compatibility

- Install: `poetry install -E all`
- Preferred checks:
  - `poetry run ruff check .`
  - `poetry run ruff format .`
  - `poetry run mypy src`
  - `poetry run pytest -q`
- Debug flags exist for the IR optimizer (`JAX2ONNX_*_DEBUG`); see the optimizer dev guide for the full matrix.
- JAX 0.7.x specifics:
  - No `jax_dynamic_shapes` auto-toggle.
  - Primitive parameters must be hashable; `construct_and_call(...).with_dtype(...)` constructs modules once per dtype.
  - Flax NNX requires attributes that hold arrays to be wrapped in `nnx.List` / `nnx.data`.

---

## Core Rules (Do Not Break)

- **IR-only** inside `converter/` and `plugins/`. Importing ONNX protobuf types here fails CI (`tests/extra_tests/framework/test_no_onnx_in_converter_plugins.py`).
- **Deterministic module construction:** never seed at import. Use `construct_and_call(...).with_requested_dtype()` and `with_rng_seed(...)` helpers so tests can rebuild deterministically for both f32/f64 variants.
- **Single-use PRNG keys:** split before handing keys to separate consumers. Turn on `jax.config.update("jax_debug_key_reuse", True)` while developing if you suspect violations.
- **Metadata parity:** tests expect `expect_graph` specs to mirror real lowering. Keep structural expectations beside metadata entries and regenerate via `scripts/emit_expect_graph.py` when behaviour changes.
- **Follow the dev guides:** IR builder, subgraph wiring, optimizer, and plugin guardrails are enforced by policy tests—read the relevant guide before editing those areas.

---

## Working Rhythm

1. Sync dependencies (`poetry install -E all`).
2. Make the change.
3. Run the focused test (e.g. `poetry run pytest -q tests/path/test_file.py::TestClass::test_case`).
4. Run the full suite before shipping.
5. Apply formatting/lints/mypy; they’re fast.

> ✅ Rule of thumb: if tests aren’t green locally, you’re not done.

---

## Policy & Checklist

- Respect repo conventions (see quick bullet list above); when in doubt, consult the dev guide for that subsystem.
- Don’t ship public API changes without updating docs, tests, and changelog.
- Prefer deleting obsolete assets in small, focused diffs.
- Before requesting review:
  - [ ] `poetry run pytest -q`
  - [ ] Added/updated tests
  - [ ] `poetry run ruff check .` & `ruff format .`
  - [ ] `poetry run mypy src`
  - [ ] Docs/notes updated if behaviour changed

---

## Handy Commands

- Focused pytest: `poetry run pytest -q tests/path/test_file.py::TestClass::test_case`
- Format touched files: `git diff --name-only | xargs poetry run ruff format`
- Emit expect_graph snippets: `poetry run python scripts/emit_expect_graph.py <testcase>`

---

## When You’re Stuck

- Capture minimal reproductions under `tests/extra_tests/framework/`.
- Enable the relevant `JAX2ONNX_*_DEBUG` env while running the failing test.
- Verify mutations persist across graph mirrors (`graph.nodes`, `_nodes`, etc.) if you’re editing IR.
- Revisit the dev guide for the subsystem you’re touching—chances are the guardrail is documented there.

---

## Glossary

- **IR:** The `onnx_ir` intermediate representation used during conversion.
- **Missing input:** Empty-name (`""`) placeholder meaning “optional input not provided” in ONNX.
- **Live list:** Container that reflects the graph state directly; mutating it changes the graph.

Welcome aboard, and thanks for keeping conversions robust across `onnx_ir` variants. When in doubt, favour clarity, tests, and abiding by the dev guides.
