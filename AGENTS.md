# AGENTS.md

Quick briefing for work in this repository. Keep this file focused on durable
rules and working commands; put implementation details in the linked guides.
Dependency requirements live in `pyproject.toml`, resolved versions in
`poetry.lock` / `package-lock.json`, hook versions in `.pre-commit-config.yaml`,
and the test matrix in `.github/workflows/ci.yml`.

## Project Map

`jax2onnx` traces JAX, Flax NNX/Linen, and Equinox callables into JAXPR, lowers
them to `onnx_ir`, and serializes ONNX at the public API boundary.

- `jax2onnx/user_interface.py`: public API, validation, and serialization.
- `jax2onnx/converter/`: tracing, dispatch, value ownership, graph/function
  scopes, optimization, and export preparation.
- `jax2onnx/plugins/`: primitive-specific semantics, tracing-time patches,
  lowerings, and testcase/example metadata.
- `jax2onnx/_compat/jax.py`: version-sensitive JAX APIs.
- `tests/extra_tests/`: maintained regression and policy tests.
  `tests/primitives/` and `tests/examples/` are generated and git-ignored;
  change plugin/example metadata or `tests/t_generator.py` instead.

Read the relevant guide before editing a subsystem:

| Area | Guide |
| --- | --- |
| Architecture and plugin contracts | [Architecture](docs/developer_guide/architecture.md), [Plugin system](docs/developer_guide/plugin_system.md) |
| Node emission and value ownership | [IR builder](docs/developer_guide/advanced_topics/onnx_ir_builder.md) |
| Structural assertions | [expect_graph](docs/developer_guide/advanced_topics/expect_graph_reference.md) |
| Functions and control flow | [Subgraph inputs](docs/developer_guide/advanced_topics/subgraph_input_handling.md) |
| Graph rewrites and typed APIs | [IR optimizer](docs/developer_guide/advanced_topics/ir_optimizer.md), [Reflection guardrails](docs/developer_guide/advanced_topics/ir_reflection_guidelines.md) |
| Precision and runtime support | [API](docs/user_guide/api.md), [Known limitations](docs/user_guide/known_limitations.md), [Dependencies and kernel gaps](docs/about/dependencies.md) |
| Generated coverage and optional models | [Coverage generation](docs/developer_guide/maintainer/coverage_generation.md), [SotA maintenance](docs/developer_guide/maintainer/sota_example_maintenance.md) |

## Setup and Compatibility

- Use Python 3.11+ and Poetry 2.x. CI covers Python 3.11–3.14.
  `poetry install` installs base and development dependencies;
  `poetry run pre-commit install` installs hooks. Add optional groups such as
  `--with test`, `--with maxtext`, or `--with maxdiffusion` only when needed.
- The lockfile resolves two JAX stacks: Python 3.11/3.12 uses JAX 0.10.2;
  Python 3.13/3.14 uses JAX 0.12.x. Compatibility changes need coverage on both.
  Declared minimum versions remain a separate CI gate; a passing locked-stack
  run does not establish minimum-version support.
- Route version-sensitive JAX APIs through `jax2onnx/_compat/jax.py`. Use
  `scan_arity(...)` for scan parameters and guard JAXPR unwrapping against
  `.jaxpr is self`. Keep compatibility handling centralized.
- Format with the Black hook pinned in `.pre-commit-config.yaml`:
  `poetry run pre-commit run black --files path/to/changed.py`.
  The Poetry environment can contain a newer Black that rewrites unrelated
  code. Ruff's hook version is also pinned separately from the Poetry version.
- Keep Python file path markers and typing conventions intact. Pre-commit
  enforces them alongside reflection and IR-builder policies. Mypy's Python
  target is 3.12 for dependency stubs; the package still supports Python 3.11.

## Core Rules

- **Keep conversion IR-only.** Do not import `onnx` or any of its submodules
  inside `converter/` or `plugins/`; use public `onnx_ir` APIs only.
  Primitive-specific lowering policy belongs in plugins; the converter owns
  graph assembly and structural optimization.
- **Use the native lowering contract for new primitive plugins.** Register with
  `@register_native_primitive(...)`, implement
  `lower(self, ctx: NativeLoweringContext, eqn: JaxprEqn) -> None`, and explicitly
  require inputs, allocate outputs, and bind every non-drop result to a value
  owned by the current graph. A matching name on a detached value is not enough.
- **Emit through the typed builder.** Prefer `ctx.builder.op(..., output=...)`
  and `op_multi_out(..., outputs=...)`. Existing dynamic shorthand takes
  `_outputs=[...]`, never a bare string. Reuse allocated output values when
  emitting into them; graph rewrites must preserve metadata, graph outputs,
  and Function signatures. Use public graph mutation APIs and shared helpers.
  Bind constants with `ctx.bind_const_for_var(...)` or builder initializer
  helpers with `name=` so Function scopes emit `Constant` nodes.
- **Precision follows the source JAX program.** Model configuration, typed
  inputs, constants, explicit conversions, and local precision settings define
  computation types. Preserve weak scalar promotion and the caller's effective
  precision settings, including delayed Function tracing. Shape-only inputs
  default to FP32 even with x64 enabled.
  Prepare FP64 models and inputs in an effective `jax.enable_x64()` context and
  validate with `allclose` under the same settings. The removed
  `enable_double_precision` keyword is rejected whenever supplied.
- **Keep canonical values intact.** Constants retain their traced dtype and
  payload. When an operator needs a work dtype, emit consumer-local casts and
  restore the source result boundary; do not mutate shared initializers or
  canonical bindings. Preserve integer arithmetic and overflow semantics.
  Check low-precision changes with caller-owned x64 both disabled and enabled.
- **Construct fixtures deterministically.** Use constructor arguments such as
  `construct_and_call(MyModule, dtype=with_requested_dtype(), rngs=with_rng_seed(0))`.
  Each dtype variant must configure the callable. Keep fixed
  `expected_output_dtypes` fixed; use `with_requested_dtype()` only for results
  that vary with the prepared dtype. Use `with_prng_key(...)` for JAX key
  arguments; do not construct random module state at import, and split keys
  before separate consumers. Primitive parameters must be hashable.
- **Keep metadata and validation aligned.** Put `expect_graph` assertions beside
  testcase metadata and update them when lowering changes. Add numeric,
  structural, and dtype checks appropriate to the change. Numeric-validation
  skips are policy-allowlisted; a successful export alone does not prove the
  runtime can load or execute it.
- **Respect runtime floors.** Select operators by target opset and source
  semantics, not the ONNX Runtime installed on the export machine. Record new
  runtime workarounds in the dependency guide; remove them only when the
  supported minimum includes the fix.
- **Keep artifacts out of git.** Do not commit `.onnx`, `.onnx.data`, generated
  test files, checkpoints, or local validation reports. Public API changes need
  docs, tests, and release notes in `docs/about/roadmap.md` (current version) or
  `docs/about/past_versions.md` (release archive).

## Working Checks

Generate tests on a fresh checkout, after metadata, registration, or optional
integration changes, and before a full-suite run:

```bash
poetry run python scripts/generate_tests.py
poetry run pytest -q tests/path/test_file.py::TestClass::test_case
```

Run the focused regression first, then the full suite before shipping code.
The convenience runner fixes path markers across the working tree (including
untracked Python files), runs pre-commit, generates tests, then runs pytest
(parallel when pytest-xdist is installed):

```bash
./scripts/run_all_checks.sh
```

Hooks can edit files: Ruff uses `--fix --unsafe-fixes`. A hook failure stops the
runner before tests; review changes and rerun until hooks and pytest complete.
Individual checks are also available:

```bash
poetry run pre-commit run --all-files
./scripts/check_typing.sh
poetry run pytest -q
```

`check_typing.sh` runs mypy and prints the registered RNG factories. For prose
and link edits, check the changed content and links, then build the docs without
regenerating tests or models. Generated coverage pages use the workflow below.
The docs build is separate from the runner; use a unique temporary destination
to avoid rewriting tracked `site/` files or another session's build:

```bash
poetry run mkdocs build --strict --site-dir "$(mktemp -d)"
```

Optional SotA and [Web-runtime checks](docs/user_guide/browser_wasm.md) require
additional setup. A default local run does not cover every CI job. Report the
checks actually completed, failures, and any untested stacks or integrations.

## Generated Docs and Debugging

- Do not hand-edit sections between `AUTOGENERATED` markers. Update metadata
  and use the appropriate generator from the coverage guide.
- Refresh component/example tables with `./scripts/generate_readme.sh`. The
  wrapper prepares optional MaxText/MaxDiffusion checkouts and runs generation
  and tests. Missing optional integrations must not silently remove coverage
  rows; follow the coverage guide for intentional removals and review the diff.
- Refresh structural snippets with
  `poetry run python scripts/emit_expect_graph.py <testcase>`.
  For stacktrace fixtures, set `JAX2ONNX_ENABLE_STACKTRACE_METADATA=1`.
- Capture minimal reproductions in the relevant `tests/extra_tests/` area.
  Optimizer debug flags are documented in the optimizer guide.
- Patched APIs must be looked up at call time in tests, for example
  `lambda x: jnp.expm1(x)`. A function reference captured before conversion can
  bypass the plugin and silently test JAX's decomposition instead.
- For dtype/kernel failures, consult the dependency gap table and known
  limitations, including BF16 runtime I/O. For examples that skip numeric
  validation, check the serialized model with
  `onnx.checker.check_model(..., full_check=True)` and test runtime loading when
  the change can affect it.
