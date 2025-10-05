# Plan: Align onnx_ir Usage with Project Conventions

1. Collect guidance and desired practices
   - Re-read `AGENTS.md` sections on IR-only policy, randomness rules, and builder usage expectations.
   - Summarize the "How to Use the ONNX IR Builder" note into actionable guardrails (Builder vs. Tape, attribute handling, initializer registration).

2. Map current implementation touchpoints
   - Inventory modules in `converter/` and `plugins/` that construct or mutate IR (search for `onnx_ir`, `Builder`, `_tape` usage).
   - Flag any direct ONNX proto imports/usages for follow-up removal per policy.

3. Define canonical builder patterns
   - Draft code snippets for common operations (single-output op, multi-output op, initializer setup, metadata edits) that follow the guidelines.
   - Ensure examples mirror RNG handling and dtype hooks mandated in `AGENTS.md` (e.g., `construct_and_call`, `with_requested_dtype`).

4. Establish validation and linting hooks
   - Add or update tests/policy checks under `tests/` to confirm no proto usage and that builder helpers stay compliant.
   - Consider adding lightweight static checks (via Ruff plugin or custom script) for discouraged patterns (e.g., bare strings in `_outputs`).

5. Update documentation and onboarding material
   - Fold the distilled guardrails into developer docs (link from `docs/`, surface in `MigrationStatus.md` if relevant).
   - Add cross-reference from plugin README/guide so contributors see the builder expectations early.

6. Schedule follow-up actions
   - Create task list for refactoring any flagged modules to the canonical patterns.
   - Plan regression tests (focused pytest targets) for each refactor to keep coverage green.

---

## Step 1 – Guidance Digest

- **IR-only policy**: `converter/` and `plugins/` must rely solely on `onnx_ir`; protobuf (`onnx`) imports belong only in top-level adapters/tests, reinforced by policy tests.
- **Randomness discipline**: never seed at import; always thread single-use PRNG keys through `construct_and_call(...)` using helpers like `with_rng_seed(...)`, `with_requested_dtype(...)` to keep plugins hashable under JAX 0.7.
- **Module construction**: avoid per-call instantiation inside traced bodies. Build modules once per dtype via `construct_and_call(...).with_dtype(...)` and reuse instances.
- **NNX semantics**: wrap array-bearing attributes with `nnx.List(...)`/`nnx.data(...)` to satisfy pytree invariants introduced in NNX ≥0.12.
- **Builder usage**: prefer `onnx_ir._tape.Builder` for scripted graph construction; fall back to `Tape.op`/direct nodes when you need explicit output reuse, naming, or non-standard attribute handling.
- **Mutation persistence**: when mutating IR graphs, ensure updates land in all node containers (`graph.nodes`, `graph._nodes`, `graph.node`) to sidestep copy-on-write differences across `onnx_ir` builds.
- **Dropout handling**: inline `training_mode=False` by splicing in missing third input and removing orphaned `Not` nodes; never delete the Dropout node itself.
- **Documentation anchors**: primary references live in `docs/design.md`, `docs/subgraph_input_handling.md`, and the builder how-to note for deeper dives.

## Step 1b – Builder Guardrails

- **Inputs & values**: create graph values up front with `ir.val(...)` or reuse existing `ir.Value` instances before passing to builder methods.
- **Initializers**: register constants through `builder.initializer(...)` so they are tracked on the owning graph and stay aligned with node outputs.
- **Opsets**: use `_domain`/`_version` kwargs to request versions, then consolidate via `builder.used_opsets` to populate `Graph.opset_imports`; reject mixed versions per domain.
- **Outputs**: pass `_outputs` as lists/tuples of names or ints; never supply bare strings (would split into characters).
- **Node metadata**: kwargs become attributes; set names/doc strings on the resulting node (`value.producer().name = ...`).
- **Multi-output ops**: accept tuple of `ir.Value`s and destructure; call `Tape.op_multi_out` when needing pre-existing outputs.
- **Attribute typing**: ambiguous cases require explicit `ir.Attr` or `ir.tensor(...)`; builder auto-converts basic Python types.
- **Graph binding**: when builder is seeded with an existing graph/function, emitted ops attach immediately, preserving naming authorities; initializers only bind to graphs, not functions.
- **Limitations**: builder cannot pass through `Tape` parameters like `overload`, `doc_string`, or `output`; switch to direct `Tape` calls for those scenarios.

---

To do next: work through Step 2 inventory.

## Step 2 – Current Usage Inventory

- **Converter core**
  - `jax2onnx/converter/ir_builder.py` now mandates `_tape.Builder`, mirroring its node/initializer state instead of hand-rolling `ir.Node` assembly.
  - `jax2onnx/converter/ir_context.py` and `conversion_api.py` still orchestrate lowering through the builder/context; targeted cleanups remain to swap residual manual `ir.Node` construction for direct builder calls.
  - `jax2onnx/converter/ir_optimizations.py` and `ir_postprocess.py` mutate graphs post-build, assuming access to raw `ir.Node` lists and handling copy-on-write semantics explicitly.
  - `jax2onnx/converter/function_scope.py` materializes `ir.Function` bodies using the same manual node construction pattern.
  - No direct `onnx` proto imports inside converter/; proto interactions live in `serde_onnx.py` and user-facing adapters.

- **Plugin system**
  - `jax2onnx/plugins/plugin_system.py` mediates between primitives and conversion contexts; it exposes `FunctionScope`/`IRBuilder` to plugins rather than `_tape.Builder`.
  - Primitive plugins across `jax2onnx/plugins/jax/**` predominantly instantiate `ir.Node` manually, often via helpers in `plugins/_utils.py`; attribute creation relies on `ir.Attr` fallbacks for cross-version compatibility.
  - Flax NNX and Equinox plugins mirror this pattern: explicit `ir.Node` creation, helper functions for attribute robustness, dtype stamping handled per module.
  - Example metadata in `plugins/examples/` reuses the plugin system but does not directly touch `_tape.Builder`.

- **Direct builder usage gaps**
  - No module currently wraps or re-exports `onnx_ir._tape.Builder`; guidance from `how_to_use_onnx_ir.md` is not yet reflected in code or docs within `jax2onnx/`.
  - Manual node assembly appears everywhere; standardized helper abstractions (`IRBuilder`, `FunctionScope`) shield most call sites but diverge from the recommended Builder/Tape layering.

- **Policy compliance check**
  - `rg "import onnx" jax2onnx/converter jax2onnx/plugins` confirms there are no proto imports under converter/plugins; existing checks remain effective.
  - Attribute helper utilities already guard against `onnx_ir` version drift (e.g., `plugins/flax/nnx/linear.py`, `plugins/flax/nnx/conv.py`).

Next: craft canonical Builder patterns (Step 3).

## Step 3 – Canonical Builder Patterns

```python
import numpy as np
import onnx_ir as ir
from onnx_ir._tape import Builder

# Single-output op with initializer registration
X = ir.val("X", dtype=ir.DataType.FLOAT, shape=[None, 128])
Y = ir.val("Y", dtype=ir.DataType.FLOAT, shape=[None, 128])

builder = Builder()
scale = builder.initializer(ir.tensor(np.array(0.5, dtype=np.float32)), name="scale")
prod = builder.Mul(X, scale, _outputs=["scaled"])
result = builder.Add(prod, Y)

# Node metadata that cannot be passed via kwargs
node = result.producer()
node.name = "bias_add"
node.doc_string = "Adds learned bias after scaling"

# Consolidate opsets before graph assembly
opset_imports = {domain or "": version for domain, version in builder.used_opsets if version}
```

```python
# Multi-output op with optional inputs and dtype placeholders
import onnx_ir as ir
from onnx_ir._tape import Builder
from jax2onnx.plugins.plugin_system import construct_and_call, with_requested_dtype, with_rng_seed

cond = ir.val("cond", dtype=ir.DataType.BOOL, shape=[])
true_graph = ir.Graph(inputs=[...], outputs=[...], nodes=[])
false_graph = ir.Graph(inputs=[...], outputs=[...], nodes=[])

builder = Builder()
then_val, else_val = builder.If(
    cond,
    then_branch=true_graph,
    else_branch=false_graph,
    _outputs=["then_val", "else_val"],
    _version=18,
)

callable_meta = construct_and_call(
    MyModule,
    hidden_features=128,
    dtype=with_requested_dtype(),
    rngs=with_rng_seed(0),
)
```

```python
# Mixing Builder with manual Tape usage when output handles must be pre-created
from onnx_ir import tape

builder = Builder()
existing = ir.Value(
    name="reshaped",
    shape=ir.Shape([1, 128]),
    type=ir.TensorType(ir.DataType.FLOAT),
)
reshape_node = tape.Tape.op(
    "Reshape",
    builder.graph_like or builder,
    inputs=[input_val, shape_val],
    output=existing,
    domain="",
    version=18,
)
```

```python
# Metadata edits performed after node creation
node = then_val.producer()
node.metadata_props = {"jax.primitive": "lax.cond"}
for value in (then_val, else_val):
    value.type = ir.TensorType(ir.DataType.FLOAT)
```

Key checks before serialization:
- All placeholders resolved via `with_requested_dtype()` / RNG helpers when exposed through `construct_and_call` metadata.
- `_outputs` arguments always provided as sequences, not bare strings.
- Optional operator slots represented with `None` in positional args to preserve indices.
- Node naming/doc strings assigned post-creation to avoid accidental attribute injection.

Progress: lax `Add`, `Mul`, `Sub`, `Div`, `Max`, `Min`, `Sin`, `Cos`, `Log`, `Tanh`, `Exp`, `Abs`, `Sqrt`, `Sign`, `Logistic`, `Cosh`, `Neg`, `Sinh`, `Eq`, `Gt`, `Ge`, `Lt`, `Ne`, `And`, `Or`, `ShiftRightLogical`, `Pow`, `Clamp`, `Rem`, `convert_element_type`, `Squeeze`, `Pad`, `Square`, and `BitwiseNot` now emit nodes through `ctx.builder`, providing reference patterns for remaining lowers.

## Step 4 – Validation & Lint Hooks

- Extend the existing policy test (`tests/policy/test_no_onnx_in_converter_plugins.py`) to also flag direct `onnx.ModelProto` references under converter/plugins by scanning AST for `onnx.ModelProto`/`onnx.helper` usage.
- Add a focused unit test under `tests/policy/` that instantiates a sample builder workflow and asserts `_outputs` rejects bare strings (exercise the helper that enforces sequence types).
- Introduce a Ruff custom rule (or simple `scripts/check_ir_builder_usage.py`) that searches for `_outputs="` and `builder.initializer(` without a `name=` argument to catch common builder misuses early.
- Leverage pytest fixtures in `tests/plugins` to ensure plugin metadata always exposes `construct_and_call(..., with_requested_dtype(), with_rng_seed(...))`; fail fast when encountering legacy `callable_factory` usage.
- Wire optional debug run (`JAX2ONNX_IROPT_DEBUG=1`) into CI smoke tests for optimizer flows mutated by builder refactors.

Next: plan doc updates and onboarding references (Step 5).

## Step 5 – Documentation / Onboarding Touchpoints

- Update `docs/design.md` with a short subsection summarizing the builder vs. tape layering, linking to `how_to_use_onnx_ir.md` for full details.
- Add a "Builder quick-reference" callout to `docs/expect_graph_reference.md` so structural test authors understand how values/nodes should be produced prior to assertions.
- Create a new `docs/dev_guides/onnx_ir_builder.md` (or migrate the existing `how_to_use_onnx_ir.md` into that location) and reference it from `README.md` and `MigrationStatus.md`.
- Surface RNG/dtype conventions in the plugin README (currently in `plugins/` root) to ensure new plugins mirror `construct_and_call` usage from day one.
- Record the validation hooks (Step 4) and their intent in `CONTRIBUTING.md` to explain why certain lint/tests exist.

Next: enumerate refactor tasks and regression coverage (Step 6).

## Step 6 – Follow-up Refactors & Regression Coverage

- Thin wrapper no longer needed: `IRBuilder` directly instantiates `_tape.Builder` and mirrors its state. Remaining work focuses on migrating residual manual `ir.Node` usage to the builder APIs.
- Incrementally migrate high-traffic plugins (start with `jax/lax` arithmetic ops, then Flax NNX linear layers) to the canonical builder helpers, adding focused pytest cases per primitive to verify op sequencing.
- Harden attribute helper modules (`plugins/flax/nnx/linear.py`, `plugins/flax/nnx/conv.py`) by routing through shared builder utilities once the wrapper is in place; add regression tests ensuring dtype/shape stamping survives conversion.
- Introduce integration tests that serialize representative graphs to protobuf via `ir.to_proto` only at the very edge, confirming converter/plugins remain protobuf-free.
- After each refactor batch, regenerate `MigrationStatus.md` and run `poetry run pytest -q` plus targeted policy tests to keep coverage green.

---

Status (2025-10-05): IRBuilder aligned with `_tape.Builder`; lax add/mul/sub migrated to builder-only lowers. Next focus: extend builder usage across remaining primitives and tighten validation hooks.
