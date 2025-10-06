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

### Converter modules touching IR
- `conversion_api.py`, `ir_context.py`, and `function_scope.py` still fabricate `ir.Node` objects directly for casts, constants, and control-flow scaffolding even though they hold a `builder`; any Builder migration plan must account for these helpers first.
- `ir_builder.py` is the only module instantiating `_tape.Builder`; it mirrors the builder’s `nodes`/`initializers` state and continues to expose manual `add_node` escape hatches for legacy call sites.
- `ir_optimizations.py` (plus `ir_postprocess.py`) intentionally works at the raw `ir.Node`/list level to rewrite existing graphs; nothing to change here, but every pass relies on `_set_nodes` keeping `graph.nodes`, `graph._nodes`, and `graph.node` in sync.
- No `import onnx` statements exist under `converter/`, confirming IR-only policy compliance for the core pipeline.

- 22 lax/random/jnp plugins already rely exclusively on `ctx.builder.*` helpers (arithmetic family, concatenate, stack, tile, gather, dynamic slice, dynamic update slice, take, random seed/fold-in/bits, etc.).
- Mixed modules remain concentrated in complex lowers (`jax/lax/conv.py`, `jax/lax/scan.py`, `jax/lax/while_loop.py`, `flax/nnx/dot_product_attention.py`, `plugin_system.py`). These should be prioritized for incremental refactors.
- The largest manual-only clusters are Flax NNX activations/pooling layers, JAX NN scalar activations, and indexing utilities such as `jax/lax/gather.py`, `jax/lax/scatter_utils.py`, `jax/numpy/take.py`.
- None of the plugin files import `onnx` protobuf helpers; all proto references are limited to docstring URLs. Policy tests remain satisfied.

**Conversion snapshot (Oct 2025)**
- LAX arithmetic + concat/stack/gather (`add`, `mul`, `neg`, `integer_pow`, `convert_element_type`, `concatenate`, `stack`, `tile`, `gather`, etc.) → builder-only.
- LAX control-flow / shape builders (`conv`, `scan`, `while_loop`, `broadcast_in_dim`) → mixed builder + manual nodes.
- LAX indexing utilities (`scatter_utils`, `transpose`) → manual-only.
- Flax NNX layers (activations, pooling, linear/conv, norms) → manual-only.
- Equinox EQX core (`linear`, `dropout`, `identity`) → mixed; builder used for inits, manual for wiring.
- JAX NN activations (`relu`, `gelu`, `softmax`, etc.) → manual-only.
- Random seeds/fold-in/bits → builder-only.

### Follow-up flags
- Manual-only plugins often duplicate constant/initializer plumbing (`_const_i64`, `ctx.add_node(ir.Node(...))`); we should identify shared helper entry points once Builder adoption begins.
- Mixed modules typically call builder for initializers but fall back to manual nodes for multi-output ops or value reuse. Cataloging those patterns (Concat shape builders, Scan/While scaffolding) will feed directly into Step 3’s canonical snippets.

Next: craft canonical Builder patterns (Step 3).

## Step 3 – Canonical Builder Patterns

### Single-output op with shared initializer
```python
import numpy as np
import onnx_ir as ir
from onnx_ir._tape import Builder

builder = Builder()
X = ir.val("X", dtype=ir.DataType.FLOAT, shape=[None, 128])
Y = ir.val("Y", dtype=ir.DataType.FLOAT, shape=[None, 128])

scale = builder.initializer(ir.tensor(np.array(0.5, dtype=np.float32)), name="scale")
prod = builder.Mul(X, scale, _outputs=["scaled"])
result = builder.Add(prod, Y, _outputs=["bias_add"])

result.producer().doc_string = "Adds learned bias after scaling"
opset_imports = {domain or "": version for domain, version in builder.used_opsets if version}
```

### Dynamic dim gather → shape tensor (for reshape/concat builders)
```python
shape_val = builder.Shape(tensor_val, _outputs=[ctx.fresh_name("shape_of_x")])
axis_idx = builder.initializer(
    ir.tensor(np.array(idx, dtype=np.int64)),
    name=ctx.fresh_name("gather_axis"),
)
dim_scalar = builder.Gather(
    shape_val,
    axis_idx,
    axis=0,
    _outputs=[ctx.fresh_name("dynamic_dim")],
)
axes = builder.initializer(
    ir.tensor(np.array([0], dtype=np.int64)),
    name=ctx.fresh_name("unsq_axes"),
)
dim_vec = builder.Unsqueeze(dim_scalar, axes, _outputs=[ctx.fresh_name("dynamic_dim_vec")])
shape_tensor = builder.Concat(
    [dim_vec, static_len_val],
    axis=0,
    _outputs=[ctx.fresh_name("shape_tensor")],
)
```

### Control-flow or other multi-output ops
```python
then_val, else_val = builder.If(
    predicate_val,
    then_branch=then_graph,
    else_branch=else_graph,
    _outputs=[ctx.fresh_name("then"), ctx.fresh_name("else")],
    _version=18,
)
for value in (then_val, else_val):
    value.type = ir.TensorType(ir.DataType.FLOAT)
    value.shape = ir.Shape((None, 128))
```

### Interop with low-level Tape when outputs must be pre-created
```python
from onnx_ir import tape

placeholder = ir.Value(
    name="reshaped",
    shape=ir.Shape([1, 128]),
    type=ir.TensorType(ir.DataType.FLOAT),
)
tape.Tape.op(
    "Reshape",
    builder.graph_like or builder,
    inputs=[input_val, shape_val],
    output=placeholder,
    domain="",
    version=18,
)
```

Key checks before serialization:
- Resolve placeholders in metadata via `with_requested_dtype()` / RNG helpers.
- Pass `_outputs` as sequences (tuple/list) rather than bare strings.
- Represent optional inputs as `None` to keep positional slots stable.
- Apply doc strings or names after node creation so builder kwargs stay attributes only when needed.

Progress: lax `Add`, `Mul`, `Sub`, `Div`, `Max`, `Min`, `Sin`, `Cos`, `Log`, `Tanh`, `Exp`, `Abs`, `Sqrt`, `Sign`, `Logistic`, `Cosh`, `Neg`, `Sinh`, `Eq`, `Gt`, `Ge`, `Lt`, `Ne`, `And`, `Or`, `ShiftRightLogical`, `Pow`, `Clamp`, `Rem`, `convert_element_type`, `Squeeze`, `Pad`, `Square`, `BitwiseNot`, `copy`, `stop_gradient`, `bitcast_convert_type`, and `integer_pow` now emit nodes through `ctx.builder`, providing reference patterns for remaining lowers.

## Step 4 – Validation & Lint Hooks

- **Policy test expansion**: extend `tests/extra_tests/framework/test_no_onnx_in_converter2_plugins2.py` to also parse AST `Call` nodes and flag:
  * Any reference to `onnx.ModelProto`, `onnx.helper`, or `onnx.shape_inference` (catches sneaky proto helpers without a top-level import).
  * `builder.initializer(...)` / `ctx.builder.add_initializer_from_*` invocations missing a `name=` keyword. Implement this via AST matching on attribute calls so false positives stay low.
- **New builder contract test**: add `tests/extra_tests/framework/test_ir_builder_contracts.py` that walks plugin/converter sources and asserts each `_outputs=` keyword is a `list`/`tuple` literal (or a variable defined as such). The helper can reuse the AST visitor to fail when `_outputs` is supplied a bare string or missing entirely for multi-value ops.
- **Static lint**: drop a lightweight checker `scripts/check_ir_builder_usage.py` (wired through CI / Ruff's `per-file-ignores`) that enforces the quick heuristics: `_outputs="` never appears, and `builder.initializer(` calls always include a `name` keyword or are wrapped in helpers that provide one. Keep it idempotent so devs can run it locally (`poetry run python scripts/check_ir_builder_usage.py`).
- **Metadata guard**: extend the plugin metadata fixture so any entry that exposes a callable must go through `construct_and_call(..., with_requested_dtype(), with_rng_seed(...))`. Reject legacy `callable_factory` usages with a clear failure message—this keeps the RNG/dtype policy observable.
- **CI surface area**: ensure the builder contract test and static checker run alongside the existing policy test (part of `poetry run pytest -q tests/extra_tests/framework` and a make target). For debugging flaky optimizer passes, keep the `JAX2ONNX_IROPT_DEBUG=1` hook documented but optional.

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
- Implement the Step 4 validation hooks: land the expanded policy test, add `tests/extra_tests/framework/test_ir_builder_contracts.py`, and wire `scripts/check_ir_builder_usage.py` into CI (pre-commit/Ruff) so regressions are caught automatically.
- Next conversion batch (indexing + scatter suite):
  1. `jax/lax/scatter_utils.py` – centralize builder-based concat/index construction.

---

Status (2025-10-05): IRBuilder aligned with `_tape.Builder`; lax add/mul/sub migrated to builder-only lowers. Next focus: extend builder usage across remaining primitives and tighten validation hooks.
