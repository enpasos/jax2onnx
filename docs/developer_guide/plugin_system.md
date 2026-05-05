# Plugin System

The plugin system is the extension point that keeps primitive-specific lowering
out of the converter core. A plugin teaches `jax2onnx` how to recognize a JAX
primitive or high-level callable during tracing, and how to lower the resulting
JAXPR equation into `onnx_ir`.

Read this page together with:

- [Architecture Overview](architecture.md) for the core/plugin boundary.
- [ONNX IR Builder Guide](advanced_topics/onnx_ir_builder.md) for builder and
  metadata rules.
- [Expect Graph Reference](advanced_topics/expect_graph_reference.md) for
  structural testcase checks.
- [ONNX Functions](../user_guide/onnx_functions.md) for user-facing
  `@onnx_function` behavior.

## Responsibilities

| Layer | Owns |
| --- | --- |
| `conversion_api` | Plugin discovery/activation, JAXPR tracing, equation dispatch, value binding checks, model finalization. |
| `PrimitiveLeafPlugin` | Primitive metadata, optional tracing-time bindings, abstract eval, and ONNX IR lowering. |
| `FunctionPlugin` | `@onnx_function` registration, function-body tracing, `FunctionScope` lowering, call-site emission. |
| `register_example` | Example/test metadata that does not introduce a new primitive. |
| Generated tests/docs | Numeric checks, structural checks, component coverage, examples, and ONNX operator coverage derived from metadata. |

The converter dispatches by `eqn.primitive.name`. It does not choose how a
primitive maps to ONNX. That mapping belongs in the plugin.

## Registry and Activation

Plugins live under `jax2onnx/plugins/`. `import_all_plugins()` recursively
imports that tree once per process. Imports are important: registration happens
as a side effect of importing plugin modules.

There are three main registries:

| Registry | Key | Value |
| --- | --- | --- |
| `PLUGIN_REGISTRY` | JAXPR primitive name string | `PrimitiveLeafPlugin` or `FunctionPlugin` instance |
| `ONNX_FUNCTION_PLUGIN_REGISTRY` | qualified Python target name | `FunctionPlugin` reference |
| `EXAMPLE_REGISTRY` | `{context}::{component}` | example metadata from `register_example(...)` |

During conversion, `_activate_plugin_worlds()` does this:

1. import all plugins;
2. apply registered function patches through `apply_monkey_patches()`;
3. enter every `PrimitiveLeafPlugin.plugin_binding()` context;
4. bind leaf-plugin abstract eval where needed;
5. backfill allowlisted transpose rules for custom primitives;
6. trace with `jax.make_jaxpr`;
7. restore all patches when tracing exits.

Nested ONNX Function body tracing uses a matching activation path so function
bodies see the same primitive bindings as the top-level trace.

## Primitive Plugins

Most primitive support is implemented as a subclass of `PrimitiveLeafPlugin`
registered with `@register_primitive(...)`.

```python
from typing import Any

import jax
import onnx_ir as ir
from jax import core

from jax2onnx.converter.typing_support import LoweringContextProtocol
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
from jax2onnx.plugins._ir_shapes import _ensure_value_metadata, _stamp_type_and_shape
from jax2onnx.plugins.plugin_system import PrimitiveLeafPlugin, register_primitive

JaxprEqn = getattr(core, "JaxprEqn", Any)


@register_primitive(
    jaxpr_primitive=jax.lax.abs_p.name,
    jax_doc="https://docs.jax.dev/en/latest/_autosummary/jax.lax.abs.html",
    onnx=[{"component": "Abs", "doc": "https://onnx.ai/onnx/operators/onnx__Abs.html"}],
    since="0.5.0",
    context="primitives.lax",
    component="abs",
    testcases=[
        {
            "testcase": "abs",
            "callable": lambda x: jax.lax.abs(x),
            "input_shapes": [(3,)],
            "post_check_onnx_graph": EG(["Abs:3"], no_unused_inputs=True),
        }
    ],
)
class AbsPlugin(PrimitiveLeafPlugin):
    def lower(self, ctx: LoweringContextProtocol, eqn: JaxprEqn) -> None:
        x_val = ctx.get_value_for_var(eqn.invars[0])
        out_val = ctx.get_value_for_var(eqn.outvars[0], name_hint=ctx.fresh_name("abs"))

        result = ctx.builder.Abs(x_val, _outputs=[out_val.name or ctx.fresh_name("abs")])
        result.type = out_val.type
        result.shape = out_val.shape
        _stamp_type_and_shape(result, tuple(getattr(eqn.outvars[0].aval, "shape", ())))
        _ensure_value_metadata(ctx, result)

        ctx.bind_value_for_var(eqn.outvars[0], result)
```

This is the common shape. More complex plugins may also define a custom JAX
primitive (`_PRIM`), `abstract_eval`, batching rules, JVP/transpose rules, and
binding specs.

## Registering Metadata

`@register_primitive(...)` stores every keyword argument as plugin metadata and
registers the plugin instance under `jaxpr_primitive`.

Important top-level metadata fields:

| Field | Purpose |
| --- | --- |
| `jaxpr_primitive` | Registry key. Must match `eqn.primitive.name`. |
| `jax_doc` | Source documentation for the JAX/Flax API. Used by generated docs. |
| `onnx` | List of ONNX components this lowering emits or depends on. |
| `since` | First jax2onnx version that supports the component. |
| `context` | Grouping namespace such as `primitives.lax` or `primitives.nnx`. |
| `component` | Stable component name used for generated tests and docs. |
| `description` | Optional human-readable summary. |
| `children` | For examples, child components used by the example. |
| `testcases` | Generated test inputs, expectations, and structural checks. |

Avoid duplicate `jaxpr_primitive` keys. The registry is a dictionary, so a later
import can overwrite an earlier entry; that is not a supported extension model.

## Binding Specs

A plugin only needs binding specs when ordinary JAX tracing would not emit the
primitive you want to lower. Direct `jax.lax` primitives often do not need
patching. High-level APIs such as `jax.numpy.*`, Flax NNX modules, or custom
wrappers often do.

New primitive plugins should implement:

```python
from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec


class SomePlugin(PrimitiveLeafPlugin):
    @classmethod
    def binding_specs(cls) -> list[AssignSpec | MonkeyPatchSpec]:
        return [
            AssignSpec("jax.numpy", "some_prim", cls._PRIM, delete_if_missing=True),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr="some_function",
                make_value=lambda original: make_wrapper(original, cls._PRIM),
            ),
        ]
```

Use `AssignSpec` to install or expose a primitive object. Use `MonkeyPatchSpec`
to wrap an existing function or method so tracing calls `cls._PRIM.bind(...)`.

Patch rules:

- Patches must be scoped. They are applied only while tracing and restored on
  exit.
- The wrapper must call the original implementation for unsupported modes, not
  silently lower an unsupported case.
- Do not seed, allocate modules, or capture mutable state inside the patch
  wrapper.
- `patch_info()` exists for older compatibility paths. Prefer
  `binding_specs()` for new primitive plugins.

## Abstract Eval, Batching, and Autodiff

Custom primitives must be usable by JAX before the converter can lower them.
At minimum, a primitive that is emitted during tracing needs abstract eval.
If the primitive can appear under `vmap`, it also needs a batching rule.
If tests or examples cover gradients, JVP/transpose support must be registered.

Guidelines:

- Define `abstract_eval(*avals, **params)` on the plugin when the output shape
  or dtype can be inferred from abstract values.
- Do not call the patched public function from `abstract_eval`; that can recurse
  back into the same primitive.
- Register batching rules explicitly for custom primitives used under `vmap`.
- Use helpers from `jax2onnx/plugins/jax/_autodiff_utils.py` for JVP and
  transpose rules instead of writing directly into JAX AD registries.
- Conversion-time AD backfill is allowlisted. Use
  `JAX2ONNX_AD_DEBUG=1` when debugging registration decisions, and
  `JAX2ONNX_DISABLE_AD_BACKFILL=1` to disable backfill.

## Lowering Contract

The lowering receives a `LoweringContextProtocol` and a JAXPR equation. The
contract is strict:

1. all non-literal inputs must already be bound before `lower(...)` runs;
2. the plugin emits connected `onnx_ir` nodes and values;
3. every non-drop output var must be bound before the equation completes;
4. returned `ir.Value` objects may be bound generically, but explicit binding is
   clearer for non-trivial lowerings.

Core context methods used most often:

| API | Use |
| --- | --- |
| `ctx.get_value_for_var(var, ...)` | Fetch or allocate the IR value for a JAX var. Literals can become constants. |
| `ctx.require_value_for_var(var, ...)` | Require an existing binding; useful when allocation would hide a bug. |
| `ctx.bind_value_for_var(var, value)` | Bind an equation output var to a produced `ir.Value`. |
| `ctx.bind_const_for_var(var, array)` | Bind a JAXPR constvar. Top-level graphs use initializers; Function bodies use `Constant` nodes. |
| `ctx.fresh_name(base)` | Create deterministic per-context names. |
| `ctx.builder` | Emit ONNX IR nodes through the project builder wrapper. |
| `ctx.try_evaluate_const(var)` | Evaluate a known constant producer when a lowering needs static data. |
| `ctx.ensure_external_flag(name, var)` | Route dynamic call parameters into graph/function inputs where needed. |

Lowering guardrails:

- Emit normal nodes through `ctx.builder`.
- Pass `_outputs=[...]`, never `_outputs="name"`.
- Stamp or preserve dtype/shape metadata on produced values.
- Reuse pre-allocated output values when identity matters.
- Keep converter/plugins IR-only; do not import ONNX protobuf types in lowering
  code.
- Use shared shape/constant helpers when available instead of ad hoc tensor
  construction.

The core validates both sides of the equation. `assert_eqn_inputs_bound` catches
unbound inputs before the plugin runs. `finalize_eqn_lowering_outputs` catches
missing or disconnected outputs after it runs.

## Function Plugins and `@onnx_function`

`@onnx_function` registers a `FunctionPlugin` under the primitive name
`onnx_fn::<qualified-target-name>`. It is user-facing, but its implementation is
part of the plugin system.

During tracing, the function plugin patches the target callable or class
`__call__` so a function primitive appears in JAXPR. During lowering, it:

1. resolves the original callable or instance;
2. builds a `FunctionKey` from target name, input avals, and capture signature;
3. reuses a cached body when the key matches;
4. otherwise creates a `FunctionScope` child `IRContext`;
5. maps parent inputs and dynamic call parameters to function inputs;
6. retraces and lowers the callable body inside the child context;
7. emits constants as `Constant` nodes because ONNX Function bodies cannot own
   initializers;
8. attaches the resulting `onnx_ir.Function` to the parent model;
9. emits a call-site node in the parent graph.

`unique=True` changes the capture signature from object identity to a stable
fingerprint of the callable/function state, so equivalent blocks can share one
Function body while different state produces separate bodies. Namespace and type
overrides are validated at registration time to avoid registry collisions.

Keep user-facing usage details in [ONNX Functions](../user_guide/onnx_functions.md).
Keep implementation details here and in tests under `tests/extra_tests/converter/`.

## Example Metadata

Use `register_example(...)` when you want generated tests and docs for a
composition of existing primitives, without introducing a new primitive.

```python
from jax2onnx.plugins.plugin_system import (
    construct_and_call,
    register_example,
    with_rng_seed,
)


register_example(
    component="CNN",
    description="A simple convolutional neural network.",
    context="examples.nnx",
    children=["nnx.Conv", "nnx.Linear", "nnx.avg_pool", "nnx.relu"],
    since="0.2.0",
    testcases=[
        {
            "testcase": "simple_cnn",
            "callable": construct_and_call(CNN, rngs=with_rng_seed(0)),
            "input_shapes": [("B", 28, 28, 1)],
            "expected_output_shapes": [("B", 10)],
        }
    ],
)
```

Examples are stored in `EXAMPLE_REGISTRY`, then consumed by the same generated
test and documentation tooling as primitive metadata.

## Testcase Metadata

Generated tests load all plugin and example metadata, expand dynamic/concrete
variants, expand f32/f64 variants, convert via public `to_onnx(...)`, run
optional structural checks, and run numeric validation when concrete
`input_values` are available.

Common testcase fields:

| Field | Purpose |
| --- | --- |
| `testcase` | Stable generated test/model name. |
| `callable` | Function or `construct_and_call(...)` object. |
| `input_shapes` | Shape specs; `"B"` triggers dynamic plus concrete variants unless restricted. |
| `input_dtypes` | Dtypes paired with `input_shapes`. |
| `input_values` | Concrete values used for tracing specs and numeric validation. |
| `input_params` | Keyword parameters supplied to the callable and optionally materialized as graph inputs. |
| `expected_output_shapes` | Output shape assertions after export. |
| `expected_output_dtypes` | Output dtype assertions after export. |
| `post_check_onnx_graph` | `expect_graph(...)` structural predicate. |
| `run_only_dynamic` | Skip the concrete variant for symbolic-shape tests. |
| `run_only_f32_variant` | Skip generated f64 variant. |
| `run_only_f64_variant` | Generate only the f64 variant. |
| `disable_float64_test` | Keep f32 only for tests that cannot safely run in f64. |
| `opset_version` | Override the default opset for this testcase. |
| `inputs_as_nchw` / `outputs_as_nchw` | Request public IO layout bridges. |
| `input_names` / `output_names` | Public graph IO name overrides. |
| `skip_numeric_validation` | Keep export/structure checks but skip runtime allclose. |
| `check_onnx_load` | Run ONNX checker after serialization. |
| `expected_number_of_function_instances` | Assert expected FunctionProto count for ONNX Function tests. |

`callable_factory` is no longer supported. Use `construct_and_call(...)` so the
test generator can instantiate a fresh callable under the requested dtype.

## Deterministic Callable Helpers

Use these helpers from `jax2onnx.plugins.plugin_system` in metadata:

| Helper | Use |
| --- | --- |
| `construct_and_call(Constructor, *args, **kwargs)` | Store a module constructor and instantiate it per generated dtype. |
| `with_requested_dtype()` | Placeholder argument resolved to the generated dtype. |
| `with_rng_seed(seed)` | Placeholder that builds fresh `nnx.Rngs(seed)` per instantiation. |
| `with_prng_key(seed)` | Placeholder that builds a fresh JAX PRNG key per instantiation. |

Do not treat dtype/RNG placeholders as methods on the factory object; they are
constructor arguments resolved later by the generated test harness.

Pass placeholders as constructor arguments instead:

```python
construct_and_call(
    MyModule,
    dtype=with_requested_dtype(),
    rngs=with_rng_seed(0),
)
```

This keeps module construction deterministic across f32/f64 variants and avoids
import-time RNG state.

## Generated Coverage

Primitive and example metadata is the source for:

- generated pytest classes under `tests/primitives/` and `tests/examples/`;
- `docs/user_guide/supported_components.md`;
- `docs/user_guide/examples.md`;
- `docs/user_guide/onnx_operator_coverage.md`.

Keep metadata user-oriented. It should describe supported components, examples,
ONNX operators, and durable tests. Release plans, temporary audits, and
one-off migration notes belong outside generated user documentation.

## Review Checklist

Before adding or changing a plugin:

- Verify the JAXPR primitive name.
- Keep primitive semantics in the plugin, not in `conversion_api`.
- Use `binding_specs()` only when tracing needs a custom primitive.
- Implement abstract eval, batching, and AD support needed by the testcase.
- Emit IR through `ctx.builder` with sequence-valued `_outputs`.
- Preserve dtype/shape metadata on produced values.
- Add focused `expect_graph(...)` checks for durable structure.
- Use `construct_and_call(...)` and RNG/dtype placeholders for module examples.
- Run the focused generated pytest target for the component.
- Run builder/policy checks when touching lowering infrastructure.

## Failure Modes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `No plugins registered for primitive ...` | Plugin module was not imported, wrong `jaxpr_primitive`, or binding did not emit the intended primitive. | Check plugin import path, registry key, and `binding_specs()`. |
| JAX tracing recurses or hangs | `abstract_eval` or a patch wrapper calls the patched public function again. | Call a `lax.*` helper, the stored original implementation, or pure shape logic. |
| `unbound input` before lowering | A nested body/function input was not mapped, or a literal/const was not bound. | Fix body input wiring or use `require_value_for_var` to expose the missing var. |
| `did not bind output` | Lowering forgot to bind or return a produced value. | Bind every non-drop outvar or return the exact `ir.Value` sequence. |
| `disconnected value` | Lowering bound a placeholder that is not produced by a graph node/input/initializer. | Emit the node with that value as output, or bind the actual produced value. |
| Extra graph inputs remain | Lowering introduced unused inputs or function params. | Add focused structural checks and inspect pruning/signature rules. |
| f64 variant fails unexpectedly | Callable construction or metadata ignores generated dtype. | Use `construct_and_call(...)` with `with_requested_dtype()` placeholders. |
