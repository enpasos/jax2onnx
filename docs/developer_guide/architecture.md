# Architecture Overview

`jax2onnx` converts JAX functions and Flax modules to ONNX by tracing a JAX
program, lowering the resulting JAXPR into `onnx_ir`, and only converting to
ONNX protobuf at the public export boundary. The converter is built around an
IR-only core plus plugins that own primitive-specific lowering semantics.

The key architectural boundary is:

- **Core converter:** tracing, value ownership, graph assembly, function scopes,
  structural optimization, export post-processing, and guardrails.
- **Plugins:** primitive registration, optional tracing-time bindings, abstract
  evaluation, ONNX operator choices, attributes, layout rules, and testcase
  metadata.

The core is not completely blind to ONNX operator names: optimizer and
post-processing passes deliberately inspect IR operator types for conservative
graph cleanup and export compatibility. What must stay out of the core is
primitive-specific lowering policy such as how `lax.conv_general_dilated`,
`nnx.Conv`, or attention layers map their semantics to ONNX nodes.

## Design Principles

1. **IR first.** `converter/` and `plugins/` operate on `onnx_ir`, not ONNX
   protobuf classes. Protobuf serialization happens in the public API facade.
2. **Plugin-owned semantics.** A plugin owns the meaning of the JAX primitive or
   high-level function it registers. The converter dispatches by primitive name.
3. **Explicit value ownership.** Every JAX variable must map to a connected
   `ir.Value`. Lowering failures are caught at the equation where a value is
   missing or disconnected.
4. **Scoped graph bodies.** ONNX Functions and control-flow bodies are built in
   child contexts so inputs, constants, symbolic dimensions, and signatures stay
   explicit.
5. **Conservative graph cleanup.** Optimizer passes are IR-only and structural.
   They may remove redundant graph artifacts, but they must preserve graph
   outputs and function signatures.

## Related Documentation

- [Plugin System Guide](plugin_system.md) - plugin authoring workflow and metadata.
- [ONNX IR Builder Guide](advanced_topics/onnx_ir_builder.md) - builder
  conventions, `_outputs`, initializers, and graph ownership rules.
- [Expect Graph Reference](advanced_topics/expect_graph_reference.md) -
  structural testcase assertions.
- [Control-Flow Body Inputs](advanced_topics/subgraph_input_handling.md) -
  body input wiring for `If`/`Loop` and function-style subgraphs.
- [IR Optimizer Guide](advanced_topics/ir_optimizer.md) - pass order and
  optimizer invariants.
- [ONNX Functions](../user_guide/onnx_functions.md) - user-facing behavior of
  `@onnx_function`.
- [Supported Components](../user_guide/supported_components.md) and
  [ONNX Operator Coverage](../user_guide/onnx_operator_coverage.md) -
  generated coverage views derived from plugin/example metadata.

## Module Map

| Module | Responsibility |
| --- | --- |
| `jax2onnx/user_interface.py` | Public `to_onnx`, `onnx_function`, return-mode handling, ONNX protobuf/file serialization. |
| `jax2onnx/converter/conversion_api.py` | Core conversion pipeline: trace, bind inputs/constants, lower JAXPR, build/finalize `onnx_ir.Model`. |
| `jax2onnx/converter/lowering_dispatch.py` | Equation-by-equation plugin dispatch, primitive call recording, stacktrace/plugin metadata staging. |
| `jax2onnx/converter/output_binding.py` | Input/output binding assertions and generic binding of returned lowering values. |
| `jax2onnx/converter/ir_context.py` | Plugin-facing context: var-to-value map, constants, graph IO, symbolic dim origins, function registry. |
| `jax2onnx/converter/ir_builder.py` | Thin wrapper around `onnx_ir` tape builder with project-specific naming, initializer, and metadata behavior. |
| `jax2onnx/converter/function_scope.py` | Child context used to stage ONNX Function bodies. |
| `jax2onnx/converter/ir_optimizations.py` | IR-only optimizer pass registry. |
| `jax2onnx/converter/ir_postprocess.py` | Export preparation: loosen intermediate shapes and promote constants when double precision is requested. |
| `jax2onnx/plugins/plugin_system.py` | Plugin registry, tracing-time patch activation, `PrimitiveLeafPlugin`, and `@onnx_function`. |
| `jax2onnx/plugins/` | Primitive and framework-specific lowering implementations plus generated-test metadata. |

## Conversion Pipeline (Detailed)

```text
Public to_onnx(...)
    |
    | user_interface.py
    | - normalize input specs, names, input_params, return mode
    v
conversion_api.to_onnx(...)
    |
    | - normalize inputs to ShapeDtypeStruct
    | - create JAX symbolic dims for string dimensions
    | - activate plugin bindings around tracing
    | - trace with jax.make_jaxpr
    v
ClosedJaxpr
    |
    | - bind constvars
    | - bind graph inputs, including optional NCHW bridges
    | - lower equations through PLUGIN_REGISTRY[primitive_name]
    | - assert every output var is bound to a connected ir.Value
    | - bind graph outputs, including optional NCHW bridges
    v
Raw onnx_ir graph
    |
    | - attach ONNX Function bodies
    | - run IR optimizer
    | - apply late attribute overrides and compatibility fixes
    | - normalize value shapes and symbolic dims
    v
Precise-shape onnx_ir.Model
    |
    | user_interface.py
    | - loosen intermediate shapes for runtime flexibility
    | - materialize referenced input_params as graph inputs
    | - apply custom input/output names
    | - return ir, proto, or file
    v
Export result
```

## Core Conversion Stages

### 1. Input Normalization and Tracing

The public API accepts shape tuples, arrays, `jax.ShapeDtypeStruct`, and related
input specs. `conversion_api` normalizes them to `ShapeDtypeStruct` instances.
String dimensions such as `"B"` are converted to JAX symbolic dimensions inside
a shared `jax.export.SymbolicScope`.

Plugin bindings are active only around tracing. This is where high-level calls
such as NNX modules, decorated `@onnx_function` targets, or plugin-specific JAX
wrappers can emit custom primitive names into the `ClosedJaxpr`. After tracing,
patches are restored.

`input_params` are passed to the traced callable as fixed keyword arguments.
If a later ONNX Function call needs one of those values as a dynamic function
input, the function plugin records that requirement and the public facade can
materialize the referenced parameter as an explicit graph input.

### 2. Context Setup

`IRContext` owns the active `IRBuilder`, the JAX-var-to-IR-value map, symbolic
dimension origins, function registry, and attribute overrides. Before lowering,
the converter:

- binds JAXPR constants as initializers in top-level graphs;
- emits `Constant` nodes instead when inside ONNX Function bodies;
- creates graph inputs for JAXPR input variables;
- inserts NCHW/NHWC bridge transposes only for explicitly requested external IO.

### 3. Equation Lowering

`lowering_dispatch.lower_jaxpr_with_plugins(...)` walks `jaxpr.eqns` in order.
For each equation it:

1. reads `eqn.primitive.name`;
2. looks up the registered plugin in `PLUGIN_REGISTRY`;
3. verifies all input vars are already bound;
4. calls the primitive or function lowering;
5. binds returned `ir.Value` objects when the plugin returns values instead of
   binding directly;
6. verifies every non-drop outvar is bound to a graph-connected value.

This makes the plugin contract small but strict: a lowering must produce the
IR values for the equation outputs and connect them to the graph.

### 4. Model Finalization

After graph outputs are bound, `IRBuilder.to_ir_model(...)` creates the
`onnx_ir.Model`. The converter then:

- attaches staged ONNX Functions to the model function store;
- runs `optimize_graph`;
- applies late attribute overrides and fills missing `Concat(axis=0)` attributes;
- normalizes value shapes so symbolic dimensions are represented as
  `ir.SymbolicDim` objects where needed.

Optimizer failures are non-fatal by default and logged. Set
`JAX2ONNX_STRICT_OPTIMIZER_FAILURES=1` when the optimizer must fail the export.

### 5. Public Export Post-Processing

The public facade calls `postprocess_ir_model(...)` before returning or
serializing. This pass loosens intermediate value shapes while preserving graph
input/output shapes, recursively handles control-flow subgraphs and ONNX
Function bodies, and promotes constants when double precision export is enabled.

Only after post-processing does the public facade apply custom IO names,
materialize referenced `input_params`, convert to `onnx.ModelProto`, or write a
file.

## Plugin Contract

Most primitive plugins subclass `PrimitiveLeafPlugin` and are registered with
`@register_primitive(...)`. A plugin normally provides:

- **Metadata:** JAX primitive name, user-facing component name, ONNX operator
  references, generated testcase metadata, and structural checks.
- **Binding specs:** optional tracing-time patches for high-level APIs. If the
  primitive is already emitted by JAX directly, no patch is needed.
- **Abstract eval:** shape/dtype inference used by JAX tracing.
- **Lowering:** ONNX IR emission through `ctx.builder` and binding of equation
  outvars.

Lowerings should follow these rules:

- Fetch inputs with `ctx.get_value_for_var(...)` or
  `ctx.require_value_for_var(...)`.
- Pre-allocate or fetch output values with `ctx.get_value_for_var(eqn.outvars[i])`
  when the output identity matters.
- Emit nodes through `ctx.builder` unless a specific `onnx_ir` feature requires
  manual `ir.Node` construction.
- Keep `_outputs` explicit and sequence-valued for builder calls.
- Stamp dtype/shape metadata on produced values when the builder cannot infer it.
- Bind outputs with `ctx.bind_value_for_var(...)` or return the produced values
  so the generic output binder can bind them.
- Never import ONNX protobuf classes from converter or plugin lowering code.

## ONNX Function Boundaries

`@onnx_function` is user-facing, but its lowering path is part of the converter
architecture. Decorating a function or class registers a `FunctionPlugin`.
During tracing, calls bind an `onnx_fn::<qualified-name>` primitive. During
lowering, the plugin:

1. builds a key from the qualified target name, input aval signatures, and
   captured/static parameters;
2. reuses a cached function definition when the key matches;
3. otherwise opens a `FunctionScope` child context;
4. maps parent values to function inputs;
5. retraces and lowers the callable body inside the child context;
6. emits constants as `Constant` nodes because Function bodies cannot own
   initializers;
7. attaches the resulting `onnx_ir.Function` to the parent model and emits a
   call-site node.

Identical calls share a function body when the function key matches. Shape,
dtype, instance, namespace, or capture changes can produce a separate function
definition.

## Control Flow

Control-flow primitives are plugins too:

- `lax.cond` lowers to ONNX `If`.
- `lax.while_loop` lowers to ONNX `Loop`.
- `lax.scan` currently lowers through ONNX `Loop`, not the ONNX `Scan` operator.

Control-flow body graphs use explicit inputs for carried state, sequence values,
and captured values. Constants inside those bodies are materialized as `Constant`
nodes. Body shape handling is deliberately conservative, and export
post-processing loosens nested body shapes where runtime shape inference needs
more flexibility.

See [Control-Flow Body Inputs](advanced_topics/subgraph_input_handling.md) for
the detailed body-input rules.

## Shapes and Symbolic Dimensions

Symbolic dimensions enter through user input specs, are preserved in JAX abstract
values, and are tracked in `IRContext` as origins from a source tensor and axis.
Plugins should not cast symbolic dimensions to Python integers.

When an ONNX operator needs a runtime shape value, plugins build it from IR
operations such as:

- `Shape(x)`
- `Gather(shape, axis)`
- `Squeeze` or `Unsqueeze`
- `Concat`
- `Reshape(x, runtime_shape)`

The `dim_as_value` path uses the recorded symbolic-dim origin to materialize a
runtime dimension from the correct tensor instead of guessing from labels alone.

## Optimizer Boundary

The optimizer runs after raw IR lowering and before late attribute overrides.
Its pass order is documented in the [IR Optimizer Guide](advanced_topics/ir_optimizer.md).

Optimizer passes may inspect ONNX op types, but they must remain structural and
IR-only. They may fold redundant casts, transposes, reshapes, constants, or dead
nodes. They must not encode the semantics of a JAX primitive that belongs in a
plugin, and they must preserve graph outputs and ONNX Function signatures.

## Structural Tests and Generated Coverage

Plugin/example metadata drives both generated tests and user-facing coverage
pages. Structural graph checks should live beside the metadata that owns the
behavior:

```python
from jax2onnx.plugins._post_check_onnx_graph import expect_graph as EG
```

Use `expect_graph(...)` to assert durable structure: operator paths, counts,
shapes, attributes, absent operators, unused inputs, and imported ONNX Function
bodies when relevant. It is not a general recursive control-flow body walker;
Loop/If body contracts should be tested directly when body arity or wiring is
the behavior under test.

When lowering structure changes, regenerate a candidate with:

```bash
poetry run python scripts/emit_expect_graph.py <testcase>
```

Then simplify the snippet before storing it in metadata. Generated coverage
tables in the user guide should remain generated artifacts, not handwritten
architecture claims.

## Architectural Guardrails

- Keep converter and plugin lowering code `onnx_ir`-only.
- Keep primitive semantics in plugins, not in `conversion_api`.
- Use `ctx.builder` for normal node emission.
- Treat graph outputs, function inputs, and symbolic-dim origins as public
  contracts during rewrites.
- Do not rely on private `onnx_ir` mirrors unless a documented compatibility
  shim owns the boundary.
- Keep plugin activation scoped to tracing and nested function-body tracing.
- Preserve deterministic module construction in testcase metadata; use
  `construct_and_call(...)` with `with_requested_dtype()`, `with_rng_seed(...)`,
  and related placeholder arguments rather than import-time seeding.

## Failure Modes

| Failure | Where it is caught | Typical fix |
| --- | --- | --- |
| Plugin binding not active during tracing | Missing primitive-specific JAXPR equation or "no plugin registered" error | Ensure the plugin declares the needed binding spec and tracing runs inside plugin activation. |
| Missing plugin for primitive | `lowering_dispatch.get_registered_lowering_plugin` | Register a plugin or rewrite the source expression to an already supported primitive. |
| Plugin reads an unbound input | `assert_eqn_inputs_bound` | Bind constants/inputs earlier or fix nested body input mapping. |
| Plugin does not bind an output | `finalize_eqn_lowering_outputs` | Bind each outvar or return the produced `ir.Value` sequence. |
| Plugin binds a disconnected value | `assert_eqn_outputs_bound` | Reuse the pre-allocated output value or emit a node whose output is the bound value. |
| Function body captures hidden state | Function key / FunctionScope behavior | Make captures explicit as static signature items or dynamic function inputs. |
| Optimizer removes too much | Focused optimizer regression tests | Make the pass more conservative and preserve outputs/signatures explicitly. |

## Maintainer Blueprint

1. Add or update a plugin when behavior depends on JAX primitive semantics.
2. Keep lowering output values connected and metadata stamped.
3. Add or update generated testcase metadata and focused `expect_graph` checks.
4. Use optimizer passes only for shared IR cleanup patterns.
5. Run focused tests first, then strict docs and broader checks for larger
   architecture changes.
