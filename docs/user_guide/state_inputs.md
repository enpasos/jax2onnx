# Model State as Inputs

By default `to_onnx` stores every weight and buffer of a model as an anonymous
constant. With `state=...` you choose arrays that are exported as **named graph
inputs with saved defaults** instead. Each chosen array is stored once, every
use reads that one tensor, and labels such as its path in the model or whether
it is trainable travel with it as tensor metadata.

This is useful when something downstream needs to know which tensor is which
parameter, for example to fine-tune from the file, to swap weights at runtime
or to audit a model. For plain inference nothing changes: the file runs with
its saved defaults.

## Quick start

```python
import re

import equinox as eqx
import jax
import jax.numpy as jnp

from jax2onnx import StateInput, to_onnx

model = eqx.nn.MLP(8, 4, 16, 2, key=jax.random.PRNGKey(0))
keystr = jax.tree_util.keystr

state = [
    StateInput(
        leaf,
        re.sub(r"\W+", "_", keystr(path)).strip("_"),  # layers_0_weight
        metadata={"pkg.example.path": keystr(path), "pkg.example.trainable": "true"},
    )
    for path, leaf in jax.tree_util.tree_leaves_with_path(model)
    if eqx.is_inexact_array(leaf)
]

onnx_model = to_onnx(
    model,
    inputs=[jax.ShapeDtypeStruct((8,), jnp.float32)],
    state=state,
)
```

`StateInput(array, name, metadata={}, overridable=True)` names one array of
the model. jax2onnx does not discover parameters and does not interpret the
metadata: you decide which arrays are state and what the labels mean.

The arrays must be **pytree leaves of the exported callable**, so pass the
model itself:

- **Equinox** and **Flax NNX** modules are pytrees; export `model`, not
  `lambda x: model(x)`.
- For **Flax Linen** or a plain function, hold the parameters in a
  `jax.tree_util.Partial`:

  ```python
  fn = jax.tree_util.Partial(lambda variables, x: module.apply(variables, x), variables)
  ```

Arrays you do not declare stay constants, exactly as before.

## What changes in the file

The computation is the same. Only the interface of the chosen tensors differs.

| | Ordinary export | With `state=` |
| --- | --- | --- |
| Graph inputs | `in_0` | `in_0`, then each state input in the order you declared it |
| A weight | initializer `const_0` | graph input `layers_0_weight` with a same-named initializer as its default |
| Labels | none | `metadata_props` on that initializer |
| A use | `MatMul(in_0, const_0)` | `MatMul(in_0, layers_0_weight)` |

Because state is an input, jax2onnx cannot fold it into other constants.
Reshapes, transposes and broadcasts of a state array therefore stay in the
graph as nodes that read the one stored tensor.

## Running and replacing state

ONNX Runtime treats these inputs as *overridable initializers*: a session runs
without feeding them and uses the saved defaults.

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(onnx_model.SerializeToString())
x = np.zeros((8,), np.float32)

session.run(None, {"in_0": x})  # saved defaults
session.run(None, {"in_0": x, "layers_0_bias": np.ones((16,), np.float32)})
```

`session.get_inputs()` lists only `in_0`; the state inputs are under
`session.get_overridable_initializers()`. ONNX Runtime logs a warning that
such an initializer "will not be treated as constant value/weight". That is
the intended behaviour here. Other runtimes may handle overridable
initializers differently.

## Fixed inference files

ONNX Runtime does not fold or prepack overridable initializers, which costs
inference speed (roughly a quarter to a third in our measurements). When the
weights are final, keep the names and labels but drop the input:

```python
StateInput(leaf, name, metadata=labels, overridable=False)
```

The array is then an ordinary initializer that still carries its name and
metadata, and the file runs as fast as an ordinary export.

## Guarantees

A state export is **the ordinary export of `g(*data, *state)`**, where `g` is
the exported callable with every declared array replaced by an argument. The
arrays are attached afterwards as the defaults of those inputs. Numerics,
dtype rules, control flow and dynamic shapes are those of the ordinary export.
The clauses below follow from this definition or guard its edges. Each one is
enforced by a test that carries its ID.

| ID | Clause |
| --- | --- |
| K1 Entries | `array` is a concrete, non-empty `jax.Array` of a real, integer or boolean dtype: not a NumPy array, Python scalar or tracer. Names are C90 identifiers, unique, and differ from every data input, output and `input_params` name. An array object is declared at most once. Metadata maps non-empty strings to strings; keys under `pkg.jax2onnx.` are reserved. Violations raise before tracing. |
| K2 Location | Each declared array is a pytree leaf of the exported callable, found by object identity. Every leaf position holding that object is replaced, so tied weights become one input and equal-valued but distinct arrays stay separate. An array that is not a leaf raises `StateCaptureError` naming it. |
| K3 State is abstract | While tracing, declared arrays are tracers, so the program cannot read their values. `float(w)`, `np.asarray(w)` and value-dependent Python control flow or shapes raise JAX's error. Work under `jax.ensure_compile_time_eval()` that involves state stays in the graph. No export bakes a value derived from state. |
| K4 No second route | If a declared array also reaches the trace as a captured constant (at the root, in a `cond`, `scan` or `while` scope, or in an ONNX function body), the export raises `StateCaptureError` naming it. |
| K5 Use | Every declared array must feed the exported computation; otherwise the export raises and names the unused entries. With `allow_unused_state=True` they are left out of the file. By K3 and K4 an omitted entry has no influence on the outputs. |
| K6 Interface | `graph.input` lists the data inputs in positional order, then the bound overridable state in declaration order. A state input has the array's dtype and static shape; tracing honours its `weak_type`. With `overridable=False` the entry is an initializer only. |
| K7 Default | Each bound entry has exactly one initializer, named `name`, bit-identical to the array (no cast, no layout change) and attached to the value that every use reads. |
| K8 Metadata | That tensor's `metadata_props` equal the declared mapping exactly; jax2onnx adds nothing. This holds for `return_mode` `proto`, `ir` and `file`, with external data and with `export_mode="web"`. |
| K9 State independence | With the payloads of the state initializers removed, the file depends on the program, the input specifications and each entry's shape, dtype, weak type, name and metadata, and not on state values. Two exports that differ only in state values are byte-identical after clearing those payloads. This rules out derived copies and value-dependent lowering. |
| K10 Semantics | For all data `x` and all overrides `s'` of matching shape and dtype, running the file with `(x, s')` matches `g(x, s')` within the tolerances of an ordinary export. Without overrides the defaults apply. |
| K11 Limits fail loudly | The export raises for an array that an `@onnx_function`-decorated module reads from itself, and for `inputs_as_nchw` indices that address state. State passed to a free `@onnx_function` as an argument works, and so do decorated modules whose own arrays are not declared. |
| K12 Neutral and repeatable | `state=None` and `state=()` produce the bytes of an export without the option. Repeating an export produces identical bytes. |

## What is not promised

- **Training or gradient semantics.** Metadata is opaque caller data, and
  `jax.lax.stop_gradient` is not encoded in the file.
- **Discovery of parameters or framework paths.** The caller lists the arrays.
- **Override support in every runtime.** The behaviour above is that of ONNX
  Runtime.
- **NumPy leaves and BF16.** NumPy arrays are rejected; convert them with
  `jnp.asarray` in the model first. BF16 state follows the
  [known runtime limits](known_limitations.md) and is not part of the test
  corpus.

A lowering that needs a compile-time constant where you supply state fails
with that lowering's own error, because state is not a constant during export.

## How this is tested

[`tests/extra_tests/test_state_inputs_contract.py`](https://github.com/enpasos/jax2onnx/blob/main/tests/extra_tests/test_state_inputs_contract.py)
is the executable form of the table above.

- One generic checker covers K6 to K10. It exports a program and the same
  program with other state values, then compares interface, defaults,
  metadata, both results against JAX, the two files byte for byte (K9), and
  the first file fed with the second program's values (K10).
- The checker runs over a corpus of programs: shared and equal-valued arrays,
  views, convolutions, `cond`, `scan`, `while_loop`, `fori_loop`, nested `jit`,
  `checkpoint`, custom JVP and VJP rules, `vmap`, compile-time evaluation,
  Flax NNX and Linen layers, ONNX functions, and FP16, FP64, INT32, BOOL,
  scalar and weak-typed state.
- Each error in K1 to K5, K8 and K11 has a negative test.
- A policy test keeps one binding point for captured constants, which is where
  K4 is checked, and a meta-test fails when this table and the tests disagree.

The [explicit state walkthrough](explicit_state_export.md) builds the same
file by hand with an Equinox RoPE model and continues with saving, reloading
and an optimizer step. A test checks that `state=` reproduces its file byte
for byte.

## Reference

::: jax2onnx.converter.state_inputs.StateInput

::: jax2onnx.converter.state_inputs.StateCaptureError
