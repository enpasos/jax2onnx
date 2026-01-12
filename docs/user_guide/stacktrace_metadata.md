# Stacktrace Metadata Levels

When `JAX2ONNX_ENABLE_STACKTRACE_METADATA` is enabled, the converter emits provenance metadata on each ONNX node. The system has two tiers so you can choose between a lightweight overview and a full debug trace.

---

## Level 1 — Lightweight

- `pkg.jax2onnx.callsite` stores the originating user function and line (`function:line`).
- `pkg.jax2onnx.plugin` stores the plugin (or lowering helper) and the line where it emitted the node (`Plugin.lower:line`).

This is the default reduced payload surfaced in tools like Netron:

![Level 1 metadata](https://github.com/user-attachments/assets/87201dae-d91b-45be-ab4c-610080a1acad)

### How to enable

```bash
JAX2ONNX_ENABLE_STACKTRACE_METADATA=1 python -m your_export_script
```

To convert a function inline:

```python
JAX2ONNX_ENABLE_STACKTRACE_METADATA=1 python - <<'PY'
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx

def fn(x):
    return jnp.sin(x)

to_onnx(
    fn,
    [jax.ShapeDtypeStruct((2,), jnp.float32)],
    return_mode="file",
    output_path="fn.onnx",
)
PY
```

Open the resulting ONNX in Netron to see the callsite/plugin metadata per node.

---

## Level 2 — Verbose

Set `JAX2ONNX_STACKTRACE_DETAIL=full` to capture the complete Python stack traces in addition to the Level 1 metadata:

- `pkg.jax2onnx.stacktrace` holds the full Python call stack at the point the node was created.
- `pkg.jax2onnx.jax_traceback` mirrors the JAX equation traceback.

![Level 2 metadata](https://github.com/user-attachments/assets/5e6c027c-e063-4e8d-a541-8195af585562)

### How to enable

```bash
JAX2ONNX_ENABLE_STACKTRACE_METADATA=1 \
JAX2ONNX_STACKTRACE_DETAIL=full \
python -m your_export_script
```

Or inline:

```python
JAX2ONNX_ENABLE_STACKTRACE_METADATA=1 \
JAX2ONNX_STACKTRACE_DETAIL=full \
python - <<'PY'
import jax
import jax.numpy as jnp
from jax2onnx import to_onnx

def fn(x):
    return jnp.sin(x)

to_onnx(
    fn,
    [jax.ShapeDtypeStruct((2,), jnp.float32)],
    return_mode="file",
    output_path="fn_full.onnx",
)
PY
```

The ONNX nodes will now carry both the lightweight metadata and the full Python/JAX trace strings for deep debugging.

Use Level 1 for routine debugging, and switch to Level 2 when you need to drill into the full call flow across user code and plugin lowerings.

---

## Background & Implementation Notes

- Originated from [issue #109](https://github.com/enpasos/jax2onnx/issues/109), requesting stacktrace metadata to help debug exported ONNX graphs.
- Metadata keys live under the `pkg.jax2onnx.*` namespace and are attached per ONNX node during lowering.
- Level 1 keeps exports lightweight (`callsite` + `plugin`), while Level 2 adds full Python/JAX stack dumps.
- Tests/fixtures must be regenerated whenever metadata changes (`JAX2ONNX_ENABLE_STACKTRACE_METADATA=1` during updates) to keep `expect_graph` references in sync.
