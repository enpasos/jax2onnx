# jax2onnx 🌟

`jax2onnx` converts your [JAX](https://docs.jax.dev/), [Flax NNX](https://flax.readthedocs.io/en/latest/), [Equinox](https://docs.kidger.site/equinox/) functions directly into the ONNX format.


![jax2onnx.svg](https://enpasos.github.io/jax2onnx/readme/images/jax2onnx.svg)

## ✨ Key Features

- **simple API**  
  Easily convert JAX callables—including Flax NNX and Equinox models—into ONNX format using `to_onnx(...)`.

- **model structure preserved**  
  With `@onnx_function`, submodules appear as named functions in the ONNX graph (e.g. in Netron). Useful for readability and reuse.

- **dynamic input support**  
  Use abstract dimensions like `'B'` or pass scalars as runtime inputs. Models stay flexible without retracing.

- **plugin-based extensibility**  
  Add support for new primitives by writing small, local plugins.

- **onnx-ir native pipeline**  
  Conversion, optimization, and post-processing all run on the typed `onnx_ir` toolkit—no protobuf juggling—and stay memory-lean before the final ONNX serialization.

- **Netron-friendly outputs**  
  Generated graphs carry shape/type annotations and a clean hierarchy, so tools like Netron stay easy to read.



---

## 🚀 Quickstart

Install and export your first model in minutes:

```bash
pip install jax2onnx
```

Convert your JAX callable to ONNX in just a few lines:

```python
from flax import nnx
from jax2onnx import to_onnx

# Define a simple MLP (from Flax docs)
class MLP(nnx.Module):
    def __init__(self, din, dmid, dout, *, rngs): 
        self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.bn = nnx.BatchNorm(dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dout, rngs=rngs) 
    def __call__(self, x): 
        x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
        return self.linear2(x)

# Instantiate model
my_callable = MLP(din=30, dmid=20, dout=10, rngs=nnx.Rngs(0))

# Export straight to disk without keeping the proto in memory
to_onnx(
    my_callable,
    [("B", 30)],
    return_mode="file",
    output_path="my_callable.onnx",
)
```
 
🔎 See it visualized:  [`my_callable.onnx`](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/my_callable.onnx)

---

## 🧠 ONNX Functions — Minimal Example

ONNX functions help encapsulate reusable subgraphs. Simply use the `@onnx_function` decorator to make your callable an ONNX function.
Just an @onnx_function decorator to make your callable an ONNX function

```python
from flax import nnx
from jax2onnx import onnx_function, to_onnx

# just an @onnx_function decorator to make your callable an ONNX function
@onnx_function
class MLPBlock(nnx.Module):
  def __init__(self, dim, *, rngs):
    self.linear1 = nnx.Linear(dim, dim, rngs=rngs)
    self.linear2 = nnx.Linear(dim, dim, rngs=rngs)
    self.batchnorm = nnx.BatchNorm(dim, rngs=rngs)
  def __call__(self, x):
    return nnx.gelu(self.linear2(self.batchnorm(nnx.gelu(self.linear1(x)))))

# Use it inside another module
class MyModel(nnx.Module):
  def __init__(self, dim, *, rngs):
    self.block1 = MLPBlock(dim, rngs=rngs)
    self.block2 = MLPBlock(dim, rngs=rngs)
  def __call__(self, x):
    return self.block2(self.block1(x))

callable = MyModel(256, rngs=nnx.Rngs(0))
to_onnx(
    callable,
    [(100, 256)],
    return_mode="file",
    output_path="model_with_function.onnx",
)
```

🔎 See it visualized: [`model_with_function.onnx`](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/model_with_function.onnx)

  
---

## SotA examples 🚀 

- Vision: [DINOv3](https://ai.meta.com/dinov3/)
  - Architecture: Equimo’s clean-room Equinox/JAX implementation, following Meta AI’s [DINOv3 paper](https://arxiv.org/abs/2508.10104)
  - Structural graphs:
    - [eqx_dinov3_vit_Ti14 ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_Ti14.onnx)
    - [eqx_dinov3_vit_Ti14_dynamic ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_Ti14_dynamic.onnx)
    - [eqx_dinov3_vit_S14 ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_S14.onnx)
    - [eqx_dinov3_vit_S14_dynamic ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_S14_dynamic.onnx)
    - [eqx_dinov3_vit_B14 ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_B14.onnx)
    - [eqx_dinov3_vit_B14_dynamic ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_B14_dynamic.onnx)
    - [eqx_dinov3_vit_S16 ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_S16.onnx)
    - [eqx_dinov3_vit_S16_dynamic ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_S16_dynamic.onnx)
  - How-to: [Getting Meta weights into jax2onnx](./docs/readme/dinov3/getting_weights.md)
  - Equivalence check: [Comparing Meta vs jax2onnx ONNX](./docs/readme/dinov3/compare_meta_vs_jax2onnx.md)
  - Optional pretrained weights (Meta AI): [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) (other variants live under the same namespace) — DINOv3 license applies; review before downloading or redistributing.

---

## 🧩 Coverage & Examples (Interactive)

> [!TIP]
> **JAX · Flax · Equinox** — explore everything that’s supported **and** see it in action.
>
> - ✅ **Support matrix**: status per component
> - 🧪 **Exact regression testcase** for each entry
> - 🔍 **One-click Netron** graph to inspect nodes, shapes, attributes
> - 🧩 **Examples that compose multiple components** (Conv→Norm→Activation→Pool, MLP w/ LayerNorm+Dropout, `reshape/transpose/concat`, `scan`/`while_loop`, `gather`/`scatter`, …)
>
> **Links:** [Open support matrix ↗](https://enpasos.github.io/jax2onnx/readme/coverage_tables#supported-jaxonnx-components) ·
> [Browse examples ↗](https://enpasos.github.io/jax2onnx/readme/coverage_tables#examples)


---

## 📅 Roadmap and Releases

### **Planned**

  * Expanding coverage of JAX, Flax NNX and Equinox components.
  * Enhancing support for **physics-based simulations**


### **Current Productive Version**

* **0.10.1** *(PyPI)*:

    * **Complex numbers:** unified packed layout (`[..., 2]`), helper stack for packing/dtype reconciliation, and the first wave of plugin coverage: elementwise (`lax`/`jnp` add/sub/mul/div), conjugation (`lax.conj`, `jnp.conj`), bilinear ops (`lax.dot_general`, `jnp.matmul`, `lax.conv_general_dilated`), plus FFTs via an ONNX-compliant DFT lowering. See `docs/dev_guides/complex_numbers.md` for the full playbook (helpers, roadmap, future optimizations).
    * Add stacktrace metadata toggles (`pkg.jax2onnx.callsite` / `pkg.jax2onnx.plugin`) with optional full Python/JAX traces.
    * `lax.dot_general`: add `Einsum` fallback 
    * `lax.broadcast_in_dim`: keep constant folding on handler infrastructure, preserve loop extent metadata, and always emit the Expand node for deterministic IR.
    * `lax.reduce_window_sum`: new Conv-based lowering that handles strides, window dilation, integer operands (via cast wrappers), and static base dilation expansion.

 

### **Past Versions**

See [`past_versions`](https://enpasos.github.io/jax2onnx/readme/past_versions) for the full release archive.
 

---

## ❓ Troubleshooting

If conversion doesn't work out of the box, it could be due to:

- **Non-dynamic function references:**  
  JAXPR-based conversion requires function references to be resolved dynamically at call-time.  
  **Solution:** Wrap your function call inside a lambda to enforce dynamic resolution:
  ```python
  my_dynamic_callable_function = lambda x: original_function(x)
  ```

- **Unsupported primitives:**  
  The callable may use a primitive not yet or not fully supported by `jax2onnx`.  
  **Solution:** Write a [plugin](https://enpasos.github.io/jax2onnx/design#plugin-op-specific) to handle the unsupported function (this is straightforward!).

Looking for provenance details while debugging? Check out the new [Stacktrace Metadata guide](docs/readme/stacktrace_metadata.md).



---

## 🤝 How to Contribute

We warmly welcome contributions!

**How you can help:**

- **Add a plugin:** Extend `jax2onnx` by writing a simple Python file in [`jax2onnx/plugins`](./jax2onnx/plugins):
  a primitive or an example. The [Plugin Quickstart](https://enpasos.github.io/jax2onnx/dev_guides/plugin_quickstart) walks through the process step-by-step.
- **Bug fixes & improvements:** PRs and issues are always welcome.
 


---


## 📌 Dependencies

**Latest supported version of major dependencies:**

| Library       | Versions |  
|:--------------|:---------| 
| `JAX`         | 0.8.0    | 
| `Flax`        | 0.12.0   | 
| `Equinox`     | 0.13.2   | 
| `onnx-ir`     | 0.1.12   | 
| `onnx`        | 1.19.1   |  
| `onnxruntime` | 1.23.2   |  

*For exact pins and extras, see `pyproject.toml`.*


---

## 📜 License

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE) for details.



---

## 🌟 Special Thanks

✨ Special thanks to [@clementpoiret](https://github.com/clementpoiret) for initiating Equinox support and for [Equimo](https://github.com/clementpoiret/equimo), which brings modern vision models—such as [DINOv3](https://ai.meta.com/dinov3/)—to JAX/Equinox.

✨ Special thanks to [@justinchuby](https://github.com/justinchuby) for introducing **onnx-ir** as a scalable and more efficient way to handle ONNX model construction.  

✨ Special thanks for example contributions to [@burakssen](https://github.com/burakssen), [@Cadynum](https://github.com/Cadynum), [@clementpoiret](https://github.com/clementpoiret) and [@PVirie](https://github.com/PVirie)

✨ Special thanks for plugin contributions to [@burakssen](https://github.com/burakssen), [@clementpoiret](https://github.com/clementpoiret), [@Clouder0](https://github.com/Clouder0), [@rakadam](https://github.com/rakadam) and [benmacadam64](https://github.com/benmacadam64)

✨ Special thanks to [@benmacadam64](https://github.com/benmacadam64) for championing the complex-number handling initiative.

✨ Special thanks to [tumaer/JAXFLUIDS](https://github.com/tumaer/JAXFLUIDS) for contributing valuable insights rooted in physics simulation use cases.

✨ Special thanks to [@lutzroeder](https://github.com/lutzroeder) for making shapes internal to ONNX function visible in his great Netron viewer.

- [ONNX: Function value_info support #1447](https://github.com/lutzroeder/netron/issues/1447)


✨ Special thanks to the community members involved in:

- [Flax Feature Request #4430](https://github.com/google/flax/issues/4430)
- [JAX Feature Request #26430](https://github.com/jax-ml/jax/issues/26430)

✨ Special thanks to [@limarta](https://github.com/limarta), whose elegant [jaxpr-to-ONNX demonstration](https://gist.github.com/limarta/855a88cc1c0163487a9dc369891147ab) significantly inspired this project.

---

**Happy converting! 🎉**
