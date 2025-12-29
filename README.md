# jax2onnx 🌟

[![CI](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml/badge.svg)](https://github.com/enpasos/jax2onnx/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/jax2onnx.svg)](https://pypi.org/project/jax2onnx/)

`jax2onnx` converts your [JAX](https://docs.jax.dev/),  [Flax NNX](https://flax.readthedocs.io/en/latest/), [Flax Linen](https://flax.readthedocs.io/en/latest/api_reference/flax.linen.html), [Equinox](https://docs.kidger.site/equinox/) functions directly into the ONNX format.


![jax2onnx.svg](https://enpasos.github.io/jax2onnx/readme/images/jax2onnx.svg)

## ✨ Key Features

- **simple API**  
  Easily convert JAX callables—including Flax NNX, Flax Linen and Equinox models—into ONNX format using `to_onnx(...)`.

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

- Language: [GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) (open-source MoE Transformer)
  - Architecture: Flax/NNX + Equinox reference stacks with gating/routing capture, MoE MLP rebuilds, and deterministic ONNX exporters (see `jax2onnx/plugins/examples/nnx/gpt_oss_flax.py` and `jax2onnx/plugins/examples/eqx/gpt_oss.py`).
  - Structural graph:
    - [gpt_oss_transformer_flax ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/nnx_gpt_oss/gpt_oss_transformer_flax.onnx)
  - How-to: [Getting GPT-OSS weights into jax2onnx](./docs/readme/gpt_oss/getting_weights.md)
  - Equivalence check: [Routing parity harness](scripts/gpt_oss_routing_parity.py) · [Flax parity tests](tests/extra_tests/test_flax_routing_parity.py) · [Equinox parity tests](tests/extra_tests/test_eqx_gpt_oss_parity.py)
  - Optional pretrained weights: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) · [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) *(weights and model cards list `license: apache-2.0`)*

- Vision: [DINOv3](https://ai.meta.com/dinov3/)
  - Architecture: Equimo’s clean-room Equinox/JAX implementation, following Meta AI’s [DINOv3 paper](https://arxiv.org/abs/2508.10104). Flax/NNX parity modules now live under `jax2onnx/plugins/examples/nnx/dinov3.py` (randomly initialised example stack for IR-only exports).
  - Structural graphs (selected examples):
    - [eqx_dinov3_vit_Ti14 ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_Ti14.onnx)
    - [eqx_dinov3_vit_Ti14_dynamic ↗](https://netron.app/?url=https://huggingface.co/enpasos/jax2onnx-models/resolve/main/examples/eqx_dino/eqx_dinov3_vit_Ti14_dynamic.onnx)
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

  * Broaden coverage of JAX, Flax NNX/Linen, and Equinox components.
  * Expand SotA example support for vision and language models.
  * Improve support for **physics-based simulations**



### **Current Productive Version**

* **0.11.0**:
  * Initial Flax Linen support: core layers (Dense/DenseGeneral, Conv/ConvTranspose/ConvLocal, pooling, BatchNorm/LayerNorm/GroupNorm/RMSNorm/InstanceNorm), Dropout, Einsum/Embed, spectral/weight norm wrappers, activation coverage (GELU plus glu/hard_*/log_*/relu6/silu-swish/tanh/normalize/one_hot), attention stack (dot_product_attention, dot_product_attention_weights, make_attention_mask/make_causal_mask, SelfAttention, MultiHeadDotProductAttention, MultiHeadAttention), recurrent stack (SimpleCell, GRUCell, MGUCell, LSTMCell, OptimizedLSTMCell, ConvLSTMCell, RNN, Bidirectional), and Linen examples (MLP/CNN/Sequential).
  * Modernized IR optimization pipeline: standard onnx_ir CSE pass adoption, removed legacy helpers/getattr patterns, and simplified tests with direct graph iteration.


 
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
| `JAX`         | 0.8.2    | 
| `Flax`        | 0.12.2   | 
| `Equinox`     | 0.13.2   | 
| `onnx-ir`     | 0.1.13   | 
| `onnx`        | 1.20.0   |  
| `onnxruntime` | 1.23.2   |  

*For exact pins and extras, see `pyproject.toml`.*


---

## 📜 License

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE) for details.



---

## 🌟 Special Thanks

✨ Special thanks to [@clementpoiret](https://github.com/clementpoiret) for initiating Equinox support and for [Equimo](https://github.com/clementpoiret/equimo), which brings modern vision models—such as [DINOv3](https://ai.meta.com/dinov3/)—to JAX/Equinox.

✨ Special thanks to [@justinchuby](https://github.com/justinchuby) for introducing **onnx-ir** as a scalable and more efficient way to handle ONNX model construction.  

✨ Special thanks to [@atveit](https://github.com/atveit) for introducing us to [gpt-oss-jax-vs-torch-numerical-comparison](https://github.com/atveit/gpt-oss-jax-vs-torch-numerical-comparison).

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
