# jax2onnx 🌟

`jax2onnx` converts your [JAX](https://docs.jax.dev/), [Flax(nnx)](https://flax.readthedocs.io/en/latest/), [Equinox](https://docs.kidger.site/equinox/) functions directly into the ONNX format.


![img.png](https://enpasos.github.io/jax2onnx/readme/images/jax2onnx.png)

## ✨ Key Features

- **Simple API**  
  Easily convert JAX callables—including Flax (NNX) and Equinox models—into ONNX format using `to_onnx(...)`.

- **Model structure preserved**  
  With `@onnx_function`, submodules appear as named functions in the ONNX graph (e.g. in Netron). Useful for readability and reuse.

- **Dynamic input support**  
  Use abstract dimensions like `'B'` or pass scalars as runtime inputs. Models stay flexible without retracing.

- **Plugin-based extensibility**  
  Add support for new primitives by writing small, local plugins.

- **Netron-friendly outputs**  
  All generated ONNX graphs include shape/type annotations and are structured for clear visualization.
---

## 🚀 Quickstart

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

## 📅 Roadmap and Releases

### **Planned Versions**

* **Ongoing**

  * Expanding coverage of JAX, Flax (NNX) and Equinox components
  * Enhancing support for **physics-based simulations**
 
* **Upcoming**

  * Advanced ONNX function support, including function reuse, optimized internal graph structure, and improved variable naming for clarity and readability



### **Upcoming Version**
 
* **0.9.0**:

  * migrated internally from a [prototype-based ONNX representation](https://github.com/onnx/onnx) to an [IR-based one](https://github.com/onnx/ir-py).
  * added a `return_mode` option in `to_onnx`:

    * `"proto"` (default) → returns an `onnx.ModelProto`
    * `"ir"` → returns the intermediate `onnx_ir.Model`
    * `"file"` → serializes directly to disk *(faster than `proto` + external save)*.
  * updated dependencies: JAX **0.7.2**, Flax **0.12.0** *(requires Python ≥3.11)*, Equinox **0.13.2**, onnx-ir **0.1.10**, onnx **1.19.1**.

 
 
### **Past Versions**

See [`docs/readme/past_versions.md`](docs/readme/past_versions.md) for the full release archive.

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
  **Solution:** Write a [plugin](#how-to-contribute) to handle the unsupported function (this is straightforward!).

---

## 🧩 Supported JAX/ONNX Components


See [`docs/readme/coverage_tables.md`](docs/readme/coverage_tables.md#supported-jaxonnx-components) for the full autogenerated support matrix, including links to every regression testcase.

---

## 🎯 Examples

See [`docs/readme/coverage_tables.md`](docs/readme/coverage_tables.md#examples) for the autogenerated examples catalog.

---

## 📌 Dependencies

**Versions of Major Dependencies:**

| Library       | Versions |  
|:--------------|:---------| 
| `JAX`         | 0.7.2    | 
| `Flax`        | 0.12.0   | 
| `Equinox`     | 0.13.2   | 
| `onnx-ir`     | 0.1.10   | 
| `onnx`        | 1.19.1   |  
| `onnxruntime` | 1.23.1   |  

*Note: For more details, check `pyproject.toml`.*

---

## ⚠️ Limitations

- Currently not all JAX/Flax or Equinox components are supported (you can easily help expand this coverage!).
- Function references need dynamic resolution at call-time.


---

## 🤝 How to Contribute

We warmly welcome contributions!

**How you can help:**

- **Add a plugin:** Extend `jax2onnx` by writing a simple Python file in [`jax2onnx/plugins`](./jax2onnx/plugins): 
a custom primitive or an example. See the [plugin quickstart](docs/design.md#roles--responsibilities) for architecture details and lowering patterns.
- **Follow builder conventions:** The [ONNX IR Builder Guide](docs/dev_guides/onnx_ir_builder.md) covers required `_outputs`/initializer naming rules and the helper script/tests that enforce them.
- **Bug fixes & improvements:** PRs and issues are always welcome.

---

## 💾 Installation

Install from PyPI:

```bash
pip install jax2onnx  
```


---

## 📜 License

This project is licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE) for details.

---

## 🌟 Special Thanks

✨ Special thanks to [@justinchuby](https://github.com/justinchuby) for introducing **onnx-ir** as a scalable and more efficient way to handle ONNX model construction.  

✨ Special thanks for example contributions to [@burakssen](https://github.com/burakssen), [@Cadynum](https://github.com/Cadynum), [@clementpoiret](https://github.com/clementpoiret) and [@PVirie](https://github.com/PVirie)

✨ Special thanks for plugin contributions to [@burakssen](https://github.com/burakssen), [@clementpoiret](https://github.com/clementpoiret) and [@Clouder0](https://github.com/Clouder0)

✨ Special thanks to [tumaer/JAXFLUIDS](https://github.com/tumaer/JAXFLUIDS) for contributing valuable insights rooted in physics simulation use cases.

✨ Special thanks to [@lutzroeder](https://github.com/lutzroeder) for making shapes internal to ONNX function visible in his great Netron viewer.

- [ONNX: Function value_info support #1447](https://github.com/lutzroeder/netron/issues/1447)


✨ Special thanks to the community members involved in:

- [Flax Feature Request #4430](https://github.com/google/flax/issues/4430)
- [JAX Feature Request #26430](https://github.com/jax-ml/jax/issues/26430)

✨ Special thanks to [@limarta](https://github.com/limarta), whose elegant [jaxpr-to-ONNX demonstration](https://gist.github.com/limarta/855a88cc1c0163487a9dc369891147ab) significantly inspired this project.

---

**Happy converting! 🎉**
