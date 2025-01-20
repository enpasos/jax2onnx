# jax2onnx

## Overview
"jax2onnx" is a tool that converts models and functions used in JAX/Flax directly into the ONNX format. 
It does this without using other AI libraries first. This gives you more control to convert JAX/Flax to ONNX. 
You can then use them with other machine learning tools and frameworks that support ONNX. If something is missing, 
you can easily add it yourself — provided it is supported yet by ONNX.

Supported Models and Functions

| Framework | Component Type      | Component Name             | Status                          | Notes                              |
|-----------|---------------------|----------------------------|----------------------------------|------------------------------------|
| **Flax**  | Module              | `flax.nnx.Linear`          | ✅ Supported                     | Converts to ONNX Gemm              |
| **Flax**  | Module              | `flax.nnx.Conv`            | ❌ Pending                       |                                    |
| **Flax**  | Module              | `flax.nnx.ConvTranspose`   | ❌ Pending                       |                                    |
| **Flax**  | Module              | `flax.nnx.MultiHeadAttention` | ❌ Pending                    |                                    |
| **Flax**  | Module              | `flax.nnx.LayerNorm`       | ❌ Pending                       |                                    |
| **Flax**  | Module              | `flax.nnx.BatchNorm`       | ❌ Pending                       |                                    |
| **Flax**  | Module              | `flax.nnx.Dropout`         | ❌ Pending                       |                                    |
| **JAX**   | Activation Function | `jax.nn.relu`              | ✅ Supported                     | Converts to ONNX Relu              |
| **JAX**   | Activation Function | `jax.nn.sigmoid`           | ✅ Supported                     | Converts to ONNX Sigmoid           |
| **JAX**   | Activation Function | `jax.nn.tanh`              | ✅ Supported                     | Converts to ONNX Tanh              |
| **JAX**   | Activation Function | `jax.nn.softmax`           | ✅ Supported                     | Converts to ONNX Softmax           |
| **JAX**   | Activation Function | `jax.nn.gelu`              | ✅ Supported                     | Converts to ONNX Gelu              |
| **JAX**   | Activation Function | `jax.nn.silu`              | ❌ Pending                       | Also known as Swish                |
| **JAX**   | Activation Function | `jax.nn.leaky_relu`        | ✅ Supported                     | Converts to ONNX LeakyRelu         |
| **JAX**   | Activation Function | `jax.nn.celu`              | ✅ Supported                     | Converts to ONNX Celu              |
| **JAX**   | Activation Function | `jax.nn.elu`               | ✅ Supported                     | Converts to ONNX Elu               |
| **JAX**   | Activation Function | `jax.nn.glu`               | ❌ Pending                       | Gated Linear Unit                  |
| **JAX**   | Activation Function | `jax.nn.hard_sigmoid`      | ⚠️ Implemented, runs but fails | Fails error tolerance during testing |
| **JAX**   | Activation Function | `jax.nn.hard_silu`         | ❌ Pending                       | Hard SiLU (Swish)                  |
| **JAX**   | Activation Function | `jax.nn.hard_swish`        | ❌ Pending                       | Alias for Hard SiLU                |
| **JAX**   | Activation Function | `jax.nn.hard_tanh`         | ❌ Pending                       | Hard Tanh Function                 |
| **JAX**   | Activation Function | `jax.nn.log_sigmoid`       | ❌ Pending                       | Log-Sigmoid Function               |
| **JAX**   | Activation Function | `jax.nn.log_softmax`       | ❌ Pending                       | Log-Softmax Function               |
| **JAX**   | Activation Function | `jax.nn.soft_sign`         | ✅ Supported                     | Converts to ONNX Softsign          |
| **JAX**   | Activation Function | `jax.nn.softplus`          | ✅ Supported                     | Converts to ONNX Softplus          |


Current Versions
* JAX: 0.4.38
* Flax: 0.10.2
* ONNX: 1.17.0

Note: for more details look into the `pyproject.toml` file

Examples

 | Component Name | Status      | Notes           |
 |----------------|-------------|-----------------|
 | `MLP`          | ✅ Supported | Linear and Relu |

## How to Contribute

If you'd like to see a model or function supported, consider contributing by adding a plugin for an existing   
module or function under the `jax2onnx/plugins` directory. Or you can add an example to the `examples` directory. 
Certainly any other improvements are welcome as well.

## Installation

To install `jax2onnx`, use the following command (t.b.d.):

```bash
pip install jax2onnx  
```

## Usage
t.b.d.
 

## License

This project is licensed under the terms of the MIT license. See the `LICENSE` file for details.

