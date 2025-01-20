# jax2onnx

## Overview
`jax2onnx` is a library designed to facilitate the conversion of models and functions implemented in JAX/Flax into the ONNX format. This allows seamless integration of JAX/Flax models with other machine learning tools and frameworks that support ONNX.

Supported Models and Functions

| Framework | Component Type     | Component Name             | Status      | Notes                 |
|-----------|--------------------|----------------------------|-------------|-----------------------|
| **Flax**  | Module             | `flax.nnx.Linear`          | ✅ Supported | Converts to ONNX Gemm |
| **Flax**  | Module             | `flax.nnx.Conv`            | ❌ Pending   |                       |
| **Flax**  | Module             | `flax.nnx.ConvTranspose`   | ❌ Pending   |                       |
| **Flax**  | Module             | `flax.nnx.MultiHeadAttention` | ❌ Pending |                       |
| **Flax**  | Module             | `flax.nnx.LayerNorm`       | ❌ Pending   |                       |
| **Flax**  | Module             | `flax.nnx.BatchNorm`       | ❌ Pending   |                       |
| **Flax**  | Module             | `flax.nnx.Dropout`         | ❌ Pending   |                       |
| **JAX**   | Activation Function| `jax.nn.relu`              | ✅ Supported | Converts to ONNX Relu |
| **JAX**   | Activation Function| `jax.nn.sigmoid`           | ❌ Pending   |                       |
| **JAX**   | Activation Function| `jax.nn.tanh`              | ❌ Pending   |  |
| **JAX**   | Activation Function| `jax.nn.softmax`           | ❌ Pending   |  |
| **JAX**   | Activation Function| `jax.nn.gelu`              | ❌ Pending   | |
| **JAX**   | Activation Function| `jax.nn.silu`              | ❌ Pending   |  |
| **JAX**   | Activation Function| `jax.nn.leaky_relu`        | ❌ Pending   |  |

Current Versions
* JAX: 0.4.38
* Flax: 0.10.2
* ONNX: 1.17.0
(Note: for more details look into the `pyproject.toml` file)

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

