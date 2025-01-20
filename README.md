# jax2onnx

## Overview
"jax2onnx" is a tool that converts models and functions used in JAX/Flax directly into the ONNX format. 
It does this without using other AI libraries first. This gives you more control to convert JAX/Flax to ONNX. 
You can then use them with other machine learning tools and frameworks that support ONNX. If something is missing, 
you can easily add it yourself — provided it is supported yet by ONNX.

Supported Models and Functions

| Framework    | Component Type                                                                                     | Component Name                                                                                                   | Status       | Notes                                                                           |
|--------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------|---------------------------------------------------------------------------------|
| **[Flax](https://flax.readthedocs.io/en/latest/)** | [Module](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module) | [`flax.nnx.Linear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear) | ✅ Supported  | Converts to [ONNX Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-13) |
| **Flax**     | Module                                                                                             | [`flax.nnx.Conv`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv.html#flax.nnx.Conv)       | ✅ Supported  | Converts to [ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11) |
| **Flax**     | Module                                                                                             | [`flax.nnx.ConvTranspose`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv.html#flax.nnx.ConvTranspose) | ❌ Pending    | Converts to [ONNX ConvTranspose](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html#convtranspose-11) |
| **Flax**     | Module                                                                                             | `flax.nnx.MultiHeadAttention`                                                                                    | ❌ Pending    |                                                                                 |
| **Flax**     | Module                                                                                             | [`flax.nnx.LayerNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm) | ❌ Pending    | Converts to [ONNX LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) |
| **Flax**     | Module                                                                                             | [`flax.nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm) | ❌ Pending    | Converts to [ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) |
| **Flax**     | Module                                                                                             | [`flax.nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/dropout.html#flax.nnx.Dropout) | ❌ Pending    | Converts to [ONNX Dropout](https://onnx.ai/onnx/operators/onnx__Dropout.html)   |
| **JAX**      | Activation Function                                                                                | `jax.nn.relu`                                                                                                    | ✅ Supported  | Converts to [ONNX Relu](https://onnx.ai/onnx/operators/onnx__Relu.html#relu-6)  |
| **JAX**      | Activation Function                                                                                | `jax.nn.sigmoid`                                                                                                 | ✅ Supported  | Converts to [ONNX Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html#sigmoid-6) |
| **JAX**      | Activation Function                                                                                | `jax.nn.tanh`                                                                                                    | ✅ Supported  | Converts to [ONNX Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html#tanh-6)  |
| **JAX**      | Activation Function                                                                                | `jax.nn.softmax`                                                                                                 | ✅ Supported  | Converts to [ONNX Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax-13) |
| **JAX**      | Activation Function                                                                                | `jax.nn.gelu`                                                                                                    | ✅ Supported  | Converts to [ONNX Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html#gelu)    |
| **JAX**      | Activation Function                                                                                | `jax.nn.silu`                                                                                                    | ❌ Pending    | Also known as Swish                                                             |
| **JAX**      | Activation Function                                                                                | `jax.nn.leaky_relu`                                                                                              | ✅ Supported  | Converts to [ONNX LeakyRelu](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html#leakyrelu-6) |
| **JAX**      | Activation Function                                                                                | `jax.nn.celu`                                                                                                    | ✅ Supported  | Converts to [ONNX Celu](https://onnx.ai/onnx/operators/onnx__Celu.html)         |
| **JAX**      | Activation Function                                                                                | `jax.nn.elu`                                                                                                     | ✅ Supported  | Converts to [ONNX Elu](https://onnx.ai/onnx/operators/onnx__Elu.html)           |
| **JAX**      | Activation Function                                                                                | `jax.nn.glu`                                                                                                     | ❌ Pending    | Gated Linear Unit                                                               |
| **JAX**      | Activation Function                                                                                | `jax.nn.hard_sigmoid`                                                                                            | Not in focus  | Converts to [ONNX HardSigmoid](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html) |
| **JAX**      | Activation Function                                                                                | `jax.nn.hard_silu`                                                                                               | Not in focus  |                                                                                 |
| **JAX**      | Activation Function                                                                                | `jax.nn.hard_swish`                                                                                              | Not in focus  |                                                                                 |
| **JAX**      | Activation Function                                                                                | `jax.nn.hard_tanh`                                                                                               | Not in focus  |                                                                                 |
| **JAX**      | Activation Function                                                                                | `jax.nn.log_sigmoid`                                                                                             | ❌ Pending    | Log-Sigmoid Function                                                            |
| **JAX**      | Activation Function                                                                                | `jax.nn.log_softmax`                                                                                             | ❌ Pending    | Log-Softmax Function                                                            |
| **JAX**      | Activation Function                                                                                | `jax.nn.soft_sign`                                                                                               | ✅ Supported  | Converts to [ONNX SoftSign](https://onnx.ai/onnx/operators/onnx__Softsign.html) |
| **JAX**      | Activation Function                                                                                | `jax.nn.softplus`                                                                                                | ✅ Supported  | Converts to [ONNX Softplus](https://onnx.ai/onnx/operators/onnx__Softplus.html) |


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

