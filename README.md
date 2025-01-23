# jax2onnx

## Overview
"jax2onnx" is a tool that converts models and functions used in JAX/Flax directly into the ONNX format. 
It does this without using other AI libraries first. This gives you more control to convert JAX/Flax to ONNX. 
You can then use them with other machine learning tools and frameworks that support ONNX. If something is missing, 
you can easily add it yourself — provided it is supported yet by ONNX.

Supported and Planned Models and Functions

| JAX Component                                                                                                                               | ONNX Component                                                                       | From   | v0.1.0 |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------ | ------ |
| [`flax.nnx.Linear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear)                            | [`Gemm`](https://onnx.ai/onnx/operators/onnx__Gemm.html)                             | v0.1.0 | ✅      |
| [`flax.nnx.Conv`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv.html#flax.nnx.Conv)                                  | [`Conv`](https://onnx.ai/onnx/operators/onnx__Conv.html)                             | v0.1.0 | ✅      |
| [`flax.nnx.ConvTranspose`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv.html#flax.nnx.ConvTranspose)                | [`ConvTranspose`](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html)           | v0.1.0 | ❌      |
| [`flax.nnx.MultiHeadAttention`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.MultiHeadAttention) | [`Attention`](https://onnx.ai/onnx/operators/onnx__Attention.html)                   | v0.1.0 | ❌      |
| [`flax.nnx.LayerNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm)               | [`LayerNormalization`](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html) | v0.1.0 | ✅      |
| [`flax.nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm)               | [`BatchNormalization`](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html) | v0.1.0 | ✅      |
| [`flax.nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/dropout.html#flax.nnx.Dropout)                         | [`Dropout`](https://onnx.ai/onnx/operators/onnx__Dropout.html)                       | v0.1.0 | ❌      |
| [`jax.nn.relu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.relu)                                                               | [`Relu`](https://onnx.ai/onnx/operators/onnx__Relu.html)                             | v0.1.0 | ✅      |
| [`jax.nn.sigmoid`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.sigmoid)                                                         | [`Sigmoid`](https://onnx.ai/onnx/operators/onnx__Sigmoid.html)                       | v0.1.0 | ✅      |
| [`jax.nn.tanh`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.tanh)                                                               | [`Tanh`](https://onnx.ai/onnx/operators/onnx__Tanh.html)                             | v0.1.0 | ✅      |
| [`jax.nn.softmax`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.softmax)                                                         | [`Softmax`](https://onnx.ai/onnx/operators/onnx__Softmax.html)                       | v0.1.0 | ✅      |
| [`jax.nn.gelu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.gelu)                                                               | [`Gelu`](https://onnx.ai/onnx/operators/onnx__Gelu.html)                             | v0.1.0 | ✅      |
| [`jax.nn.silu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.silu)                                                               | N/A                                                                                  | v0.1.0 | ❌      |
| [`jax.nn.leaky_relu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.leaky_relu)                                                   | [`LeakyRelu`](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html)                   | v0.1.0 | ✅      |
| [`jax.nn.celu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.celu)                                                               | [`Celu`](https://onnx.ai/onnx/operators/onnx__Celu.html)                             | v0.1.0 | ✅      |
| [`jax.nn.elu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.elu)                                                                 | [`Elu`](https://onnx.ai/onnx/operators/onnx__Elu.html)                               | v0.1.0 | ✅      |
| [`jax.nn.glu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.glu)                                                                 | N/A                                                                                  | v0.1.0 | ❌      |
| [`jax.nn.hard_sigmoid`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_sigmoid)                                               | [`HardSigmoid`](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html)               | v0.1.0 | ➖      |
| [`jax.nn.hard_silu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_silu)                                                     | N/A                                                                                  | v0.1.0 | ➖      |
| [`jax.nn.hard_swish`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_swish)                                                   | N/A                                                                                  | v0.1.0 | ➖      |
| [`jax.nn.hard_tanh`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_tanh)                                                     | N/A                                                                                  | v0.1.0 | ➖      |
| [`jax.nn.log_sigmoid`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.log_sigmoid)                                                 | N/A                                                                                  | v0.1.0 | ✅      |
| [`jax.nn.log_softmax`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.log_softmax)                                                 | [`LogSoftmax`](https://onnx.ai/onnx/operators/onnx__LogSoftmax.html)                 | v0.1.0 | ✅      |
✅ = implemented with unit test for eval\
❌ = planned but not implemented, yet\
➖ = not planned for the current planned version

Examples

| Component Name | Description                                   | From   | v0.1.0 |
 |----------------|-----------------------------------------------|--------|--------|
| `MLP`          | Linear and Relu                               | v0.1.0 |    ✅    |
| `MNIST - CNN`  | CNN based MNIST classification                | v0.1.0 |     ❌   |
| `MNIST - ViT`  | Vision Transformer based MNIST classification | v0.1.0 |     ❌   |




Versions of Major Dependencies

| Library       | jax2onnx v0.1.0 | 
|---------------|-----------------| 
| `JAX`         | v0.4.38         | 
| `Flax`        | v0.10.2         | 
| `onnx`        | v1.17.0         |  
| `onnxruntime` | v1.20.1         |  

onnxruntime

Note: for more details look into the `pyproject.toml` file


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

