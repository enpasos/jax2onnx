# jax2onnx

## Overview
"jax2onnx" is a tool that converts models and functions used in JAX/Flax directly into the ONNX format. 
It does this without using other AI libraries first. This gives you more control to convert JAX/Flax to ONNX. 
You can then use them with other machine learning tools and frameworks that support ONNX. If something is missing, 
you can easily add it yourself — provided it is supported yet by ONNX.

Supported Models and Functions

| JAX Component                                                                                                   | ONNX Component                                                                                 | Status       |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------|
| [`flax.nnx.Linear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear) | [ONNX Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-13)                            | ✅ Supported  |
| [`flax.nnx.Conv`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv.html#flax.nnx.Conv)       | [ONNX Conv](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11)                            | ✅ Supported  |
| [`flax.nnx.ConvTranspose`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/conv.html#flax.nnx.ConvTranspose) | [ONNX ConvTranspose](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html#convtranspose-11) | ❌ Pending    |
| [`flax.nnx.MultiHeadAttention`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.MultiHeadAttention) | [ONNX Attention](https://onnx.ai/onnx/operators/onnx__Attention.html)                          | ❌ Pending    |
| [`flax.nnx.LayerNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm) | [ONNX LayerNormalization](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html)        | ✅ Supported  |
| [`flax.nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm) | [ONNX BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html)        | ✅ Supported  |
| [`flax.nnx.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/dropout.html#flax.nnx.Dropout) | [ONNX Dropout](https://onnx.ai/onnx/operators/onnx__Dropout.html)                              | ❌ Pending    |
| [`jax.nn.relu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.relu)                                     | [ONNX Relu](https://onnx.ai/onnx/operators/onnx__Relu.html#relu-6)                             | ✅ Supported  |
| [`jax.nn.sigmoid`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.sigmoid)                               | [ONNX Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html#sigmoid-6)                    | ✅ Supported  |
| [`jax.nn.tanh`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.tanh)                                     | [ONNX Tanh](https://onnx.ai/onnx/operators/onnx__Tanh.html#tanh-6)                             | ✅ Supported  |
| [`jax.nn.softmax`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.softmax)                               | [ONNX Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax-13)                   | ✅ Supported  |
| [`jax.nn.gelu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.gelu)                                     | [ONNX Gelu](https://onnx.ai/onnx/operators/onnx__Gelu.html#gelu)                               | ✅ Supported  |
| [`jax.nn.silu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.silu)                                     |                                                                                                | ❌ Pending    |
| [`jax.nn.leaky_relu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.leaky_relu)                         | [ONNX LeakyRelu](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html#leakyrelu-6)              | ✅ Supported  |
| [`jax.nn.celu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.celu)                                     | [ONNX Celu](https://onnx.ai/onnx/operators/onnx__Celu.html)                                    | ✅ Supported  |
| [`jax.nn.elu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.elu)                                       | [ONNX Elu](https://onnx.ai/onnx/operators/onnx__Elu.html)                                      | ✅ Supported  |
| [`jax.nn.glu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.glu)                                       |                                                                                                | ❌ Pending    |
| [`jax.nn.hard_sigmoid`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_sigmoid)                     | [ONNX HardSigmoid](https://onnx.ai/onnx/operators/onnx__HardSigmoid.html)                      | Not in focus |
| [`jax.nn.hard_silu`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_silu)                           |                                                                                                | Not in focus |
| [`jax.nn.hard_swish`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_swish)                         |                                                                                                | Not in focus |
| [`jax.nn.hard_tanh`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.hard_tanh)                           |                                                                                                | Not in focus |
| [`jax.nn.log_sigmoid`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.log_sigmoid)                       | no LogSigmoid on ONNX, using Softplus as a workaround                                          | ✅ Supported   |
| [`jax.nn.log_softmax`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.log_softmax)                       | [ONNX LogSoftmax](https://onnx.ai/onnx/operators/onnx__LogSoftmax.html)                        | ✅ Supported  |
| [`jax.nn.soft_sign`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.soft_sign)                           | [ONNX SoftSign](https://onnx.ai/onnx/operators/onnx__Softsign.html)                            | ✅ Supported  |
| [`jax.nn.softplus`](https://jax.readthedocs.io/en/latest/jax.nn.html#jax.nn.softplus)                             | [ONNX Softplus](https://onnx.ai/onnx/operators/onnx__Softplus.html)                            | ✅ Supported  |
| [`jax.numpy.add`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.add.html)                           | [ONNX Add](https://onnx.ai/onnx/operators/onnx__Add.html)                                      | ✅ Supported  |
| [`jax.numpy.identity`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.identity.html)                 | [ONNX Identity](https://onnx.ai/onnx/operators/onnx__Identity.html)                           | ❌ Pending  |

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

