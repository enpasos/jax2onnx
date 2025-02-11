# file: jax2onnx/transpose_utils.py


# Conversion functions for JAX and ONNX shape formats
def jax_shape_to_onnx_shape(jax_shape: tuple):
    if len(jax_shape) == 4:
        return tuple(
            [jax_shape[0], jax_shape[3], jax_shape[1], jax_shape[2]]
        )  # (B, C, H, W)
    return tuple(jax_shape)


def onnx_shape_to_jax_shape(onnx_shape: tuple):
    if len(onnx_shape) == 4:
        return tuple(
            [onnx_shape[0], onnx_shape[2], onnx_shape[3], onnx_shape[1]]
        )  # (B, H, W, C)
    return tuple(onnx_shape)
