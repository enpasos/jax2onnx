import jax.numpy as jnp
import onnx
from jax2onnx import to_onnx
from jax2onnx.plugin_system import onnx_function
from logging_config import configure_logging


configure_logging()


@onnx_function
def fn(x):
    return jnp.squeeze(x, (-1, -3))


def symbolic_batch_dim_is_preserved():
    # Use abstracted axes with a symbolic name "B"

    # Convert the function to ONNX
    model = to_onnx(fn=lambda x: fn(x), inputs=[(1, "B", 1)])

    onnx.save_model(model, "docs/onnx/test_symbolic_dim_name_in_fn.onnx")

    print(model)


# add main
if __name__ == "__main__":
    symbolic_batch_dim_is_preserved()
