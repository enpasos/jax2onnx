# file: jax2onnx/plugins/examples/nnx/fori_loop.py

import jax
from jax2onnx.plugin_system import register_example


def model_fn(x):
    steps = 5

    def body_func(index, args):
        x, counter = args
        x += 0.1 * x**2
        counter += 1
        return (x, counter)

    args = (x, 0)
    y, _ = jax.lax.fori_loop(0, steps, body_func, args)

    return y


register_example(
    component="ForiLoop",
    description="fori_loop example",
    since="v0.5.1",
    context="examples.nnx",
    children=[],
    testcases=[
        {
            "testcase": "fori_loop_counter",
            "callable": lambda x: model_fn(x),
            "input_shapes": [(1,)],
        },
    ],
)


# main function
# def main():
#     x = jnp.array([1.0, 2.0], dtype=jnp.float32)
#     y = model_fn(x)


# if __name__ == "__main__":
#     main()
