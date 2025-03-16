# file: jax2onnx/examples/cnn.py
from flax import nnx


# from https://github.com/google/flax/blob/main/README.md
class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def get_metadata() -> dict:
    """Return test parameters for verifying the ONNX conversion of the CNN model."""
    return {
        "component": "CNN",
        "description": "A simple convolutional neural network (CNN).",
        "since": "v0.1.0",
        "context": "examples.nnx",
        "children": [
            "nnx.Conv",
            "nnx.Linear",
            "nnx.avg_pool",
            "nnx.relu",
            "lax.reshape",
        ],
        "testcases": [
            {
                "testcase": "cnn",
                "callable": CNN(rngs=nnx.Rngs(0)),
                "input_shapes": [(2, 28, 28, 1)],
            }
        ],
    }
