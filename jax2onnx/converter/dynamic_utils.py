import numpy as np

INT64_MAX = np.iinfo(np.int64).max


def encode_dims(seq):
    """Encode a sequence of dims for ONNX: int stays, symbolic becomes INT64_MAX."""
    return np.asarray(
        [d if isinstance(d, int) else INT64_MAX for d in seq], dtype=np.int64
    )
