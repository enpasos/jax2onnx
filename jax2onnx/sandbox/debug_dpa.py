# debug_dpa.py
import jax
import numpy as np
from jax import numpy as jnp
from jax import nn


@jax.jit
def debug_dpa(q, k, v, mask):
    # 1) Q·Kᵀ / √H
    logits = jnp.einsum("btnh,bsnh->bnts", q, k) / jnp.sqrt(q.shape[-1])
    jax.debug.print("logits:\n{}", logits)

    # 2) mask out invalid → −1e9
    not_mask = ~mask
    additive = not_mask.astype(q.dtype) * -1e9
    masked_logits = logits + additive
    jax.debug.print("masked_logits:\n{}", masked_logits)

    # 3) softmax
    weights = nn.softmax(masked_logits, axis=-1)
    jax.debug.print("weights:\n{}", weights)

    # 4) weights·V
    out = jnp.einsum("bnts,bsnh->btnh", weights, v)
    jax.debug.print("output:\n{}", out)

    return out


# --- test inputs (now with the right shapes) ---
Q = np.array([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=np.float32)  # (1,1,1,4)
K = np.array(
    [
        [  # batch index 0
            [[1.0, 0.0, 0.0, 0.0]],  # time–key 0
            [[0.0, 1.0, 0.0, 0.0]],  # time–key 1
        ]
    ],
    dtype=np.float32,
)  # (1,2,1,4)
V = np.array(
    [[[[10.0, 20.0, 30.0, 40.0]], [[50.0, 60.0, 70.0, 80.0]]]], dtype=np.float32
)  # (1,2,1,4)
M = np.array([[[[True, False]]]], dtype=bool)  # (1,1,1,2)

# run & block so we see the prints
_ = debug_dpa(Q, K, V, M).block_until_ready()
