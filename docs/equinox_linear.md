# Using Equinox’s `Linear` with Higher-Rank Inputs

Equinox’s neural network modules (including `equinox.nn.Linear`) are designed to operate on a **single input element** (no implicit batch handling). By default, a Linear layer expects an input vector of shape `(in_features,)` and produces an output of shape `(out_features,)`. If your data has extra dimensions (e.g. a batch dimension, time steps, image pixels, etc.), you should **vectorize or reshape** the computation rather than relying on implicit broadcasting. Below are a few common patterns with code examples:

## 1. Batched Inputs via `jax.vmap`

For batched data, the recommended approach is to apply the Linear in a vectorized way using JAX’s `vmap`. This maps the layer over the batch dimension, applying it independently to each sample. For example, given an input array `x` of shape `(batch_size, in_features)`, you can do the following:

```python
import jax
import equinox as eqx

key = jax.random.PRNGKey(0)
linear = eqx.nn.Linear(input_size, output_size, key=key)  # expects (in_features,) input

# Apply to each batch element using vmap:
y = jax.vmap(linear)(x)  # x shape (batch_size, input_size) -> y shape (batch_size, output_size)
```

This usage is indeed the intended pattern. The Equinox FAQ confirms that *“every layer in `eqx.nn` acts on a single batch element, and you can apply them to batches by calling `jax.vmap`”*. The result is that the linear transformation is applied row-by-row to `x`, producing an output of shape `(batch_size, out_features)` as expected.

## 2. Multiple or Nested Dimensions (e.g. Sequence or Spatial Data)

If your input has **more than one extra dimension** (for example, a time-series or image data with shape `(batch, sequence_length, features)` or `(batch, height*width, channels)`), you can *nest* `vmap` or reshape the data. In practice, you might apply one `vmap` for the batch and another for the additional dimension:

* **Nested `vmap` approach:** You can wrap the Linear layer in a second `vmap` to handle a sequence or spatial dimension. For instance, to apply a Linear to every time step of a sequence (length `T`) for each item in a batch, you could do:

  ```python
  linear = eqx.nn.Linear(in_features, out_features, key=key)

  # First, vectorize linear across the sequence dimension (assume input shape (T, in_features))
  apply_to_sequence = jax.vmap(linear)  # now takes input shape (T, in_features) -> (T, out_features)

  # Next, vectorize across the batch dimension:
  Y = jax.vmap(apply_to_sequence)(X)  # X shape (batch, T, in_features) -> Y shape (batch, T, out_features)
  ```

  In words, we first map the linear layer over the sequence axis, then map that *over the batch*. This effectively broadcasts the Linear across both dimensions. Patrick Kidger (Equinox’s author) describes this pattern as “**just wrap it in a second vmap**” for an extra dimension.

* **Reshape approach:** Alternatively, you can reshape your input to collapse multiple dimensions into one, apply the linear, and then reshape back. For example, if `inputs` has shape `(B, P, in_features)` (where `B` = batch size and `P` = number of positions or time steps), you could combine the first two axes, apply the linear across that flattened axis, and then restore the original structure:

  ```python
  B, P, F = inputs.shape  # batch, positions, features
  outputs_flat = jax.vmap(linear)(inputs.reshape(B * P, F))   # shape (B*P, out_features)
  outputs = outputs_flat.reshape(B, P, out_features)         # back to (B, P, out_features)
  ```

  This achieves the same result as nested vmaps in a manual way. (Ensure that the linear’s bias is handled correctly in this case; since Equinox’s Linear doesn’t automatically broadcast the bias across extra dimensions, using `vmap` as above avoids shape mismatches.)

**Example – applying Linear to sequence data:** In a transformer-like architecture, one can apply `vmap` to feed each time step through a linear layer. For instance, an Equinox MLP might apply a linear to each token embedding in a sequence with `jax.vmap`:

```python
# Inside an Equinox MLP module's __call__, given x of shape (T, embed_dim):
x = jax.vmap(self.c_fc)(x)      # c_fc is eqx.nn.Linear; input (T, embed_dim) -> output (T, hidden_dim)
x = jax.vmap(self.swiglu)(x)    # apply activation per time step
x = jax.vmap(self.c_proj)(x)    # another Linear: (T, hidden_dim) -> (T, embed_dim_out)
```

This snippet (from a nanoGPT rewrite in JAX/Equinox) shows three layers each vmapped over a sequence length `T`. Here each `vmap` takes care of mapping a layer over the 0th dimension of `x` (the time dimension), so each token’s feature vector is processed independently, yielding an output of shape `(T, out_features)` for each layer.

## 3. Sequential Processing with `jax.lax.scan`

When you need to apply a Linear layer *iteratively* (e.g. in an RNN or other recurrence), you can use JAX’s `lax.scan` to carry state through time while still using the Linear on single inputs. Inside the scan’s step function, you call the Linear normally on a 1-D input. For example:

```python
linear = eqx.nn.Linear(in_features, out_features, key=key)

def step(carry, x_t):
    y_t = linear(x_t)           # x_t has shape (in_features,)
    return carry, y_t           # y_t is (out_features,)

init_carry = initial_state
carry_out, y_sequence = jax.lax.scan(step, init_carry, sequence_data)
# sequence_data shape: (T, in_features); y_sequence shape: (T, out_features)
```

Each time step `x_t` is a vector of length `in_features`, and the Linear produces a vector of length `out_features`. The `lax.scan` stitches these together so that `y_sequence` has shape `(T, out_features)`. In Equinox, **no special “scan” wrapper is needed** – you can directly use JAX transformations like `vmap` or `scan` with Equinox modules because they are just pure functions/PyTrees. (Unlike some frameworks that require specialized `scan` utilities, Equinox works with JAX’s transforms out-of-the-box.) For instance, the Equinox RNN example uses `jax.lax.scan` to loop over sequence steps with a recurrent cell, and then calls a Linear on the final hidden state. This demonstrates that higher-rank sequence inputs can be processed sequentially, and the Linear layer is applied when needed to a single-step output (which by that point is just a vector).

---

**Summary:** Equinox’s `nn.Linear` expects a 1D input, so **you manage higher-rank inputs by explicitly mapping or reshaping**. Use `jax.vmap` to **batch-apply** the linear over extra axes (batch, time, etc.), even nesting `vmap` for multiple dimensions. Alternatively, reshape your data to 2D (flattening extra dims), apply the linear, and reshape back. And for sequential logic, integrate the Linear inside a `jax.scan` or loop, which handles one step at a time. These approaches ensure the linear transformation is correctly applied across higher-rank tensors without violating Equinox’s design (no implicit batch broadcasting). The result is equivalent to how other frameworks apply dense layers to batches or sequences, but in Equinox you just make the batching explicit with JAX’s functional tools.

**Sources:**

* Equinox documentation – Linear layer usage and FAQ on batch dimensions
* Stack Overflow – Patrick Kidger on using `vmap` (including for sequence dims)
* Example code from a JAX/Equinox GPT implementation – vmapping Linear over sequence
* Equinox RNN training example – using `lax.scan` with an Equinox model and Linear layer
