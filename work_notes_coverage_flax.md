# Work Notes: Flax API Coverage Checklist

## Scope
- Source list: code-annotated API entries linked from Flax API index: `https://flax.readthedocs.io/en/latest/api_reference/index.html`
- Coverage signal: `jax_doc`, `jaxpr_primitive`, and `component` metadata in `jax2onnx/plugins/**/*.py`.

Regenerate with:

```bash
poetry run python scripts/generate_flax_api_coverage.py
```

## Snapshot
- Total API entries discovered: `258`
- In-scope neural entries (`flax.linen/*`, `flax.nnx/nn/*`): `75`
- Covered (direct Flax plugin): `40`
- Covered (indirect via lower-level plugins): `12`
- Composite/helper entries: `23`
- Out-of-scope entries: `183`
- Missing dedicated Flax coverage: `0`

## Priority Gap Queue
- No in-scope missing entries currently detected by this heuristic.

## Full Checklist
Legend: `covered`, `covered_indirect`, `composite`, `out_of_scope`, `missing`.

| Checklist | Flax API Entry | Status | Modules (signals) | Notes |
|:--|:--|:--|:--|:--|
| [x] | `flax.core.frozen_dict.FrozenDict` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.FrozenDict.copy` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.FrozenDict.pop` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.FrozenDict.pretty_repr` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.FrozenDict.unfreeze` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.copy` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.freeze` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.pop` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.pretty_repr` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.core.frozen_dict.unfreeze` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.All` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Any` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.BatchNorm` | `covered` | `flax/linen/batch_norm, flax/nnx/batch_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.BatchStat` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Cache` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Carry` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Conv` | `covered` | `flax/linen/conv, flax/nnx/conv` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.ConvTranspose` | `covered` | `flax/linen/conv_transpose` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Data` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Dict` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Dropout` | `covered` | `flax/linen/dropout, flax/nnx/dropout` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Einsum` | `covered` | `flax/linen/einsum, flax/nnx/einsum` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Embed` | `covered` | `flax/linen/embed, flax/nnx/embed` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Everything` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.FlatState` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.GraphDef` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.GroupNorm` | `covered` | `flax/linen/group_norm, flax/nnx/group_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.InstanceNorm` | `covered` | `flax/linen/instance_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Intermediate` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.LayerNorm` | `covered` | `flax/linen/layer_norm, flax/nnx/layer_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Linear` | `covered` | `flax/nnx/linear` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.LinearGeneral` | `covered` | `flax/nnx/linear_general` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.List` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.List.append` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.List.insert` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.LoRA` | `covered` | `flax/nnx/lora` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.LoRALinear` | `covered` | `flax/nnx/lora` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Module` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.eval` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.iter_children` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.iter_modules` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.perturb` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.set_attributes` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.sow` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Module.train` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.MultiHeadAttention` | `covered` | `flax/linen/multi_head_attention` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.Not` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Nothing` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Object` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.OfType` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Param` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.PathContains` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Pytree` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RMSNorm` | `covered` | `flax/linen/rms_norm, flax/nnx/rms_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.RngStream` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.ball` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.bernoulli` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.beta` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.binomial` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.bits` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.categorical` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.cauchy` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.chisquare` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.choice` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.dirichlet` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.double_sided_maxwell` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.exponential` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.f` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.gamma` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.generalized_normal` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.geometric` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.gumbel` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.laplace` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.loggamma` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.logistic` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.lognormal` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.maxwell` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.multinomial` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.multivariate_normal` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.normal` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.orthogonal` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.pareto` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.permutation` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.poisson` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.rademacher` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.randint` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.rayleigh` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.t` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.triangular` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.truncated_normal` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.uniform` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.wald` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.RngStream.weibull_min` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Rngs` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Rngs.__init__` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Sequential` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.SpectralNorm` | `covered` | `flax/linen/spectral_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.State` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Static` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.TrainState` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.TrainState.replace` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.UpdateContext` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Variable` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Variable.del_metadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Variable.get_metadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Variable.has_metadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Variable.set_metadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.Variable.type` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.VariableMetadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.WeightNorm` | `covered` | `flax/linen/weight_norm` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.WithTag` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.__call__` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.add_axis` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.get_partition_spec` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.remove_axis` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.replace` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.replace_boxed` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.to_nnx_variable` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.NNXMeta.unbox` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.ToLinen` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.ToLinen.__call__` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.ToNNX` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.ToNNX.__call__` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.ToNNX.lazy_init` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.bridge.to_linen` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.cached_partial` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.call` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.celu` | `covered_indirect` | `jax/nn/celu` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.check_pytree` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.clone` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.combine_masks` | `covered` | `flax/nnx/combine_masks` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.cond` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.current_update_context` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.custom_vjp` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.data` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.display` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.dot_product_attention` | `covered` | `flax/linen/dot_product_attention, flax/nnx/dot_product_attention` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.elu` | `covered` | `flax/nnx/elu` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.eval_shape` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.filter_state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.filterlib.to_predicate` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.find_duplicates` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.flatten` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.fori_loop` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.fork_rngs` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.from_flat_state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.gelu` | `covered` | `flax/nnx/gelu` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.get_named_sharding` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.get_partition_spec` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.glu` | `covered` | `flax/nnx/glu` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.grad` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.graph` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.graphdef` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.hard_sigmoid` | `covered_indirect` | `jax/nn/hard_sigmoid` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.hard_silu` | `covered_indirect` | `jax/nn/hard_swish` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.hard_swish` | `covered_indirect` | `jax/nn/hard_swish` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.hard_tanh` | `covered` | `flax/nnx/hard_tanh` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.identity` | `covered_indirect` | `equinox/eqx/nn/identity, jax/nn/identity` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.initializers.constant` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.delta_orthogonal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.glorot_normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.glorot_uniform` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.he_normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.he_uniform` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.kaiming_normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.kaiming_uniform` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.lecun_normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.lecun_uniform` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.ones` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.ones_init` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.orthogonal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.truncated_normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.uniform` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.variance_scaling` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.xavier_normal` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.xavier_uniform` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.zeros` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.initializers.zeros_init` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.is_data` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.iter_children` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.iter_graph` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.iter_modules` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.jit` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.jvp` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.leaky_relu` | `covered` | `flax/nnx/leaky_relu` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.log_sigmoid` | `covered` | `flax/nnx/log_sigmoid` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.log_softmax` | `covered` | `flax/nnx/log_softmax` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.logsumexp` | `covered_indirect` | `jax/nn/logsumexp` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.make_attention_mask` | `covered` | `flax/linen/dot_product_attention` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.make_causal_mask` | `covered` | `flax/linen/dot_product_attention` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.map_state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.merge` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.merge_state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.metrics.Accuracy` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.metrics.Average` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.metrics.Metric` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.metrics.MultiMetric` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.metrics.Welford` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.nn.dtypes.canonicalize_dtype` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.nn.dtypes.promote_dtype` | `composite` | `-` | Helper/config API; no standalone ONNX plugin expected. |
| [x] | `flax.nnx.nn.recurrent.Bidirectional` | `covered` | `flax/linen/recurrent` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.nn.recurrent.GRUCell` | `covered` | `flax/linen/recurrent` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.nn.recurrent.LSTMCell` | `covered` | `flax/linen/recurrent` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.nn.recurrent.OptimizedLSTMCell` | `covered` | `flax/linen/recurrent` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.nn.recurrent.RNN` | `covered` | `flax/linen/recurrent` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.nn.recurrent.SimpleCell` | `covered` | `flax/linen/recurrent` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.nn.recurrent.flip_sequences` | `covered` | `flax/nnx/flip_sequences` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.one_hot` | `covered_indirect` | `jax/nn/one_hot` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.optimizer.Optimizer` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.pop` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.pure` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.recursive_map` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.register_data_type` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.register_variable_name` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.relu` | `covered` | `flax/nnx/relu` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.remat` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.replace_by_pure_dict` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.reseed` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.restore_int_paths` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.scan` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.selu` | `covered_indirect` | `jax/nn/selu` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.set_metadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.shard_map` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.sigmoid` | `covered` | `flax/nnx/sigmoid` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.silu` | `covered_indirect` | `jax/nn/silu` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.soft_sign` | `covered_indirect` | `jax/nn/softsign` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.softmax` | `covered` | `flax/nnx/softmax` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.softplus` | `covered` | `flax/nnx/softplus` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.split` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.split_rngs` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.split_state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.standardize` | `covered_indirect` | `jax/nn/standardize` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.static` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.swish` | `covered_indirect` | `jax/nn/silu` | Covered indirectly via lower-level JAX plugin signals. |
| [x] | `flax.nnx.switch` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.tanh` | `covered` | `flax/nnx/tanh` | Direct Flax plugin signal (jax_doc/jaxpr/component). |
| [x] | `flax.nnx.to_flat_state` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.to_pure_dict` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.unflatten` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.update` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.update_context` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.value_and_grad` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.variable_name_from_type` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.variable_type_from_name` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.variables` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.vjp` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.vmap` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.while_loop` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.with_metadata` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.nnx.with_partitioning` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.struct.PyTreeNode` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.struct.dataclass` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.training.train_state.TrainState` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.training.train_state.TrainState.apply_gradients` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |
| [x] | `flax.training.train_state.TrainState.create` | `out_of_scope` | `-` | Outside neural-module surface; no dedicated converter plugin expected. |

## Next Steps
1. Prioritize missing in-scope Flax NN entries from the gap queue.
2. Add metadata testcases for each new plugin and regenerate tests (`scripts/generate_tests.py`).
3. Re-run this script after each plugin batch to keep coverage docs in sync.

