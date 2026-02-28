# Work Notes: Equinox NN Coverage Checklist

## Scope
- Source list: all `equinox.nn.*` API anchors discovered from: `https://docs.kidger.site/equinox/api/nn/`
- Coverage signal: `jax_doc`, `jaxpr_primitive`, and `component` metadata in `jax2onnx/plugins/**/*.py`.

Regenerate with:

```bash
poetry run python scripts/generate_eqx_nn_coverage.py
```

## Snapshot
- Total Equinox nn API entries: `128`
- Covered (direct Equinox plugin): `40`
- Covered (indirect signal): `0`
- Composite/helper entries: `74`
- Out-of-scope state/inference entries: `14`
- Missing dedicated Equinox coverage: `0`

## Priority Gap Queue
- No missing entries detected by this heuristic.

## Full Checklist
Legend: `covered`, `covered_indirect`, `composite`, `out_of_scope`, `missing`.

| Checklist | Equinox API Entry | Status | Modules (signals) | Notes |
|:--|:--|:--|:--|:--|
| [x] | `equinox.nn.AdaptiveAvgPool1d` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptiveAvgPool1d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveAvgPool1d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveAvgPool2d` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptiveAvgPool2d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveAvgPool2d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveAvgPool3d` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptiveAvgPool3d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveAvgPool3d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveMaxPool1d` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptiveMaxPool1d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveMaxPool1d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveMaxPool2d` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptiveMaxPool2d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveMaxPool2d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveMaxPool3d` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptiveMaxPool3d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptiveMaxPool3d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptivePool` | `covered` | `equinox/eqx/nn/adaptive_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AdaptivePool.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AdaptivePool.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AvgPool1d` | `covered` | `equinox/eqx/nn/avg_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AvgPool1d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AvgPool1d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AvgPool2d` | `covered` | `equinox/eqx/nn/avg_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AvgPool2d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AvgPool2d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AvgPool3d` | `covered` | `equinox/eqx/nn/avg_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.AvgPool3d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.AvgPool3d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.BatchNorm` | `covered` | `equinox/eqx/nn/batch_norm` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.BatchNorm.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.BatchNorm.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Conv` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Conv.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Conv.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Conv1d` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Conv1d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Conv2d` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Conv2d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Conv3d` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Conv3d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.ConvTranspose` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.ConvTranspose.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.ConvTranspose.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.ConvTranspose1d` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.ConvTranspose1d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.ConvTranspose2d` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.ConvTranspose2d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.ConvTranspose3d` | `covered` | `equinox/eqx/nn/conv` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.ConvTranspose3d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Dropout` | `covered` | `equinox/eqx/nn/dropout` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Dropout.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Dropout.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Embedding` | `covered` | `equinox/eqx/nn/embedding` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Embedding.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Embedding.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.GRUCell` | `covered` | `equinox/eqx/nn/recurrent` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.GRUCell.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.GRUCell.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.GroupNorm` | `covered` | `equinox/eqx/nn/group_norm` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.GroupNorm.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.GroupNorm.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Identity` | `covered` | `equinox/eqx/nn/identity` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Identity.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Identity.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.LSTMCell` | `covered` | `equinox/eqx/nn/recurrent` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.LSTMCell.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.LSTMCell.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Lambda` | `covered` | `equinox/eqx/nn/lambda` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Lambda.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Lambda.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.LayerNorm` | `covered` | `equinox/eqx/nn/layer_norm` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.LayerNorm.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.LayerNorm.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Linear` | `covered` | `equinox/eqx/nn/linear` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Linear.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Linear.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MLP` | `covered` | `equinox/eqx/nn/linear` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.MLP.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MLP.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MaxPool1d` | `covered` | `equinox/eqx/nn/max_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.MaxPool1d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MaxPool1d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MaxPool2d` | `covered` | `equinox/eqx/nn/max_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.MaxPool2d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MaxPool2d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MaxPool3d` | `covered` | `equinox/eqx/nn/max_pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.MaxPool3d.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MaxPool3d.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MultiheadAttention` | `covered` | `equinox/eqx/nn/multihead_attention` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.MultiheadAttention.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.MultiheadAttention.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.PReLU` | `covered` | `equinox/eqx/nn/prelu` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.PReLU.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.PReLU.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Pool` | `covered` | `equinox/eqx/nn/pool` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Pool.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Pool.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.RMSNorm` | `covered` | `equinox/eqx/nn/rms_norm` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.RMSNorm.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.RMSNorm.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.RotaryPositionalEmbedding` | `covered` | `equinox/eqx/nn/rotary_positional_embedding` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.RotaryPositionalEmbedding.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.RotaryPositionalEmbedding.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Sequential` | `covered` | `equinox/eqx/nn/sequential` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.Sequential.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Sequential.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.Shared` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.Shared.__call__` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.Shared.__init__` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.SpectralNorm` | `covered` | `equinox/eqx/nn/spectral_norm` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.SpectralNorm.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.SpectralNorm.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.State` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.State.get` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.State.set` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.State.substate` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.State.update` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.StateIndex` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.StateIndex.__init__` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.StatefulLayer` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.StatefulLayer.is_stateful` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.WeightNorm` | `covered` | `equinox/eqx/nn/weight_norm` | Direct Equinox plugin signal (jax_doc/jaxpr/component). |
| [x] | `equinox.nn.WeightNorm.__call__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.WeightNorm.__init__` | `composite` | `-` | Class helper/dunder API; no standalone plugin expected. |
| [x] | `equinox.nn.inference_mode` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |
| [x] | `equinox.nn.make_with_state` | `out_of_scope` | `-` | State/inference helper surface; no standalone converter plugin expected. |

## Next Steps
1. Implement missing Equinox nn entries from the priority queue.
2. Add metadata testcases for each new plugin and regenerate tests (`scripts/generate_tests.py`).
3. Re-run this script after each batch to keep docs and work notes in sync.

