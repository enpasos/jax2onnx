# Work Notes: JAX NN Coverage Checklist

## Scope
- Source list: autosummary entries on `https://docs.jax.dev/en/latest/jax.nn.html`.
- Coverage signal:
  - `jax_doc` metadata in `jax2onnx/plugins/**/*.py`
  - `jaxpr_primitive` registrations in `jax2onnx/plugins/**/*.py`
  - known alias mappings for `jax.nn` public names

## Snapshot (2026-02-28)
- Total docs entries: `35`
- Covered (direct plugin signal): `32`
- Covered (alias/indirect): `2`
- Helper/composite entries: `1`
- Missing dedicated coverage: `0`

## Documented Coverage Adjustments
- `soft_sign` is already covered by [`softsign.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/softsign.py) (component `soft_sign`).
- `hard_silu` is covered indirectly by [`hard_swish.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/hard_swish.py).
- `swish` is covered indirectly by [`silu.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/silu.py).
- `get_scaled_dot_general_config` is a helper/config API; no standalone converter plugin is expected.

## Missing Plugin Queue

### Phase 1: Unary and low-risk ops
- [x] `relu6`
- [x] `hard_tanh`
- [x] `tanh`
- [x] `log_sigmoid`
- [x] `sparse_plus`
- [x] `sparse_sigmoid`
- [x] `squareplus`

### Phase 2: Medium complexity functional ops
- [x] `glu`
- [x] `log1mexp`
- [x] `logmeanexp`

### Phase 3: Advanced attention/quantized ops
- [x] `scaled_dot_general`
- [x] `scaled_matmul`

## Implemented In This Batch
- Added [`relu6.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/relu6.py)
- Added [`hard_tanh.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/hard_tanh.py)
- Added [`tanh.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/tanh.py)
- Added [`log_sigmoid.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/log_sigmoid.py)
- Added [`sparse_plus.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/sparse_plus.py)
- Added [`sparse_sigmoid.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/sparse_sigmoid.py)
- Added [`squareplus.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/squareplus.py)
- Added [`glu.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/glu.py)
- Added [`log1mexp.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/log1mexp.py)
- Added [`logmeanexp.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/logmeanexp.py)
- Added [`scaled_dot_general.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/scaled_dot_general.py)
- Added [`scaled_matmul.py`](/home/enpasos/projects/jax2onnx/jax2onnx/plugins/jax/nn/scaled_matmul.py)

## Implementation Plan Notes
- Prefer `lower_unary_elementwise(...)` for straightforward unary lowers.
- Use `register_jvp_via_jax_jvp(...)` for new primitives unless a custom JVP is required.
- Add metadata testcases per plugin with:
  - one static shape case
  - one dynamic shape case (where meaningful)
  - `post_check_onnx_graph` assertions
- Regenerate generated tests after plugin additions:

```bash
poetry run python scripts/generate_tests.py
```

- Validate incrementally:

```bash
poetry run pytest -q tests/primitives/test_nn.py
```

## Definition of Done Per Operator
- New plugin registered under `context="primitives.nn"`.
- Testcases pass in `tests/primitives/test_nn.py`.
- Coverage notes updated in this file.
- Follow-up: mirror into MkDocs `docs/user_guide/jax_nn_coverage.md`.
