# Handover Task List: MaxText Integration in jax2onnx

**Objective**: Enable MaxText models (Llama2, Gemma, etc.) as supported examples in `jax2onnx` to verify export capabilities.

## 1. Current Status
- **MaxText Source**: Cloned at `tmp/maxtext` (referenced by `jax2onnx/plugins/examples/maxtext/maxtext_models.py`).
- **Dependencies**: 
    - Installed via pip: `flax`, `omegaconf`, `orbax-checkpoint`, `transformers`, `sentencepiece`, `tensorflow-cpu`, `tensorboardX`, `onnx-ir`.
    - Note: `tensorflow-cpu` is required because MaxText uses `tensorboard` and some TF utilities for data loading/logging.
- **Implementation**:
    - Created `jax2onnx/plugins/examples/maxtext/maxtext_models.py`.
    - This file dynamically discovers MaxText configs in `tmp/maxtext/src/MaxText/configs/models` and registers them using `jax2onnx.plugins.plugin_system.register_example`.
    - Configured to use minimal inference settings (batch_size=1, seq_len=32, no checkpointing) to facilitate testing.

## 2. Immediate Next Steps
- [ ] **Run Verification**: Execute the tests to see if the models instantiate and export correctly.
    ```bash
    pytest -v -k maxtext
    ```
- [ ] **Fix Missing Dependencies**: If `ImportError`s persist (e.g. `gcsfs`, `jsonschema`, etc.), install them. MaxText has a heavy dependency footprint.
- [ ] **Debug Export Failures**: 
    - Watch for JAX primitive lowering errors. MaxText uses modern Flax/NNX patterns and custom layers (like `DinoRoPE`, `LayerScale`) that might need custom handlers in `jax2onnx`.
    - If `AbstractEval` errors occur, check `jax2onnx/plugins/examples/maxtext/maxtext_models.py` and potentially add mocks or simplifications to the model config `argv`.

## 3. Future Work
- [ ] **Expand Model Coverage**:
    - Currently `SELECTED_MODELS` in `maxtext_models.py` is limited to `["llama2-7b.yml", "gemma-2b.yml"]`.
    - Once these pass, remove the filter to support all compatible models.
- [ ] **CI Integration**: 
    - `tmp/maxtext` is local. For CI, MaxText should likely be installed as a proper dependency or submodule.
- [ ] **Mocking**: 
    - Considerations for mocking `pyconfig` or `Mesh` creation if hardware dependent errors arise on standard runners.

## 4. Key Files
- `jax2onnx/plugins/examples/maxtext/maxtext_models.py`: Main registration logic.
- `tmp/maxtext/src/MaxText/`: Source code of MaxText.
- `jax2onnx/plugins/plugin_system.py`: Reference for `register_example`.
