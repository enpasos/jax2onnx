# Work Notes – Duplicate Value Names in Full GPT‑OSS Export

## Summary

Attempting to export the full GPT‑OSS 20B checkpoint to ONNX produces an invalid graph: ONNX Runtime rejects `/tmp/gpt_oss_transformer_flax_full.onnx` with `Duplicate definition of name (add_out_5)` (plus `add_out_6`, `select_out_1`, `select_out_2`). Inspection confirms that those four tensor names appear 24 times each—once per transformer block—so the model violates the unique-name requirement before it even reaches ORT.

This is not an ONNX quirk; it is a systematic naming bug in our exporter. The Baseline5 (2-layer) artifact hides the issue because it does not repeat the affected helpers often enough to clash, but the moment we exercise all layers the graph becomes invalid.

## Reproduction

1. **Export the full checkpoint (already staged under `~/.cache/gpt_oss/gpt-oss-20b/`):**
   ```bash
   JAX_PLATFORM_NAME=cpu \
   poetry run python scripts/export_flax_gpt_oss_to_onnx.py \
     --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params.msgpack \
     --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params.config.json \
     --output /tmp/gpt_oss_transformer_flax_full.onnx \
     --sequence-length 32 \
     --skip-validation
   ```
   This emits `/tmp/gpt_oss_transformer_flax_full.onnx` (~795 KB) + `/tmp/gpt_oss_transformer_flax_full.onnx.data` (~78 GB). The warnings about float64→float32 casts are harmless.

2. **Attempt parity / any ORT load:**
   ```bash
   JAX_PLATFORM_NAME=cpu ORT_LOG_SEVERITY_LEVEL=1 \
   poetry run python scripts/run_flax_gpt_oss_onnx.py \
     --prompt "What is the capital of France?" \
     --params ~/.cache/gpt_oss/gpt-oss-20b/flax_params.msgpack \
     --config ~/.cache/gpt_oss/gpt-oss-20b/flax_params.config.json \
     --onnx /tmp/gpt_oss_transformer_flax_full.onnx \
     --sequence-length 32
   ```
   ORT aborts immediately:
   ```
   onnxruntime_pybind11_state.Fail: [ONNXRuntimeError] : 1 : FAIL :
   Load model ... failed: This is an invalid model. Error: Duplicate definition of name (add_out_5).
   ```

3. **Minimal inspection script (pure Python, no ORT) to confirm:**
   ```bash
   poetry run python - <<'PY'
   import onnx
   model = onnx.load("/tmp/gpt_oss_transformer_flax_full.onnx", load_external_data=False)
   counts = {}
   for node in model.graph.node:
       for out in node.output:
           counts[out] = counts.get(out, 0) + 1
   dups = {name: freq for name, freq in counts.items() if freq > 1}
   print(f"duplicate outputs: {len(dups)}")
   for name, freq in sorted(dups.items()):
       print(f"{name}: {freq}")
   PY
   ```
   Output:
   ```
   duplicate outputs: 4
   add_out_5: 24
   add_out_6: 24
   select_out_1: 24
   select_out_2: 24
   ```

4. **Trace producers/consumers to show they are different nodes reusing the same name:**
   ```bash
   poetry run python - <<'PY'
   import onnx
   model = onnx.load("/tmp/gpt_oss_transformer_flax_full.onnx", load_external_data=False)
   targets = {"add_out_5", "add_out_6", "select_out_1", "select_out_2"}
   for name in targets:
       print(f"=== {name} ===")
       for idx, node in enumerate(model.graph.node):
           if name in node.output:
               print("producer", idx, node.name, node.op_type)
           elif name in node.input:
               print("consumer", idx, node.name, node.op_type)
       print()
   PY
   ```
   Each tensor has 24 distinct `Add`/`Where` producers and 24 `GreaterOrEqual`/downstream consumers, proving the duplicate naming is systemic across every block.

## Next Steps for Root-Cause Analysis

- Inspect the lowering path for the rotary/positional helpers (likely `jax.lax.add`, `jax.lax.select`, and the dimension-expression utilities) to see why the same `Var` → `ir.Value` gets rebound with a fixed `name_hint` (`ctx.fresh_name("add_out")`) but never uniquified per block.
- Consider adding a temporary validation pass before serialization that de-duplicates value names (even by suffixing) so large exports can proceed while we design the long-term fix.
- Longer-term, ensure every lowering routine either respects existing producers or calls `ctx.fresh_name(...)` whenever it emits a new node, even if it is processing the same JAX var in different loop iterations/blocks.

This note should give us a reproducible starting point to design a durable solution (e.g., IR-wide name uniquification + targeted plugin fixes). Together with the parity docs, it completes the context needed for architectural discussion.
