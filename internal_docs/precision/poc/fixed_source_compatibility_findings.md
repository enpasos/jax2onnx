# Fixed-source parameter compatibility

Status: read-only compatibility classification, 2026-10-18. All 22 exact raw
`input_params` rows were instantiated with their original factories and dtypes,
exported through public LEGACY, serialized/reloaded and checked on both stacks.
All **44 serialized checks passed in total**, 22 per stack. Another **20 manual
checks passed in total**, ten per stack. This establishes the migration baseline,
not SOURCE support or stochastic numerical equivalence.

## Preserve the actual capabilities

Sixteen rows have a BOOL input reaching a live positive-rate Dropout control.
Five GPT rows have live control ports but zero dropout rates, making the flag
irrelevant to their outputs. One SpectralNorm row supports only the fixed
`update_stats=False` calculation. An input name, Function formal or Identity
sink alone does not establish dynamic behavior.

The exact row mapping below preserves registration origin and raw ordinal;
similarly named examples and the two `dropout_call_params` rows remain distinct.
`dynamic` means an existing live positive-rate control, `zero-rate` means the
flag-insensitive GPT configuration, and `static` means fixed False.

| Registration origin | Ordinal | Testcase | Contract |
| --- | ---: | --- | --- |
| `example::examples.gpt::GPT` | 0 | `gpt` | zero-rate |
| `example::examples.gpt::GPT_Embeddings` | 0 | `gpt_embeddings` | zero-rate |
| `example::examples.gpt::GPT_MLP` | 0 | `gpt_mlp` | zero-rate |
| `example::examples.gpt::GPT_TransformerBlock` | 0 | `gpt_block` | zero-rate |
| `example::examples.gpt::GPT_TransformerStack` | 0 | `gpt_transformer_stack` | zero-rate |
| `example::examples.linen::LinenMLP` | 2 | `simple_linen_mlp_with_call_params` | dynamic |
| `example::examples.nnx::MLP` | 2 | `simple_mlp_with_call_params` | dynamic |
| `example::examples.onnx_functions::onnx_functions_013` | 0 | `013_vit_conv_embedding_with_call_params` | dynamic |
| `example::examples.onnx_functions::onnx_functions_014` | 0 | `014_one_function_with_input_param_with_default_value` | dynamic |
| `example::examples.onnx_functions::onnx_functions_015` | 0 | `015_one_function_with_input_param_without_default_value` | dynamic |
| `example::examples.vit::TransformerBlock` | 0 | `vit_transformer_block` | dynamic |
| `example::examples.vit::TransformerBlockFlatten` | 0 | `vit_transformer_block_flat` | dynamic |
| `example::examples.vit::TransformerStack` | 0 | `vit_transformer_stack` | dynamic |
| `example::examples.vit::TransformerStackFlatten` | 0 | `vit_transformer_stack_flat` | dynamic |
| `example::examples.vit::VisionTransformer` | 0 | `vit_model_conv_embedding` | dynamic |
| `example::examples.vit::VisionTransformer` | 1 | `vit_model_patch_embedding` | dynamic |
| `example::examples.vit::VisionTransformerFlatten` | 0 | `vit_model_conv_embedding_flat` | dynamic |
| `example::examples.vit::VisionTransformerFlatten` | 1 | `vit_model_patch_embedding_flat` | dynamic |
| `plugin::eqx.nn.dropout` | 2 | `eqx_dropout_dynamic_inference` | dynamic; training key prerequisite |
| `plugin::linen.dropout` | 1 | `dropout_call_params` | dynamic |
| `plugin::linen.spectral_norm` | 0 | `spectral_norm_dense` | static |
| `plugin::nnx.dropout` | 1 | `dropout_call_params` | dynamic |

The GPT model and embeddings retain INT32 `[B,1024]` positional inputs; other
positional inputs retain their original FLOAT shapes. BOOL parameters remain
scalar, including the Equinox ndarray's declared type and bytes. Exported `B`
remains symbolic; original typed evaluation used concrete `B=1`. Successful
original leaves are non-weak FP32. The complete shapes, Function signatures,
live control paths and factory recipes are retained in the local JSON evidence.

## Source and state prerequisites

Original unpatched typed evaluation covers concrete True and False on both
stacks, except the independently invalid subcases below. NNX modules are lifted
graph arguments to `nnx.eval_shape`; Linen bridge wrappers retain the underlying
module and RNG. Closing over mutable NNX RNG state in plain `jax.eval_shape`
would be an invalid probe preparation, not a source defect.

Small original Dropout, MLP and 014/015 examples also ran eagerly. Positive-rate
examples changed results between True and False. NNX False calls advance the
existing RNG stream; True consumes no RNG. Linen bridges retain a fixed key,
so repeated False calls can reproduce the same mask. These are distinct source
randomness contracts, not interchangeable numerical references. Large GPT/ViT
numerical forward passes were not run.

- The exact Equinox callable has `key=None`: True works, while False raises
  because training requires a key. An externally supplied key makes the
  unchanged original training call valid. Preserve its dynamic adapter and
  runtime BOOL port; do not replace it with permanently fixed inference.
- SpectralNorm False exports without a runtime parameter. True requires mutable
  `batch_stats` in original JAX and is also rejected by the existing export
  patch. Use `partial(model, update_stats=False)` for the supported fixed source.
  Do not invent a previously supported dynamic state-update capability.

## Whole-model runtime meaning

The manual dependency walk follows actual operands into Functions, lexical
captures into If branches, and Loop formals/carried recurrence. Root-only
Python-if, where and numeric-scale cases are currently static. Forwarded
Function/If/Loop flags reach live Where controls. Unused formals and Identity
sinks do not; omitted defaults and explicit False literals stay local.

A mixed root/Function counterexample demonstrates the migration requirement:
legacy export at True, executed with False, returns `[4,6]` where original False
returns `[24,26]`. A late synthetic flag cannot repair an already specialized
root. Separately, a legacy forwarded numeric scale is incorrectly materialized
as BOOL and fails the checker; numeric parameters must not use a Boolean helper.

C4c must trace one prepared root with real typed runtime arguments, carry those
actual operands through framework adapters and Function/If/Loop scopes, and
derive names/reports from the same input mapping. Existing NNX/Linen/Equinox
Dropout adapters must keep live controls, rather than synthesizing inputs by
name after tracing. Defaults and explicit literals retain their normal meaning.

For fixed sources with direct Python branches, an external `lax.cond` wrapper
can call the complete unchanged source with concrete True/False values. Earlier
bounded evidence covers an actual deterministic wrapper under CPU optimization
off/all; it does not establish arbitrary framework-state or training semantics.
Static binding is appropriate only for intended fixed configuration.

The first implementation packet may cover scalar BOOLs, but all remaining
parameter types and all 22 rows stay in the final acceptance inventory. Full
SOURCE, numerical and stochastic-model acceptance remains open. Whether an
external preparation facility must also transform hardcoded FP32 calculations
to FP64 is still awaiting the user's answer; no such API is assumed here.

## Evidence location

The two stacks used Python 3.12.9/JAX 0.11.2/Flax 0.12.10 and
Python 3.11.12/JAX 0.10.2/Flax 0.12.8, both with Equinox 0.13.8 and bounded CPU
threads. Local evidence is
`/tmp/jax2onnx-c4-fixed-source-classification.{md,json}`, supplemented by
`/tmp/jax2onnx-c4-manual-bool-liveness-{py311,py312}.json` and
`/tmp/jax2onnx-c4-original-bool-prereqs-{py311,py312}.json`.
These temporary probe artifacts are not maintained repository commands.
The [work plan](../precision_implementation_work_plan.md) records implementation
and final acceptance gates; this classification made no production changes.
