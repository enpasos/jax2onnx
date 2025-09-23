# Migration Status

This file is auto-generated. Run `poetry run python scripts/generate_migration_status.py` to refresh after adding or converting plugins/examples.

## examples.eqx

### MlpExample
- Coverage: ✅ complete
- Legacy contexts: examples2.eqx
- Converter2 contexts: examples2.eqx
- Legacy testcases: mlp_batched_training_mode, mlp_inference_mode, mlp_training_mode
- Converter2 testcases: mlp_batched_training_mode, mlp_inference_mode, mlp_training_mode

### SimpleLinearExample
- Coverage: ✅ complete
- Legacy contexts: examples2.eqx
- Converter2 contexts: examples2.eqx
- Legacy testcases: nn_linear, simple_linear
- Converter2 testcases: nn_linear, simple_linear

## examples.gpt

### GPT
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: gpt

### GPT_Attention
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: gpt_attention

### GPT_CausalSelfAttention
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: causal_self_attention

### GPT_Embeddings
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: gpt_embeddings

### GPT_Head
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: gpt_head

### GPT_MLP
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: gpt_mlp

### GPT_PositionEmbedding
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: position_embedding

### GPT_TokenEmbedding
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: token_embedding

### GPT_TransformerBlock
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: gpt_block

### GPT_TransformerStack
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: transformer_stack

### broadcast_add
- Coverage: ❌ missing
- Legacy contexts: examples.gpt
- Legacy testcases: broadcast_add_dynamic

## examples.jaxfluids

### cfl_timestep
- Coverage: ❌ missing
- Legacy contexts: examples.jaxfluids
- Legacy testcases: cfl_timestep_f64

### weno_reconstruction
- Coverage: ❌ missing
- Legacy contexts: examples.jaxfluids
- Legacy testcases: weno_reconstruction_f64

## examples.jnp

### fori_loop_test
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: fori_loop_test, fori_loop_test_f64

### issue18_abs
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: abs_fn

### issue18_arange
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: arange_fn

### issue18_fori_loop
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: fori_loop_fn

### issue18_linspace
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: linspace_fn

### issue18_scan
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: scan_fn

### issue18_sign
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: sign_fn

### issue18_where
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: where_fn

### issue18_while_loop
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: while_loop_fn

### select_test
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: select_test_all_options, select_test_default_case, select_test_scalar_select_option_0, select_test_scalar_select_option_1, select_test_scalar_select_option_2

### sort_test
- Coverage: ❌ missing
- Legacy contexts: examples.jnp
- Legacy testcases: sort_test_basic

## examples.lax

### cond_scatter_add_mul
- Coverage: ✅ complete
- Legacy contexts: examples2.lax
- Converter2 contexts: examples2.lax
- Legacy testcases: cond_scatter_add_mul_f64_a, cond_scatter_add_mul_f64_b
- Converter2 testcases: cond_scatter_add_mul_f64_a, cond_scatter_add_mul_f64_b

### cond_scatter_repro
- Coverage: ❌ missing
- Legacy contexts: examples.lax
- Legacy testcases: cond_scatter_repro_f64

### remat2
- Coverage: ❌ missing
- Legacy contexts: examples.lax
- Legacy testcases: checkpoint_scalar_f32

### scatter_window
- Coverage: ❌ missing
- Legacy contexts: examples.lax
- Legacy testcases: scatter_window_update_f64_example

## examples.nnx

### AutoEncoder
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: simple_autoencoder

### CNN
- Coverage: ✅ complete
- Legacy contexts: examples2.nnx
- Converter2 contexts: examples2.nnx
- Legacy testcases: simple_cnn, simple_cnn_static
- Converter2 testcases: simple_cnn, simple_cnn_static

### CNN2
- Coverage: -
- Legacy contexts: examples.nnx

### ForiLoop
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: fori_loop_counter

### GRUCell
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: gru_cell_basic

### MLP
- Coverage: ✅ complete
- Legacy contexts: examples2.nnx
- Converter2 contexts: examples2.nnx
- Legacy testcases: simple_mlp, simple_mlp_static, simple_mlp_with_call_params
- Converter2 testcases: simple_mlp, simple_mlp_static, simple_mlp_with_call_params

### MultiHeadAttention
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: multihead_attention_2_nnx, multihead_attention_nn, multihead_attention_nnx

### SequentialReLU
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: sequential_double_relu

### SequentialWithResidual
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: sequential_nested_with_residual

### TransformerDecoderWithSequential
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: tiny_decoder_with_sequential, tiny_decoder_with_sequential_and_full_dynamic_shapes

### TransformerDecoderWithoutSequential
- Coverage: ❌ missing
- Legacy contexts: examples.nnx
- Legacy testcases: tiny_decoder_without_sequential

## examples.onnx_functions

### onnx_functions_000
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 000_one_function_on_outer_layer
- Converter2 testcases: 000_one_function_on_outer_layer

### onnx_functions_001
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 001_one_function_inner
- Converter2 testcases: 001_one_function_inner

### onnx_functions_002
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 002_two_nested_functions
- Converter2 testcases: 002_two_nested_functions

### onnx_functions_003
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 003_two_simple_nested_functions
- Converter2 testcases: 003_two_simple_nested_functions

### onnx_functions_004
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 004_nested_function_plus_component
- Converter2 testcases: 004_nested_function_plus_component

### onnx_functions_005
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 005_nested_function_plus_component
- Converter2 testcases: 005_nested_function_plus_component

### onnx_functions_006
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 006_one_function_outer
- Converter2 testcases: 006_one_function_outer

### onnx_functions_007
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 007_transformer_block
- Converter2 testcases: 007_transformer_block

### onnx_functions_008
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 008_transformer_block
- Converter2 testcases: 008_transformer_block

### onnx_functions_009
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 009_transformer_block
- Converter2 testcases: 009_transformer_block

### onnx_functions_010
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 010_transformer_stack
- Converter2 testcases: 010_transformer_stack

### onnx_functions_012
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 012_vit_conv_embedding
- Converter2 testcases: 012_vit_conv_embedding

### onnx_functions_013
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 013_vit_conv_embedding_with_call_params, 013_vit_conv_embedding_with_internal_call_params
- Converter2 testcases: 013_vit_conv_embedding_with_call_params, 013_vit_conv_embedding_with_internal_call_params

### onnx_functions_014
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 014_one_function_with_input_param_with_default_value, 014_one_function_without_input_param_with_default_value
- Converter2 testcases: 014_one_function_with_input_param_with_default_value, 014_one_function_without_input_param_with_default_value

### onnx_functions_015
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 015_one_function_with_input_param_without_default_value
- Converter2 testcases: 015_one_function_with_input_param_without_default_value

### onnx_functions_016
- Coverage: ✅ complete
- Legacy contexts: examples2.onnx_functions
- Converter2 contexts: examples2.onnx_functions
- Legacy testcases: 016_internal_function_with_input_param_with_default_value
- Converter2 testcases: 016_internal_function_with_input_param_with_default_value

## examples.vit

### ClassificationHead
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: classification_head

### ClassificationHeadFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: classification_head_flat

### ConcatClsToken
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: concat_cls_token

### ConcatClsTokenFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: concat_cls_token_flat

### ConvEmbedding
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: mnist_conv_embedding

### ConvEmbeddingFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: mnist_conv_embedding_flat

### FeedForward
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: feed_forward

### FeedForwardFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: feed_forward_flat

### GetToken
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: get_token

### GetTokenFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: get_token_flat

### PatchEmbedding
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: patch_embedding

### PatchEmbeddingFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: patch_embedding_flat

### PositionalEmbedding
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: positional_embedding

### PositionalEmbeddingFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: positional_embedding_flat

### TransformerBlock
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: transformer_block

### TransformerBlockFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: transformer_block_flat

### TransformerStack
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: transformer_stack

### TransformerStackFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: transformer_stack_flat

### VisionTransformer
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: vit_conv_embedding, vit_patch_embedding

### VisionTransformerFlatten
- Coverage: ❌ missing
- Legacy contexts: examples.vit
- Legacy testcases: vit_conv_embedding_flat, vit_patch_embedding_flat

## primitives.core

### custom_jvp_generic
- Coverage: ✅ complete
- Legacy contexts: primitives.core
- Converter2 contexts: primitives2.core
- Legacy testcases: custom_jvp_square
- Converter2 testcases: custom_jvp_square

### dim_as_value
- Coverage: ✅ complete
- Legacy contexts: primitives.core
- Converter2 contexts: primitives2.core
- Legacy testcases: dim_as_value
- Converter2 testcases: dim_as_value

## primitives.debug

### debug_callback
- Coverage: -
- Legacy contexts: primitives.debug

## primitives.eqx

### dropout
- Coverage: ✅ complete
- Legacy contexts: primitives.eqx
- Converter2 contexts: primitives2.eqx
- Legacy testcases: eqx_dropout_batched_inference, eqx_dropout_dynamic_inference, eqx_dropout_inference_mode, eqx_dropout_training_mode
- Converter2 testcases: eqx_dropout_batched_inference, eqx_dropout_dynamic_inference, eqx_dropout_inference_mode, eqx_dropout_training_mode

### identity
- Coverage: ✅ complete
- Legacy contexts: primitives.eqx
- Converter2 contexts: primitives2.eqx
- Legacy testcases: eqx_identity_static, eqx_identity_symbolic_batch
- Converter2 testcases: eqx_identity_static, eqx_identity_symbolic_batch

### layer_norm
- Coverage: ✅ complete
- Legacy contexts: primitives.eqx
- Converter2 contexts: primitives2.eqx
- Legacy testcases: batched_layer_norm, layer_norm, layer_norm_multiaxis, layer_norm_no_bias_no_scale
- Converter2 testcases: batched_layer_norm, layer_norm, layer_norm_multiaxis, layer_norm_no_bias_no_scale

### linear
- Coverage: ✅ complete
- Legacy contexts: primitives.eqx
- Converter2 contexts: primitives2.eqx
- Legacy testcases: eqx_linear_high_rank, eqx_linear_no_bias_symbolic_batch, eqx_linear_no_bias_vector, eqx_linear_symbolic_batch, eqx_linear_vector
- Converter2 testcases: eqx_linear_high_rank, eqx_linear_no_bias_symbolic_batch, eqx_linear_no_bias_vector, eqx_linear_symbolic_batch, eqx_linear_vector

## primitives.jnp

### add
- Coverage: ⚠️ partial (missing: add)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: add
- Converter2 testcases: jnp_add_broadcast, jnp_add_vector

### arange
- Coverage: ✅ complete
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: arange_data_dependent_indices, arange_float_concrete_input_val, arange_start_stop_concrete_input_val, arange_start_stop_step_concrete_input_val, arange_static_empty_result_neg_step, arange_static_empty_result_pos_step, arange_static_float_step_explicit_dtype, arange_static_float_step_inferred_dtype, arange_static_large_numbers_int, arange_static_negative_step, arange_static_start_equals_stop, arange_static_start_stop_int, arange_static_start_stop_step_int, arange_static_stop_only_float, arange_static_stop_only_int, arange_static_stop_zero, arange_stop_only_concrete_input_val
- Converter2 testcases: arange_data_dependent_indices, arange_float_concrete_input_val, arange_start_stop_concrete_input_val, arange_start_stop_step_concrete_input_val, arange_static_empty_result_neg_step, arange_static_empty_result_pos_step, arange_static_float_step_explicit_dtype, arange_static_float_step_inferred_dtype, arange_static_large_numbers_int, arange_static_negative_step, arange_static_start_equals_stop, arange_static_start_stop_int, arange_static_start_stop_step_int, arange_static_stop_only_float, arange_static_stop_only_int, arange_static_stop_zero, arange_stop_only_concrete_input_val

### clip
- Coverage: ✅ complete
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: clip_broadcast_bounds, clip_f32_scalar_bounds_no_upcast_f64_mode, clip_i32_scalar_bounds, clip_only_lower, clip_only_upper
- Converter2 testcases: clip_broadcast_bounds, clip_f32_scalar_bounds_no_upcast_f64_mode, clip_i32_scalar_bounds, clip_only_lower, clip_only_upper

### concatenate
- Coverage: ⚠️ partial (missing: concatenate_abstract_middle_dim, concatenate_basic, concatenate_mixed_dtypes, concatenate_tile_and_symbolic, concatenate_with_explicit_dtype, concatenate_with_explicit_dtype_casts_inputs)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: concatenate_abstract_middle_dim, concatenate_basic, concatenate_mixed_dtypes, concatenate_tile_and_symbolic, concatenate_with_explicit_dtype, concatenate_with_explicit_dtype_casts_inputs
- Converter2 testcases: jnp_concatenate_basic, jnp_concatenate_dtype

### cumsum
- Coverage: ⚠️ partial (missing: cumsum_axis2_i32, cumsum_axis2_reverse_i32)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: cumsum_axis2_i32, cumsum_axis2_reverse_i32
- Converter2 testcases: jnp_cumsum_axis1, jnp_cumsum_reverse_dtype

### einsum
- Coverage: ⚠️ partial (missing: einsum_attention_logits_batched, einsum_attention_logits_batched_rank_mismatch, einsum_attention_logits_orig, einsum_attention_output_batched, einsum_attention_output_orig, einsum_batch_transpose, einsum_diag, einsum_ellipsis_rank_mismatch, einsum_matrix_matrix, einsum_matrix_vector, einsum_multi_operand, einsum_sum_reduce, einsum_transpose, einsum_vector_dot)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: einsum_attention_logits_batched, einsum_attention_logits_batched_rank_mismatch, einsum_attention_logits_orig, einsum_attention_output_batched, einsum_attention_output_orig, einsum_batch_transpose, einsum_diag, einsum_ellipsis_rank_mismatch, einsum_matrix_matrix, einsum_matrix_vector, einsum_multi_operand, einsum_sum_reduce, einsum_transpose, einsum_vector_dot
- Converter2 testcases: einsum_matmul, einsum_vecdot

### linspace
- Coverage: ⚠️ partial (missing: linspace_static_basic, linspace_static_endpoint_false, linspace_static_int_inputs_default_dtype, linspace_static_num_0, linspace_static_num_1)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: linspace_static_basic, linspace_static_endpoint_false, linspace_static_int_inputs_default_dtype, linspace_static_num_0, linspace_static_num_1
- Converter2 testcases: linspace_basic_f32, linspace_endpoint_false_i32, linspace_num_one, linspace_num_zero

### matmul
- Coverage: ⚠️ partial (missing: matmul_1d, matmul_1d_2d, matmul_2d, matmul_2d_1d, matmul_3d, matmul_dynamic, matmul_dynamic_a)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: matmul_1d, matmul_1d_2d, matmul_2d, matmul_2d_1d, matmul_3d, matmul_dynamic, matmul_dynamic_a
- Converter2 testcases: matmul_basic, matmul_batch, matmul_vector_matrix

### pow
- Coverage: ⚠️ partial (missing: pow_jnp_pow)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: pow_jnp_pow
- Converter2 testcases: jnp_pow_vector

### power
- Coverage: ⚠️ partial (missing: pow_jnp_power)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: pow_jnp_power
- Converter2 testcases: jnp_power_vector

### prod
- Coverage: ⚠️ partial (missing: basic_prod, prod_with_axis, prod_with_keepdims)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: basic_prod, prod_with_axis, prod_with_keepdims
- Converter2 testcases: jnp_prod_axis, jnp_prod_basic, jnp_prod_keepdims

### reshape
- Coverage: ⚠️ partial (missing: reshape_1, reshape_2, reshape_3, reshape_4, reshape_cnn, reshape_from_scalar, reshape_to_scalar, reshape_valid_flatten_trailing, reshape_with_target_shape_from_symbolic_dim_computation)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: reshape_1, reshape_2, reshape_3, reshape_4, reshape_cnn, reshape_from_scalar, reshape_to_scalar, reshape_valid_flatten_trailing, reshape_with_target_shape_from_symbolic_dim_computation
- Converter2 testcases: reshape_basic, reshape_infer, reshape_symbolic_flatten

### select
- Coverage: ⚠️ partial (missing: select_broadcast, select_gpt2_attention_mask, select_simple)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: select_broadcast, select_gpt2_attention_mask, select_simple
- Converter2 testcases: select_basic

### shape
- Coverage: ⚠️ partial (missing: shape_dynamic)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: shape_basic, shape_dynamic
- Converter2 testcases: shape_basic

### sort
- Coverage: ⚠️ partial (missing: sort_1d, sort_2d_axis0)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: sort_1d, sort_2d_axis0
- Converter2 testcases: sort_basic

### split
- Coverage: ⚠️ partial (missing: split_by_indices, split_by_indices_symbolic, split_by_sections)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: split_by_indices, split_by_indices_symbolic, split_by_sections
- Converter2 testcases: split_indices_numpy, split_sections

### squeeze
- Coverage: ⚠️ partial (missing: squeeze_all_dims, squeeze_dynamic_and_negative_axis, squeeze_dynamic_batch, squeeze_multiple_dims, squeeze_negative_axis, squeeze_negative_axis_tuple, squeeze_single_dim, squeeze_vit_output)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: squeeze_all_dims, squeeze_dynamic_and_negative_axis, squeeze_dynamic_batch, squeeze_multiple_dims, squeeze_negative_axis, squeeze_negative_axis_tuple, squeeze_single_dim, squeeze_vit_output
- Converter2 testcases: squeeze_all, squeeze_axis0, squeeze_negative

### stack
- Coverage: ⚠️ partial (missing: stack_axis_0, stack_axis_1, stack_negative_axis, stack_scalars)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: stack_axis_0, stack_axis_1, stack_negative_axis, stack_scalars
- Converter2 testcases: jnp_stack_axis0, jnp_stack_axis1, jnp_stack_negative_axis, jnp_stack_scalars

### take
- Coverage: ✅ complete
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: take_data_dependent_indices
- Converter2 testcases: take_basic_axis1, take_data_dependent_indices

### tile
- Coverage: ⚠️ partial (missing: tile_a, tile_b, tile_c, tile_d, tile_dynamic_input, tile_dynamic_input_static, tile_pad, tile_param_symbolic, tile_repeats, tile_with_symbolic_repeats, tile_with_symbolic_repeats_static)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: tile_a, tile_b, tile_c, tile_d, tile_dynamic_input, tile_dynamic_input_static, tile_pad, tile_param_symbolic, tile_repeats, tile_with_symbolic_repeats, tile_with_symbolic_repeats_static
- Converter2 testcases: jnp_tile_basic, jnp_tile_pad_rank, jnp_tile_scalar_repeats, jnp_tile_symbolic

### transpose
- Coverage: ⚠️ partial (missing: transpose_3d, transpose_4d, transpose_no_axes, transpose_reverse, transpose_square_matrix)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: transpose_3d, transpose_4d, transpose_basic, transpose_high_dim, transpose_no_axes, transpose_reverse, transpose_square_matrix
- Converter2 testcases: transpose_basic, transpose_high_dim, transpose_reverse_default

### unstack
- Coverage: ⚠️ partial (missing: unstack_axis_0, unstack_axis_1, unstack_negative_axis)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: unstack_axis_0, unstack_axis_1, unstack_negative_axis
- Converter2 testcases: unstack_axis0, unstack_axis1

### where
- Coverage: ⚠️ partial (missing: where_A, where_B, where_broadcast, where_dtype_mismatch_f64_vs_i32_promote, where_gpt_mask_scores_literal_else, where_gpt_mask_scores_scalar_else, where_int_condition_cast, where_jax_int_literals_broadcast_f64_mode, where_literal_else_pyfloat, where_multidim_condition_scalar_branches_broadcast, where_simple)
- Legacy contexts: primitives.jnp
- Converter2 contexts: primitives2.jnp
- Legacy testcases: where_A, where_B, where_broadcast, where_dtype_mismatch_f64_vs_i32_promote, where_gpt_mask_scores_literal_else, where_gpt_mask_scores_scalar_else, where_int_condition_cast, where_jax_int_literals_broadcast_f64_mode, where_literal_else_pyfloat, where_multidim_condition_scalar_branches_broadcast, where_simple
- Converter2 testcases: jnp_where_basic, jnp_where_broadcast, jnp_where_scalar_else

## primitives.lax

### abs
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: abs
- Converter2 testcases: abs

### add
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: add, add_const
- Converter2 testcases: add, add_const

### add_any
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: add_any_via_jvp_on_mul
- Converter2 testcases: add_any_via_jvp_on_mul

### and
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: and_bool, and_int
- Converter2 testcases: and_bool, and_int

### argmax
- Coverage: ⚠️ partial (missing: argmax_boolean_input_axis0_specific_values, argmax_boolean_input_axis1_specific_values, argmax_boolean_random_input_axis0, argmax_float_axis0, argmax_float_axis1)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: argmax_boolean_input_axis0_specific_values, argmax_boolean_input_axis1_specific_values, argmax_boolean_random_input_axis0, argmax_float_axis0, argmax_float_axis1
- Converter2 testcases: argmax_axis0, argmax_bool

### argmin
- Coverage: ⚠️ partial (missing: argmin_test1, argmin_test2)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: argmin_test1, argmin_test2
- Converter2 testcases: argmin_axis0, argmin_axis1

### bitwise_not
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: bitwise_not_bool, bitwise_not_i32
- Converter2 testcases: bitwise_not_bool, bitwise_not_i32

### broadcast_in_dim
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: broadcast_in_dim, broadcast_in_dim_2d_to_3d, broadcast_in_dim_batch, broadcast_in_dim_dynamic_B, broadcast_in_dim_scalar
- Converter2 testcases: broadcast_in_dim, broadcast_in_dim_2d_to_3d, broadcast_in_dim_batch, broadcast_in_dim_dynamic_B, broadcast_in_dim_scalar

### clamp
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: clamp_i32_scalar_bounds, clamp_pyint_bounds_promote_to_x_dtype, clamp_scalar_float_bounds_match_x, clamp_vector_bounds_match
- Converter2 testcases: clamp_i32_scalar_bounds, clamp_pyint_bounds_promote_to_x_dtype, clamp_scalar_float_bounds_match_x, clamp_vector_bounds_match

### concatenate
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: concatenate, concatenate_3d, concatenate_axis0, concatenate_axis1, concatenate_internal_int32_then_cast_to_f32_zeroarg
- Converter2 testcases: concatenate, concatenate_3d, concatenate_axis0, concatenate_axis1, concatenate_internal_int32_then_cast_to_f32_zeroarg

### cond
- Coverage: ⚠️ partial (missing: cond_internal_constant_f64, cond_with_scatter)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: cond_internal_constant_f64, cond_multiple_operands_in_tuple, cond_my_new_complex_scenario, cond_nested_conditional, cond_passthrough_identity, cond_scalar, cond_variables, cond_with_scatter
- Converter2 testcases: cond_multiple_operands_in_tuple, cond_my_new_complex_scenario, cond_nested_conditional, cond_passthrough_identity, cond_scalar, cond_variables

### conv
- Coverage: ⚠️ partial (missing: conv, conv2, conv_general_dilated_nhwc_output)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: conv, conv2, conv_general_dilated_nhwc_output
- Converter2 testcases: conv_nchw, conv_nhwc

### convert_element_type
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: convert_element_type
- Converter2 testcases: convert_element_type

### copy
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: copy_float32_array, copy_int64_scalar
- Converter2 testcases: copy_float32_array, copy_int64_scalar

### cos
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: cos
- Converter2 testcases: cos

### cosh
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: cosh
- Converter2 testcases: cosh

### cumsum
- Coverage: ⚠️ partial (missing: cumsum_f32_axism1_reverse, cumsum_i32_axis2)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: cumsum_f32_axism1_reverse, cumsum_i32_axis2
- Converter2 testcases: cumsum_axis2, cumsum_reverse_last_axis

### device_put
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: device_put_array, device_put_scalar
- Converter2 testcases: device_put_array, device_put_scalar

### div
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: div
- Converter2 testcases: div, div_const

### dot_general
- Coverage: ⚠️ partial (missing: dot_general, dot_general_lhs1_rhs1)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: dot_general, dot_general_lhs1_rhs1
- Converter2 testcases: dot_contract_min, dot_contract_nm

### dynamic_slice
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: dynamic_slice_2d, dynamic_slice_3d, dynamic_slice_test1, dynamic_slice_vit_like
- Converter2 testcases: dynamic_slice_2d, dynamic_slice_3d, dynamic_slice_test1, dynamic_slice_vit_like

### dynamic_update_slice
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: dus_1d_block_update, dus_1d_scalar_update, dus_2d_block_update, dus_3d_block_update, dus_4d_block_update
- Converter2 testcases: dus_1d_block_update, dus_1d_scalar_update, dus_2d_block_update, dus_3d_block_update, dus_4d_block_update

### eq
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: eq
- Converter2 testcases: eq

### exp
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: exp
- Converter2 testcases: exp

### fori_loop
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: fori_loop_counter, fori_loop_example, fori_loop_test, fori_loop_test_f64, fori_loop_vector, fori_loop_zero
- Converter2 testcases: fori_loop_counter, fori_loop_example, fori_loop_test, fori_loop_test_f64, fori_loop_vector, fori_loop_zero

### gather
- Coverage: ⚠️ partial (missing: gather_f64_data_i32_indices_cast_and_output_is_f64, gather_f64_data_i64_indices_output_is_f64)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: gather_dynamic_batch_simple_index, gather_f64_data_i32_indices_cast_and_output_is_f64, gather_f64_data_i64_indices_output_is_f64, gather_static, gather_trig_where_pipeline_f64_indices_i32, gather_trig_where_pipeline_f64_indices_i64
- Converter2 testcases: gather_dynamic_batch_simple_index, gather_static, gather_trig_where_pipeline_f64_indices_i32, gather_trig_where_pipeline_f64_indices_i64

### greater_equal
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: greater_equal
- Converter2 testcases: greater_equal

### gt
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: gt
- Converter2 testcases: gt

### integer_pow
- Coverage: ⚠️ partial (missing: integer_pow)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: integer_pow
- Converter2 testcases: integer_pow_square

### iota
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: broadcasted_iota, iota_float32, iota_int32
- Converter2 testcases: broadcasted_iota, iota_float32, iota_int32

### log
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: log
- Converter2 testcases: log

### logistic
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: lax_logistic_basic
- Converter2 testcases: lax_logistic_basic

### lt
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: lt
- Converter2 testcases: lt

### max
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: max
- Converter2 testcases: max

### min
- Coverage: ⚠️ partial (missing: min_test1)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: min_test1
- Converter2 testcases: min

### mul
- Coverage: ⚠️ partial (missing: mul_pyfloat_promotes_to_array_dtype_f64, mul_scalar_broadcast_promote_to_f64, mul_test1, mul_test2)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: mul_pyfloat_promotes_to_array_dtype_f64, mul_scalar_broadcast_promote_to_f64, mul_test1, mul_test2
- Converter2 testcases: mul, mul_const

### ne
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: ne
- Converter2 testcases: ne

### neg
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: neg
- Converter2 testcases: neg

### pad
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: pad_const_1d, pad_const_2d, pad_const_2d_cval, pad_inside_nested_scan_smoke_f64, pad_inside_scan_smoke_f64
- Converter2 testcases: pad_const_1d, pad_const_2d, pad_const_2d_cval, pad_inside_nested_scan_smoke_f64, pad_inside_scan_smoke_f64

### pjit
- Coverage: -
- Converter2 contexts: primitives2.lax

### pow
- Coverage: ⚠️ partial (missing: pow_lax)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: pow_lax
- Converter2 testcases: pow_basic

### reduce_and
- Coverage: ⚠️ partial (missing: reduce_and_all_true, reduce_and_one_false)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_and_all_true, reduce_and_keepdims, reduce_and_one_false
- Converter2 testcases: reduce_and_all_axes, reduce_and_axis0, reduce_and_keepdims

### reduce_max
- Coverage: ⚠️ partial (missing: reduce_max, reduce_max_allaxes)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_max, reduce_max_allaxes, reduce_max_axes_input, reduce_max_keepdims
- Converter2 testcases: reduce_max_all_axes, reduce_max_axes_input, reduce_max_axis0, reduce_max_keepdims

### reduce_min
- Coverage: ⚠️ partial (missing: reduce_min, reduce_min_allaxes)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_min, reduce_min_allaxes, reduce_min_keepdims
- Converter2 testcases: reduce_min_all_axes, reduce_min_axis0, reduce_min_keepdims

### reduce_or
- Coverage: ⚠️ partial (missing: reduce_or_all_false, reduce_or_one_true)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_or_all_false, reduce_or_keepdims, reduce_or_one_true
- Converter2 testcases: reduce_or_all_axes, reduce_or_axis0, reduce_or_keepdims

### reduce_prod
- Coverage: ⚠️ partial (missing: reduce_prod, reduce_prod_allaxes, reduce_prod_dtype, reduce_prod_dtype_f64)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_prod, reduce_prod_allaxes, reduce_prod_dtype, reduce_prod_dtype_f64, reduce_prod_keepdims
- Converter2 testcases: reduce_prod_all_axes, reduce_prod_axis0, reduce_prod_dtype_promotion, reduce_prod_keepdims

### reduce_sum
- Coverage: ⚠️ partial (missing: reduce_sum, reduce_sum_allaxes, reduce_sum_dtype, reduce_sum_dtype_f64)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_sum, reduce_sum_allaxes, reduce_sum_dtype, reduce_sum_dtype_f64, reduce_sum_keepdims
- Converter2 testcases: reduce_sum_all_axes, reduce_sum_axis0, reduce_sum_dtype_promotion, reduce_sum_keepdims

### reduce_xor
- Coverage: ⚠️ partial (missing: reduce_xor_all_false, reduce_xor_one_true, reduce_xor_two_true)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reduce_xor_all_false, reduce_xor_keepdims, reduce_xor_one_true, reduce_xor_two_true
- Converter2 testcases: reduce_xor_all_axes, reduce_xor_axis0, reduce_xor_keepdims

### rem
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: rem_float, rem_float_neg, rem_int, rem_int_neg
- Converter2 testcases: rem_float, rem_float_neg, rem_int, rem_int_neg

### remat2
- Coverage: -
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax

### reshape
- Coverage: ⚠️ partial (missing: reshape_with_target_shape_from_symbolic_dim_computation)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: reshape, reshape_merge_symbolic_with_static_and_check_name, reshape_valid_flatten_trailing, reshape_valid_squeeze_middle_dim_from_problematic_source, reshape_with_inferred_dimension_from_input, reshape_with_inferred_dimension_from_input_dynamic, reshape_with_target_shape_from_symbolic_dim_computation
- Converter2 testcases: reshape, reshape_after_transpose_folds_const_shape, reshape_flatten_trailing_folds_const_shape, reshape_merge_symbolic_with_static_and_check_name, reshape_valid_flatten_trailing, reshape_valid_squeeze_middle_dim_from_problematic_source, reshape_with_inferred_dimension_from_input, reshape_with_inferred_dimension_from_input_dynamic

### rev
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: rev_matrix_axes01, rev_vector
- Converter2 testcases: rev_matrix_axes01, rev_vector

### scan
- Coverage: ⚠️ partial (missing: scan_captured_scalar, scan_captured_scalar_f64, scan_captured_scalar_with_xs, scan_captured_vector_with_xs_f64, scan_carry_only, scan_cumsum, scan_fn, scan_jit_no_xs, scan_jit_no_xs_f64, scan_matrix_carry_multidim_xs, scan_multiple_carry, scan_multiple_sequences, scan_nested_len_mismatch, scan_nested_len_mismatch_f64, scan_no_xs, scan_rank0_sequence_vectorized, scan_rank0_sequence_vectorized_f64, scan_two_diff_lengths, scan_two_diff_lengths_broadcast, scan_two_diff_lengths_broadcast_f64, scan_two_diff_lengths_f64, scan_two_diff_lengths_with_broadcast)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: scan_captured_scalar, scan_captured_scalar_f64, scan_captured_scalar_with_xs, scan_captured_vector_with_xs_f64, scan_carry_only, scan_cumsum, scan_fn, scan_jit_no_xs, scan_jit_no_xs_f64, scan_matrix_carry_multidim_xs, scan_multiple_carry, scan_multiple_sequences, scan_nested_len_mismatch, scan_nested_len_mismatch_f64, scan_no_xs, scan_rank0_sequence_vectorized, scan_rank0_sequence_vectorized_f64, scan_two_diff_lengths, scan_two_diff_lengths_broadcast, scan_two_diff_lengths_broadcast_f64, scan_two_diff_lengths_f64, scan_two_diff_lengths_with_broadcast
- Converter2 testcases: scan_identity_slice_helper

### scatter
- Coverage: ⚠️ partial (missing: scatter_clip_2d_window_at_edge, scatter_correct_axis_determination, scatter_depth2_fp64_type_mismatch, scatter_depth2_mixed_dtypes_fp_mismatch, scatter_depth2_mixed_dtypes_fp_mismatch_f64, scatter_from_user_warning_shapes_valid_jax, scatter_set_axis0, scatter_set_middle, scatter_simple_2d_window_out_of_bounds, scatter_static_slice_set_f64, scatter_updates_slice_needed_axis0, scatter_user_error_scenario_precise, scatter_window_update_depth3_shapes_ok, scatter_window_update_f64)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: scatter_clip_2d_window_at_edge, scatter_correct_axis_determination, scatter_depth2_fp64_type_mismatch, scatter_depth2_mixed_dtypes_fp_mismatch, scatter_depth2_mixed_dtypes_fp_mismatch_f64, scatter_from_user_warning_shapes_valid_jax, scatter_set_axis0, scatter_set_middle, scatter_simple_2d_window_out_of_bounds, scatter_static_slice_set_f64, scatter_updates_slice_needed_axis0, scatter_user_error_scenario_precise, scatter_window_update_depth3_shapes_ok, scatter_window_update_f64
- Converter2 testcases: scatter_set_single, scatter_set_vector

### scatter_add
- Coverage: ⚠️ partial (missing: scatter_add_batch_updates_1d_operand, scatter_add_depth2_depth2_helper_regression, scatter_add_fluids_pattern_updates_5_4_1_1, scatter_add_fp64_dtype_mismatch, scatter_add_in_cond_float64, scatter_add_mismatched_window_dims_from_user_report, scatter_add_mismatched_window_dims_from_user_report2, scatter_add_mismatched_window_dims_from_user_report3, scatter_add_simple_1d, scatter_add_window_2d_operand_1d_indices, scatter_depth2_fp64_type_mismatch)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: scatter_add_batch_updates_1d_operand, scatter_add_depth2_depth2_helper_regression, scatter_add_fluids_pattern_updates_5_4_1_1, scatter_add_fp64_dtype_mismatch, scatter_add_in_cond_float64, scatter_add_mismatched_window_dims_from_user_report, scatter_add_mismatched_window_dims_from_user_report2, scatter_add_mismatched_window_dims_from_user_report3, scatter_add_simple_1d, scatter_add_window_2d_operand_1d_indices, scatter_depth2_fp64_type_mismatch
- Converter2 testcases: scatter_add_scalar, scatter_add_vector

### scatter_max
- Coverage: ⚠️ partial (missing: scatter_max_batch_updates_1d_operand, scatter_max_depth2_helper_regression_fp64, scatter_max_fp64_dtype_path_check, scatter_max_simple_1d, scatter_max_window_2d_operand_1d_indices)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: scatter_max_batch_updates_1d_operand, scatter_max_depth2_helper_regression_fp64, scatter_max_fp64_dtype_path_check, scatter_max_simple_1d, scatter_max_window_2d_operand_1d_indices
- Converter2 testcases: scatter_max_simple

### scatter_min
- Coverage: ⚠️ partial (missing: scatter_min_batch_updates_1d_operand, scatter_min_depth2_helper_regression_fp64, scatter_min_fp64_dtype_path_check, scatter_min_simple_1d, scatter_min_window_2d_operand_1d_indices)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: scatter_min_batch_updates_1d_operand, scatter_min_depth2_helper_regression_fp64, scatter_min_fp64_dtype_path_check, scatter_min_simple_1d, scatter_min_window_2d_operand_1d_indices
- Converter2 testcases: scatter_min_simple

### scatter_mul
- Coverage: ⚠️ partial (missing: scatter_mul_batch_updates_1d_operand, scatter_mul_fluids_pattern_updates_5_4_1_1, scatter_mul_in_cond_float64, scatter_mul_mismatched_window_dims_from_user_report, scatter_mul_mismatched_window_dims_from_user_report2, scatter_mul_mismatched_window_dims_from_user_report3, scatter_mul_simple_1d, scatter_mul_window_2d_operand_1d_indices)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: scatter_mul_batch_updates_1d_operand, scatter_mul_fluids_pattern_updates_5_4_1_1, scatter_mul_in_cond_float64, scatter_mul_mismatched_window_dims_from_user_report, scatter_mul_mismatched_window_dims_from_user_report2, scatter_mul_mismatched_window_dims_from_user_report3, scatter_mul_simple_1d, scatter_mul_window_2d_operand_1d_indices
- Converter2 testcases: scatter_mul_simple

### select
- Coverage: ⚠️ partial (missing: select_simple)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: select_mask_scores_tensor_else, select_simple
- Converter2 testcases: select_basic, select_mask_scores_tensor_else, select_mask_scores_tensor_else_dynamic, select_mask_scores_tensor_else_dynamic_f64, select_mask_scores_tensor_else_f64

### select_n
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: select_n_bool_predicate_scalar_broadcast, select_n_bool_predicate_two_cases_float, select_n_bool_predicate_two_cases_int, select_n_int_indices_four_cases, select_n_int_indices_three_cases
- Converter2 testcases: select_n_bool_predicate_scalar_broadcast, select_n_bool_predicate_two_cases_float, select_n_bool_predicate_two_cases_int, select_n_int_indices_four_cases, select_n_int_indices_three_cases

### sign
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: sign
- Converter2 testcases: sign

### sin
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: sin
- Converter2 testcases: sin

### sinh
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: sinh
- Converter2 testcases: sinh

### slice
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: slice_3d_none_strides, slice_scan_axis_drop, slice_test1
- Converter2 testcases: slice_3d_none_strides, slice_scan_axis_drop, slice_test1

### sort
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: sort_1d, sort_2d
- Converter2 testcases: sort_1d, sort_2d

### split
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: lax_split_equal_parts, lax_split_unequal_parts
- Converter2 testcases: lax_split_equal_parts, lax_split_unequal_parts

### sqrt
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: sqrt
- Converter2 testcases: sqrt

### square
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: square
- Converter2 testcases: square

### squeeze
- Coverage: ⚠️ partial (missing: lax_squeeze_multiple_axes, lax_squeeze_no_op_empty_dims, lax_squeeze_problem_case_input_squeeze_all_dims_explicitly, lax_squeeze_problem_case_input_squeeze_axes_0_2, lax_squeeze_problem_case_input_squeeze_only_axis_0, lax_squeeze_specific_axis_0)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: lax_squeeze_multiple_axes, lax_squeeze_no_op_empty_dims, lax_squeeze_problem_case_input_squeeze_all_dims_explicitly, lax_squeeze_problem_case_input_squeeze_axes_0_2, lax_squeeze_problem_case_input_squeeze_only_axis_0, lax_squeeze_specific_axis_0
- Converter2 testcases: squeeze_all_unit_dims_default, squeeze_single_axis

### stop_gradient
- Coverage: ⚠️ partial (missing: stop_gradient)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: stop_gradient
- Converter2 testcases: stop_gradient_basic

### sub
- Coverage: ⚠️ partial (missing: sub_test1, sub_test2)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: sub_test1, sub_test2
- Converter2 testcases: sub, sub_const

### tanh
- Coverage: ✅ complete
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: tanh
- Converter2 testcases: tanh

### transpose
- Coverage: ⚠️ partial (missing: transpose_basic)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: transpose_basic
- Converter2 testcases: transpose_nhwc_to_nchw

### while_loop
- Coverage: ⚠️ partial (missing: while_loop_4d_and_scalar_state, while_loop_basic, while_loop_captured_tracer, while_loop_closure_topo, while_loop_cnn_scalar_state_bug, while_loop_counter, while_loop_f64, while_loop_mixed_rank, while_loop_multi_state_f32, while_loop_multi_state_f64, while_loop_nnx_repro, while_loop_no_loop_output_reused_as_input, while_loop_renamed_passthrough, while_loop_tracer_passthrough, while_loop_two_state, while_loop_vector, while_loop_with_closure, while_loop_with_scalar_state)
- Legacy contexts: primitives.lax
- Converter2 contexts: primitives2.lax
- Legacy testcases: while_loop_4d_and_scalar_state, while_loop_basic, while_loop_captured_tracer, while_loop_closure_topo, while_loop_cnn_scalar_state_bug, while_loop_counter, while_loop_f64, while_loop_mixed_rank, while_loop_multi_state_f32, while_loop_multi_state_f64, while_loop_nnx_repro, while_loop_no_loop_output_reused_as_input, while_loop_renamed_passthrough, while_loop_tracer_passthrough, while_loop_two_state, while_loop_vector, while_loop_with_closure, while_loop_with_scalar_state
- Converter2 testcases: while_scalar_counter, while_tuple_state

## primitives.nn

### celu
- Coverage: ⚠️ partial (missing: jaxnn_celu, jaxnn_celu_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_celu, jaxnn_celu_1
- Converter2 testcases: jaxnn_celu_alpha_custom, jaxnn_celu_alpha_default

### dot_product_attention
- Coverage: ⚠️ partial (missing: dpa_axis1, dpa_basic, dpa_batch1_seq2, dpa_batch4_seq16, dpa_batch8_seq4, dpa_diff_heads_embed, dpa_float64, dpa_heads1_embed4, dpa_heads8_embed8, dpa_mask_none, dpa_mostly_false, dpa_one_false, dpa_positional_bias_mask, dpa_tiny_mask_all_valid, dpa_tiny_mask_mixed, dpa_with_causal_mask, dpa_with_local_window_mask, dpa_with_padding_mask, dpa_with_tensor_mask)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: dpa_axis1, dpa_basic, dpa_batch1_seq2, dpa_batch4_seq16, dpa_batch8_seq4, dpa_diff_heads_embed, dpa_float64, dpa_heads1_embed4, dpa_heads8_embed8, dpa_mask_none, dpa_mostly_false, dpa_one_false, dpa_positional_bias_mask, dpa_tiny_mask_all_valid, dpa_tiny_mask_mixed, dpa_with_causal_mask, dpa_with_local_window_mask, dpa_with_padding_mask, dpa_with_tensor_mask
- Converter2 testcases: jaxnn_dpa_basic, jaxnn_dpa_causal, jaxnn_dpa_mask

### elu
- Coverage: ⚠️ partial (missing: jaxnn_elu, jaxnn_elu_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_elu, jaxnn_elu_1
- Converter2 testcases: jaxnn_elu_custom_alpha, jaxnn_elu_default

### gelu
- Coverage: ⚠️ partial (missing: jaxnn_gelu, jaxnn_gelu_1, jaxnn_gelu_approx)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_gelu, jaxnn_gelu_1, jaxnn_gelu_approx
- Converter2 testcases: jaxnn_gelu_exact, jaxnn_gelu_tanh

### identity
- Coverage: ⚠️ partial (missing: jaxnn_identity, jaxnn_identity_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_identity, jaxnn_identity_1
- Converter2 testcases: jaxnn_identity_basic, jaxnn_identity_dynamic

### leaky_relu
- Coverage: ⚠️ partial (missing: jaxnn_leaky_relu, jaxnn_leaky_relu_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_leaky_relu, jaxnn_leaky_relu_1
- Converter2 testcases: jaxnn_leaky_relu_custom, jaxnn_leaky_relu_default

### mish
- Coverage: ⚠️ partial (missing: jaxnn_mish, jaxnn_mish_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_mish, jaxnn_mish_1
- Converter2 testcases: jaxnn_mish_basic

### relu
- Coverage: ⚠️ partial (missing: jaxnn_relu, jaxnn_relu_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_relu, jaxnn_relu_1
- Converter2 testcases: jaxnn_relu_basic, jaxnn_relu_dynamic

### selu
- Coverage: ⚠️ partial (missing: jaxnn_selu, jaxnn_selu_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_selu, jaxnn_selu_1
- Converter2 testcases: jaxnn_selu_basic

### sigmoid
- Coverage: ⚠️ partial (missing: jaxnn_sigmoid, jaxnn_sigmoid_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_sigmoid, jaxnn_sigmoid_1
- Converter2 testcases: jaxnn_sigmoid_basic, jaxnn_sigmoid_dynamic

### soft_sign
- Coverage: ⚠️ partial (missing: jaxnn_soft_sign, jaxnn_soft_sign_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_soft_sign, jaxnn_soft_sign_1
- Converter2 testcases: jaxnn_softsign_basic

### softmax
- Coverage: ⚠️ partial (missing: softmax, softmax_2d, softmax_3d)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: softmax, softmax_2d, softmax_3d
- Converter2 testcases: jaxnn_softmax_axis, jaxnn_softmax_default

### softplus
- Coverage: ⚠️ partial (missing: jaxnn_softplus, jaxnn_softplus_1)
- Legacy contexts: primitives.nn
- Converter2 contexts: primitives2.nn
- Legacy testcases: jaxnn_softplus, jaxnn_softplus_1
- Converter2 testcases: jaxnn_softplus_basic

### truncated_normal
- Coverage: ❌ missing
- Legacy contexts: primitives.nn
- Legacy testcases: flax_dense_like_init, initializer, random_truncated_normal_positional

## primitives.nnx

### avg_pool
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: avg_pool, avg_pool_count_include_pad_false, avg_pool_default_padding, avg_pool_same_padding, avg_pool_stride1, avg_pool_stride_none, avg_pool_win3x3_stride2
- Converter2 testcases: avg_pool, avg_pool_count_include_pad_false, avg_pool_default_padding, avg_pool_same_padding, avg_pool_stride1, avg_pool_stride_none, avg_pool_win3x3_stride2

### batch_norm
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: batch_norm_3d, batch_norm_4d, batch_norm_4d_no_bias_no_scale, batch_norm_bias_no_scale, batch_norm_bias_scale, batch_norm_no_bias_no_scale, batch_norm_no_bias_scale
- Converter2 testcases: batch_norm_3d, batch_norm_4d, batch_norm_4d_no_bias_no_scale, batch_norm_bias_no_scale, batch_norm_bias_scale, batch_norm_no_bias_no_scale, batch_norm_no_bias_scale

### conv
- Coverage: ⚠️ partial (missing: conv_1d_large_kernel_on_4d, conv_2d_asymmetric_on_5d, conv_2d_large_dilation, conv_2d_large_stride, conv_2d_mixed_params, conv_2d_same_padding_mixed_dilation, conv_3d_stride)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: conv_1d, conv_1d_complex_on_4d, conv_1d_dilation, conv_1d_group_conv_more_dims, conv_1d_group_on_higher_dim, conv_1d_high_dilation_on_3d, conv_1d_kernel_1, conv_1d_large_kernel, conv_1d_large_kernel_on_4d, conv_1d_more_1d_inputs, conv_1d_more_2d_inputs, conv_1d_same_padding_on_3d, conv_1d_stride_dilation, conv_1d_unit_group_on_multi_dim, conv_1d_wide_input, conv_2d_asymmetric_dilation, conv_2d_asymmetric_kernel, conv_2d_asymmetric_on_5d, conv_2d_asymmetric_stride, conv_2d_complex_on_5d, conv_2d_depthwise, conv_2d_group_conv, conv_2d_group_stride_dilation, conv_2d_kernel_1x1, conv_2d_large_dilation, conv_2d_large_stride, conv_2d_many_channels, conv_2d_mixed_params, conv_2d_same_padding_mixed_dilation, conv_2d_small_input, conv_3d_asymmetric, conv_3d_basic, conv_3d_dilation, conv_3d_group_complex, conv_3d_stride, conv_basic_bias, conv_basic_bias_2, conv_basic_bias_3, conv_different_kernel, conv_float64, conv_large_batch, conv_no_bias, conv_single_batch, conv_stride1, conv_stride2, conv_stride2_bias, conv_valid_padding
- Converter2 testcases: conv_1d, conv_1d_complex_on_4d, conv_1d_dilation, conv_1d_group_conv_more_dims, conv_1d_group_on_higher_dim, conv_1d_high_dilation_on_3d, conv_1d_kernel_1, conv_1d_large_kernel, conv_1d_more_1d_inputs, conv_1d_more_2d_inputs, conv_1d_same_padding_on_3d, conv_1d_stride_dilation, conv_1d_unit_group_on_multi_dim, conv_1d_wide_input, conv_2d_asymmetric_dilation, conv_2d_asymmetric_kernel, conv_2d_asymmetric_stride, conv_2d_complex_on_5d, conv_2d_depthwise, conv_2d_group_conv, conv_2d_group_stride_dilation, conv_2d_kernel_1x1, conv_2d_many_channels, conv_2d_small_input, conv_3d_asymmetric, conv_3d_basic, conv_3d_dilation, conv_3d_group_complex, conv_basic_bias, conv_basic_bias_2, conv_basic_bias_3, conv_different_kernel, conv_float64, conv_large_batch, conv_no_bias, conv_single_batch, conv_stride1, conv_stride2, conv_stride2_bias, conv_valid_padding

### dot_product_attention
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: dpa_basic, dpa_with_bias, dpa_with_causal_mask, dpa_with_mask_and_bias, dpa_with_tensor_mask
- Converter2 testcases: dpa_basic, dpa_with_bias, dpa_with_causal_mask, dpa_with_mask_and_bias, dpa_with_tensor_mask

### dropout
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: dropout_call_params, dropout_init_params
- Converter2 testcases: dropout_call_params, dropout_init_params

### einsum
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: einsum_module_no_bias, einsum_module_with_bias
- Converter2 testcases: einsum_module_no_bias, einsum_module_with_bias

### elu
- Coverage: ⚠️ partial (missing: elu)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: elu
- Converter2 testcases: elu_alpha, elu_default

### embed
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: positional_embedding, token_embedding
- Converter2 testcases: positional_embedding, token_embedding

### gelu
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: gelu, gelu_1, gelu_2, gelu_3
- Converter2 testcases: gelu, gelu_1, gelu_2, gelu_3, gelu_4, gelu_5

### group_norm
- Coverage: ⚠️ partial (missing: group_norm, group_norm_bias_no_scale, group_norm_bias_scale, group_norm_no_bias_no_scale, group_norm_no_bias_scale)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: group_norm, group_norm_bias_no_scale, group_norm_bias_scale, group_norm_no_bias_no_scale, group_norm_no_bias_scale
- Converter2 testcases: group_norm_no_bias, group_norm_no_scale, group_norm_rank2, group_norm_rank4

### layer_norm
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: layer_norm, layer_norm_bias_no_scale, layer_norm_bias_scale, layer_norm_multiaxis, layer_norm_negative_axis_no_div, layer_norm_no_bias_no_scale, layer_norm_no_bias_scale, layer_norm_symbolic_batch
- Converter2 testcases: layer_norm, layer_norm_bias_no_scale, layer_norm_bias_scale, layer_norm_multiaxis, layer_norm_negative_axis_no_div, layer_norm_no_bias_no_scale, layer_norm_no_bias_scale, layer_norm_symbolic_batch, layer_norm_symbolic_batch_seq10_feat3, layer_norm_symbolic_batch_seq10_feat3_2

### leaky_relu
- Coverage: ⚠️ partial (missing: leaky_relu)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: leaky_relu
- Converter2 testcases: leaky_relu_custom, leaky_relu_default

### linear
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: linear_high_rank, linear_high_rank_no_bias, linear_merge_symbolic_dim, linear_no_bias, linear_symbolic_batch
- Converter2 testcases: linear_high_rank, linear_high_rank_no_bias, linear_high_rank_static, linear_merge_symbolic_dim, linear_no_bias, linear_symbolic_batch

### linear_general
- Coverage: ⚠️ partial (missing: dynamic_batch_and_feature_dims)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: dynamic_batch_and_feature_dims, linear_general, linear_general_2, linear_general_3, linear_general_4, linear_general_abstract_eval_axes, linear_general_abstract_eval_axes_pair, linear_general_merge_symbolic_dim
- Converter2 testcases: linear_general, linear_general_2, linear_general_3, linear_general_4, linear_general_abstract_eval_axes, linear_general_abstract_eval_axes_pair, linear_general_dynamic_batch_and_feature_dims, linear_general_merge_symbolic_dim

### log_softmax
- Coverage: ⚠️ partial (missing: log_softmax)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: log_softmax
- Converter2 testcases: log_softmax_axis0, log_softmax_default_axis

### max_pool
- Coverage: ⚠️ partial (missing: max_pool, max_pool_same_padding)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: max_pool, max_pool_same_padding
- Converter2 testcases: max_pool_basic, max_pool_same

### relu
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: relu_1d, relu_4d
- Converter2 testcases: relu_1d, relu_4d

### rms_norm
- Coverage: ⚠️ partial (missing: rms_norm_4d_dynamic, rms_norm_4d_dynamic_no_scale, rms_norm_use_scale_false)
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: rms_norm_4d_dynamic, rms_norm_4d_dynamic_no_scale, rms_norm_basic, rms_norm_use_scale_false
- Converter2 testcases: rms_norm_basic, rms_norm_no_scale, rms_norm_rank4

### sigmoid
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: sigmoid
- Converter2 testcases: sigmoid

### softmax
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: softmax
- Converter2 testcases: softmax

### softplus
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: softplus
- Converter2 testcases: softplus

### tanh
- Coverage: ✅ complete
- Legacy contexts: primitives.nnx
- Converter2 contexts: primitives2.nnx
- Legacy testcases: tanh
- Converter2 testcases: tanh
