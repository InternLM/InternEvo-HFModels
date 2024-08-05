# Adapting HuggingFace Models for InternEvo Packed Training and ISP

## Background

When HuggingFace models are being integrated with the InternEvo framework, we want packed training and ISP be supproted to:
1. Improve GPU computation utilization (reduce wasting computation on meaningless padded tokens)
2. Support training with long sequences

This requires adapting the models to support:
1. Packed training
2. ISP (Intern Sequence Parallelism) training

## Supporting Packed Training

### For modeling_internlm.py

Step 1. Obtain `cu_seqlens` and `max_seqlen` from gpc for the current batch.

```python
use_packed_dataset = gpc.config.data.get("use_packed_dataset", False)

if use_packed_dataset:
    assert bsz == 1, "hidden_states should be packed into bsz=1 when use_packed_dataset=True"
    cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
    max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
```

Step 2. For `use_packed_dataset=True`, reuse InternEvo's `apply_rotary_emb`. Otherwise, use the original code logic.

```python
if use_packed_dataset:
    cos, sin = self.rotary_emb(value_states, max_seqlen)
    cos = cos[position_ids].squeeze(0)
    sin = sin[position_ids].squeeze(0)
    assert sin.shape == cos.shape, "cos and sin must have the same shape"
    _, rotary_dim = cos.shape
    rotary_dim_half = rotary_dim // 2
    cos_half = cos[:q_len, :rotary_dim_half]
    sin_half = sin[:q_len, :rotary_dim_half]
    query_states = apply_rotary_emb(query_states, cos_half, sin_half)
    key_states = apply_rotary_emb(key_states, cos_half, sin_half) 
```

Step 3. Pass `cu_seqlens` and `max_seqlen` to flash attention's varlen kernel for variable-length attention calculation.

```python   
if use_packed_dataset:
    attn_output = flash_attn_varlen_func(
        query_states.flatten(0, 1),
        key_states.flatten(0, 1),
        value_states.flatten(0, 1),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        0.0,
        softmax_scale=None,
        causal=True,
        return_attn_probs=False,
    ).unsqueeze(0)
```


### For modeling_internlm2.py

1. Obtain `cu_seqlens` and `max_seqlen` from gpc for the current batch.

```python
use_packed_dataset = gpc.config.data.get("use_packed_dataset", False)

if use_packed_dataset:
    assert bsz == 1, "hidden_states should be packed into bsz=1 when use_packed_dataset=True"
    cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
    max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
```

2. Pass `cu_seqlens` and `max_seqlen` to flash attention's varlen kernel for variable-length attention calculation.

```python
if use_packed_dataset:
    attn_output = flash_attn_varlen_func(
        query_states.flatten(0, 1),
        key_states.flatten(0, 1),
        value_states.flatten(0, 1),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        dropout_rate,
        softmax_scale=None,
        causal=True,
        return_attn_probs=False,
    ).unsqueeze(0)
```


## Supporting ISP Training

### Weight Parallel Support

1. Replace `nn.Linear` with `new_linear` function in the modeling file.
2. Specify correct names for parameters to be split (w1, w2, w3, wq, wk, wv, wo, head).

### Sequence Parallel Support

1. Replace `nn.Embedding` with `Embedding1D` in the modeling file.
2. Use `auto_wrap_func_distributed_attention` decorator for attention calculation functions.
3. Replace attention calculation functions based on whether it's a packed dataset or not.

### Setting Parameter Attributes

1. For parameters not involved in ISP splitting (e.g., RMSNorm, MoE model parameters), set the attribute to `IS_REPLICA_ZERO_PARALLEL`.

### Configuration File Changes

Update the `parallel` configuration in `config.py`:

```python
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=2, mode="isp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=2, overlap=False, memory_pool=True),
)
```

- Set `tensor` size for sequence parallel size
- Set `weight` size for weight parallel size
- Set `overlap` to `False` in the `weight` dictionary

