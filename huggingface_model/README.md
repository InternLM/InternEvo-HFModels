# Adapting HuggingFace Models for InternEvo Packed and ISP Training

## Background

When HuggingFace models are being integrated with the InternEvo framework, we want packed training and ISP be supproted to:
1. Improve GPU computation utilization (reduce wasting computation on meaningless padded tokens)
2. Support training with long sequences (use the latest parallel techniques from InternEvo framework)

This requires adapting the models to support:
1. Packed training
2. ISP (Intern Sequence Parallelism) training

## Supporting Packed Training

### Example for modeling_internlm.py

Step 1. Obtain `cu_seqlens` and `max_seqlen` from `gpc` for the current batch.

```python
use_packed_dataset = gpc.config.data.get("use_packed_dataset", False)

if use_packed_dataset:
    assert bsz == 1, "hidden_states should be packed into bsz=1 when use_packed_dataset=True"
    cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
    max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
```

Optional Step 2. If the rotary embedding logic cannot meet the requirement of packed training, please use InternEvo `apply_rotary_emb`. 
Otherwise, just use the original logic and skip this step.

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

Step 3. Pass `cu_seqlens` and `max_seqlen` to flash attention varlen kernel for variable-length attention calculation.

```python
if use_packed_dataset:
    attn_output = isp_flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        causal=True,
    )
```


### Example for modeling_internlm2.py

Step 1. Obtain `cu_seqlens` and `max_seqlen` from gpc for the current batch.

```python
use_packed_dataset = gpc.config.data.get("use_packed_dataset", False)

if use_packed_dataset:
    assert bsz == 1, "hidden_states should be packed into bsz=1 when use_packed_dataset=True"
    cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
    max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
```

Step 2. Pass `cu_seqlens` and `max_seqlen` to flash attention varlen kernel for variable-length attention calculation.

```python
if use_packed_dataset:
    attn_output = isp_flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        causal=True,
    )
```


## Supporting ISP Training

### Automatic dispatch

For simplicity, you can just create model with `hf_model_dispatch` like that:

```
model = initialize_model(model_dispatch_func=hf_model_dispatch)
```

And you can also modify `huggingface_model/dispatch_utils/__init__.py` to add custom patterns for automatic dispatch.

For the config, you need to set ISP size like that:

```python
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=2, mode="isp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=1, overlap=False, memory_pool=True),
)
```

- Set `tensor` size and mode for ISP.

### Manual code adaption dispatch

T.B.A.