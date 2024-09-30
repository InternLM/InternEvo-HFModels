# Introduction :muscle:

Democratizing Huggingface model training with InternEvo framework, via the capability of Variable-Length and Intern Sequence Parallel training.

![InternEvo_and_Huggingface](doc/images/InternEvo_and_Huggingface.png)


# Support Matrix :heartpulse:

## Modelzoo

| Model     | Supported         | Variable-Length Training | Intern Sequence Parallel |
|-----------|:-----------------:|:------------------------:|:------------------------:|
| [Baichuan2](huggingface_model/baichuan_inc/Baichuan2_7B_Base) |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |
| [Flux](huggingface_model/flux)      |:white_check_mark: |:x:                       |:x:                       |
| [InternLM1](huggingface_model/internlm/internlm_7b) |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |
| [InternLM2](huggingface_model/internlm/internlm2_7b) |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |
| [LLaVA1.5](huggingface_model/llava_hf/llava_1_5_7b_hf)  |:white_check_mark: |:x:                       |:x:                       |
| [Mamba](huggingface_model/mamba)     |:white_check_mark: |:x:                       |:x:                       |
| [Llama2](huggingface_model/meta_llama/Llama_2_7b_hf)    |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |
| [Mixtral](huggingface_model/mistralai/mixtral_8x7B_v0_1)   |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |
| [Qwen2](huggingface_model/Qwen/Qwen2_7B)     |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |
| [Yi](huggingface_model/Yi)        |:white_check_mark: |:white_check_mark:        |:white_check_mark:        |


# Usage :rocket:

## How to enable InternEvo Variable-Length and Intern Sequence Parallel training for huggingface models?

### Option1: apply the already prepared `internevo.patch`

Take huggingface_model InternLM2 for example,

```bash
cd huggingface_model/internlm/internlm2_7b
patch modeling_internlm2.py internevo.patch
```

### Option2: manual modification to modeling file

Take huggingface_model InternLM2 for example, you just need to make ~10 lines of code changes to the attention `forward()` calculation. All you need is to get `cu_seqlens` and `max_seqlen` from `gpc.config.data` and then pass them to `isp_flash_attn_varlen_func` or `isp_flash_attn_func`.

``` python
+        use_packed_dataset = gpc.config.data.get("use_packed_dataset", False)
+        if use_packed_dataset:
+            assert bsz == 1, "hidden_states should be packed into bsz=1 when use_packed_dataset=True"
+            cu_seqlens = gpc.config.data[f"cu_seqlens_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
+            max_seqlen = gpc.config.data[f"max_seqlen_data_rank{gpc.get_local_rank(ParallelMode.DATA)}"]
+
         qkv_states = self.wqkv(hidden_states)
 
         qkv_states = rearrange(
@@ -473,9 +484,28 @@
             key_states = key_states.to(target_dtype)
             value_states = value_states.to(target_dtype)
 
-        attn_output = self._flash_attention_forward(
-            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
-        )
+        if use_packed_dataset:
+            attn_output = isp_flash_attn_varlen_func(
+                query_states,
+                key_states,
+                value_states,
+                cu_seqlens,
+                cu_seqlens,
+                max_seqlen,
+                max_seqlen,
+                attention_dropout=dropout_rate,
+                softmax_scale=None,
+                causal=True,
+            )
+        else:
+            attn_output = isp_flash_attn_func(
+                query_states, 
+                key_states, 
+                value_states, 
+                attention_dropout=dropout_rate, 
+                softmax_scale=None, 
+                causal=True,
+            )
```
