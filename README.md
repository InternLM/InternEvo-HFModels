# Usage

## How to enable InternEvo pack and ISP training for huggingface models?

### Option1: apply the already prepared `internevo.patch`

Take huggingface_model InternLM2 for example,

```bash
cd huggingface_model/internlm/internlm2_7b
patch modeling_internlm2.py internevo.patch
```

### Option2: manual modification to modeling file

Take huggingface_model InternLM2 for example, you just need to make ~10 lines of code changes to the `attention` forward() calculation. All you need is to get `cu_seqlens` and `max_seqlen` from `gpc.config.data` and then pass them to `isp_flash_attn_varlen_func` or `isp_flash_attn_func`.

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