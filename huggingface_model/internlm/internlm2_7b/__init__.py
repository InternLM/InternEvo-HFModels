import torch
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal

from .configuration_internlm2 import InternLM2Config
from .modeling_internlm2 import InternLM2ForCausalLM


def reset_attn_parameters(layer_idx, layer, use_scaled_init=True, std=0.02):
    for name, param in layer.attention.named_parameters():
        if param.ndim == 1:  # bias
            param.data.zero_()
        elif "wq" in name or "wk" in name or "wv" in name:
            normal_(std=std)(param.data)
        elif use_scaled_init:  # wo
            scaled_init_method_normal(sigma=std, num_layers=layer_idx + 1)(param.data)
        else:  # wo
            normal_(std=std)(param.data)

    for name, param in layer.feed_forward.named_parameters():
        if use_scaled_init:
            scaled_init_method_normal(sigma=std, num_layers=layer_idx + 1)(param.data)
        else:
            normal_(std=std)(param.data)


def reset_parameters(model, std=0.02):
    with torch.no_grad():
        for _, param in model.model.tok_embeddings.named_parameters():
            normal_(std=std)(param)
        for layer_idx, layer in enumerate(model.model.layers):
            reset_attn_parameters(layer_idx, layer)
        for _, param in model.output.named_parameters():
            normal_(std=std)(param)


__all__ = ["InternLM2Config", "InternLM2ForCausalLM", "reset_parameters"]
