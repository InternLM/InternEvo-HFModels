import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import init

from ..math import attention, rope

from internlm.model.modules.linear import new_linear
from internlm.core.context import (
    IS_REPLICA_EXPERT_DATA_PARALLEL,
    IS_REPLICA_ZERO_PARALLEL,
    IS_TENSOR_EXPERT_DATA_PARALLEL,
    IS_TENSOR_ZERO_PARALLEL,
    IS_WEIGHT_EXPERT_DATA_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
    ParallelMode,
)
from internlm.utils.parallel import is_using_isp

from internlm.initialize.initialize_tensor import normal_

def set_parallel_attr(module, parallel_attr):
    for p in module.parameters():
        setattr(p, parallel_attr, True)


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, device, dtype):
        super().__init__()
        # self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.in_layer = new_linear("w1", in_features=in_dim, out_features=hidden_dim, bias=True, device=device, dtype=dtype)
        self.silu = nn.SiLU()
        # self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_layer = new_linear("w1", in_features=hidden_dim, out_features=hidden_dim, bias=True, device=device, dtype=dtype)
        if is_using_isp():
            set_parallel_attr(self.in_layer, IS_WEIGHT_ZERO_PARALLEL)
            set_parallel_attr(self.out_layer, IS_WEIGHT_ZERO_PARALLEL)
        self.init_func = normal_
        
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.in_layer.named_parameters():
                self.init_func(std=0.02)(param.data)
            for name, param in self.out_layer.named_parameters():
                self.init_func(std=0.02)(param.data)
            

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.scale)
        
    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, device = None, dtype = None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = new_linear("w1", dim, dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.norm = QKNorm(head_dim)
        # self.proj = nn.Linear(dim, dim)
        self.proj = new_linear("w1", dim, dim, device=device, dtype=dtype)
        
        if is_using_isp():
            set_parallel_attr(self.qkv, IS_WEIGHT_ZERO_PARALLEL)
            set_parallel_attr(self.proj, IS_WEIGHT_ZERO_PARALLEL)

        self.init_func = normal_
        
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.qkv.named_parameters():
                self.init_func(std=0.02)(param.data)
            for name, param in self.proj.named_parameters():
                self.init_func(std=0.02)(param.data)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, device, dtype):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        # self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
        self.lin = new_linear("w1", in_features=dim, out_features=self.multiplier * dim, bias=True, device=device, dtype=dtype)
        if is_using_isp():
            set_parallel_attr(self.lin, IS_WEIGHT_ZERO_PARALLEL)
        
        self.init_func = normal_
        
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.lin.named_parameters():
                self.init_func(std=0.02)(param.data)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, device = None, dtype = None):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True, device=device, dtype=dtype)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            # nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            new_linear("w1", hidden_size, mlp_hidden_dim, bias=True, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            # nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
            new_linear("w1", mlp_hidden_dim, hidden_size, bias=True, device=device, dtype=dtype)
        )
        
        if is_using_isp():
            set_parallel_attr(self.img_mlp[0], IS_WEIGHT_ZERO_PARALLEL)
            set_parallel_attr(self.img_mlp[2], IS_WEIGHT_ZERO_PARALLEL)

        self.txt_mod = Modulation(hidden_size, double=True, device=device, dtype=dtype)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, device=device, dtype=dtype)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            # nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            new_linear("w1", hidden_size, mlp_hidden_dim, bias=True, device=device, dtype=dtype),
            nn.GELU(approximate="tanh"),
            # nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
            new_linear("w1", mlp_hidden_dim, hidden_size, bias=True, device=device, dtype=dtype),
        )
        
        if is_using_isp():
            set_parallel_attr(self.txt_mlp[0], IS_WEIGHT_ZERO_PARALLEL)
            set_parallel_attr(self.txt_mlp[2], IS_WEIGHT_ZERO_PARALLEL)

        self.init_func = normal_
        
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.txt_mlp[0].named_parameters():
                self.init_func(std=0.02)(param.data)
            for name, param in self.txt_mlp[2].named_parameters():
                self.init_func(std=0.02)(param.data)

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        device = None,
        dtype = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        # self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear1 = new_linear("w1", hidden_size, hidden_size * 3 + self.mlp_hidden_dim, device=device, dtype=dtype)
        # proj and mlp_out
        # self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.linear2 = new_linear("w1", hidden_size + self.mlp_hidden_dim, hidden_size, device=device, dtype=dtype)
        
        if is_using_isp():
            set_parallel_attr(self.linear1, IS_WEIGHT_ZERO_PARALLEL)
            set_parallel_attr(self.linear2, IS_WEIGHT_ZERO_PARALLEL)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False, device=device, dtype=dtype)

        self.init_func = normal_
        
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.linear1.named_parameters():
                self.init_func(std=0.02)(param.data)
            for name, param in self.linear2.named_parameters():
                self.init_func(std=0.02)(param.data)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, device, dtype):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.linear = new_linear("w1", hidden_size, patch_size * patch_size * out_channels, bias=True, device=device, dtype=dtype)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), new_linear("w1", hidden_size, 2 * hidden_size, bias=True, device=device, dtype=dtype))
        
        if is_using_isp():
            set_parallel_attr(self.linear, IS_WEIGHT_ZERO_PARALLEL)
            set_parallel_attr(self.adaLN_modulation[1], IS_WEIGHT_ZERO_PARALLEL)
        
        self.init_func = normal_
        
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.linear.named_parameters():
                self.init_func(std=0.02)(param.data)
            for name, param in self.adaLN_modulation[1].named_parameters():
                self.init_func(std=0.02)(param.data)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
