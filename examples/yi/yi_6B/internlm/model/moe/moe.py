from typing import Optional

import torch

from internlm.core.context import global_context as gpc
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.moe.gshard_layer import GShardMOELayer
from internlm.model.moe.megablock.megablock_dmoe import MegaBlockdMoE
from internlm.model.moe.megablock.megablock_moe import MegaBlockMoE
from internlm.utils.logger import get_logger

# global llm logger
logger = get_logger(__file__)


def new_moe_layer(moe_type: str, **kwargs):
    if moe_type == "GShard":
        return GShardMOELayer(**kwargs)
    elif moe_type == "MegaBlock":
        return MegaBlockMoE(**kwargs)
    elif moe_type == "MegaBlock-D":
        return MegaBlockdMoE(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {moe_type}")


class MoE(torch.nn.Module):
    """Initialize an MoE layer.

    Arguments:
        hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.
        num_experts (int, optional): default=1, the total number of experts per layer.
        ep_size (int, optional): default=1, number of ranks in the expert parallel world or group.
        k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.
        capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.
        eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.
        min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.
        noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample'
                                            or 'None'.
        using_default_moe (bool, optional): default=True, whether to use the default MoE layer.
        drop_tokens (bool, optional): default=True, whether to drop tokens - (setting to False is equivalent to
                                        infinite capacity).
        use_rts (bool, optional): default=True, whether to use Random Token Selection.
        moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
                                          (https://arxiv.org/abs/2201.05596) layer.
        residual_mlp (torch.nn.Module, optional): default=None, the torch module that defines the residual MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        num_experts: int = 1,
        ep_size=1,
        use_residual: bool = False,
        device=None,
        dtype=None,
    ):

        super().__init__()

        if not hasattr(gpc.config, "moe"):
            gpc.config.moe = dict()

        self.moe_layer = new_moe_layer(
            moe_type=gpc.config.model.moe_type,
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_experts=num_experts,
            ep_group=ep_group,
            ep_size=ep_size,
            device=device,
            dtype=dtype,
            **(gpc.config.moe),
        )

        # residual network, see https://arxiv.org/pdf/2201.05596.pdf, seems useful for convergence
        self.use_residual = use_residual
        if self.use_residual:
            self.residual_mlp = new_feed_forward(
                in_features=in_features,
                hidden_features=hidden_features,
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
            )
            # coefficient is used for weighted sum of the output of expert and residual mlp
            self.coefficient = torch.nn.Linear(in_features, 2)

    def forward(self, hidden_states, used_token=None):
        """MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.moe_layer(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.residual_mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = torch.nn.functional.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.moe_layer.l_aux, self.moe_layer.exp_counts
