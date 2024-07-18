"""
A simple operator selector, used for compatibility with different platforms such as CUDA and Ascend,
as well as whether to enable flash-attn operator optimization, may be replaced by a more comprehensive
operator compatibility layer in the future.

This file implements support for the cross entropy operators.
"""

from torch import nn

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

try:
    from flash_attn.losses.cross_entropy import (
        CrossEntropyLoss as FlashCrossEntropyLoss,
    )

    flash_cross_entropy_impl = True
except (ModuleNotFoundError, ImportError):
    flash_cross_entropy_impl = False

logger = get_logger(__file__)
internlm_accelerator = get_accelerator()


# TODO: ops是否需要实现更加统一的形式
def new_cross_entropy(
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0,
    parallel_output: bool = False,
    **kwargs,
):
    if parallel_output:
        assert (
            gpc.config.model.get("use_flash_attn", False) and flash_cross_entropy_impl
        ), "Only flash cross entropy support parallel_output"
        assert (
            internlm_accelerator.get_accelerator_backend() is AcceleratorType.GPU
        ), "flash cross entropy only support gpu backend"

        return FlashCrossEntropyLoss(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            process_group=gpc.get_group(ParallelMode.TENSOR),
        )
    else:
        if gpc.is_rank_for_log():
            logger.warning(
                "Use nn.CrossEntropyLoss rather than flashattn CrossEntropyLoss."
                "parallel_output must be set false. Please note this!"
            )
        kwargs.pop("inplace_backward", None)
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing, **kwargs
        )
