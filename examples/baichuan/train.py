#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.monitor import internevo_monitor
from internlm.utils.common import parse_args, get_current_device

from huggingface_model.baichuan.configuration_baichuan import BaichuanConfig
from huggingface_model.baichuan.modeling_baichuan import BaichuanForCausalLM


@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # initialize model
    model = BaichuanForCausalLM(
        BaichuanConfig(
            return_dict=False, 
        )
    ).to(get_current_device())

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # build trainer
    trainer = TrainerBuilder(model, train_dl, val_dls, **(vars(args) | {"dataset_types": dataset_types}))

    # training
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)