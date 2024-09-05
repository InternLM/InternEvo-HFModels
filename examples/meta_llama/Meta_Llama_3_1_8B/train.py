#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.model.registry import hf_config_initializer, model_initializer
from internlm.monitor import internevo_monitor
from internlm.train import initialize_model
from internlm.utils.common import parse_args

from huggingface_model.meta_llama.Meta_Llama_3_1_8B.configuration_llama import (
    LlamaConfig,
)
from huggingface_model.meta_llama.Meta_Llama_3_1_8B.modeling_llama import (
    LlamaForCausalLM,
)


@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # register huggingface model and config for InternEvo
    model_initializer.register_module(gpc.config.model_type, LlamaForCausalLM)
    hf_config_initializer.register_module(gpc.config.model_type, LlamaConfig)

    # initialize model
    model = initialize_model()

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # initialize kwargs
    kwargs = vars(args) | {"dataset_types": dataset_types}

    # build trainer
    trainer = TrainerBuilder(model, train_dl, val_dls, **kwargs)

    # training
    trainer.fit()


if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(
        config=args.config,
        launcher=args.launcher,
        master_port=args.port,
        seed=args.seed,
    )
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)
