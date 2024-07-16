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
from internlm.train import initialize_model
from internlm.utils.common import parse_args
from internlm.model.registry import model_initializer, hf_config_initializer

from huggingface_model.internlm.internlm2_7b.modeling_internlm2 import InternLM2ForCausalLM
from huggingface_model.internlm.internlm2_7b.configuration_internlm2 import InternLM2Config


@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # register huggingface model and config for InternEvo
    model_initializer.register_module(gpc.config.model_type, InternLM2ForCausalLM)
    hf_config_initializer.register_module(gpc.config.model_type, InternLM2Config)
    if gpc.config.model_type == "hf":
        hf_config_builder = hf_config_initializer.get_module(module_name=gpc.config.model_type)
        hf_cfg = hf_config_builder(return_dict=False)
        gpc.config.model.num_layers = hf_cfg.num_hidden_layers
        gpc.config.model.hidden_size = hf_cfg.hidden_size
        gpc.config.model.num_attention_heads = hf_cfg.num_attention_heads
        gpc.config.model.mlp_ratio = hf_cfg.intermediate_size / hf_cfg.hidden_size
        gpc.config.model.vocab_size = hf_cfg.vocab_size

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
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)
