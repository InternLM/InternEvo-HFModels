#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from internlm.core.context import global_context as gpc
from internlm.initialize import initialize_distributed_env
from internlm.utils.common import parse_args, get_current_device, enable_pytorch_expandable_segments

from flux.util import (configs, load_ae, load_clip,
                       load_flow_model, load_t5) 
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from dataset import loader
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from internlm.core.context import (
    IS_REPLICA_ZERO_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
)

from internlm.train.pipeline import initialize_optimizer, initialize_parallel_communicator
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger
from datetime import datetime

from dataset import DummyClsDataset
from internlm.checkpoint.checkpoint_manager import CheckpointManager
from internlm.data.train_state import get_train_state

logger = get_logger(__file__)

def get_models(model_cfg: dict, device, is_schnell: bool):
    
    name = model_cfg.model_name
    
    t5 = load_t5(
            tokenizer_path=model_cfg.t5_tokenizer, 
            model_path=model_cfg.t5_ckpt,
            device=device, max_length=256 if is_schnell else 512)
    
    clip = load_clip(
            tokenizer_path=model_cfg.clip_tokenizer, 
            model_path=model_cfg.clip_ckpt,
            device=device)
    
    model = load_flow_model(name, device=device).to(device)
    
    vae = load_ae(
            name, 
            ckpt_path=model_cfg.vae_ckpt,
            ).to(device)
    
    for name, p in model.named_parameters():
        if not hasattr(p, IS_WEIGHT_ZERO_PARALLEL):
            setattr(p, IS_REPLICA_ZERO_PARALLEL, True)
    
    return model, vae, t5, clip


def main(args):
    
    # obtain the data config
    data_cfg = gpc.config.flux.data
    
    # obtain the model config
    model_cfg = gpc.config.flux.model
    
    if model_cfg.weight_dtype == "bfloat16":
        weight_dtype = torch.bfloat16
    
    is_schnell = model_cfg.model_name == "flux-schnell"
    
    device = get_current_device()
    
    dit, vae, t5, clip = get_models(model_cfg=model_cfg, device=device, is_schnell=is_schnell)

    isp_communicator = initialize_parallel_communicator(dit)
    
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(torch.bfloat16)
    dit.train()
    
    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(dit, isp_communicator)
    
    # train_dataset = DummyClsDataset([3, 256, 256])

    # sampler = DistributedSampler(
    #         train_dataset,
    #         num_replicas=1, # important
    #         rank=0, # important
    #         shuffle=True,
    #         seed=global_seed
    #     )

    # train_dataloader = Data.DataLoader(dataset=train_dataset,
    #                                batch_size=1,
    #                                shuffle=False,
    #                                sampler=sampler,
    #                                num_workers=4,
    #                                pin_memory=True,
    #                                drop_last=True)
    
    train_dataloader = loader(train_batch_size=data_cfg.batch_size, num_workers=data_cfg.num_workers, img_dir=data_cfg.train_folder, img_size=data_cfg.img_size)
    train_iter = iter(train_dataloader)
    
    with open(args.config, "r") as f:
        config_lines = f.readlines()

    # initialize the checkpoint manager
    ckpt_manager = CheckpointManager(
                    ckpt_config=gpc.config.ckpt,
                    model=dit,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    train_dl=train_dataloader,
                    model_config=gpc.config.model,
                    model_config_file="".join(config_lines),
                    feishu_address=gpc.config.monitor.alert.feishu_alert_address,
                )

    train_state = get_train_state(train_dataloader)


    for step in range(0, data_cfg.total_steps):
        gpc.step = step
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        img, prompts = batch[0], batch[1]

        with torch.no_grad():
            x_1 = vae.encode(img.to(device))
            inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


        bs = img.shape[0]
        t = torch.sigmoid(torch.randn((bs,), device=device))
        x_0 = torch.randn_like(x_1).to(device)
        x_t = (1 - t) * x_1 + t * x_0
        guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

        # Predict the noise residual and compute loss
        model_pred = dit(img=x_t.to(weight_dtype),
                        img_ids=inp['img_ids'].to(weight_dtype),
                        txt=inp['txt'].to(weight_dtype),
                        txt_ids=inp['txt_ids'].to(weight_dtype),
                        y=inp['vec'].to(weight_dtype),
                        timesteps=t.to(weight_dtype),
                        guidance=guidance_vec.to(weight_dtype),)
        loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

        # Backpropagate
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        if gpc.is_rank_for_log():
            logger.info(f"{datetime.now()}: step = {step}, loss = {loss}")
    
        if ckpt_manager.try_save_checkpoint(train_state):
            ckpt_manager.wait_async_upload_finish()



if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)