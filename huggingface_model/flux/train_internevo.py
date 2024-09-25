#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import torch
import torch.nn.functional as F


from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.monitor import internevo_monitor
from internlm.utils.common import parse_args, get_current_device, enable_pytorch_expandable_segments

from flux.util import (configs, load_ae, load_clip,
                       load_flow_model, load_t5) 
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from dataset import loader
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as Data
from internlm.solver.activation_checkpoint import activation_checkpoint

from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)

from internlm.core.context import (
    IS_REPLICA_ZERO_PARALLEL,
    IS_WEIGHT_ZERO_PARALLEL,
)

from internlm.train.pipeline import initialize_optimizer, initialize_parallel_communicator

from dataset import DummyClsDataset

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512).to(device)
    clip = load_clip(device).to(device)
    clip.requires_grad_(False)
    model = load_flow_model(name, device=device).to(device)
    vae = load_ae(name, device="cpu" if offload else device).to(device)
    return model, vae, t5, clip


def main(args):
    enable_pytorch_expandable_segments()
    weight_dtype = torch.bfloat16
    global_seed = 1234
    
    # is_schnell = args.model_name == "flux-schnell"
    is_schnell = True
    
    device = get_current_device()
    
    dit, vae, t5, clip = get_models(name="flux-schnell", device=device, offload=False, is_schnell=is_schnell)
    for p in dit.parameters():
        if not hasattr(p, IS_WEIGHT_ZERO_PARALLEL):
            setattr(p, IS_REPLICA_ZERO_PARALLEL, True)
    isp_communicator = initialize_parallel_communicator(dit)
    
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(torch.bfloat16)
    dit.train()
    
    # optimizer_cls = torch.optim.AdamW
    
    # optimizer = optimizer_cls(
    #     [p for p in dit.parameters() if p.requires_grad],
    #     # lr=args.learning_rate,
    #     lr=2e-5,
    #     # betas=(args.adam_beta1, args.adam_beta2),
    #     betas=(0.9, 0.999),
    #     # weight_decay=args.adam_weight_decay,
    #     weight_decay=0.01,
    #     # eps=args.adam_epsilon,
    #     eps=1e-8,
    # )
    
    optimizer, beta2_scheduler, lr_scheduler = initialize_optimizer(dit)
    
    # train_dataloader = loader(**args.data_config)

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    #     num_training_steps=args.max_train_steps * accelerator.num_processes,
    # )
    
    train_dataset = DummyClsDataset([3, 256, 256])

    sampler = DistributedSampler(
            train_dataset,
            num_replicas=1, # important
            rank=0, # important
            shuffle=True,
            seed=global_seed
        )

    train_dataloader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   sampler=sampler,
                                   num_workers=4,
                                   pin_memory=True,
                                   drop_last=True)
    
    num_steps = 10

    train_dataloader = iter(train_dataloader)

    for step in range(0, num_steps):
        batch = next(train_dataloader)
        img, prompts = batch["model_input"], batch["video_prompts"]

        with torch.no_grad():
            x_1 = vae.encode(img.to(device))
            inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


        bs = img.shape[0]
        t = torch.sigmoid(torch.randn((bs,), device=device))
        x_0 = torch.randn_like(x_1).to(device)
        x_t = (1 - t) * x_1 + t * x_0
        bsz = x_1.shape[0]
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
        if gpc.get_global_rank() == 0:
            print(f"step = {step}, loss = {loss}", flush=True)

        # Gather the losses across all processes for logging (if we use distributed training).
        # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
        # train_loss += avg_loss.item() / args.gradient_accumulation_steps

        # Backpropagate
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        optimizer.zero_grad()



if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)