# InternEvo-HFModels

[English](./README.md) |
[简体中文](./README-zh-Hans.md)

This directory contains the model modeling and configuration files in the Hugging Face format, as well as the sample train.py file and configuration files for training these models with InternEvo.

## Installation of InternEvo
Refer to the [InternEvo Installation Documentation](https://github.com/InternLM/InternEvo/blob/develop/doc/install.md)

## Code Download
Download the files from InternEvo-HFModels to your local machine:
```bash
git clone https://github.com/InternLM/InternEvo-HFModels.git
```

## Start Training
Run the specified train.py and config.py configuration files to start training according to the model you need, for example:
```bash
srun -p llm_s -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python examples/internlm_model/train.py --config examples/internlm_model/config.py                             
```
