# Info

## model:
https://huggingface.co/internlm/internlm-7b/blob/main/modeling_internlm.py

## config:
https://huggingface.co/internlm/internlm-7b/blob/main/configuration_internlm.py

## commit id:
96e127d08d851a88cac736a9b091dd953ae1b873


# Usage

## How to apply InternEvo patch to support pack and ISP training?
```bash
patch modeling_internlm.py internevo.patch
```