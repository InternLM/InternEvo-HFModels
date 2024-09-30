# Info

## model:
https://huggingface.co/internlm/internlm2-7b/blob/main/modeling_internlm2.py

## config:
https://huggingface.co/internlm/internlm2-7b/blob/main/configuration_internlm2.py

## commit id:
3909fd287ddfc6d9c9ac4b4e57b101ad7fd8d204


# Usage

## How to apply InternEvo patch to support Variable-Length and Intern Sequence Parallel training?
```bash
patch modeling_internlm2.py internevo.patch
```