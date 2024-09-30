# Info

## model:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

## config:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/configuration_qwen2.py

## commit id:
d6ffe74dfa577b5e7d12e48aa1c686ad8d3ef557


# Usage

## How to apply InternEvo patch to support Variable-Length and Intern Sequence Parallel training?
```bash
patch modeling_qwen2.py internevo.patch
```