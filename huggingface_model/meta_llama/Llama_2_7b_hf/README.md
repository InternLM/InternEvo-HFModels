# Info

## model:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

## config:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py

## commit id:
e39b6c1c7cdc890b6849b8c9de545fc9590ba871


# Usage

## How to apply InternEvo patch to support pack and ISP training?
```bash
patch modeling_llama.py internevo.patch
```