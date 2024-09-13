# Info

## model:
https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/modeling_baichuan.py

## config:
https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/configuration_baichuan.py

## commit id:
f9d4d8dd2f7a3dbede3bda3b0cf0224e9272bbe5


# Usage

## How to apply InternEvo patch to support pack and ISP training?
```bash
patch modeling_baichuan.py internevo.patch
```