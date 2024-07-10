# InternEvo-HFModels

[English](./README.md) |
[简体中文](./README-zh-Hans.md)

该文件夹下包含了 huggingface 格式的模型modeling及configuration文件，以及使用InternEvo训练这些模型的样例train.py文件和配置文件。

## 安装InternEvo
参考[InternEvo安装文档](https://github.com/InternLM/InternEvo/blob/develop/doc/install.md)

## 代码下载
将InternEvo-HFModels中的文件下载到本地：
```bash
git clone https://github.com/InternLM/InternEvo-HFModels.git
```

## 启动训练
根据需要运行的模型，选择指定的train.py及config.py配置文件启动训练，如：
```bash
srun -p llm_s -N 1 -n 8 --ntasks-per-node=8 --gpus-per-task=1 python examples/internlm_model/train.py --config examples/internlm_model/config.py 
```
