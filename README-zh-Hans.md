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

## isp并行
如果需要开启isp并行模式训练，需要在启动训练前，修改config.py文件，将tensor并行模式改为isp，修改如下：
```bash
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=2, mode="isp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=2, overlap=True, memory_pool=True),
)
```
其中，tensor中的size为序列并行的大小，weight中的size为isp模式中，权重并行的大小。

需要修改模型modeling文件，将head、attention计算以及mlp中涉及的linear初始化函数改为使用InternEvo提供的new_linear()函数。以internlm模型的modeling文件为例，修改如下：
```bash
from internlm.model.modules.linear import new_linear

class InternLMMLP(nn.Module):
         super().__init__()
         self.gate_proj = new_linear("w1", hidden_size, intermediate_size, bias=False)
         self.down_proj = new_linear("w2", intermediate_size, hidden_size, bias=False)
         self.up_proj = new_linear("w3", hidden_size, intermediate_size, bias=False)
         self.act_fn = ACT2FN[hidden_act]
 
class InternLMAttention(nn.Module):
         self.q_proj = new_linear("wq", self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
         self.k_proj = new_linear("wk", self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
         self.v_proj = new_linear("wv", self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
         self.o_proj = new_linear("wo", self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
 
class InternLMForCausalLM(InternLMPreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
         self.model = InternLMModel(config)
 
         self.lm_head = new_linear("head", config.hidden_size, config.vocab_size, bias=False)
```
new_linear()函数的第一个参数标志参数的名称，可接受的名称范围为："head"、"output"、"wqkv"、"wq"、"wk"、"wv"、"wkv"、"w1"、"w3"、"w13"、"wo"、"out_proj"、"w2"，根据实际情况修改。

