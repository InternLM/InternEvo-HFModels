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

## isp Parallel
For parallel training in ISP mode, the config.py file needs to be modified before starting the training to change the tensor parallel mode to ISP. The modification is as follows:
```bash
parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=2, mode="isp"),
    pipeline=dict(size=1, interleaved_overlap=True),
    weight=dict(size=2, overlap=True, memory_pool=True),
)
```
Here, the size value in tensor is the size of sequence parallelism, and the size value in weight is the size of weight parallelism in ISP mode.

The modeling file of the model needs to be modified to use the new_linear() function provided by InternEvo for the initialization of head, attention calculations, and mlp in the linear function. Taking the modeling file of the InternLM model as an example, the modification is as follows:
```bash
from internlm.model.modules.linear import new_linear

class InternLMMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_proj = new_linear("w1", hidden_size, intermediate_size, bias=False)
        self.down_proj = new_linear("w2", intermediate_size, hidden_size, bias=False)
        self.up_proj = new_linear("w3", hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

class InternLMAttention(nn.Module):
    def __init__(self, config: InternLMConfig):
        super().__init__()
        self.q_proj = new_linear("wq", self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = new_linear("wk", self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.v_proj = new_linear("wv", self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.o_proj = new_linear("wo", self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

class InternLMForCausalLM(InternLMPreTrainedModel):
    def __init__self, config):
        super().__init__(config)
        self.model = InternLMModel(config)

        self.lm_head = new_linear("head", config.hidden_size, config.vocab_size, bias=False)
```
The first parameter of the new_linear() function indicates the name of the parameter and can be one of the following: "head", "output", "wqkv", "wq", "wk", "wv", "wkv", "w1", "w3", "w13", "wo", "out_proj", "w2". Modify according to the actual situation.

