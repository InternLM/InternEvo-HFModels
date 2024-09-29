from torch import Tensor, nn
from transformers import (CLIPTextModel, CLIPTokenizer, #T5EncoderModel,
                          T5Tokenizer)
from transformers import AutoConfig, CLIPTextConfig

from t5.modeling_t5 import T5EncoderModel

class HFEmbedder(nn.Module):
    def __init__(self, tokenizer_path: str, model_path: str, max_length: int, is_clip: bool, **hf_kwargs):
        super().__init__()
        self.is_clip = is_clip
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"

        if self.is_clip:
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=max_length)
            # config = CLIPTextConfig.from_pretrained(version)
            # self.hf_module = CLIPTextModel(config)
            # version = "/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/flux_weight/text_encoder"
            self.hf_module = CLIPTextModel.from_pretrained(model_path)
        else:                
            # version = "/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/flux_weight/tokenizer_2"
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, max_length=max_length)
            # config = AutoConfig.from_pretrained(version, trust_remote_code=True)
            # self.hf_module = T5EncoderModel(config)
            # version = "/mnt/petrelfs/xiongyingtong/InternEvo-HFModels/huggingface_model/flux/flux_weight/text_encoder_2"
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(model_path)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key]
