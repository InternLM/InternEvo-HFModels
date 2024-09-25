from transformers import (CLIPTextModel, CLIPTokenizer, T5EncoderModel,
                          T5Tokenizer)

clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", max_length=77)
clip_hf_module: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

t5_tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", max_length=256)
t5_hf_module: T5EncoderModel = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")

print("successfully")