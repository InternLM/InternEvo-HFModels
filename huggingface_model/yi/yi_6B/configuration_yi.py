
"""Yi model configuration"""


from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)
YI_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


# Modified from transformers.model.llama.configuration_llama.LlamaConfig
class YiConfig(PretrainedConfig):
    model_type = "yi"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(  # pylint: disable=W0102
        self,
        architectures=["LlamaForCausalLM"],
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=4096,
        initializer_range=0.02,
        intermediate_size=11008,
        max_position_embeddings=4096,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=4,
        pad_token_id=0,
        pretraining_tp=1,
        rms_norm_eps=1e-05,
        rope_theta=5000000.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        transformers_version="4.34.0",
        use_cache=True,
        vocab_size=64000,
        attention_dropout=0.0,
        attention_bias=False,
        mlp_bias=False,
        rope_scaling=None,
        hidden_dropout_prob=0.0,
        attn_implementation=None,
        **kwargs,
    ):
        super().__init__(
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
        )
        self.architectures = architectures
        self.pretraining_tp = pretraining_tp
        self.rope_theta = rope_theta
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.mlp_bias = mlp_bias
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attn_implementation = attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = "eager"
        
