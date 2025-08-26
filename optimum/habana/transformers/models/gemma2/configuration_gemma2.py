# from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config


class Gemma2Config(Gemma2Config):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma2/configuration_gemma2.py#L25
    Changes:
    - add layer_type (taken from transformers 4.55): https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gemma2/configuration_gemma2.py#L175-L179
    """

    def __init__(
        self,
        vocab_size=256000,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        cache_implementation="hybrid",
        layer_types=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_activation,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            eos_token_id,
            bos_token_id,
            tie_word_embeddings,
            rope_theta,
            attention_bias,
            attention_dropout,
            query_pre_attn_scalar,
            sliding_window,
            final_logit_softcapping,
            attn_logit_softcapping,
            cache_implementation,
            **kwargs,
        )

        self.layer_types = layer_types
        # breakpoint()
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
