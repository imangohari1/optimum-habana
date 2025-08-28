# from transformers.modeling_rope_utils import rope_config_validation
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig


class Gemma3TextConfig(Gemma3TextConfig):
    """
    Copied from: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/gemma3/configuration_gemma3.py
    Changes:
    - add layer_type (taken from transformers 4.55): https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gemma3/configuration_gemma3.py#L233-L237
    """

    def __init__(
        self,
        vocab_size=262_208,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=1_000_000.0,
        attention_bias=False,
        attention_dropout=0.0,
        query_pre_attn_scalar=256,
        sliding_window=4096,
        final_logit_softcapping=None,
        attn_logit_softcapping=None,
        cache_implementation="hybrid",
        rope_scaling=None,
        rope_local_base_freq=10_000.0,
        sliding_window_pattern=6,
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
            rope_scaling,
            rope_local_base_freq,
            sliding_window_pattern,
            **kwargs,
        )

        self.layer_types = layer_types
        # breakpoint()
        self._sliding_window_pattern = sliding_window_pattern if sliding_window_pattern is not None else 6

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % self._sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
