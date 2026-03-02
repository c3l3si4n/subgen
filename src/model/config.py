"""
LlamaConfig builder for ~148M parameter causal language model.

Architecture: Llama-style with RoPE, RMSNorm, SwiGLU, GQA.
"""

from transformers import LlamaConfig, LlamaForCausalLM


def build_config(
    vocab_size: int = 8192,
    hidden_size: int = 1024,
    intermediate_size: int = 2816,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 16,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 512,
    hidden_act: str = "silu",
    tie_word_embeddings: bool = True,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    rope_theta: float = 500_000.0,
    variant: str = "default",
) -> LlamaConfig:
    """Build a LlamaConfig for the subdomain generation model.

    Args:
        variant: Architecture variant to use.
            "default" — 12 layers × 1024 hidden (~148M params)
            "wide"    — 8 layers × 1280 hidden (~148M params, same budget)
    """
    if variant == "wide":
        hidden_size = 1280
        intermediate_size = 3584  # ~2.8× hidden, matches SwiGLU convention
        num_hidden_layers = 8
        num_attention_heads = 20
        num_key_value_heads = 4

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        hidden_act=hidden_act,
        tie_word_embeddings=tie_word_embeddings,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        rms_norm_eps=1e-5,
        rope_theta=rope_theta,
    )
    return config


def build_model(config: LlamaConfig | None = None) -> LlamaForCausalLM:
    """Initialize a LlamaForCausalLM from scratch."""
    if config is None:
        config = build_config()
    model = LlamaForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")
    return model


if __name__ == "__main__":
    model = build_model()
