from dataclasses import dataclass

@dataclass
class HWDebugConfig:
    """
    HW Debug AI - 10B Parameter Model Configuration
    Domain-specific LLM for hardware-software debugging
    """
    # Model scale
    n_params_approx: str = "10B"
    n_layers: int = 48
    n_embd: int = 5120
    n_heads: int = 40
    n_kv_heads: int = 40           # Full MHA (no GQA)
    head_dim: int = 128            # n_embd // n_heads

    # MLP
    mlp_hidden: int = 20480        # SwiGLU intermediate dim
    mlp_type: str = "swiglu"

    # Context
    block_size: int = 4096
    vocab_size: int = 32000        # 32K SentencePiece tokenizer

    # Normalization and position
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    pos_encoding: str = "rope"
    rope_theta: float = 10000.0

    # Training precision
    dtype: str = "bfloat16"

    # Tokenizer
    tokenizer_path: str = "gs://rohith-llm-checkpoints/hw_tok.model"

    # Checkpoints
    checkpoint_dir: str = "gs://rohith-llm-checkpoints/"

    def __post_init__(self):
        assert self.n_embd % self.n_heads == 0, "n_embd must be divisible by n_heads"
        assert self.head_dim == self.n_embd // self.n_heads

    def num_parameters(self) -> int:
        """Approximate parameter count."""
        embed = self.vocab_size * self.n_embd
        attn = self.n_layers * (4 * self.n_embd * self.n_embd)
        mlp = self.n_layers * (3 * self.n_embd * self.mlp_hidden)
        norm = self.n_layers * 2 * self.n_embd
        head = self.n_embd * self.vocab_size
        return embed + attn + mlp + norm + head


# 1.6B proof-of-concept config (completed, 500K iterations)
HWDebugConfig1B = HWDebugConfig(
    n_params_approx="1.6B",
    n_layers=24,
    n_embd=2048,
    n_heads=16,
    n_kv_heads=16,
    head_dim=128,
    mlp_hidden=8192,
)

# 10B production config (pretraining in progress)
HWDebugConfig10B = HWDebugConfig()

if __name__ == "__main__":
    cfg = HWDebugConfig10B
    params = cfg.num_parameters()
    print(f"Model: {cfg.n_params_approx}")
    print(f"Layers: {cfg.n_layers}")
    print(f"Hidden dim: {cfg.n_embd}")
    print(f"Heads: {cfg.n_heads}")
    print(f"MLP hidden: {cfg.mlp_hidden}")
    print(f"Context: {cfg.block_size} tokens")
    print(f"Vocab: {cfg.vocab_size}")
    print(f"Approx params: {params / 1e9:.2f}B")
