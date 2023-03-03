"""Modeling code for a GPT-style language model."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from fancy_einsum import einsum


@dataclass
class GPTConfig:
    """Configuration for a GPT-style language model."""

    d_model: int
    n_heads: int
    n_layers: int
    d_vocab: int
    n_ctx: int
    d_head: Optional[int] = None
    d_mlp: Optional[int] = None
    d_vocab_out: Optional[int] = None
    use_mlp: bool = True
    use_norm: bool = True
    norm_eps: float = 1e-5
    act_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu
    weight_std: float = 0.02
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.d_head is not None:
            assert (
                self.d_model // self.n_heads == self.d_head
            ), "d_model must be divisible by n_heads into d_head"
        else:
            assert (
                self.d_model % self.n_heads == 0
            ), "d_model must be divisible by n_heads"
            self.d_head = self.d_model // self.n_heads
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        if self.d_vocab_out is None:
            self.d_vocab_out = self.d_vocab


class TokEmbed(nn.Module):
    """Token Embedding Layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.W_E = nn.Parameter(
            torch.empty(
                config.d_vocab,
                config.d_model,
                device=config.device,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W_E, std=self.config.weight_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.W_E)


class Unembed(nn.Module):
    """Unembedding Layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert (
            config.d_vocab_out is not None
        ), "d_vocab_out must be specified for unembedding"
        self.W_U = nn.Parameter(
            torch.empty(
                config.d_vocab_out,
                config.d_model,
                device=config.device,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W_U, std=self.config.weight_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.W_U)


class PosEmbed(nn.Module):
    """Positional Embedding Layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.W_P = nn.Parameter(
            torch.empty(
                config.n_ctx,
                config.d_model,
                device=config.device,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.W_P, std=self.config.weight_std)

    def forward(self, x: torch.Tensor, cache_offset: int = 0) -> torch.Tensor:
        return F.embedding(
            torch.arange(
                cache_offset,
                x.shape[1] + cache_offset,
                device=x.device,
            ),
            self.W_P,
        )


class RMSNorm(nn.Module):
    """RMS Norm without affine transform."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.config.norm_eps
        )


@dataclass
class KVCacheEntry:
    """Cache entry for key and value tensors."""

    past_k: torch.Tensor
    past_v: torch.Tensor

    @classmethod
    def init_entry(cls, config: GPTConfig, batch_size: int):
        assert config.d_head is not None
        return cls(
            past_k=torch.empty(
                batch_size,
                config.n_heads,
                0,
                config.d_head,
                device=config.device,
            ),
            past_v=torch.empty(
                batch_size,
                config.n_heads,
                0,
                config.d_head,
                device=config.device,
            ),
        )

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self.past_k = torch.cat([self.past_k, k], dim=2)
        self.past_v = torch.cat([self.past_v, v], dim=2)
        return self.past_k, self.past_v


@dataclass
class KVCache:
    """Cache for key and value tensors for a model."""

    entries: List[KVCacheEntry]

    def __getitem__(self, i: int) -> KVCacheEntry:
        return self.entries[i]

    def __len__(self):
        return len(self.entries)

    @classmethod
    def init_cache(cls, config: GPTConfig, batch_size: int):
        return cls(
            [
                KVCacheEntry.init_entry(config, batch_size)
                for _ in range(config.n_layers)
            ]
        )


class Attention(nn.Module):
    """Multi-headed attention layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.d_head is not None
        self.W_Q = nn.Parameter(
            torch.empty(
                config.n_heads,
                config.d_head,
                config.d_model,
                device=config.device,
            )
        )
        self.W_K = nn.Parameter(
            torch.empty(
                config.n_heads,
                config.d_head,
                config.d_model,
                device=config.device,
            )
        )
        self.W_V = nn.Parameter(
            torch.empty(
                config.n_heads,
                config.d_head,
                config.d_model,
                device=config.device,
            )
        )
        self.W_O = nn.Parameter(
            torch.empty(
                config.n_heads,
                config.d_model,
                config.d_head,
                device=config.device,
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(w, std=self.config.weight_std)

    def forward(self, x: torch.Tensor, cache_entry: Optional[KVCacheEntry] = None):
        seq_len = x.shape[1]

        q = einsum(
            "... seq d_model, n_heads d_head d_model -> ... n_heads seq d_head",
            x,
            self.W_Q,
        )
        k = einsum(
            "... seq d_model, n_heads d_head d_model -> ... n_heads seq d_head",
            x,
            self.W_K,
        )
        v = einsum(
            "... seq d_model, n_heads d_head d_model -> ... n_heads seq d_head",
            x,
            self.W_V,
        )

        if cache_entry is not None:
            cache_start = cache_entry.past_k.shape[2]
            if cache_start > 0:
                assert seq_len == 1, "Cache can only be used for single step decoding"

            k, v = cache_entry.update(k, v)
        else:
            cache_start = 0

        attn = einsum(
            "... seq_q d_head, ... seq_k d_head -> ... seq_q seq_k",
            q,
            k,
        )
        attn = attn / (self.config.d_head**0.5)  # type: ignore
        mask = (
            torch.arange(
                cache_start,
                cache_start + seq_len,
                device=attn.device,
            )[:, None]
            >= torch.arange(cache_start + seq_len, device=attn.device)[None, :]
        )
        attn.masked_fill_(~mask, -torch.finfo(attn.dtype).max)
        attn = F.softmax(attn, dim=-1)

        combined_v = einsum(
            "... seq_q seq_k, ... seq_k d_head -> ... seq_q d_head",
            attn,
            v,
        )

        out = einsum(
            "... n_heads seq d_head, n_heads d_model d_head -> ... n_heads seq d_model",
            combined_v,
            self.W_O,
        )

        out = einops.reduce(
            out,
            "... n_heads seq d_model -> ... seq d_model",
            "sum",
        )

        return out


class MLP(nn.Module):
    """Position-wise feed-forward layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.d_mlp is not None
        self.W_in = nn.Parameter(
            torch.empty(config.d_mlp, config.d_model, device=config.device)
        )
        self.W_out = nn.Parameter(
            torch.empty(config.d_model, config.d_mlp, device=config.device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.W_in, self.W_out]:
            nn.init.normal_(w, std=self.config.weight_std)
            w = w.to(self.config.device)

    def forward(self, x: torch.Tensor):
        x = F.linear(x, self.W_in)
        x = self.config.act_fn(x)
        x = F.linear(x, self.W_out)
        return x


class Block(nn.Module):
    """Transformer block, with Attention and MLP Layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.attn = Attention(config)
        self.norm_1 = RMSNorm(config) if config.use_norm else nn.Identity()
        self.mlp = MLP(config)
        self.norm_2 = RMSNorm(config) if config.use_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cache_entry: Optional[KVCacheEntry] = None,
    ):
        x = x + self.attn(self.norm_1(x), cache_entry)
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):
    """GPT model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = TokEmbed(config)
        self.pos_emb = PosEmbed(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config) if config.use_norm else nn.Identity()
        self.unembed = Unembed(config)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None):
        x = self.tok_emb(x) + self.pos_emb(
            x,
            cache[0].past_k.shape[2] if cache is not None else 0,
        )
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                cache[i] if cache is not None else None,
            )
        x = self.final_norm(x)
        x = self.unembed(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
    ):
        batch_size, seq_len = prompt.shape

        if use_cache:
            cache = KVCache.init_cache(self.config, prompt.shape[0])
        else:
            cache = None

        out = prompt
        for i in range(max_new_tokens):
            if prompt.shape[-1] > self.config.n_ctx:
                break
            logits = self(prompt, cache)
            logits = logits[:, -1, :]
            if sample:
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(
                        logits,
                        top_k,
                        dim=-1,
                    )
                    logits = torch.where(
                        logits < top_k_logits[:, -1:],
                        torch.full_like(logits, -torch.finfo(logits.dtype).max),
                        logits,
                    )

                if top_p is not None:
                    probs = F.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(
                        probs,
                        descending=True,
                    )
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(
                        indices_to_remove,
                        -torch.finfo(logits.dtype).max,
                    )

                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            prompt = (
                next_token
                if cache is not None
                else torch.cat([prompt, next_token], dim=1)
            )
            out = torch.cat([out, next_token], dim=1)

        return out
