from math import sqrt
from typing import Optional

import torch.nn.functional as F
import torch
from torch import nn, Tensor
from einops import rearrange

from .util import RMSNorm, SwiGLU


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.scale = sqrt(d_model / n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def apply_rotary_emb(x: Tensor, freqs: Tensor) -> Tensor:
        return torch.view_as_real(
            torch.view_as_complex(x.unflatten(-1, (-1, 2))) *
            freqs.unsqueeze(0)
        )

    def forward(self, x: Tensor, freqs: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        B, L, _ = x.shape

        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(x), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(x), "b l (n d) -> b n l d", n=self.n_heads)

        q = self.apply_rotary_emb(q, freqs)
        k = self.apply_rotary_emb(k, freqs)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask.view(B, 1, -1, L)

        return self.W_o(
            rearrange(
                self.dropout(F.softmax(attn, dim=-1)) @ v, "b n l d -> b l (n d)"
            )
        )


class TransformerBlock(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int,
            d_hidden: Optional[int] = None, norm_eps: float = 1e-6,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(TransformerBlock, self).__init__()
        self.attn = Attention(d_model, n_heads, attn_dropout)
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps),
            SwiGLU(d_model, d_hidden),
            nn.Dropout(ffn_dropout)
        )

    def forward(self, x: Tensor, freqs: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(
            self.attn_norm(x), freqs, attention_mask
        )

        return x + self.ffn(x)


class Transformer(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int, n_layers: int, vocab_size: int, max_length: int,
            padding_token: int = 0,
            d_hidden: Optional[int] = None, norm_eps: float = 1e-6,
            attn_dropout: float = 0.0, ffn_dropout: float = 0.0
    ):
        super(Transformer, self).__init__()
        assert d_model % 2 == 0
        self.d_model = d_model

        self._set_buffers(max_length)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_token)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_hidden, norm_eps, attn_dropout, ffn_dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            RMSNorm(d_model, eps=norm_eps),
            nn.Linear(d_model, vocab_size)
        )

    def _set_buffers(self, max_length: int):
        self.max_length = max_length

        freqs = 1.0 / (1e5 ** (2 * torch.arange(self.d_model // 2) / self.d_model))
        t = torch.arange(self.max_length)
        freqs = torch.outer(t, freqs)

        self.register_buffer(
            "freqs",
            torch.polar(torch.ones_like(freqs), freqs),
            persistent=False
        )

        self.register_buffer(
            "attention_mask",
            torch.triu(
                torch.ones(self.max_length, self.max_length), diagonal=1
            ),
            persistent=False
        )

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[1]

        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, self.freqs[:L], self.attention_mask[:L, :L])

        return self.head(x)
