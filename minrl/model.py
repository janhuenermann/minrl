import os

import torch
from torch import Tensor
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from minrl.modeling.lora import LoRAEmbedding
from minrl.utils import flash_decode


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        qk_norm=False,
        qkv_bias=False,
    ):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_proj = nn.Linear(hidden_size, self.head_dim * num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.head_dim * num_kv_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.head_dim * num_kv_heads, bias=qkv_bias)
        self.o_proj = nn.Linear(self.head_dim * num_heads, hidden_size, bias=False)
        self.q_norm = RmsNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RmsNorm(self.head_dim) if qk_norm else nn.Identity()
        self.gqa = num_kv_heads != num_heads

    def forward(self, x: Tensor, pos_emb: Tensor, cache_pos: Tensor | None = None):
        b, n, _ = x.shape

        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, n, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, n, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cache_pos is not None:
            rotary_cos, rotary_sin = pos_emb
            y = flash_decode(
                q=q,
                k=k,
                v=v,
                k_cache=self.key_states,
                v_cache=self.value_states,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                cache_seqlens=cache_pos,
            )

        else:
            q, k = rotate(q, k, pos_emb)
            q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.gqa)
            y = y.transpose(1, 2)

        y = self.o_proj(y.flatten(2))
        return y

    def init_kv_cache(self, batch_size: int, seq_len: int):
        x = self.q_proj.weight
        self.register_buffer(
            "key_states",
            x.new_empty((batch_size, seq_len, self.num_kv_heads, self.head_dim)),
            persistent=False,
        )
        self.register_buffer(
            "value_states",
            x.new_empty((batch_size, seq_len, self.num_kv_heads, self.head_dim)),
            persistent=False,
        )

    def release_kv_cache(self):
        if "key_states" in self._buffers:
            del self.key_states
        if "value_states" in self._buffers:
            del self.value_states


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor):
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        return self.down_proj(F.silu(x1) * x2)


class RmsNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim,)

    def forward(self, x: Tensor):
        output = F.rms_norm(x.to(torch.float32), self.normalized_shape, eps=self.eps)
        return output.type_as(x) * self.weight


class Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        qkv_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Attention(
            hidden_size, num_heads, num_kv_heads, head_dim, qk_norm, qkv_bias
        )
        self.mlp = Mlp(hidden_size, intermediate_size)

    def forward(self, x: Tensor, pos_emb: tuple[Tensor, Tensor], cache_pos: Tensor | None = None):
        x = x + self.self_attn(self.input_layernorm(x), pos_emb, cache_pos)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def precompute_rotary(dim: int, max_seq_len: int, rope_theta: float = 500000.0):
    inv_freq = rope_theta ** (-torch.arange(0, dim, 2, dtype=torch.float32, device="cpu") / dim)
    # TODO: this is necessary to match float16 HF implementation, but is it good?
    inv_freq = inv_freq.to(torch.float16)
    position = torch.arange(max_seq_len, device="cpu")

    with torch.autocast(device_type=position.device.type, enabled=False):
        w = inv_freq[None, :].float() * position[:, None].float()
        cos = w.cos()
        sin = w.sin()

    return torch.stack((cos, sin), dim=0)


class LanguageModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 500000.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.rope_theta = rope_theta
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads if head_dim is None else head_dim
        self.layers = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    qk_norm,
                    qkv_bias,
                    rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.norm = RmsNorm(hidden_size, eps=rms_norm_eps)
        self.register_buffer(
            "rotary", precompute_rotary(self.head_dim, max_seq_len, rope_theta), persistent=False
        )

    def forward(
        self,
        x: Tensor,
        cache_pos: Tensor | None = None,
        return_next_token_logits=False,
        return_logits=False,
        return_sample=False,
    ):
        if not torch.is_floating_point(x):
            x = self.embed_tokens(x)

        for layer in self.layers:
            x = layer(x, self.rotary, cache_pos)

        x = self.norm(x)

        if return_sample:
            x = x[:, -1:]

            if isinstance(self.embed_tokens, LoRAEmbedding):
                logits = self.embed_tokens.dot_product(x)
            else:
                logits = torch.matmul(x, self.embed_tokens.weight.t())

            return sample(logits)
        if return_next_token_logits:
            return torch.matmul(x[:, -1:], self.embed_tokens.weight.t())
        if return_logits:
            return torch.matmul(x, self.embed_tokens.weight.t())

        return x

    def decode_prefill(self, x: Tensor, cache_pos: Tensor | None = None):
        return self.forward(x, cache_pos=cache_pos, return_sample=True)

    def decode_one(self, x: Tensor, cache_pos: Tensor | None = None):
        assert x.ndim == 1, "Please pass a tensor of shape (batch_size,)."
        return self.forward(x[:, None], cache_pos=cache_pos, return_sample=True)

    def init_kv_cache(self, batch_size: int, seq_len=1024):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(batch_size, seq_len)
        return self

    def release_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.release_kv_cache()
        return self

    @classmethod
    def from_hf(
        cls,
        model_name: str,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **overrides,
    ):
        if model_name not in MODEL_TO_CONFIG:
            raise ValueError(f"Model {model_name} not found in MODEL_TO_CONFIG")

        config = MODEL_TO_CONFIG[model_name].copy()
        config.update(overrides)

        with torch.device("meta"):
            model = cls(**config)

        repo_path = snapshot_download(repo_id=model_name, allow_patterns=["*.safetensors"])

        state_dict = {}
        for name in os.listdir(repo_path):
            if name.endswith(".safetensors"):
                state_dict.update(load_file(os.path.join(repo_path, name)))

        state_dict = _strip_prefix(state_dict, "model.")
        model.load_state_dict(state_dict, assign=True)

        if device is not None or dtype is not None:
            model.to(device=device, dtype=dtype)

        return model


def to_compiled(model: LanguageModel):
    model.forward = torch.compile(model.forward, dynamic=True, fullgraph=True)
    # Dynamic and no cuda graphs for prefill
    model.decode_prefill = torch.compile(model.decode_prefill, dynamic=True, fullgraph=True)
    # Not dynamic but cuda graph for decode one
    model.decode_one = torch.compile(model.decode_one, fullgraph=True, mode="reduce-overhead")
    return model


def generate(
    model: LanguageModel,
    input_ids: Tensor,
    generation_length: int,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    early_stopping: bool = True,
) -> Tensor:
    if pad_token_id is None:
        pad_token_id = -1

    input_mask = input_ids != pad_token_id
    input_mask_diff = input_mask.long().diff(1, dim=1)
    if torch.any(input_mask_diff > 0):
        raise ValueError("input_ids must be right padded")
    if torch.any((input_mask_diff != 0).sum(-1) > 1):
        raise ValueError("input_ids can only have padding at the end")

    input_len = input_mask.sum(dim=1)
    min_input_len = input_len.min().item()
    max_input_len = input_len.max().item()
    seqlen = max_input_len + generation_length

    B = input_ids.size(0)
    tokens = torch.full((B, seqlen), pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
    is_finished = torch.zeros((B,), device=input_ids.device, dtype=torch.bool)

    # Copy prompt into scratchpad
    tokens[:, :max_input_len] = input_ids[:, :max_input_len]

    prev_pos = 0
    cache_pos = torch.zeros((B,), device=input_ids.device, dtype=torch.int32)
    for cur_pos in range(min_input_len, seqlen):
        if prev_pos == 0:
            generated_ids = model.decode_prefill(tokens[:, :cur_pos], cache_pos)
        else:
            generated_ids = model.decode_one(tokens[:, cur_pos - 1].contiguous(), cache_pos)

        generated_ids = generated_ids.squeeze(-1)

        cur_ids = tokens[:, cur_pos]
        is_generated_token = cur_ids == pad_token_id
        next_ids = torch.where(is_generated_token, generated_ids, cur_ids)
        next_ids = torch.where(is_finished, pad_token_id, next_ids)

        tokens[:, cur_pos] = next_ids
        prev_pos = cur_pos
        cache_pos.fill_(cur_pos)

        if eos_token_id is not None:
            is_finished = is_finished | ((next_ids == eos_token_id) & is_generated_token)
            if early_stopping and is_finished.all():
                # This comes at a slight cost of performance due to additional cuda sync
                break

    return tokens


def rotate(q: Tensor, k: Tensor, pos_emb: Tensor):
    cos, sin = pos_emb[:, : q.size(1), None, :]

    q1, q2 = q.chunk(2, dim=-1)
    q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)

    k1, k2 = k.chunk(2, dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

    return q, k


def sample(logits, temperature: float = 1.0, top_p: float | None = None, top_k: int | None = 50):
    if temperature == 0.0:
        return logits.argmax(-1)

    # Optionally crop the logits to only the top k options
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[..., -1:]] = float("-inf")

    probs = torch.softmax(logits / temperature, dim=-1)

    # Optionally crop the probabilities to only the top p options
    if top_p is not None:
        sorted_probs = torch.sort(probs, dim=-1, descending=True).values
        pivot_index = (sorted_probs.cumsum(-1) > top_p).max(-1, True).indices
        pivot = sorted_probs.gather(-1, pivot_index)
        probs[probs < pivot] = 0
        probs = probs / probs.sum(dim=-1, keepdim=True)

    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1)


def _strip_prefix(state_dict, prefix: str):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


MODEL_TO_CONFIG = {
    "Qwen/Qwen2.5-3B-Instruct": {
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_heads": 16,
        "num_kv_heads": 2,
        "num_layers": 36,
        "qkv_bias": True,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_heads": 12,
        "num_kv_heads": 2,
        "num_layers": 28,
        "qkv_bias": True,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "Qwen/Qwen3-1.7B": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_layers": 28,
        "qk_norm": True,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "Qwen/Qwen3-1.7B-Base": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_layers": 28,
        "qk_norm": True,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
    },
    "Qwen/Qwen3-0.6B": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_layers": 28,
        "qk_norm": True,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "head_dim": 128,
    },
    "Qwen/Qwen3-0.6B-Base": {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_heads": 16,
        "num_kv_heads": 8,
        "num_layers": 28,
        "qk_norm": True,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "head_dim": 128,
    },
}
