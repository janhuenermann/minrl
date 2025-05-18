from contextlib import contextmanager
import math
import re

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, rank: int, in_size: int, out_size: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_size, in_size))
        self.bias = nn.Parameter(torch.empty(out_size)) if bias else None
        self.lora_A = nn.Parameter(torch.empty(rank, in_size))
        self.lora_B = nn.Parameter(torch.zeros(out_size, rank))
        self.use_lora = True
        self.scale = 1.0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.use_lora:
            y += self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return y

    @classmethod
    def wrap(cls, layer: nn.Linear, rank=8, dtype=torch.float32):
        lora = LoRALinear(
            rank=rank,
            in_size=layer.in_features,
            out_size=layer.out_features,
            bias=layer.bias is not None,
        )

        lora.weight.data = layer.weight.data
        lora.weight.requires_grad = False

        if layer.bias is not None:
            lora.bias.data = layer.bias.data
            lora.bias.requires_grad = False

        lora.lora_A.to(dtype=dtype)
        lora.lora_B.to(dtype=dtype)

        lora.to(device=layer.weight.device)

        return lora

    @classmethod
    def find_and_replace(
        cls, model: nn.Module, pattern: str, rank=8, dtype=torch.float32
    ) -> nn.Module:
        for parent_name, parent in model.named_modules():
            for child_name, child in parent.named_children():
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                if re.search(pattern, full_name):
                    if not isinstance(child, torch.nn.Linear):
                        raise ValueError(f"Pattern {pattern} matched non-linear layer {full_name}")

                    setattr(parent, child_name, cls.wrap(child, rank=rank, dtype=dtype))
        return model


class LoRAEmbedding(nn.Module):
    def __init__(self, rank: int, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.lora_A = nn.Parameter(torch.zeros(rank, num_embeddings))
        self.lora_B = nn.Parameter(torch.empty(embedding_dim, rank))
        self.use_lora = True
        self.scale = 1.0
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        y = F.embedding(x, self.weight)
        if self.use_lora:
            y += self.scale * F.linear(F.embedding(x, self.lora_A.T), self.lora_B)
        return y

    def scores(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.weight)
        if self.use_lora:
            y += self.scale * F.linear(F.linear(x, self.lora_B.T), self.lora_A.T)
        return y

    @classmethod
    def wrap(cls, layer: nn.Embedding, rank=8, dtype=torch.float32):
        lora = LoRAEmbedding(
            rank=rank, num_embeddings=layer.num_embeddings, embedding_dim=layer.embedding_dim
        )

        lora.weight.data = layer.weight.data
        lora.weight.requires_grad = False

        lora.lora_A.to(dtype=dtype)
        lora.lora_B.to(dtype=dtype)

        lora.to(device=layer.weight.device)

        return lora

    @classmethod
    def find_and_replace(cls, model: nn.Module, rank=8, dtype=torch.float32) -> nn.Module:
        for parent_name, parent in model.named_modules():
            for child_name, child in parent.named_children():
                if isinstance(child, torch.nn.Embedding):
                    print(f"Replacing {parent_name}.{child_name} with LoRAEmbedding")
                    setattr(parent, child_name, cls.wrap(child, rank=rank, dtype=dtype))
        return model


def set_lora_mode(model: nn.Module, enabled: bool = True):
    before = None
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding)):
            assert (
                before is None or before == module.use_lora
            ), "LoRA mode must be consistent across all LoRA modules"
            before = module.use_lora
            module.use_lora = enabled

    return before


@contextmanager
def lora_mode(model: nn.Module, enabled: bool = True):
    """
    Context manager to enable or disable LoRA mode for a model.
    """
    before = set_lora_mode(model, enabled)
    try:
        yield
    finally:
        if before is not None and before != enabled:
            set_lora_mode(model, before)
