from contextlib import contextmanager

import torch
from torch import Tensor
from flash_attn import flash_attn_with_kvcache
from tabulate import tabulate


def flash_decode(q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin):
    """
    Flash attention with key-value cache. This is a compilable version of `flash_attn_with_kvcache`.
    """
    return torch.ops.flash_attn._flash_attn_with_kvcache(
        q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin
    )


@torch.library.custom_op(
    "flash_attn::_flash_attn_with_kvcache", mutates_args=("k_cache", "v_cache"), device_types="cuda"
)
def flash_attn_kvcache_wrapper(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    k: Tensor,
    v: Tensor,
    cache_seqlens: Tensor,
    rotary_cos: Tensor,
    rotary_sin: Tensor,
) -> Tensor:
    return flash_attn_with_kvcache(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        causal=True,
        rotary_interleaved=False,
    )


@torch.library.register_fake("flash_attn::_flash_attn_with_kvcache")
def flash_attn_kvcache_wrapper_fake(
    q, k_cache, v_cache, k, v, cache_seqlens, rotary_cos, rotary_sin
):
    return torch.empty_like(q)


class TrainingTimer:
    def __init__(self):
        self.start_events: dict[str, torch.cuda.Event] = {}
        self.stop_events: dict[str, torch.cuda.Event] = {}

    @contextmanager
    def __call__(self, name):
        if name not in self.start_events:
            self.start_events[name] = torch.cuda.Event(enable_timing=True)
            self.stop_events[name] = torch.cuda.Event(enable_timing=True)

        self.start_events[name].record()
        yield
        self.stop_events[name].record()

    def get_timings(self):
        torch.cuda.synchronize()
        return {
            name: self.start_events[name].elapsed_time(self.stop_events[name])
            for name in self.start_events
        }

    def print_timings(self):
        table = []
        # Headings
        table.append(["Name", "Time (ms)"])
        # Data
        for name, time in self.get_timings().items():
            table.append([name, f"{time:.2f}"])
        # Print the table
        print(tabulate(table, headers="firstrow", tablefmt="grid"))
        self.start_events.clear()
        self.stop_events.clear()
