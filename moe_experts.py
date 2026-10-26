import math

import torch
import torch.nn.functional as F
from torch import nn


class Experts(nn.Module):
    def __init__(self, hf_experts: nn.Module):
        super().__init__()

        self.num_experts = hf_experts.num_experts

        self.gate_up_proj = hf_experts.gate_up_proj  # [E, 2h, d], h — ширина эксперта
        self.down_proj = hf_experts.down_proj  # [E, d, h]

    def _weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.dtype]:
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
        else:
            dtype = x.dtype

        return self.gate_up_proj.to(dtype), self.down_proj.to(dtype), dtype


class PackedExperts(Experts):
    def __init__(self, hf_experts: nn.Module, capacity_factor: float = 1.25):
        super().__init__(hf_experts)

        self.capacity_factor = capacity_factor
        self.drop_rate = None

    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """
        x: [tokens, d]
        top_k_index: [tokens, k]
        top_k_weights: [tokens, k]
        returns: [tokens, d]
        """
        tokens, dim = x.shape
        k = top_k_index.shape[-1]
        routes = tokens * k
        capacity = max(1, math.ceil(self.capacity_factor * routes / self.num_experts))

        ids = top_k_index.reshape(-1)  # [routes]
        sorted_ids, order = ids.sort(stable=True)  # [routes], [routes]
        experts = torch.arange(self.num_experts, device=x.device)  # [E]

        starts = torch.searchsorted(sorted_ids, experts)  # [E]
        pos = starts[:, None] + torch.arange(capacity, device=x.device)[None, :]  # [E, C]
        safe = pos.clamp(max=routes - 1)  # [E, C]
        keep = (pos < routes) & (sorted_ids[safe] == experts[:, None])  # [E, C]

        w_gate_up, w_down, dtype = self._weights(x)

        packed = x[order[safe] // k].to(dtype).masked_fill(~keep[..., None], 0)  # [E, C, d]
        gate, up = torch.bmm(packed, w_gate_up.transpose(1, 2)).chunk(2, dim=-1)  # [E, C, h], [E, C, h]
        out = torch.bmm(F.silu(gate) * up, w_down.transpose(1, 2))  # [E, C, d]

        slot = order.argsort() - starts[ids]  # [routes]
        dropped = slot >= capacity  # [routes]
        self.drop_rate = dropped.float().mean().detach()  # []

        out = out[ids, slot.clamp(max=capacity - 1)].masked_fill(dropped[:, None], 0)  # [routes, d]
        out = out * top_k_weights.reshape(-1, 1).to(out.dtype)  # [routes, d]

        return out.reshape(tokens, k, dim).sum(1).to(x.dtype)  # [tokens, d]


class GroupedMMExperts(Experts):
    # Сортировка маршрутов по эксперту + F.grouped_mm
    # Нужен torch 2.15.0 RC
    @torch.compiler.disable
    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        """
        x: [tokens, d]
        top_k_index: [tokens, k]
        top_k_weights: [tokens, k]
        returns: [tokens, d]
        """
        tokens, dim = x.shape
        k = top_k_index.shape[-1]

        ids = top_k_index.reshape(-1)  # [routes]
        sorted_ids, perm = ids.sort()  # [routes], [routes]

        counts = torch.histc(sorted_ids.float(), bins=self.num_experts, min=0, max=self.num_experts - 1)  # [E]
        offs = counts.cumsum(0).to(torch.int32)  # [E]

        w_gate_up, w_down, dtype = self._weights(x)

        xs = x[perm // k].to(dtype)  # [routes, d]
        gate, up = F.grouped_mm(xs, w_gate_up.transpose(1, 2), offs=offs).chunk(2, dim=-1)  # [routes, h], [routes, h]
        out = F.grouped_mm(F.silu(gate) * up, w_down.transpose(1, 2), offs=offs)  # [routes, d]
        out = out * top_k_weights.reshape(-1)[perm, None].to(out.dtype)  # [routes, d]

        result = torch.zeros(tokens, dim, dtype=out.dtype, device=x.device)  # [tokens, d]

        return result.index_add_(0, perm // k, out).to(x.dtype)  # [tokens, d]


def swap_experts(model: nn.Module, impl: str) -> nn.Module:
    cls = {"packed": PackedExperts, "grouped": GroupedMMExperts}[impl]

    for layer in model.model.layers:
        layer.mlp.experts = cls(layer.mlp.experts)

    return model
