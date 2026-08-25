"""
Hand-rolled LoRA: explicit A/B tensors per target nn.Linear, kept accessible
so NSCE can project A before it's folded into the base weight (peft's
merge_and_unload() hides this, which is why the earlier prototype couldn't
use peft directly). Merging is done in-place on self.base.weight -- no full
model checkpoint is written to disk between steps, which matters given this
machine's disk budget.

Convention: y = W x + b + scaling * (x @ A^T) @ B^T
  A: [r, in_features], B: [out_features, r], scaling = alpha / r
  dW = scaling * B @ A   (this is why protecting A's row space protects dW's
  row space -- see lib/nsce.py for why that's the mechanism that matters)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, in_f, device=device, dtype=dtype) * (1.0 / r ** 0.5))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=device, dtype=dtype))

    def forward(self, x):
        return self.base(x) + self.scaling * ((x @ self.A.T) @ self.B.T)

    def reset(self):
        with torch.no_grad():
            self.A.copy_(torch.randn_like(self.A) * (1.0 / self.r ** 0.5))
            self.B.zero_()

    @torch.no_grad()
    def delta_weight(self, A_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        A = self.A if A_override is None else A_override
        return self.scaling * (self.B @ A)

    @torch.no_grad()
    def merge_into_base(self, A_override: Optional[torch.Tensor] = None) -> None:
        self.base.weight.add_(self.delta_weight(A_override).to(self.base.weight.dtype))


def inject_lora(model: nn.Module, target_suffixes: Tuple[str, ...], r: int, alpha: float) -> Dict[str, LoRALinear]:
    adapters: Dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suf) for suf in target_suffixes):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = LoRALinear(module, r=r, alpha=alpha)
        setattr(parent, attr, wrapped)
        adapters[name] = wrapped
    return adapters


def trainable_parameters(adapters: Dict[str, LoRALinear]) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for ad in adapters.values():
        params += [ad.A, ad.B]
    return params


@torch.no_grad()
def capture_activations(
    adapters: Dict[str, LoRALinear], model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Run one forward pass, capturing each target module's *input* activation
    (the vector the module's A matrix actually acts on) for every real (non-pad) token."""
    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return hook

    for name, ad in adapters.items():
        handles.append(ad.base.register_forward_pre_hook(make_hook(name)))

    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()

    mask = attention_mask.bool()
    out: Dict[str, torch.Tensor] = {}
    for name, act in captured.items():
        # act: [batch, seq, in_features] -> flatten to real tokens only
        out[name] = act[mask]
    return out
