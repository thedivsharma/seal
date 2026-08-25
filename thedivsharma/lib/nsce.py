"""
NSCE v2 -- Null-Space Constrained Edits, redone against real activation
statistics rather than reusing prior adapters' own A row-space (that was
the earlier prototype's proxy, and its weakness: "protect whatever
direction the optimizer happened to move in last time" isn't the same
claim as "protect the directions that matter for the fact itself").

This follows AlphaEdit (Fang et al., ICLR 2025) -- the exact method the
SEAL paper's own Limitations section (arXiv 2506.10943, §5) names as a
candidate fix for the catastrophic forgetting its Figure 6 documents.
AlphaEdit's core idea: protect the input directions that matter for
previously-taught content, measured from real forward-pass activations on
that content, not from an adapter's own optimizer trajectory.

For each target nn.Linear (a q_proj/v_proj inside one transformer layer),
we accumulate an uncentered covariance matrix

    C = sum_t x_t x_t^T

over every previously-merged self-edit's source-passage token activations
x_t (the actual input to that Linear). The protected subspace U is the top
eigenvectors of C, kept up to an energy threshold (fraction of variance
explained) and hard-capped by a rank budget. That budget directly targets
the prototype's failure mode: protected rank there grew unboundedly
(2,240 -> 22,400 over 10 steps), which -- left unchecked -- eventually
protects the entire input space and starves every future self-edit of room
to write anything new. Capping it trades some retention for guaranteed
plasticity; see thedivsharma/README.md for the resulting curve.

A new self-edit's LoRA A [r, in_features] is projected orthogonal to U
before its delta (scaling * B @ A) is merged into the base weights. Because
dW = scaling * B @ A, and A_orth's rows are (by construction) orthogonal to
every column of U, we get dW @ u ~= 0 for every direction u in the
protected subspace regardless of what B is doing -- i.e. the merged update
leaves those directions untouched to first order.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


class NullSpaceTracker:
    def __init__(self, energy_threshold: float = 0.9, rank_budget: Optional[int] = 64, eps: float = 1e-6):
        self.energy_threshold = energy_threshold
        self.rank_budget = rank_budget
        self.eps = eps
        self._C: Dict[str, torch.Tensor] = {}
        self._U: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def collect(self, module_name: str, activations: torch.Tensor) -> None:
        """activations: [N_tokens, in_features], real (non-pad) tokens only."""
        x = activations.detach().to(torch.float32).cpu()
        cov = x.T @ x
        self._C[module_name] = self._C.get(module_name, torch.zeros_like(cov)) + cov
        self._refresh_basis(module_name)

    def _refresh_basis(self, module_name: str) -> None:
        C = self._C[module_name]
        eigvals, eigvecs = torch.linalg.eigh(C)
        order = torch.argsort(eigvals, descending=True)
        eigvals, eigvecs = eigvals[order].clamp_min(0.0), eigvecs[:, order]
        total = eigvals.sum().item()
        if total <= 0:
            self._U[module_name] = eigvecs[:, :0]
            return
        cumulative = torch.cumsum(eigvals, dim=0) / total
        k = int(torch.searchsorted(cumulative, torch.tensor(self.energy_threshold)).item()) + 1
        k = max(1, k)
        if self.rank_budget is not None:
            k = min(k, self.rank_budget)
        self._U[module_name] = eigvecs[:, :k].contiguous()

    def project(self, module_name: str, A: torch.Tensor) -> torch.Tensor:
        U = self._U.get(module_name)
        if U is None or U.shape[1] == 0:
            return A
        U = U.to(device=A.device, dtype=A.dtype)
        coeff = A @ U
        return A - coeff @ U.T

    def collision_fraction(self, module_name: str, A: torch.Tensor) -> float:
        """Diagnostic: what fraction of this *raw* (pre-projection) self-edit's
        update energy already lies in the protected subspace -- i.e. how much a
        plain merge would have overwritten here had NSCE not intervened."""
        U = self._U.get(module_name)
        if U is None or U.shape[1] == 0:
            return 0.0
        U = U.to(device=A.device, dtype=A.dtype)
        coeff = A @ U
        overlap = (coeff ** 2).sum().item()
        total = (A ** 2).sum().item()
        return overlap / total if total > 0 else 0.0

    def subspace_rank(self, module_name: str) -> int:
        U = self._U.get(module_name)
        return 0 if U is None else U.shape[1]

    def total_protected_rank(self) -> int:
        return sum(u.shape[1] for u in self._U.values())

    def mean_collision_fraction(self, adapters_A: Dict[str, torch.Tensor]) -> float:
        vals = [self.collision_fraction(name, A) for name, A in adapters_A.items()]
        return sum(vals) / len(vals) if vals else 0.0
