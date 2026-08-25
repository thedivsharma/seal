"""Simplest plausible competitor baseline: interleave a sample of previously-seen
articles' own self-edit training sequences into the current step's training batch.
Exists so NSCE has to beat something cheaper than itself, not just plain merging."""
import random
from typing import Any, Dict, List


class ReplayBuffer:
    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def add(self, title: str, train_sequences: List[str]) -> None:
        self._items.append({"title": title, "sequences": train_sequences})

    def sample(self, n: int, rng: random.Random) -> List[str]:
        if not self._items:
            return []
        picks = rng.sample(self._items, min(n, len(self._items)))
        out: List[str] = []
        for p in picks:
            out.append(rng.choice(p["sequences"]))
        return out
