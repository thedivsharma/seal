"""Deterministic sampling of real SQuAD articles for continual self-edit sequences."""
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_articles(path: str, min_questions: int = 8) -> List[Dict[str, Any]]:
    data = json.load(open(path, encoding="utf-8"))
    return [a for a in data if len(a.get("questions", [])) >= min_questions]


def sample_sequence(
    articles: List[Dict[str, Any]], k: int, questions_per_article: int, seed: int
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    chosen = rng.sample(articles, k)
    out = []
    for a in chosen:
        qs = rng.sample(a["questions"], questions_per_article) if len(a["questions"]) > questions_per_article else list(a["questions"])
        out.append({"title": a["title"], "context": a["context"], "questions": qs})
    return out
