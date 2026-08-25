"""
Grading for closed-book SQuAD-style QA.

Primary/default: standard SQuAD F1 + exact-match (dataset's own canonical
metric, deterministic, no external dependency). Optional upgrade: Azure
OpenAI (Azure AI Foundry) semantic yes/no grading, matching how the real
SEAL repo uses GPT-4.1 in general-knowledge/src/utils.py::grade_with_gpt4 --
swapped here for Azure's REST contract (api-key header, deployment-routed).

The grader mode is resolved ONCE per run (see resolve_grader) so a run's
results are internally consistent even if env vars change mid-run.
"""
from __future__ import annotations

import os
import re
import string
from collections import Counter
from typing import Optional

import requests

# --------------------------- SQuAD F1 / EM ----------------------------- #

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def grade_f1em(pred: str, gold: str, f1_threshold: float = 0.5) -> bool:
    if not pred.strip():
        return False
    if exact_match(pred, gold) or f1_score(pred, gold) >= f1_threshold:
        return True
    # A small instruct model answering freely (not extractively) often embeds
    # the right short fact inside a longer sentence -- e.g. gold "114" inside
    # "The Quran has 114 suras, ..." -- which token-level F1 punishes hard for
    # verbosity even though the fact is correct. Reward-model-free substring
    # containment on the normalized strings catches these without needing an
    # LLM judge; it's more permissive than F1 but still requires the exact
    # gold phrase to appear, so it won't reward an unrelated ramble.
    gold_n, pred_n = _normalize(gold), _normalize(pred)
    return bool(gold_n) and gold_n in pred_n


# --------------------------- Azure OpenAI grading ----------------------- #

GRADE_TEMPLATE = (
    "You are a grading assistant. Your job is to determine whether a student's answer correctly "
    "answers the question based solely on the provided gold answer. Do not use any outside knowledge. "
    "The student answer can include additional information, but it must at least fully convey the gold "
    "answer and must not contradict it. Ignore style, phrasing, or extra details that do not affect "
    "correctness. Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'."
)


class AzureGrader:
    def __init__(self):
        self.key = os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    @property
    def available(self) -> bool:
        return bool(self.key and self.endpoint and self.deployment)

    def grade(self, question: str, gold: str, pred: str) -> Optional[bool]:
        if not self.available:
            return None
        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )
        prompt = GRADE_TEMPLATE.format(question=question, gold=gold, pred=pred.strip())
        try:
            r = requests.post(
                url,
                headers={"api-key": self.key, "content-type": "application/json"},
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
                timeout=30,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip().lower()
            has_yes = bool(re.search(r"\byes\b", text))
            has_no = bool(re.search(r"\bno\b", text))
            return has_yes and not has_no
        except Exception:
            return None


def resolve_grader() -> str:
    """Decide grading mode once, at run start. Returns 'azure' or 'f1em'."""
    g = AzureGrader()
    return "azure" if g.available else "f1em"


class Grader:
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or resolve_grader()
        self._azure = AzureGrader() if self.mode == "azure" else None

    def grade(self, question: str, gold: str, pred: str) -> bool:
        if self._azure is not None:
            result = self._azure.grade(question, gold, pred)
            if result is not None:
                return result
            # transient failure -> fall back for this call only
        return grade_f1em(pred, gold)
