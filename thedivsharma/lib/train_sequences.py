"""
Ported verbatim (split_newlines path only) from general-knowledge/src/utils.py
so continual self-edits here are preprocessed identically to the real SEAL
pipeline's own continual_self_edits.py driver.
"""
import re
from typing import List

MAX_TRAIN_SEQS_PER_COMPLETION = 30
TRAINING_SEQUENCE_TEMPLATE = "{title}\n{completion_text}"


def _split_segments(text: str) -> List[str]:
    return [seg.strip() for seg in text.split("---") if seg.strip()]


def build_train_sequences(
    completion_raw: str,
    context: str,
    title: str,
    *,
    split_newlines: bool = True,
    add_context: bool = True,
) -> List[str]:
    m = re.search(r"\nImplications:\s*", completion_raw)
    if m:
        completion_raw = completion_raw[m.end():].lstrip()

    segs = _split_segments(completion_raw) or [completion_raw.strip()]
    if split_newlines:
        if re.search(r'Question\s+\d+:', completion_raw) and re.search(r'Answer\s*:', completion_raw):
            segs = re.split(r'\n(?=Question\s+\d+:)', completion_raw.strip())
            if not segs[0].lstrip().startswith("Question"):
                segs[0] = "Question 1: " + segs[0].strip()
        else:
            segs = [ln.strip() for seg in segs for ln in seg.splitlines() if ln.strip()]
            if len(segs) > 1 and segs[1].startswith("1."):
                segs = segs[1:]

    if len(segs) > MAX_TRAIN_SEQS_PER_COMPLETION:
        segs = segs[:MAX_TRAIN_SEQS_PER_COMPLETION]
    segs = [s for s in segs if s.strip()]
    seqs = [TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=s) for s in segs]
    if add_context:
        seqs.append(TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=context.strip()))
    return seqs
