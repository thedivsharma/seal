"""
Continual self-edit driver for one (seed, mode) run. mode in {baseline, replay, nsce}.

Mirrors general-knowledge/src/continual/continual_self_edits.py's experiment
design (lower-triangular accuracy matrix over K sequential self-edits, each
merged permanently into the base weights) but keeps the model resident in
memory across the whole sequence and merges LoRA deltas in-place (lib/lora.py)
instead of writing a full merged checkpoint to disk every step -- this
machine has ~6GB free, a 1.5B model in fp32 is ~6GB, so per-step checkpoints
were not an option.

Per Phase 2 of the roadmap: only the merge step should differ between
conditions, so the null-space tracker's diagnostic (collision_fraction) is
computed identically in all three modes -- only `nsce` actually acts on it
(projects A before merging). That makes the collision-fraction comparison
across conditions apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from lib.grading import Grader
from lib.lora import LoRALinear
from lib.model_utils import (
    answer_questions,
    collect_context_activations,
    free_memory,
    get_device,
    load_model,
    setup_lora,
    train_lora_step,
)
from lib.nsce import NullSpaceTracker
from lib.replay import ReplayBuffer
from lib.train_sequences import build_train_sequences

REPO_ROOT = Path(__file__).resolve().parent


def qualify_questions(title: str, questions: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [{"question": f"Topic: {title}\n{q['question']}", "answer": q["answer"]} for q in questions]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", choices=["baseline", "replay", "nsce"], required=True)
    p.add_argument("--cache_dir", default=str(REPO_ROOT / "results" / "cache"))
    p.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "runs"))
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--replay_n", type=int, default=2)
    p.add_argument("--nsce_energy_threshold", type=float, default=0.9)
    p.add_argument("--nsce_rank_budget", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=4)
    args = p.parse_args()

    cache_path = Path(args.cache_dir) / f"seed{args.seed}.json"
    cache = json.loads(cache_path.read_text())
    seq = cache["articles"]
    K = len(seq)
    model_name = cache["model_name"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{args.seed}_{args.mode}.json"

    grader = Grader()
    print(f"[grader] mode={grader.mode}")

    device = get_device()
    print(f"[model] loading {model_name} on {device} ...")
    model, tokenizer = load_model(model_name, device)
    adapters: Dict[str, LoRALinear] = setup_lora(model, r=args.lora_r, alpha=args.lora_alpha)
    print(f"[lora] {len(adapters)} target modules")

    tracker = NullSpaceTracker(energy_threshold=args.nsce_energy_threshold, rank_budget=args.nsce_rank_budget)
    replay_buffer = ReplayBuffer()
    rng = random.Random(args.seed + 7919)

    # Row 0: from cache (untouched base model), graded here for consistency with this run's grader.
    R = K + 1
    mat_vals: List[List[List[float]]] = [[[] for _ in range(K)] for _ in range(R)]
    for i, item in enumerate(seq):
        preds = item["row0_predictions"]
        correct = [grader.grade(q["question"], q["answer"], p) for q, p in zip(item["questions"], preds)]
        mat_vals[0][i].append(sum(correct) / len(correct))

    diagnostics = {"collision_fraction": [], "protected_rank": []}

    for k in range(K):
        item = seq[k]
        t_step = time.time()
        train_sequences = build_train_sequences(item["self_edit"], item["context"], item["title"], split_newlines=True)

        if args.mode == "replay":
            train_sequences = train_sequences + replay_buffer.sample(args.replay_n, rng)

        final_loss = train_lora_step(model, tokenizer, adapters, device, train_sequences, epochs=args.epochs, lr=args.lr)
        diagnostics.setdefault("final_loss", []).append(final_loss)

        with torch.no_grad():
            A_snapshot = {name: ad.A.detach().clone() for name, ad in adapters.items()}
        collision = tracker.mean_collision_fraction(A_snapshot)
        diagnostics["collision_fraction"].append(collision)
        del A_snapshot

        for name, ad in adapters.items():
            if args.mode == "nsce":
                A_orth = tracker.project(name, ad.A.detach())
                ad.merge_into_base(A_override=A_orth)
            else:
                ad.merge_into_base()
        free_memory()

        # Extend the protected subspace with this article's real activations on the
        # just-updated model (all modes track this for diagnostic parity; only nsce
        # ever calls .project() above). Each captured activation is moved to CPU
        # inside tracker.collect() immediately, so nothing here should linger on
        # the MPS device past this block -- explicit del + free_memory() to be sure.
        acts = collect_context_activations(model, tokenizer, device, adapters, item["title"], item["context"])
        for name, act in list(acts.items()):
            tracker.collect(name, act)
            del act
        acts.clear()
        del acts
        diagnostics["protected_rank"].append(tracker.total_protected_rank())
        free_memory()

        replay_buffer.add(item["title"], train_sequences)

        # Evaluate on all articles seen so far (0..k).
        eval_questions: List[Dict[str, str]] = []
        spans = []
        cum = 0
        for i in range(k + 1):
            qs = qualify_questions(seq[i]["title"], seq[i]["questions"])
            eval_questions.extend(qs)
            spans.append((cum, cum + len(qs)))
            cum += len(qs)

        preds = answer_questions(model, tokenizer, device, eval_questions, batch_size=args.eval_batch_size)

        for i, (s, e) in enumerate(spans):
            gold = seq[i]["questions"]
            p_slice = preds[s:e]
            correct = [grader.grade(q["question"], q["answer"], p) for q, p in zip(gold, p_slice)]
            acc = sum(correct) / len(correct)
            mat_vals[k + 1][i].append(acc)
        del preds
        free_memory()

        print(
            f"[{args.mode} seed{args.seed}] step {k+1}/{K} '{item['title'][:40]}' "
            f"collision={collision:.3f} protected_rank={diagnostics['protected_rank'][-1]} "
            f"acc_row=[{', '.join(f'{mat_vals[k+1][i][0]:.2f}' for i in range(k+1))}] "
            f"({time.time()-t_step:.1f}s)"
        )

        # Checkpoint after every step (resumable, inspectable mid-run).
        checkpoint = {
            "seed": args.seed,
            "mode": args.mode,
            "grader": grader.mode,
            "k_articles": K,
            "steps_completed": k + 1,
            "args": vars(args),
            "titles": [a["title"] for a in seq],
            "mean_matrix": [[sum(c) / len(c) if c else None for c in row] for row in mat_vals],
            "diagnostics": diagnostics,
        }
        out_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False))

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"[done] {args.mode} seed={args.seed} -> {out_path}")


if __name__ == "__main__":
    main()
