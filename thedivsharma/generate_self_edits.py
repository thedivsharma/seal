"""
Pre-step, run once per seed: sample K real SQuAD articles, generate one
self-edit per article via local Ollama, and evaluate the untouched base
model on all K articles' questions (mode-independent "row 0"). Cached so
baseline / replay / nsce conditions for a given seed all train on and are
evaluated against exactly the same self-edits and the same row-0 numbers --
the only thing that should differ between conditions is the merge step.
"""
import argparse
import json
import time
from pathlib import Path

from lib.grading import resolve_grader
from lib.model_utils import answer_questions, get_device, load_model
from lib.self_edit_gen import generate_self_edit
from lib.squad_data import load_articles, sample_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--k_articles", type=int, default=15)
    p.add_argument("--questions_per_article", type=int, default=8)
    p.add_argument("--dataset", default=str(REPO_ROOT / "general-knowledge/data/squad_train.json"))
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--self_edit_model", default="qwen2.5:1.5b")
    p.add_argument("--output_dir", default=str(Path(__file__).parent / "results" / "cache"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed{args.seed}.json"
    if out_path.exists():
        print(f"[skip] cache already exists: {out_path}")
        return

    print(f"[grader] resolved mode: {resolve_grader()}")

    articles = load_articles(args.dataset, min_questions=args.questions_per_article)
    print(f"[data] {len(articles)} candidate articles (>= {args.questions_per_article} questions)")
    seq = sample_sequence(articles, args.k_articles, args.questions_per_article, seed=args.seed)

    print(f"[self-edit] generating {len(seq)} self-edits via ollama/{args.self_edit_model} ...")
    for i, item in enumerate(seq):
        t0 = time.time()
        item["self_edit"] = generate_self_edit(
            item["title"], item["context"], model=args.self_edit_model, temperature=1.0, seed=args.seed * 1000 + i
        )
        print(f"  [{i+1}/{len(seq)}] {item['title'][:60]!r} ({time.time()-t0:.1f}s, {len(item['self_edit'])} chars)")

    print("[row0] loading base model for untouched-baseline eval ...")
    device = get_device()
    model, tokenizer = load_model(args.model_name, device)

    all_questions = []
    spans = []
    cum = 0
    for item in seq:
        all_questions.extend(
            [{"question": f"Topic: {item['title']}\n{q['question']}", "answer": q["answer"]} for q in item["questions"]]
        )
        spans.append((cum, cum + len(item["questions"])))
        cum += len(item["questions"])

    t0 = time.time()
    preds = answer_questions(model, tokenizer, device, all_questions, batch_size=8)
    print(f"[row0] answered {len(all_questions)} questions in {time.time()-t0:.1f}s")

    for item, (s, e) in zip(seq, spans):
        item["row0_predictions"] = preds[s:e]

    del model
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    cache = {
        "seed": args.seed,
        "k_articles": args.k_articles,
        "questions_per_article": args.questions_per_article,
        "model_name": args.model_name,
        "self_edit_model": args.self_edit_model,
        "articles": seq,
    }
    out_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
