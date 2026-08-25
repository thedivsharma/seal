"""
Orchestrates the full experiment: for each seed, generate (or reuse) the
self-edit cache, then run baseline / replay / nsce sequentially (one process
each, so MPS memory is fully released between runs regardless of anything
left resident from the previous one). Resumable: skips any (seed, mode) run
that already completed all K steps, and skips cache generation if the cache
file already exists.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/anaconda3/envs/seal_env/bin/python"

SEEDS = [1, 2, 3]
MODES = ["baseline", "replay", "nsce"]
K_ARTICLES = 15
QUESTIONS_PER_ARTICLE = 8


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    print(f"  -> exit {result.returncode} ({time.time()-t0:.0f}s)", flush=True)
    return result.returncode == 0


def cache_ready(seed: int) -> bool:
    p = REPO_ROOT / "results" / "cache" / f"seed{seed}.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return len(data.get("articles", [])) == K_ARTICLES
    except Exception:
        return False


def run_ready(seed: int, mode: str) -> bool:
    p = REPO_ROOT / "results" / "runs" / f"seed{seed}_{mode}.json"
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        return data.get("steps_completed") == K_ARTICLES
    except Exception:
        return False


def main():
    failures = []
    for seed in SEEDS:
        if cache_ready(seed):
            print(f"[skip] cache seed{seed} already complete")
        else:
            ok = run([
                PYTHON, "generate_self_edits.py",
                "--seed", str(seed),
                "--k_articles", str(K_ARTICLES),
                "--questions_per_article", str(QUESTIONS_PER_ARTICLE),
            ])
            if not ok:
                failures.append(f"cache seed{seed}")
                continue

        for mode in MODES:
            if run_ready(seed, mode):
                print(f"[skip] run seed{seed}_{mode} already complete")
                continue
            ok = run([PYTHON, "run_continual.py", "--seed", str(seed), "--mode", mode, "--eval_batch_size", "8"])
            if not ok:
                failures.append(f"run seed{seed}_{mode}")

    print("\n=== SUITE COMPLETE ===")
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)
    print("all runs completed successfully")


if __name__ == "__main__":
    main()
