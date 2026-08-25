"""
Aggregate results/runs/seed{S}_{mode}.json across seeds into a per-mode
comparison, mirroring the SEAL paper's own catastrophic-forgetting analysis
(Figure 6 / Table 5): for article i, row (i+1) column i is "accuracy right
after this article was taught" (plasticity) and row K column i is "accuracy
at the very end of the sequence" (retention). The ratio of the two is the
per-article retention metric; averaging over articles and seeds with SEM
gives the headline baseline-vs-replay-vs-nsce comparison.
"""
import json
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "results" / "runs"
MODES = ["baseline", "replay", "nsce"]


def sem(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return st.stdev(vals) / (len(vals) ** 0.5)


def load_runs(mode: str) -> List[Dict[str, Any]]:
    runs = []
    for p in sorted(RUNS_DIR.glob(f"seed*_{mode}.json")):
        data = json.loads(p.read_text())
        if data.get("steps_completed") == len(data["mean_matrix"]) - 1:
            runs.append(data)
    return runs


def per_run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    mat = run["mean_matrix"]
    K = len(mat) - 1
    diag, final, ratios = [], [], []
    for i in range(K):
        d = mat[i + 1][i]
        f = mat[K][i]
        if d is None or f is None:
            continue
        diag.append(d)
        final.append(f)
        if d > 0:
            ratios.append(f / d)
    diag_curve = [mat[i + 1][i] for i in range(K)]
    final_row = mat[K][:K]
    return {
        "seed": run["seed"],
        "diag_mean": st.mean(diag) if diag else 0.0,
        "final_mean": st.mean(final) if final else 0.0,
        "retention_ratio_mean": st.mean(ratios) if ratios else None,
        "diag_curve": diag_curve,
        "final_row": final_row,
        "collision_fraction": run["diagnostics"].get("collision_fraction", []),
        "protected_rank": run["diagnostics"].get("protected_rank", []),
        "final_loss": run["diagnostics"].get("final_loss", []),
    }


def aggregate(mode: str) -> Optional[Dict[str, Any]]:
    runs = load_runs(mode)
    if not runs:
        return None
    per_run = [per_run_metrics(r) for r in runs]
    diag_means = [m["diag_mean"] for m in per_run]
    final_means = [m["final_mean"] for m in per_run]
    ratio_means = [m["retention_ratio_mean"] for m in per_run if m["retention_ratio_mean"] is not None]
    mean_collision = [st.mean(m["collision_fraction"]) for m in per_run if m["collision_fraction"]]

    return {
        "mode": mode,
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "plasticity_mean_accuracy": st.mean(diag_means),
        "plasticity_sem": sem(diag_means),
        "retained_mean_accuracy": st.mean(final_means),
        "retained_sem": sem(final_means),
        "retention_ratio_mean": st.mean(ratio_means) if ratio_means else None,
        "retention_ratio_sem": sem(ratio_means) if ratio_means else None,
        "mean_collision_fraction": st.mean(mean_collision) if mean_collision else None,
        "per_seed": per_run,
    }


def main():
    results = {mode: aggregate(mode) for mode in MODES}

    print("=" * 78)
    print("NSCE vs baseline vs replay -- aggregate over seeds")
    print("=" * 78)
    for mode in MODES:
        r = results[mode]
        if r is None:
            print(f"{mode:10s}  (no completed runs yet)")
            continue
        ratio = r["retention_ratio_mean"]
        ratio_s = f"{ratio*100:6.1f}%" if ratio is not None else "   n/a"
        print(
            f"{mode:10s}  n_seeds={r['n_seeds']}  "
            f"plasticity(just-learned)={r['plasticity_mean_accuracy']*100:5.1f}% (+/-{r['plasticity_sem']*100:.1f})  "
            f"retained(at end)={r['retained_mean_accuracy']*100:5.1f}% (+/-{r['retained_sem']*100:.1f})  "
            f"retention_ratio={ratio_s}"
        )
        if r["mean_collision_fraction"] is not None:
            print(f"             mean collision fraction (pre-projection): {r['mean_collision_fraction']*100:.1f}%")

    out_path = REPO_ROOT / "results" / "comparison_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
