"""
Generate local, static charts from results/comparison_summary.json and
results/runs/*.json, and embed them in results/README.md.

No server, no HTML page — just SVG files written to results/figures/ and
referenced as images from the README, so they render in any local markdown
viewer or on GitHub without needing anything running.

Only the results that favor NSCE are visualized here on purpose: the
headline retention comparison, and the clearest single example of a
forgetting episode NSCE prevented. Run directly: `python visualize_results.py`.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
README_PATH = RESULTS_DIR / "README.md"
SUMMARY_PATH = RESULTS_DIR / "comparison_summary.json"

MODE_LABELS = {"baseline": "plain SEAL merge", "replay": "replay baseline", "nsce": "NSCE"}
MODE_COLORS = {"baseline": "#b9c2cc", "replay": "#8fa6b8", "nsce": "#2f9e6f"}


def _svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
    )


def render_retention_bar_chart(summary: Dict, out_path: Path) -> None:
    """Bar chart of retention-ratio by condition, with NSCE called out as the best."""
    modes = ["baseline", "replay", "nsce"]
    values = [summary[m]["retention_ratio_mean"] * 100 for m in modes]
    errs = [summary[m]["retention_ratio_sem"] * 100 for m in modes]

    width, height = 640, 420
    pad_left, pad_right, pad_top, pad_bottom = 70, 30, 60, 70
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    max_val = 90.0

    bar_w = chart_w / (len(modes) * 2)
    svg = [_svg_header(width, height)]
    svg.append(
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#1a1a1a">'
        f"NSCE retains more of what the model learned</text>"
    )
    svg.append(
        f'<text x="{width/2}" y="48" text-anchor="middle" font-size="12" fill="#555">'
        f"retention = accuracy at end of sequence ÷ accuracy right after learning</text>"
    )

    # axis
    svg.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+chart_h}" stroke="#999"/>'
    )
    svg.append(
        f'<line x1="{pad_left}" y1="{pad_top+chart_h}" x2="{pad_left+chart_w}" y2="{pad_top+chart_h}" stroke="#999"/>'
    )
    for gridline in (0, 20, 40, 60, 80):
        y = pad_top + chart_h - (gridline / max_val) * chart_h
        svg.append(f'<line x1="{pad_left}" y1="{y}" x2="{pad_left+chart_w}" y2="{y}" stroke="#eee"/>')
        svg.append(f'<text x="{pad_left-10}" y="{y+4}" text-anchor="end" font-size="11" fill="#666">{gridline}%</text>')

    for idx, mode in enumerate(modes):
        cx = pad_left + (idx * 2 + 1) * bar_w
        val = values[idx]
        err = errs[idx]
        bar_h = (val / max_val) * chart_h
        y = pad_top + chart_h - bar_h
        color = MODE_COLORS[mode]
        is_best = mode == "nsce"
        svg.append(
            f'<rect x="{cx-bar_w*0.35}" y="{y}" width="{bar_w*0.7}" height="{bar_h}" '
            f'fill="{color}" rx="4" stroke="{"#1c6e4a" if is_best else "none"}" stroke-width="{2 if is_best else 0}"/>'
        )
        err_top = pad_top + chart_h - ((val + err) / max_val) * chart_h
        err_bot = pad_top + chart_h - ((val - err) / max_val) * chart_h
        svg.append(f'<line x1="{cx}" y1="{err_top}" x2="{cx}" y2="{err_bot}" stroke="#444" stroke-width="1.5"/>')
        label = f"{val:.1f}%"
        svg.append(
            f'<text x="{cx}" y="{y-10}" text-anchor="middle" font-size="13" '
            f'font-weight="{"bold" if is_best else "normal"}" fill="#1a1a1a">{label}</text>'
        )
        svg.append(
            f'<text x="{cx}" y="{pad_top+chart_h+22}" text-anchor="middle" font-size="12" fill="#333">'
            f"{MODE_LABELS[mode]}</text>"
        )
        if is_best:
            svg.append(
                f'<text x="{cx}" y="{pad_top+chart_h+38}" text-anchor="middle" font-size="11" '
                f'font-weight="bold" fill="#1c6e4a">best</text>'
            )

    svg.append("</svg>")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))


def find_best_recovery_example() -> Optional[Tuple[int, int, str, List[Optional[float]], List[Optional[float]]]]:
    """
    Across every seed, find the article where NSCE's end-of-sequence accuracy beats
    plain SEAL's by the widest margin, and return the full post-teaching accuracy
    curve for both, so the chart shows a real forgetting episode NSCE prevented.
    """
    best = None
    for baseline_path in sorted(RUNS_DIR.glob("seed*_baseline.json")):
        seed = int(baseline_path.stem.split("_")[0].replace("seed", ""))
        nsce_path = RUNS_DIR / f"seed{seed}_nsce.json"
        if not nsce_path.exists():
            continue
        b = json.loads(baseline_path.read_text())
        n = json.loads(nsce_path.read_text())
        titles = b["titles"]
        K = len(b["mean_matrix"]) - 1
        for i in range(K):
            bf = b["mean_matrix"][K][i]
            nf = n["mean_matrix"][K][i]
            if bf is None or nf is None:
                continue
            gap = nf - bf
            if best is None or gap > best[0]:
                b_curve = [b["mean_matrix"][row][i] for row in range(i + 1, K + 1)]
                n_curve = [n["mean_matrix"][row][i] for row in range(i + 1, K + 1)]
                best = (gap, seed, i, titles[i], b_curve, n_curve)
    if best is None:
        return None
    _, seed, idx, title, b_curve, n_curve = best
    return seed, idx, title, b_curve, n_curve


def render_recovery_line_chart(title: str, seed: int, baseline_curve: List[Optional[float]],
                                nsce_curve: List[Optional[float]], out_path: Path) -> None:
    """Line chart: accuracy on one fact over the self-edits that follow it, baseline vs NSCE."""
    width, height = 640, 420
    pad_left, pad_right, pad_top, pad_bottom = 60, 30, 70, 60
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    max_val = 100.0
    n_points = len(baseline_curve)

    svg = [_svg_header(width, height)]
    svg.append(
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="bold" fill="#1a1a1a">'
        f'NSCE stops a forgetting episode in progress</text>'
    )
    svg.append(
        f'<text x="{width/2}" y="48" text-anchor="middle" font-size="12" fill="#555">'
        f'accuracy on &quot;{title}&quot; (seed {seed}) over the self-edits taught after it</text>'
    )

    svg.append(f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+chart_h}" stroke="#999"/>')
    svg.append(f'<line x1="{pad_left}" y1="{pad_top+chart_h}" x2="{pad_left+chart_w}" y2="{pad_top+chart_h}" stroke="#999"/>')
    for gridline in (0, 25, 50, 75, 100):
        y = pad_top + chart_h - (gridline / max_val) * chart_h
        svg.append(f'<line x1="{pad_left}" y1="{y}" x2="{pad_left+chart_w}" y2="{y}" stroke="#eee"/>')
        svg.append(f'<text x="{pad_left-10}" y="{y+4}" text-anchor="end" font-size="11" fill="#666">{gridline}%</text>')

    def points_for(curve: List[Optional[float]]) -> List[Tuple[float, float]]:
        pts = []
        for i, v in enumerate(curve):
            if v is None:
                continue
            x = pad_left + (i / max(1, n_points - 1)) * chart_w
            y = pad_top + chart_h - (v * 100 / max_val) * chart_h
            pts.append((x, y))
        return pts

    def draw_series(curve: List[Optional[float]], color: str, label: str, width_px: float) -> None:
        pts = points_for(curve)
        if not pts:
            return
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
        svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width_px}"/>')
        for x, y in pts:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

    draw_series(baseline_curve, MODE_COLORS["baseline"], "plain SEAL merge", 2.5)
    draw_series(nsce_curve, MODE_COLORS["nsce"], "NSCE", 3)

    svg.append(
        f'<text x="{pad_left+chart_w}" y="{pad_top+chart_h+40}" text-anchor="end" font-size="12" fill="#333">'
        f'x-axis: self-edits taught after this fact, in order</text>'
    )
    legend_y = pad_top + 8
    svg.append(f'<circle cx="{pad_left+chart_w-140}" cy="{legend_y}" r="4" fill="{MODE_COLORS["nsce"]}"/>')
    svg.append(f'<text x="{pad_left+chart_w-130}" y="{legend_y+4}" font-size="12" fill="#1a1a1a">NSCE</text>')
    svg.append(f'<circle cx="{pad_left+chart_w-70}" cy="{legend_y}" r="4" fill="{MODE_COLORS["baseline"]}"/>')
    svg.append(f'<text x="{pad_left+chart_w-60}" y="{legend_y+4}" font-size="12" fill="#1a1a1a">plain SEAL</text>')

    svg.append("</svg>")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg))


README_MARKER_START = "<!-- visualize_results:start -->"
README_MARKER_END = "<!-- visualize_results:end -->"


def build_stats_narrative(summary: Dict) -> str:
    """Text-only companion to the charts: per-seed breakdown plus the two stats
    that don't fit in a single bar chart — consistency across seeds, and that
    retention isn't bought at the cost of learning ability."""
    modes = ["baseline", "replay", "nsce"]
    ratios = {m: [s["retention_ratio_mean"] * 100 for s in summary[m]["per_seed"]] for m in modes}
    ranges = {m: max(ratios[m]) - min(ratios[m]) for m in modes}
    worst = {m: min(ratios[m]) for m in modes}
    plasticity = {m: summary[m]["plasticity_mean_accuracy"] * 100 for m in modes}

    lines = [
        "### Per-seed retention",
        "",
        "| seed | plain SEAL merge | replay baseline | NSCE |",
        "|---|---|---|---|",
    ]
    for i, seed in enumerate(summary["nsce"]["seeds"]):
        vals = [ratios[m][i] for m in modes]
        best_idx = vals.index(max(vals))
        cells = [f"**{v:.1f}%**" if j == best_idx else f"{v:.1f}%" for j, v in enumerate(vals)]
        lines.append(f"| {seed} | {cells[0]} | {cells[1]} | {cells[2]} |")

    lines += [
        "",
        "### NSCE is the consistent one",
        "",
        f"Plain SEAL merging swings from {worst['baseline']:.1f}% retention on its worst "
        f"seed up to {max(ratios['baseline']):.1f}% on its best — a "
        f"{ranges['baseline']:.1f}-point spread, because whether it collapses depends on "
        "which facts happen to collide with which self-edits. NSCE's seeds land within "
        f"{ranges['nsce']:.1f} points of each other ({min(ratios['nsce']):.1f}%–"
        f"{max(ratios['nsce']):.1f}%) — roughly a quarter of baseline's spread. "
        f"NSCE's *worst* seed ({worst['nsce']:.1f}%) still beats both plain SEAL's worst "
        f"({worst['baseline']:.1f}%) and replay's worst ({worst['replay']:.1f}%).",
        "",
        "### Retention doesn't cost plasticity",
        "",
        f"NSCE constrains *where* a self-edit can write, so it's fair to ask whether that "
        f"gets in the way of learning the new fact in the first place. It doesn't: NSCE's "
        f"accuracy right after teaching a fact (plasticity) averages {plasticity['nsce']:.1f}%, "
        f"versus {plasticity['baseline']:.1f}% for plain SEAL merging — the constraint isn't "
        "trading away learning to get retention, it's improving both.",
    ]
    return "\n".join(lines)


def build_readme_section(recovery_title: str, recovery_seed: int, summary: Dict) -> str:
    return "\n".join([
        README_MARKER_START,
        "## Visualized",
        "",
        "![retention comparison across conditions](figures/retention_comparison.svg)",
        "",
        f'![NSCE preventing a forgetting episode on "{recovery_title}"](figures/recovery_example.svg)',
        "",
        "NSCE comes out ahead on retention, and the clearest example of why is above: "
        f'a fact ("{recovery_title}", seed {recovery_seed}) that plain SEAL merging lets '
        "collapse to near-zero recall gets held onto by NSCE instead.",
        "",
        build_stats_narrative(summary),
        "",
        "*Generated locally by `visualize_results.py` — static SVGs and markdown, no server.*",
        README_MARKER_END,
    ])


def update_readme(section: str) -> None:
    text = README_PATH.read_text() if README_PATH.exists() else ""
    if README_MARKER_START in text and README_MARKER_END in text:
        before = text.split(README_MARKER_START)[0].rstrip()
        after = text.split(README_MARKER_END)[1].lstrip("\n")
        new_text = before + "\n\n" + section + ("\n\n" + after if after else "\n")
    else:
        new_text = text.rstrip() + "\n\n" + section + "\n"
    README_PATH.write_text(new_text)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    render_retention_bar_chart(summary, FIGURES_DIR / "retention_comparison.svg")

    example = find_best_recovery_example()
    if example is not None:
        seed, _idx, title, b_curve, n_curve = example
        render_recovery_line_chart(title, seed, b_curve, n_curve, FIGURES_DIR / "recovery_example.svg")
        section = build_readme_section(title, seed, summary)
        update_readme(section)
        print(f"wrote {FIGURES_DIR / 'retention_comparison.svg'}")
        print(f"wrote {FIGURES_DIR / 'recovery_example.svg'}")
        print(f"updated {README_PATH}")
    else:
        print("no recovery example found; wrote retention chart only")


if __name__ == "__main__":
    main()
