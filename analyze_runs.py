"""Summarise and plot the Experiment 2 run JSON files.

Run from anywhere: python analyze_runs.py
Only files with a run config (context_len and seed) are analysed.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def metric_label(metric):
    return "Validation BPC" if metric == "val_bpc" else "Test BPC"


def finite_number(value, label, path):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path}: {label} must be a finite number")
    return value


def load_runs(results_dir):
    # Both the currently committed flat layout and the notebooks' results/runs layout.
    paths = sorted(set(results_dir.glob("*.json")) | set((results_dir / "runs").glob("*.json")))
    runs = {}
    skipped = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("config"), dict) or not {"context_len", "seed"} <= data["config"].keys():
            skipped.append(path.name)
            continue
        context = data["config"]["context_len"]
        seed = data["config"]["seed"]
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (context, seed)) or context <= 0:
            raise ValueError(f"{path}: context_len must be a positive integer and seed an integer")
        key = (context, seed)
        if key in runs:
            raise ValueError(f"Duplicate ctx{context}/seed{seed}: {runs[key]['path']} and {path}")
        for field in ("val_bpc", "test_bpc", "train_seconds", "best_step"):
            if field in data:
                finite_number(data[field], field, path)
        history = data.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"{path}: history must be a list")
        for point in history:
            if not isinstance(point, dict):
                raise ValueError(f"{path}: history entries must be objects")
            for field in ("step", "train_loss", "val_loss"):
                if field in point:
                    finite_number(point[field], f"history.{field}", path)
        runs[key] = {"path": path, "data": data}
    return runs, skipped


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bpc(rows, metric, output_dir):
    contexts = sorted({row["context_len"] for row in rows if row[metric] is not None})
    if not contexts:
        return False
    groups = {ctx: [row for row in rows if row["context_len"] == ctx and row[metric] is not None] for ctx in contexts}
    fig, ax = plt.subplots(figsize=(8, 5))
    for ctx in contexts:
        values = [row[metric] for row in groups[ctx]]
        ax.scatter([ctx] * len(values), values, color="#32658a", alpha=0.7, s=48, zorder=3)
        mean = statistics.mean(values)
        ax.scatter(ctx, mean, marker="D", color="#c35c35", s=70, zorder=4)
        if len(values) > 1:
            sd = statistics.stdev(values)
            ax.errorbar(ctx, mean, yerr=sd, color="#c35c35", capsize=5, zorder=2)
    ax.set_xticks(contexts)
    ax.set_xlabel("Context length (characters)")
    ax.set_ylabel(metric_label(metric) + " (lower is better)")
    ax.set_title(f"{metric_label(metric)} by context length")
    ax.grid(axis="y", alpha=0.25)
    ax.scatter([], [], color="#32658a", label="Individual seed")
    ax.scatter([], [], marker="D", color="#c35c35", label="Mean ± sample SD (n ≥ 2)")
    ax.legend()
    save_plot(fig, output_dir / f"{metric}_by_context.png")
    return True


def plot_paired(rows, metric, output_dir):
    by_seed = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            by_seed[row["seed"]].append(row)
    paired = {seed: sorted(values, key=lambda row: row["context_len"]) for seed, values in by_seed.items() if len(values) >= 2}
    if not paired:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for seed, values in sorted(paired.items()):
        ax.plot([row["context_len"] for row in values], [row[metric] for row in values], marker="o", label=f"Seed {seed}")
    ax.set_xticks(sorted({row["context_len"] for values in paired.values() for row in values}))
    ax.set_xlabel("Context length (characters)")
    ax.set_ylabel(metric_label(metric) + " (lower is better)")
    ax.set_title("Matched seed comparisons")
    ax.grid(alpha=0.25)
    ax.legend()
    save_plot(fig, output_dir / f"{metric}_paired_seeds.png")
    return True


def plot_history(runs, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for (context, seed), run in sorted(runs.items()):
        points = [(p["step"], p["val_loss"]) for p in run["data"]["history"] if "step" in p and "val_loss" in p]
        if points:
            points.sort()
            ax.plot(*zip(*points), label=f"ctx{context} seed{seed}", alpha=0.8)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("Training step")
    ax.set_ylabel("Sampled validation loss (nats)")
    ax.set_title("Validation loss during training")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    save_plot(fig, output_dir / "validation_loss_history.png")
    return True


def analyse(results_dir, output_dir, metric):
    runs, skipped = load_runs(results_dir)
    if not runs:
        raise ValueError(f"No run JSON files found in {results_dir} or {results_dir / 'runs'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (context, seed), run in sorted(runs.items()):
        data = run["data"]
        rows.append({
            "context_len": context,
            "seed": seed,
            "val_bpc": data.get("val_bpc"),
            "test_bpc": data.get("test_bpc"),
            "best_step": data.get("best_step"),
            "train_minutes": round(data["train_seconds"] / 60, 3) if "train_seconds" in data else None,
            "file": str(run["path"].relative_to(results_dir)),
        })
    write_csv(output_dir / "runs.csv", list(rows[0]), rows)
    groups = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            groups[row["context_len"]].append(row[metric])
    summary = [{
        "context_len": ctx,
        "n": len(values),
        f"mean_{metric}": round(statistics.mean(values), 6) if values else None,
        f"sd_{metric}": round(statistics.stdev(values), 6) if len(values) > 1 else None,
        f"min_{metric}": min(values) if values else None,
        f"max_{metric}": max(values) if values else None,
    } for ctx, values in sorted(groups.items())]
    write_csv(output_dir / f"{metric}_summary.csv", ["context_len", "n", f"mean_{metric}", f"sd_{metric}", f"min_{metric}", f"max_{metric}"], summary)

    paired = []
    contexts = sorted(groups)
    lookup = {(row["context_len"], row["seed"]): row[metric] for row in rows if row[metric] is not None}
    for i, shorter in enumerate(contexts):
        for longer in contexts[i + 1:]:
            seeds = sorted({seed for ctx, seed in lookup if ctx == shorter} & {seed for ctx, seed in lookup if ctx == longer})
            for seed in seeds:
                paired.append({"shorter_context": shorter, "longer_context": longer, "seed": seed,
                               "shorter_bpc": lookup[shorter, seed], "longer_bpc": lookup[longer, seed],
                               "delta_bpc": lookup[longer, seed] - lookup[shorter, seed]})
    write_csv(output_dir / f"{metric}_paired.csv", ["shorter_context", "longer_context", "seed", "shorter_bpc", "longer_bpc", "delta_bpc"], paired)

    plots = []
    if plot_bpc(rows, metric, output_dir):
        plots.append(f"{metric}_by_context.png")
    if plot_paired(rows, metric, output_dir):
        plots.append(f"{metric}_paired_seeds.png")
    if plot_history(runs, output_dir):
        plots.append("validation_loss_history.png")

    try:
        source = results_dir.resolve().relative_to(ROOT)
    except ValueError:
        source = results_dir
    lines = [f"# Experiment 2: {metric_label(metric)} analysis", "", f"Loaded {len(rows)} run(s) from `{source}`.", ""]
    if skipped:
        lines += ["Skipped non-run JSON: " + ", ".join(skipped) + ".", ""]
    if metric == "val_bpc":
        lines += ["Validation BPC is measured on the fixed validation targets. It is not a test-set result.", ""]
    lines += ["| Context | Scored runs | Mean BPC | Sample SD |", "| ---: | ---: | ---: | ---: |"]
    for row in summary:
        lines.append(f"| {row['context_len']} | {row['n']} | {fmt(row[f'mean_{metric}'])} | {fmt(row[f'sd_{metric}'])} |")
    if not summary:
        lines += ["", f"No runs currently contain `{metric}`."]
    else:
        lines += ["", "SD is shown only where at least two seeds are available. Means with different seed coverage are descriptive and should be interpreted cautiously."]
        lowest = min(summary, key=lambda row: row[f"mean_{metric}"])
        lines += ["", f"The lowest observed mean is at context {lowest['context_len']} ({fmt(lowest[f'mean_{metric}'])} BPC across {lowest['n']} seed(s)). This is a provisional validation comparison until the planned runs are complete."]
    lines += ["", "## Matched seeds", "", "Delta = longer-context BPC minus shorter-context BPC; a negative value favours the longer context.", ""]
    if paired:
        lines += ["| Contexts | Seed | Delta BPC |", "| --- | ---: | ---: |"]
        for pair in paired:
            lines.append(f"| {pair['shorter_context']} → {pair['longer_context']} | {pair['seed']} | {pair['delta_bpc']:+.4f} |")
        comparisons = defaultdict(list)
        for pair in paired:
            comparisons[(pair["shorter_context"], pair["longer_context"])].append(pair["delta_bpc"])
        lines += [""]
        for (shorter, longer), deltas in sorted(comparisons.items()):
            lines.append(f"{shorter} → {longer}: mean matched delta {statistics.mean(deltas):+.4f} BPC across {len(deltas)} seed(s).")
    else:
        lines.append("No matched seeds are available yet.")
    lines += ["", "The training-history plot uses sampled validation loss in nats, which is distinct from fixed-target BPC.", ""]
    if plots:
        lines += ["## Plots", ""] + [f"![{name}](./{name})" for name in plots] + [""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return len(rows), len(groups), plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Directory holding run JSONs (flat or runs/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "analysis", help="Directory for report, CSVs and PNGs")
    parser.add_argument("--metric", choices=("val_bpc", "test_bpc"), default="val_bpc")
    args = parser.parse_args()
    try:
        count, contexts, plots = analyse(args.results_dir, args.output_dir, args.metric)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Analysed {count} runs across {contexts} scored contexts. Wrote {args.output_dir / 'report.md'} and {len(plots)} plots.")


if __name__ == "__main__":
    main()
