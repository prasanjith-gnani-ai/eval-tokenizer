"""
=============================================================================
compare_tokenizers.py  —  Load & Compare Saved Evaluation Results
=============================================================================
Reads the CSV files produced by evaluate_tokenizer.py (one per model) and
generates comparison tables + publication-quality plots.

NO re-evaluation. NO re-downloading of data. Just analysis.

Workflow:
    1. Run evaluate_tokenizer.py once per model.
       Each run saves  results/results_<model_name>.csv
    2. Run this script to combine and visualise all saved CSVs.

Usage:
    python compare_tokenizers.py --results_dir results/
    python compare_tokenizers.py --results_dir results/ --output_dir plots/
=============================================================================
"""

import argparse
import os
import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = [
    "#2196F3",  # blue
    "#FF5722",  # deep orange
    "#4CAF50",  # green
    "#9C27B0",  # purple
    "#FF9800",  # amber
    "#00BCD4",  # cyan
    "#F44336",  # red
    "#607D8B",  # blue grey
]
FERTILITY_BANDS = [(1.5, "#4CAF50"), (2.5, "#FFC107"), (4.0, "#FF5722"), (99, "#B71C1C")]

plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(results_dir: str) -> pd.DataFrame:
    """
    Scan results_dir for files matching results_*.csv.
    Each file = one model's evaluation run.
    The model name is extracted from the filename: results_<model_name>.csv
    Returns a single combined DataFrame with a 'model' column.
    """
    pattern = os.path.join(results_dir, "results_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No results_*.csv files found in '{results_dir}'.\n"
            "Run evaluate_tokenizer.py first with OUTPUT_FILENAME set."
        )

    frames = []
    for f in files:
        model_name = Path(f).stem.replace("results_", "")
        df = pd.read_csv(f)
        # Inject model name if not already in CSV
        if "model" not in df.columns or df["model"].isna().all():
            df["model"] = model_name
        else:
            # Use filename-derived name as a clean label
            df["model"] = model_name
        frames.append(df)
        print(f"  Loaded: {f}  ({len(df)} languages, model='{model_name}')")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot to languages x models for a given metric."""
    return df.pivot_table(index=["lang_code","lang_name"], columns="model",
                          values=metric, aggfunc="mean").reset_index()


# =============================================================================
# CONSOLE TABLE
# =============================================================================

def print_comparison_table(df: pd.DataFrame, metric: str = "fertility",
                            lower_is_better: bool = True):
    piv = pivot(df, metric)
    models = [c for c in piv.columns if c not in ("lang_code","lang_name")]

    rows = []
    for _, row in piv.iterrows():
        vals = [row.get(m, np.nan) for m in models]
        valid_vals = [v for v in vals if not np.isnan(v)]
        best_val = (min(valid_vals) if lower_is_better else max(valid_vals)) if valid_vals else None

        cells = [row["lang_code"], row["lang_name"]]
        for m, v in zip(models, vals):
            if np.isnan(v):
                cells.append("N/A")
            elif best_val is not None and abs(v - best_val) < 1e-9:
                cells.append(f"*{v:.3f}*")
            else:
                cells.append(f"{v:.3f}")
        rows.append(cells)

    direction = "lower=better" if lower_is_better else "higher=better"
    print(f"\n{'='*70}")
    print(f"  {metric.upper().replace('_',' ')}  ({direction})  * = best per row")
    print(f"{'='*70}")
    print(tabulate(rows, headers=["Code","Language"] + models,
                   tablefmt="rounded_outline"))


# =============================================================================
# PLOT 1 - Grouped Bar Chart: Fertility per Language
# =============================================================================

def plot_fertility_bars(df: pd.DataFrame, output_dir: str):
    piv    = pivot(df, "fertility")
    models = [c for c in piv.columns if c not in ("lang_code","lang_name")]
    langs  = piv["lang_name"].tolist()
    n_langs  = len(langs)
    n_models = len(models)

    x     = np.arange(n_langs)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(18, n_langs * 0.9), 7))

    for i, (model, color) in enumerate(zip(models, PALETTE)):
        vals   = piv[model].values
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width * 0.9, label=model,
                        color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val) and val > 1.0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.1f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color="#333333")

    # Fertility band reference lines
    for threshold, color, label in [
        (1.5, "#4CAF50", "Excellent (<1.5)"),
        (2.5, "#FFC107", "Good (<2.5)"),
        (4.0, "#FF5722", "Marginal (<4.0)")
    ]:
        ax.axhline(threshold, color=color, linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"-- {label}")

    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fertility  (tokens / word)", fontsize=11)
    ax.set_title("Tokenizer Fertility Across Indic Languages\n"
                 "(lower is better — 1.0 = perfect, each word = 1 token)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9,
              framealpha=0.9, ncol=1)
    ax.set_ylim(0, None)
    fig.tight_layout()

    path = os.path.join(output_dir, "01_fertility_bars.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 2 - Heatmap: Fertility (languages x models)
# =============================================================================

def plot_fertility_heatmap(df: pd.DataFrame, output_dir: str):
    piv    = pivot(df, "fertility")
    models = [c for c in piv.columns if c not in ("lang_code","lang_name")]
    matrix = piv[models].values.T
    langs  = piv["lang_name"].tolist()

    fig, ax = plt.subplots(figsize=(max(16, len(langs)*0.8), max(4, len(models)*0.9 + 1)))
    cmap = sns.diverging_palette(130, 10, as_cmap=True)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        xticklabels=langs,
        yticklabels=models,
        linewidths=0.4,
        linecolor="#cccccc",
        cbar_kws={"label": "Fertility (lower=better)", "shrink": 0.6},
        vmin=1.0,
        vmax=8.0,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("Fertility Heatmap — Tokenizers x Indic Languages",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()

    path = os.path.join(output_dir, "02_fertility_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 3 - Radar Chart: Average metrics per model
# =============================================================================

def plot_radar(df: pd.DataFrame, output_dir: str):
    METRICS = {
        "fertility":         ("Fertility\n(inv)",          True,  10.0),
        "bytes_per_token":   ("Bytes/Token",               False, 12.0),
        "vocab_coverage":    ("Vocab\nCoverage",           False,  1.0),
        "nsl":               ("NSL\n(inv)",                True,   5.0),
        "continuation_rate": ("Continuation\nRate (inv)",  True,   1.0),
    }

    models = df["model"].unique().tolist()
    labels = [v[0] for v in METRICS.values()]
    n_axes = len(labels)
    angles = np.linspace(0, 2*np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for model, color in zip(models, PALETTE):
        mdf = df[df["model"] == model]
        vals = []
        for metric, (_, invert, scale) in METRICS.items():
            avg  = mdf[metric].mean()
            norm = min(avg / scale, 1.0)
            if invert:
                norm = 1.0 - norm
            vals.append(norm)
        vals += vals[:1]

        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Tokenizer Quality Radar\n(outward = better on all axes)",
                 fontsize=13, fontweight="bold", y=1.12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "03_radar_chart.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 4 - Box Plot: Fertility distribution per model
# =============================================================================

def plot_fertility_boxplot(df: pd.DataFrame, output_dir: str):
    models     = df["model"].unique().tolist()
    model_data = [df[df["model"] == m]["fertility"].values for m in models]

    fig, ax = plt.subplots(figsize=(max(8, len(models)*2), 6))

    bp = ax.boxplot(model_data, patch_artist=True, notch=False,
                    medianprops={"color":"black","linewidth":2},
                    whiskerprops={"linewidth":1.2},
                    capprops={"linewidth":1.2})

    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (data, color) in enumerate(zip(model_data, PALETTE), start=1):
        jitter = np.random.normal(0, 0.07, size=len(data))
        ax.scatter(np.full_like(data, float(i)) + jitter, data,
                   color=color, alpha=0.5, s=30, zorder=3,
                   edgecolors="white", linewidth=0.5)

    for threshold, color in [(1.5,"#4CAF50"), (2.5,"#FFC107"), (4.0,"#FF5722")]:
        ax.axhline(threshold, linestyle="--", color=color, linewidth=1.2, alpha=0.7)

    ax.set_xticks(range(1, len(models)+1))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Fertility  (tokens / word)", fontsize=11)
    ax.set_title("Fertility Distribution per Tokenizer\n"
                 "across all Indic languages  (lower = better)",
                 fontsize=13, fontweight="bold", pad=10)
    patches = [mpatches.Patch(color=c, label=l, alpha=0.6)
               for c, l in zip(PALETTE[:len(models)], models)]
    ax.legend(handles=patches, loc="upper right", fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "04_fertility_boxplot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 5 - Scatter: Fertility vs. Vocab Coverage
# =============================================================================

def plot_fertility_vs_coverage(df: pd.DataFrame, output_dir: str):
    models = df["model"].unique().tolist()
    fig, ax = plt.subplots(figsize=(9, 6))

    for model, color in zip(models, PALETTE):
        mdf = df[df["model"] == model]
        ax.scatter(
            mdf["vocab_coverage"],
            mdf["fertility"],
            c=color, label=model, alpha=0.75, s=60,
            edgecolors="white", linewidth=0.8, zorder=3,
        )
        x, y = mdf["vocab_coverage"].values, mdf["fertility"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 2:
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            xline = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xline, p(xline), color=color, linewidth=1.5,
                    linestyle="--", alpha=0.6)

    ax.set_xlabel("Vocab Coverage  (fraction of script chars in vocab)", fontsize=10)
    ax.set_ylabel("Fertility  (tokens / word)", fontsize=10)
    ax.set_title("Fertility vs. Vocabulary Coverage\n"
                 "Higher coverage -> lower fertility  (dashed = trend per model)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()

    path = os.path.join(output_dir, "05_fertility_vs_coverage.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 6 - Stacked Bar: Grade distribution per model
# =============================================================================

def plot_grade_distribution(df: pd.DataFrame, output_dir: str):
    def grade(f):
        if f < 1.5: return "Excellent (<1.5)"
        if f < 2.5: return "Good (1.5-2.5)"
        if f < 4.0: return "Marginal (2.5-4.0)"
        return "Poor (>4.0)"

    df = df.copy()
    df["grade"] = df["fertility"].apply(grade)
    models = df["model"].unique().tolist()

    grade_order  = ["Excellent (<1.5)","Good (1.5-2.5)","Marginal (2.5-4.0)","Poor (>4.0)"]
    grade_colors = ["#4CAF50","#FFC107","#FF5722","#B71C1C"]

    counts = (df.groupby(["model","grade"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=grade_order, fill_value=0))

    fig, ax = plt.subplots(figsize=(max(8, len(models)*2), 6))
    bottom = np.zeros(len(models))
    xs     = np.arange(len(models))

    for grade_label, color in zip(grade_order, grade_colors):
        vals = (counts.reindex(models)[grade_label].values
                if grade_label in counts.columns else np.zeros(len(models)))
        bars = ax.bar(xs, vals, bottom=bottom, color=color,
                      label=grade_label, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_y() + bar.get_height()/2,
                        str(int(val)), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        bottom += vals

    ax.set_xticks(xs)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Number of languages", fontsize=11)
    ax.set_title("Fertility Grade Distribution per Tokenizer\n"
                 "(how many languages fall into each quality band)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()

    path = os.path.join(output_dir, "06_grade_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 7 - Line Chart: NSL per language, all models
# =============================================================================

def plot_nsl_lines(df: pd.DataFrame, output_dir: str):
    piv    = pivot(df, "nsl")
    models = [c for c in piv.columns if c not in ("lang_code","lang_name")]
    langs  = piv["lang_name"].tolist()
    x      = np.arange(len(langs))

    fig, ax = plt.subplots(figsize=(max(16, len(langs)*0.85), 6))

    for model, color in zip(models, PALETTE):
        vals = piv[model].values
        ax.plot(x, vals, "o-", label=model, color=color,
                linewidth=2, markersize=5, alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--",
               label="NSL = 1.0  (reference mT5 baseline)")
    ax.fill_between(x, 0, 1.0, alpha=0.04, color="green")

    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("NSL  (model tokens / mT5 tokens)", fontsize=11)
    ax.set_title("Normalized Sequence Length (NSL) per Language\n"
                 "NSL < 1.0 = more efficient than mT5  (green zone = good)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1), fontsize=9)
    fig.tight_layout()

    path = os.path.join(output_dir, "07_nsl_lines.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 8 - Summary: Normalised average per model
# =============================================================================

def plot_summary_bars(df: pd.DataFrame, output_dir: str):
    METRICS = [
        ("fertility",         "Fertility\n(inv)",          True),
        ("bytes_per_token",   "Bytes/Token",               False),
        ("vocab_coverage",    "Vocab\nCoverage",           False),
        ("continuation_rate", "Continuation\nRate (inv)",  True),
        ("nsl",               "NSL\n(inv)",                True),
    ]

    models    = df["model"].unique().tolist()
    n_models  = len(models)
    n_metrics = len(METRICS)
    x     = np.arange(n_metrics)
    width = 0.7 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model, color) in enumerate(zip(models, PALETTE)):
        mdf    = df[df["model"] == model]
        offset = (i - n_models/2 + 0.5) * width
        vals   = []
        for metric, label, invert in METRICS:
            col_max = df[metric].max()
            col_min = df[metric].min()
            avg  = mdf[metric].mean()
            norm = (avg - col_min) / (col_max - col_min + 1e-9)
            if invert:
                norm = 1.0 - norm
            vals.append(norm)
        ax.bar(x + offset, vals, width * 0.9, label=model,
               color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in METRICS], fontsize=10)
    ax.set_ylabel("Normalised Score  (taller = better)", fontsize=10)
    ax.set_title("Average Normalised Scores per Tokenizer\n"
                 "(metrics inverted where needed so taller bar always = better)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()

    path = os.path.join(output_dir, "08_summary_bars.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(df: pd.DataFrame):
    rows = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        rows.append([
            model,
            f"{mdf['fertility'].mean():.3f}",
            f"{mdf['bytes_per_token'].mean():.3f}",
            f"{mdf['vocab_coverage'].mean():.3f}",
            f"{mdf['nsl'].mean():.3f}",
            f"{mdf['continuation_rate'].mean():.3f}",
            f"{(mdf['fertility'] < 2.5).sum()} / {len(mdf)}",
        ])
    rows.sort(key=lambda r: float(r[1]))

    print(f"\n{'='*80}")
    print("  AGGREGATE SUMMARY  (sorted by avg fertility)")
    print(f"{'='*80}")
    print(tabulate(
        rows,
        headers=["Model", "Avg Fertility", "Avg BPT",
                 "Avg VocabCov", "Avg NSL", "Avg ContRate",
                 "Langs F<2.5"],
        tablefmt="rounded_outline"
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir",  default="plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("  Tokenizer Comparison — Loading saved evaluation results")
    print(f"{'='*70}\n")

    df     = load_results(args.results_dir)
    models = df["model"].unique().tolist()
    n_langs = df["lang_code"].nunique()
    print(f"\n  Models loaded   : {models}")
    print(f"  Languages found : {n_langs}")

    print_summary_table(df)
    for metric, lib in [("fertility", True), ("bytes_per_token", False),
                         ("vocab_coverage", False), ("nsl", True)]:
        print_comparison_table(df, metric, lib)

    print(f"\n  Generating 8 plots -> {args.output_dir}/\n")
    plot_fertility_bars(df,        args.output_dir)
    plot_fertility_heatmap(df,     args.output_dir)
    plot_radar(df,                 args.output_dir)
    plot_fertility_boxplot(df,     args.output_dir)
    plot_fertility_vs_coverage(df, args.output_dir)
    plot_grade_distribution(df,    args.output_dir)
    plot_nsl_lines(df,             args.output_dir)
    plot_summary_bars(df,          args.output_dir)

    merged = os.path.join(args.output_dir, "all_results_merged.csv")
    df.to_csv(merged, index=False)
    print(f"\n  Merged CSV saved -> {merged}")
    print(f"\n  Done. {len(models)} models x {n_langs} languages x 8 plots\n")


if __name__ == "__main__":
    main()