#!/usr/bin/env python3
"""
Phase 11.2 - Evaluation: Station-Level Results Visualization
-------------------------------------------------------------
This is the second of two Phase 11 evaluation scripts and must be run
after Phase 11.1 (evaluate_model_target_station.py). It reads the
long-format station-level evaluation CSV produced by Phase 11.1 and
generates PNG heatmaps, difference heatmaps, and a grouped bar chart
for the selected split and metric. All plots are written to a plots/
subdirectory under the Phase 11.1 output root. This script uses only
pandas and Plotly; no model or training artifacts are loaded.

Purpose
-------
The Phase 11.1 long CSV is complete but not suited for rapid pattern
recognition. This script makes it easy to spot which targets are
forecastable at which stations, where learned models beat climatology,
whether gains are broadly distributed or concentrated at a few stations,
and how persistence compares to learned models.

The six benchmark models (in display order) are:
- climatology, persistence, logistic_regression,
  lightgbm_main, lightgbm_tuned_simple, lstm

Supported metrics: f2, precision, recall, pr_auc, brier, hss.

Plot outputs (default args: 6 models, metric=f2, split=test)
--------------------------------------------------------------
Per-model heatmaps (one per model):
  Target x station grid colored by --metric.
  Color scale: Viridis for f2/precision/recall/pr_auc/brier;
  RdBu (diverging, centered at 0) for hss.
  Color limits set from p5-p95 of actual values (not full 0-1 range)
  to improve visual contrast. Written as:
    {model}_{split}_{metric}_heatmap.png

Difference heatmaps (produced conditionally):
  LightGBM minus Climatology: produced when both lightgbm_main and
  climatology are in --models. RdBu scale, symmetric at 0,
  ±p95(abs(diff)). Written as:
    lightgbm_minus_climatology_{split}_{metric}_heatmap.png

  LightGBM Tuned minus LightGBM: produced when both
  lightgbm_tuned_simple and lightgbm_main are in --models. Same
  color logic. Written as:
    lightgbm_tuned_minus_main_{split}_{metric}_heatmap.png

Mean station-level bar chart (one per run):
  Grouped bar chart: x=target, color=model, y=mean(metric) across
  stations. Written as:
    mean_station_{metric}_by_target_{split}.png

All PNGs are rendered at width=1400, height=700, scale=2 (2800x1400
effective resolution) using Plotly's kaleido backend.

CSV outputs (one per plot, written alongside PNGs):
  filtered_metrics_{split}.csv            (post-filter long metrics)
  {model}_{split}_{metric}_heatmap_source.csv  (per-model subset)
  lightgbm_minus_climatology_{split}_{metric}.csv
  lightgbm_tuned_minus_main_{split}_{metric}.csv
  mean_station_{metric}_by_target_{split}.csv

Skip behavior
--------------
Soft informational only: if plots/ directory exists and --force is not
set, "[info] output exists" is printed, but ensure_dir() is called and
all plots are regenerated. There is no true skip in this script.

WORK FLOW
---------
Step 1: Parse CLI args. Resolve output directory.
Step 2: Read target_station_metrics_long.csv via read_long_metrics().
Step 3: normalize_order(): apply Categorical ordering for target
        (TARGET_ORDER), station (alphabetical), and model (MODEL_ORDER
        with remaining models appended sorted).
Step 4: Filter to --split and --models. Raise ValueError if empty.
        Write filtered_metrics_{split}.csv.
Step 5: For each model in --models:
        - Filter to model rows.
        - Write heatmap source CSV.
        - skill_color_limits(): compute zmin/zmax from p5/p95 with
          metric-specific logic (brier: no clamping; hss: symmetric;
          others: clamped [0,1]).
        - make_heatmap(): pivot to target x station, render Plotly
          imshow, save PNG.
Step 6: If lightgbm_main and climatology both present:
        build_difference_df() -> make_heatmap() with RdBu scale.
        Write difference CSV and PNG.
Step 7: If lightgbm_tuned_simple and lightgbm_main both present:
        Same as Step 6 for tuned vs untuned difference.
Step 8: make_mean_bar(): group by [target, model], mean metric,
        render grouped bar chart. Write summary CSV and PNG.

INPUT DATA
----------
- data/reports/evaluation/target_station/target_station_metrics_long.csv
  Written by Phase 11.1. Required. The pivoted CSV and run_config.json
  in the same directory are not read by this script.
- CLI optional: --eval-root  (default: data/reports/evaluation/
                               target_station)
               --split       (default: test)
               --models      (default: all 6 models)
               --metric      (default: f2; choices: f2, precision,
                              recall, pr_auc, brier, hss)
               --force       (soft skip if not set)

OUTPUT DATA
-----------
All written to data/reports/evaluation/target_station/plots/.

Per-model (2 files each x 6 models = 12):
  {model}_{split}_{metric}_heatmap.png
  {model}_{split}_{metric}_heatmap_source.csv

Difference heatmaps (2 files each x up to 2 pairs = up to 8):
  lightgbm_minus_climatology_{split}_{metric}_heatmap.png
  lightgbm_minus_climatology_{split}_{metric}.csv
  lightgbm_tuned_minus_main_{split}_{metric}_heatmap.png
  lightgbm_tuned_minus_main_{split}_{metric}.csv

Summary (3 files):
  mean_station_{metric}_by_target_{split}.png
  mean_station_{metric}_by_target_{split}.csv
  filtered_metrics_{split}.csv

Total default output: 19 files (12 + 4 + 3).

TYPICAL USAGE
-------------
Default run (test split, f2, all models):
  uv run src/evaluation/plot_target_station_results.py --force

Restrict to test split and subset of models:
  uv run src/evaluation/plot_target_station_results.py \
    --split test \
    --models climatology persistence lightgbm_main \
             lightgbm_tuned_simple lstm \
    --force

NEXT PIPELINE PHASE
-------------------
- Phase 11.2 depends on Phase 11.1 completing first.
- Phase 11.2 is the final script in the documented pipeline.
- PNG and CSV outputs from plots/ are used directly for paper figures
  and supplementary reporting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

DEFAULT_EVAL_ROOT = Path("data/reports/evaluation/target_station")
DEFAULT_OUTPUT_DIRNAME = "plots"

TARGET_ORDER = [
    "target_geq_p95_30m",
    "target_geq_p99_30m",
    "target_geq_p95_60m",
    "target_geq_p99_60m",
    "target_geq_p95_120m",
    "target_geq_p99_120m",
]

MODEL_ORDER = [
    "climatology",
    "persistence",
    "logistic_regression",
    "lightgbm_main",
    "lightgbm_tuned_simple",
    "lstm",
]

MODEL_LABELS = {
    "climatology": "Climatology",
    "persistence": "Persistence",
    "logistic_regression": "Logistic Regression",
    "lightgbm_main": "LightGBM",
    "lightgbm_tuned_simple": "LightGBM Tuned",
    "lstm": "LSTM",
}

COLOR_SCALE_SKILL = "Viridis"
COLOR_SCALE_DIFF = "RdBu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot target x station evaluation results from saved evaluation CSVs."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help="Directory containing target_station_metrics_long.csv.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split to visualize, e.g. test or val.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "climatology",
            "persistence",
            "logistic_regression",
            "lightgbm_main",
            "lightgbm_tuned_simple",
            "lstm"
        ],
        help="Models to include in plots.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f2",
        choices=["f2", "precision", "recall", "pr_auc", "brier", "hss"],
        help="Primary metric for heatmaps and summary bars.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing plot outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_long_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows found in {path}")
    return df


def normalize_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = pd.Categorical(out["target"], categories=TARGET_ORDER, ordered=True)

    stations = sorted(out["station"].dropna().astype(str).unique().tolist())
    out["station"] = pd.Categorical(out["station"], categories=stations, ordered=True)

    model_categories = [m for m in MODEL_ORDER if m in out["model"].unique()]
    remaining = [m for m in out["model"].unique() if m not in model_categories]
    out["model"] = pd.Categorical(
        out["model"],
        categories=model_categories + sorted(remaining),
        ordered=True,
    )
    return out


def save_plot(fig, out_path: Path) -> None:
    fig.write_image(str(out_path), scale=2, width=1400, height=700)


def finite_series(values: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce")
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return out.astype(float)


def skill_color_limits(values: pd.Series, metric: str) -> tuple[float | None, float | None, float | None]:
    s = finite_series(values)
    if s.empty:
        return None, None, None

    if metric == "brier":
        lo = float(s.quantile(0.05))
        hi = float(s.quantile(0.95))
        if lo == hi:
            lo = float(s.min())
            hi = float(s.max())
        if lo == hi:
            lo = max(0.0, lo - 0.01)
            hi = min(1.0, hi + 0.01)
        return lo, hi, None

    if metric == "hss":
        lo = float(s.quantile(0.05))
        hi = float(s.quantile(0.95))
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        vmax = max(abs(lo), abs(hi), 0.05)
        return -vmax, vmax, 0.0

    lo = float(s.quantile(0.05))
    hi = float(s.quantile(0.95))
    lo = max(0.0, lo)
    hi = min(1.0, hi)

    if lo == hi:
        lo = float(max(0.0, s.min() - 0.01))
        hi = float(min(1.0, s.max() + 0.01))
    if lo == hi:
        lo, hi = 0.0, 1.0

    return lo, hi, None

def make_heatmap(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    out_path: Path,
    color_continuous_scale: str,
    zmin: float | None = None,
    zmax: float | None = None,
    midpoint: float | None = None,
) -> None:
    pivot = df.pivot(index="target", columns="station", values=value_col)
    pivot = pivot.reindex(TARGET_ORDER)

    fig = px.imshow(
        pivot,
        labels={"x": "Station", "y": "Target", "color": value_col},
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale=color_continuous_scale,
        zmin=zmin,
        zmax=zmax,
        aspect="auto",
        text_auto=".2f",
    )

    if midpoint is not None:
        fig.update_coloraxes(cmid=midpoint)

    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
        coloraxis_colorbar=dict(title=value_col),
    )
    fig.update_xaxes(side="bottom", tickangle=45)
    fig.update_yaxes(categoryorder="array", categoryarray=TARGET_ORDER)
    save_plot(fig, out_path)


def build_difference_df(
    df: pd.DataFrame,
    *,
    split: str,
    metric: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()
    keep = sub[sub["model"].isin([model_a, model_b])].copy()
    if keep.empty:
        return pd.DataFrame()

    wide = keep.pivot_table(
        index=["target", "station"],
        columns="model",
        values=metric,
        aggfunc="first",
        observed=False,
    ).reset_index()

    if model_a not in wide.columns or model_b not in wide.columns:
        return pd.DataFrame()

    wide[f"{model_a}_minus_{model_b}"] = wide[model_a] - wide[model_b]
    return wide


def make_mean_bar(
    df: pd.DataFrame,
    *,
    split: str,
    metric: str,
    out_path: Path,
) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()
    if sub.empty:
        raise ValueError(f"No rows found for split={split}")

    summary = (
        sub.groupby(["target", "model"], as_index=False, observed=False)[metric]
        .mean()
        .sort_values(["target", "model"])
    )

    summary["model_label"] = summary["model"].astype(str).map(MODEL_LABELS)
    summary["model_label"] = summary["model_label"].fillna(summary["model"].astype(str))

    fig = px.bar(
        summary,
        x="target",
        y=metric,
        color="model_label",
        barmode="group",
        category_orders={"target": TARGET_ORDER},
        labels={"target": "Target", metric: f"Mean station-level {metric}", "model_label": "Model"},
        title=f"Mean station-level {metric} by target ({split})",
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=60),
        legend_title_text="Model",
    )
    save_plot(fig, out_path)
    return summary


def main() -> None:
    args = parse_args()

    long_path = args.eval_root / "target_station_metrics_long.csv"
    out_dir = args.eval_root / DEFAULT_OUTPUT_DIRNAME

    if out_dir.exists() and not args.force:
        print(f"[info] output exists: {out_dir}")
    ensure_dir(out_dir)

    df = read_long_metrics(long_path)
    df = normalize_order(df)

    df = df[df["split"] == args.split].copy()
    df = df[df["model"].astype(str).isin(args.models)].copy()

    if df.empty:
        raise ValueError("No rows remain after split/model filtering.")

    df.to_csv(out_dir / f"filtered_metrics_{args.split}.csv", index=False)

    metric = args.metric

    for model_name in args.models:
        sub = df[df["model"].astype(str) == model_name].copy()
        if sub.empty:
            continue

        label = MODEL_LABELS.get(model_name, model_name)
        sub.to_csv(out_dir / f"{model_name}_{args.split}_{metric}_heatmap_source.csv", index=False)

        zmin, zmax, midpoint = skill_color_limits(sub[metric], metric)

        color_scale = COLOR_SCALE_SKILL
        if metric == "hss":
            color_scale = COLOR_SCALE_DIFF

        make_heatmap(
            sub,
            value_col=metric,
            title=f"{label} {args.split} {metric} by target x station",
            out_path=out_dir / f"{model_name}_{args.split}_{metric}_heatmap.png",
            color_continuous_scale=color_scale,
            zmin=zmin,
            zmax=zmax,
            midpoint=midpoint,
        )
        if "lightgbm_main" in args.models and "climatology" in args.models:
            diff_df = build_difference_df(
                df,
                split=args.split,
                metric=metric,
                model_a="lightgbm_main",
                model_b="climatology",
            )
            if not diff_df.empty:
                diff_col = "lightgbm_main_minus_climatology"
                diff_df.to_csv(out_dir / f"lightgbm_minus_climatology_{args.split}_{metric}.csv", index=False)

                diff_vals = finite_series(diff_df[diff_col])
                if diff_vals.empty:
                    vmax = 0.05
                else:
                    vmax = float(diff_vals.abs().quantile(0.95))
                    vmax = max(vmax, 0.05)

                make_heatmap(
                    diff_df,
                    value_col=diff_col,
                    title=f"LightGBM - Climatology {args.split} {metric} by target x station",
                    out_path=out_dir / f"lightgbm_minus_climatology_{args.split}_{metric}_heatmap.png",
                    color_continuous_scale=COLOR_SCALE_DIFF,
                    zmin=-vmax,
                    zmax=vmax,
                    midpoint=0.0,
                )

    if "lightgbm_tuned_simple" in args.models and "lightgbm_main" in args.models:
        diff_df = build_difference_df(
            df,
            split=args.split,
            metric=metric,
            model_a="lightgbm_tuned_simple",
            model_b="lightgbm_main",
        )
        if not diff_df.empty:
            diff_col = "lightgbm_tuned_simple_minus_lightgbm_main"
            diff_df.to_csv(out_dir / f"lightgbm_tuned_minus_main_{args.split}_{metric}.csv", index=False)

            diff_vals = finite_series(diff_df[diff_col])
            if diff_vals.empty:
                vmax = 0.05
            else:
                vmax = float(diff_vals.abs().quantile(0.95))
                vmax = max(vmax, 0.05)

            make_heatmap(
                diff_df,
                value_col=diff_col,
                title=f"LightGBM Tuned - LightGBM {args.split} {metric} by target x station",
                out_path=out_dir / f"lightgbm_tuned_minus_main_{args.split}_{metric}_heatmap.png",
                color_continuous_scale=COLOR_SCALE_DIFF,
                zmin=-vmax,
                zmax=vmax,
                midpoint=0.0,
            )

    mean_summary = make_mean_bar(
        df,
        split=args.split,
        metric=metric,
        out_path=out_dir / f"mean_station_{metric}_by_target_{args.split}.png",
    )
    mean_summary.to_csv(out_dir / f"mean_station_{metric}_by_target_{args.split}.csv", index=False)

    print("\nDone.")
    print(f"Plots written to: {out_dir}")


if __name__ == "__main__":
    main()
