#!/usr/bin/env python3
"""C4/en vs GBNC coupling experiment for adverb roots.

Coupling reference (slope=1 standard):
  - Orange line: log10(C4) = log10(GBNC) + ref_intercept (slope fixed at 1)
  - ref_intercept: constrained OLS intercept (= mean log-ratio center)
  - slope / intercept: free log-log fit, reported as deviation from the standard
  - 5x/10x bands and coverage: relative to the slope=1 reference line

Data processing follows S2.Frequency comparison.ipynb:
  - GBNC: 10-year window mean counts from yearly data
  - C4/en: static corpus counts (same across windows)

Data:
  - adv_c4_appear_times.csv: C4/en counts (phrase, appear times)
  - adv_gbnc_appear_counts.csv: GBNC yearly counts (year × phrase)
  - adv_gbnc_appear_times.csv: GBNC 1800–2019 corpus mean (summary reference)

Usage:
    python 10-plot_C4_GBNC.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, NullFormatter
from scipy import stats

HERE = Path(__file__).resolve().parent
DEFAULT_C4_PATH = HERE / "adv_c4_appear_times.csv"
DEFAULT_GBNC_YEARLY_PATH = HERE / "adv_gbnc_appear_counts.csv"
DEFAULT_GBNC_MEAN_PATH = HERE / "adv_gbnc_appear_times.csv"
DEFAULT_OUTPUT_DIR = HERE / "outputs_c4_gbnc_coupling"

START_YEAR = 1800
END_YEAR = 2019
WINDOW = 10
MIN_COUNT = 100
BAND_FACTORS = (2, 5, 10)
REFERENCE_SLOPE = 1.0
COVERAGE_THRESHOLD = 0.8
HIGHLIGHT_WINDOW = "2010-2019"
FIGURE_FORMAT = "svg"
FIGURE_DPI = 400  # 位图输出分辨率；SVG 为矢量，印刷时可任意缩放
INSET_BOUNDS = (0.58, 0.10, 0.38, 0.28)

COLORS = ["#f79779", "#6671b5"]
PLOT_X_RANGE = (1e2, 1e9)  # log-log 显示范围：覆盖 GBNC/C4 全频 adverb 配对（至 ~10^8）


@ticker.FuncFormatter
def log10_formatter(x: float, pos: int) -> str:
    if x <= 0:
        return ""
    return f"{int(round(np.log10(x)))}"


def normalize_phrase(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def load_appear_times(path: Path) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    phrase_col = next(
        (lower_cols[c] for c in ["phrase", "word", "ngram", "term"] if c in lower_cols),
        df.columns[0],
    )
    count_col = next(
        (
            lower_cols[c]
            for c in ["appear times", "appear_times", "count", "counts", "df"]
            if c in lower_cols
        ),
        df.columns[1],
    )
    out = df[[phrase_col, count_col]].copy()
    out[phrase_col] = out[phrase_col].map(normalize_phrase)
    out[count_col] = pd.to_numeric(out[count_col], errors="coerce")
    out = out.dropna(subset=[phrase_col, count_col])
    return (
        out.groupby(phrase_col, as_index=True)[count_col]
        .max()
        .astype(float)
    )


def load_gbnc_yearly(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    year_col = next(
        (c for c in df.columns if str(c).strip().lower() in {"year", "years"}),
        df.columns[0],
    )
    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df[df[year_col].between(START_YEAR, END_YEAR)]
    value_cols = [c for c in df.columns if c != year_col]
    df.columns = [year_col] + [normalize_phrase(c) for c in value_cols]
    return df.set_index(year_col).apply(pd.to_numeric, errors="coerce").sort_index()


def aggregate_windows(counts: pd.DataFrame) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for start in range(START_YEAR, END_YEAR + 1, WINDOW):
        end = min(start + WINDOW - 1, END_YEAR)
        sub = counts.loc[(counts.index >= start) & (counts.index <= end)]
        if not sub.empty:
            result[f"{start}-{end}"] = sub.mean(axis=0, skipna=True)
    return result


def find_threshold_crossing_year(
    summary_df: pd.DataFrame,
    *,
    threshold: float = COVERAGE_THRESHOLD,
    column: str = "within_5x",
) -> float | None:
    """Linear interpolation for the year where coverage first reaches threshold."""
    if summary_df.empty or column not in summary_df.columns:
        return None

    ordered = summary_df.sort_values("mid_year")
    years = ordered["mid_year"].to_numpy(dtype=float)
    values = ordered[column].to_numpy(dtype=float)

    for idx in range(len(years) - 1):
        y0, y1 = values[idx], values[idx + 1]
        x0, x1 = years[idx], years[idx + 1]
        if y0 < threshold <= y1:
            if y1 == y0:
                return float(x0)
            return float(x0 + (threshold - y0) / (y1 - y0) * (x1 - x0))
    return None


def crossing_year_display(crossing_year: float | None) -> int | None:
    """Ceiling of interpolated crossing year for axis labels and shading."""
    if crossing_year is None:
        return None
    return int(math.ceil(crossing_year))


def _format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    if p_value < 0.01:
        return "p < 0.01"
    return f"p = {p_value:.3f}"


def _ols_fit_legend_label(summary: pd.Series) -> str:
    slope = float(summary["slope"])
    r2 = float(summary["r2"])
    return f"OLS fit (slope = {slope:.2f}, $R^2$ = {r2:.2f})"


def coupling_for_pair(
    gbnc: pd.Series,
    c4: pd.Series,
    *,
    min_count: float = MIN_COUNT,
    band_factors: tuple[float, ...] = BAND_FACTORS,
) -> tuple[pd.Series, pd.DataFrame]:
    pair = pd.concat([gbnc.rename("gbnc"), c4.rename("c4_en")], axis=1, join="inner").dropna()
    pair = pair[(pair["gbnc"] >= min_count) & (pair["c4_en"] >= min_count)]

    summary: dict[str, float | int] = {
        "n_phrases": len(pair),
        "slope": np.nan,
        "slope_se": np.nan,
        "slope_ci_low": np.nan,
        "slope_ci_high": np.nan,
        "slope_pvalue": np.nan,
        "slope_deviation": np.nan,
        "intercept": np.nan,
        "ref_intercept": np.nan,
        "r2": np.nan,
        "pearson_log": np.nan,
        "spearman_log": np.nan,
    }
    for factor in band_factors:
        summary[f"within_{factor}x"] = np.nan

    if len(pair) < 3:
        return pd.Series(summary), pair

    lx = np.log10(pair["gbnc"].astype(float))
    ly = np.log10(pair["c4_en"].astype(float))
    reg = stats.linregress(lx, ly)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    n = len(lx)
    t_crit = float(stats.t.ppf(0.975, n - 2))
    slope_se = float(reg.stderr)
    ref_intercept = float(np.mean(ly - REFERENCE_SLOPE * lx))
    ref_fitted = REFERENCE_SLOPE * lx + ref_intercept
    residual = ly - ref_fitted

    summary.update(
        {
            "slope": slope,
            "slope_se": slope_se,
            "slope_ci_low": slope - t_crit * slope_se,
            "slope_ci_high": slope + t_crit * slope_se,
            "slope_pvalue": float(reg.pvalue),
            "slope_deviation": slope - REFERENCE_SLOPE,
            "intercept": intercept,
            "ref_intercept": ref_intercept,
            "r2": float(reg.rvalue ** 2),
            "pearson_log": stats.pearsonr(lx, ly).statistic,
            "spearman_log": stats.spearmanr(lx, ly).statistic,
        }
    )
    for factor in band_factors:
        summary[f"within_{factor}x"] = float((residual.abs() <= np.log10(factor)).mean())

    pair = pair.assign(
        log_gbnc=lx,
        log_c4_en=ly,
        fitted_log_c4_en=ref_fitted,
        residual_log10=residual,
    )
    return pd.Series(summary), pair


def fit_band_lines(ref_intercept: float) -> dict[str, np.ndarray]:
    x_values = np.array(PLOT_X_RANGE, dtype=float)
    y_fit = 10 ** (REFERENCE_SLOPE * np.log10(x_values) + ref_intercept)
    y5_down = 0.2 * y_fit
    y5_up = 5.0 * y_fit
    y10_down = 0.1 * y_fit
    y10_up = 10.0 * y_fit
    return {
        "x_values": x_values,
        "y_fit": y_fit,
        "y5_down": y5_down,
        "y5_up": y5_up,
        "y10_down": y10_down,
        "y10_up": y10_up,
    }


def style_log_axes(ax: plt.Axes) -> None:
    """Log-log axes with major/minor ticks on all four sides."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(PLOT_X_RANGE)
    ax.set_ylim(PLOT_X_RANGE)

    major = LogLocator(base=10.0)
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(major)
        axis.set_minor_locator(minor)
        axis.set_major_formatter(log10_formatter)
        axis.set_minor_formatter(NullFormatter())

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.minorticks_on()

    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        bottom=True,
        top=True,
        labelbottom=True,
        labeltop=False,
        labelsize=10,
        width=0.8,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=True,
        right=True,
        labelleft=True,
        labelright=False,
        labelsize=10,
        width=0.8,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)


def _raise_axes_over_inset(ax: plt.Axes, inset_ax: plt.Axes) -> None:
    """Draw host spines/ticks above inset background so bottom ticks are not hidden."""
    ax.set_zorder(1)
    inset_ax.set_zorder(2)
    ax.patch.set_visible(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_zorder(10)
    for spine in ax.spines.values():
        spine.set_zorder(10)


def draw_coverage_trend_on_axes(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    crossing_year: float | None,
    *,
    inset: bool = False,
) -> None:
    crossing_tick = crossing_year_display(crossing_year)
    label_size = 8 if inset else 10
    marker_size = 3 if inset else 4
    line_width = 1.0 if inset else 1.5

    if inset:
        ax.margins(0)

    ax.plot(
        summary_df["mid_year"],
        summary_df["within_5x"],
        color="#6671b5",
        linewidth=line_width,
        alpha=0.7,
        marker="o",
        markersize=marker_size,
    )
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_ylim(0.5, 0.96)
    if inset:
        x_ticks: list[float | int] = [START_YEAR, END_YEAR]
        x_labels = [str(START_YEAR), str(END_YEAR)]
        if (
            crossing_year is not None
            and crossing_tick is not None
            and START_YEAR < crossing_year < END_YEAR
        ):
            x_ticks.insert(1, crossing_year)
            x_labels.insert(1, str(crossing_tick))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text() == str(crossing_tick):
                tick_label.set_fontweight("bold")
    else:
        x_ticks = [1800, 1900, END_YEAR]
        if crossing_tick is not None and START_YEAR < crossing_tick < END_YEAR:
            x_ticks.insert(2, crossing_tick)
        ax.set_xticks(x_ticks)
    ax.set_xlabel("Decade", fontsize=label_size)
    if inset:
        ax.set_ylabel("Coverage Ratio", fontsize=label_size)
        ax.set_yticks([0.5, COVERAGE_THRESHOLD, 0.9])
    else:
        ax.set_ylabel("Rate (Covered)", fontsize=label_size)
        ax.set_yticks([0.5, 0.6, COVERAGE_THRESHOLD, 0.9])
    ax.axhline(
        COVERAGE_THRESHOLD,
        color="#6d6e70",
        linewidth=1.5 if inset else 2,
        linestyle="--",
        dashes=(1, 1),
        alpha=0.8,
    )
    if crossing_tick is not None:
        if inset and crossing_year is not None:
            ax.axvspan(crossing_tick, END_YEAR, alpha=0.5, color="#fac0ae", linewidth=0, zorder=0)
            ax.axvline(
                crossing_year,
                color="#262626",
                linewidth=0.8,
                zorder=4,
            )
        else:
            ax.axvspan(crossing_tick, END_YEAR, alpha=0.5, color="#fac0ae", linewidth=0)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        bottom=True,
        top=False,
        labelbottom=True,
        labeltop=False,
        labelsize=label_size,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
        labelsize=label_size,
    )
    ax.grid(False)


def _band_y(x: float, factor: float, ref_intercept: float) -> float:
    return factor * 10 ** (np.log10(x) + ref_intercept)


LOG_LABEL_RIGHT_PAD = 0.38  # 标注距右轴的最小 log10 留白


def _find_upper_right_label_x(
    pair: pd.DataFrame,
    ref_intercept: float,
) -> float:
    """在右上空白区选取点密度最低的 x（沿参考线以上，不过度贴边）。"""
    gbnc = pair["gbnc"].astype(float).to_numpy()
    log_g = np.log10(gbnc)
    log_c = np.log10(pair["c4_en"].astype(float).to_numpy())
    x_lo, x_hi = PLOT_X_RANGE
    log_x_cap = np.log10(x_hi) - LOG_LABEL_RIGHT_PAD

    x_min = max(float(10 ** np.percentile(log_g, 88)), x_lo * 30.0)
    x_max = min(
        float(10 ** np.percentile(log_g, 96) * 2.2),
        10.0 ** log_x_cap,
        float(gbnc.max()) * 0.35,
    )
    if x_max <= x_min * 1.2:
        x_max = x_min * 3.0

    candidates = np.logspace(np.log10(x_min), np.log10(x_max), 28)
    best_x = float(candidates[-1])
    best_score = -np.inf
    log_window = 0.45

    for x in candidates:
        lg = np.log10(x)
        ref_log_y = lg + ref_intercept
        above = log_c >= ref_log_y - 0.15
        near = np.abs(log_g - lg) < log_window
        density = int(np.sum(above & near))
        score = -density + 0.03 * (lg - np.log10(x_min))
        if score > best_score:
            best_score = score
            best_x = float(x)

    return min(best_x, 10.0 ** log_x_cap)


def _annotate_tolerance_badges(
    ax: plt.Axes,
    ref_intercept: float,
    pair: pd.DataFrame | None = None,
) -> None:
    """5× / 10× 标注：自适应置于数据稀疏的右上方，沿对应 band 边界。"""
    if pair is None or pair.empty:
        return

    text_kw = dict(
        ha="left", va="center", clip_on=True, fontfamily="sans-serif", fontsize=8,
    )
    x_label = _find_upper_right_label_x(pair, ref_intercept)
    dx = x_label * 0.035

    labels = (
        (0.2, "5x", "#666666", 0.92),
        (0.1, "10x", "#999999", 0.90),
    )
    for factor, text, color, y_scale in labels:
        y = _band_y(x_label, factor, ref_intercept) * y_scale
        ax.text(x_label + dx, y, text, color=color, **text_kw)


def _save_figure(fig: plt.Figure, output_path: Path, *, dpi: int) -> None:
    fig.savefig(
        output_path,
        format=output_path.suffix.lstrip(".") or FIGURE_FORMAT,
        bbox_inches="tight",
        dpi=dpi,
    )
    plt.close(fig)


def plot_coupling_scatter(
    pair: pd.DataFrame,
    summary: pd.Series,
    *,
    output_path: Path,
    title: str,
    c4_label: str = "C4/en",
    trend_df: pd.DataFrame | None = None,
    crossing_year: float | None = None,
    figure_dpi: int = FIGURE_DPI,
) -> None:
    ref_intercept = float(summary["ref_intercept"])
    bands = fit_band_lines(ref_intercept)
    x_values = bands["x_values"]

    in_mask = pair["residual_log10"].abs() <= np.log10(5.0)
    in_x = pair.loc[in_mask, "gbnc"].tolist()
    in_y = pair.loc[in_mask, "c4_en"].tolist()
    out_x = pair.loc[~in_mask, "gbnc"].tolist()
    out_y = pair.loc[~in_mask, "c4_en"].tolist()

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot(
        x_values,
        bands["y_fit"],
        color="#d95f02",
        alpha=0.8,
        linewidth=1.5,
        label="Reference (slope = 1)",
    )
    ax.plot(
        x_values,
        10 ** (summary["slope"] * np.log10(x_values) + summary["intercept"]),
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=_ols_fit_legend_label(summary),
    )
    ax.plot(x_values, bands["y5_up"], color="#E6E6E6", alpha=0.8, linewidth=0)
    ax.plot(x_values, bands["y5_down"], color="#E6E6E6", alpha=0.8, linewidth=0)
    ax.fill_between(x_values, bands["y5_up"], bands["y5_down"], color="#E6E6E6", alpha=0.8)
    ax.plot(
        x_values,
        bands["y10_up"],
        color="#f79779",
        alpha=0.6,
        linewidth=1,
        linestyle="--",
        dashes=(3, 3),
    )
    ax.plot(
        x_values,
        bands["y10_down"],
        color="#f79779",
        alpha=0.6,
        linewidth=1,
        linestyle="--",
        dashes=(3, 3),
    )
    ax.fill_between(x_values, bands["y5_up"], bands["y10_up"], color="#F5F5F5", alpha=0.8)
    ax.fill_between(x_values, bands["y10_down"], bands["y5_down"], color="#F5F5F5", alpha=0.8)

    ax.scatter(
        in_x,
        in_y,
        s=6,
        c=COLORS[0],
        alpha=0.25,
        label=f"Covered phrases ({title})",
    )
    ax.scatter(
        out_x,
        out_y,
        s=6,
        c=COLORS[1],
        alpha=0.25,
        label=f"Uncovered phrases ({title})",
    )

    ax.set_xlabel("GBNC occurrences (log)", fontsize=12)
    ax.set_ylabel("C4/EN occurrences (log)", fontsize=12)
    style_log_axes(ax)
    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
    )

    if trend_df is not None and not trend_df.empty:
        inset_ax = ax.inset_axes(INSET_BOUNDS)
        inset_ax.set_facecolor("white")
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("#262626")
            spine.set_linewidth(0.8)
        draw_coverage_trend_on_axes(
            inset_ax,
            trend_df,
            crossing_year,
            inset=True,
        )
        _raise_axes_over_inset(ax, inset_ax)
        style_log_axes(ax)

    _annotate_tolerance_badges(ax, ref_intercept, pair)

    plt.tight_layout()
    _save_figure(fig, output_path, dpi=figure_dpi)


def plot_coverage_trend(
    summary_df: pd.DataFrame,
    output_path: Path,
    *,
    figure_dpi: int = FIGURE_DPI,
) -> float | None:
    crossing_year = find_threshold_crossing_year(summary_df)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    draw_coverage_trend_on_axes(ax, summary_df, crossing_year, inset=False)
    plt.tight_layout()
    _save_figure(fig, output_path, dpi=figure_dpi)
    return crossing_year


def run_coupling_experiment(
    c4_path: Path,
    gbnc_yearly_path: Path,
    gbnc_mean_path: Path,
    output_dir: Path,
    *,
    figure_dpi: int = FIGURE_DPI,
    figure_format: str = FIGURE_FORMAT,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], float | None]:
    output_dir.mkdir(parents=True, exist_ok=True)

    c4_counts = load_appear_times(c4_path)
    gbnc_yearly = load_gbnc_yearly(gbnc_yearly_path)
    gbnc_windows = aggregate_windows(gbnc_yearly)

    rows: list[pd.Series] = []
    paired: dict[str, pd.DataFrame] = {}

    labels = sorted(gbnc_windows, key=lambda s: int(s.split("-")[0]))
    for label in labels:
        summary, pair = coupling_for_pair(gbnc_windows[label], c4_counts)
        start, end = [int(v) for v in label.split("-")]
        summary["window"] = label
        summary["mid_year"] = (start + end) / 2
        rows.append(summary)
        paired[label] = pair
        pair.to_csv(output_dir / f"paired_counts_{label}.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        cols = ["window", "mid_year"] + [
            c for c in summary_df.columns if c not in {"window", "mid_year"}
        ]
        summary_df = summary_df[cols]
    summary_df.to_csv(output_dir / "coupling_summary.csv", index=False, encoding="utf-8-sig")

    crossing_year = find_threshold_crossing_year(summary_df) if not summary_df.empty else None

    # Reference: GBNC 1800-2019 mean vs C4 (static summary files)
    gbnc_mean = load_appear_times(gbnc_mean_path)
    mean_summary, mean_pair = coupling_for_pair(gbnc_mean, c4_counts)
    mean_summary["window"] = "1800-2019_mean"
    mean_summary["mid_year"] = (START_YEAR + END_YEAR) / 2
    mean_pair.to_csv(output_dir / "paired_counts_1800-2019_mean.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([mean_summary]).to_csv(
        output_dir / "coupling_summary_1800-2019_mean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if HIGHLIGHT_WINDOW in paired:
        plot_coupling_scatter(
            paired[HIGHLIGHT_WINDOW],
            summary_df.loc[summary_df["window"] == HIGHLIGHT_WINDOW].iloc[0],
            output_path=output_dir / f"scatter_{HIGHLIGHT_WINDOW}.{figure_format}",
            title=HIGHLIGHT_WINDOW,
            trend_df=summary_df,
            crossing_year=crossing_year,
            figure_dpi=figure_dpi,
        )

    plot_coupling_scatter(
        mean_pair,
        mean_summary,
        output_path=output_dir / f"scatter_1800-2019_mean.{figure_format}",
        title="1800-2019 mean",
        figure_dpi=figure_dpi,
    )

    if not summary_df.empty:
        plot_coverage_trend(
            summary_df,
            output_dir / f"coupling_trend_within_5x.{figure_format}",
            figure_dpi=figure_dpi,
        )

    return summary_df, paired, crossing_year


def main() -> None:
    parser = argparse.ArgumentParser(description="C4/en vs GBNC coupling experiment.")
    parser.add_argument("--c4", type=Path, default=DEFAULT_C4_PATH)
    parser.add_argument("--gbnc-yearly", type=Path, default=DEFAULT_GBNC_YEARLY_PATH)
    parser.add_argument("--gbnc-mean", type=Path, default=DEFAULT_GBNC_MEAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_DPI,
        help=f"输出 DPI（位图有效，默认 {FIGURE_DPI}）",
    )
    parser.add_argument(
        "--format",
        dest="figure_format",
        default=FIGURE_FORMAT,
        choices=("svg", "png", "pdf"),
        help="输出格式（png/pdf 时 --dpi 决定像素分辨率）",
    )
    args = parser.parse_args()

    plt.rcParams["figure.dpi"] = args.dpi
    plt.rcParams["savefig.dpi"] = args.dpi
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "path"
    sns.set_theme(style="white", font="Arial")
    plt.rcParams.update(
        {
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    summary_df, paired, crossing_year = run_coupling_experiment(
        args.c4,
        args.gbnc_yearly,
        args.gbnc_mean,
        args.output_dir,
        figure_dpi=args.dpi,
        figure_format=args.figure_format,
    )

    print(f"c4:          {args.c4}")
    print(f"gbnc yearly: {args.gbnc_yearly}")
    print(f"gbnc mean:   {args.gbnc_mean}")
    print(f"output:      {args.output_dir}")
    print(f"format/dpi:  {args.figure_format} @ {args.dpi} dpi")
    print()
    if summary_df.empty:
        print("No coupling windows produced.")
        return

    print(summary_df.to_string(index=False))
    if crossing_year is not None:
        crossing_tick = crossing_year_display(crossing_year)
        print()
        print(
            f"within_5x crosses {COVERAGE_THRESHOLD:.1f} at year "
            f"{crossing_year:.1f} (display/shade from {crossing_tick} to {END_YEAR})"
        )
    if HIGHLIGHT_WINDOW in paired:
        highlight = summary_df.loc[summary_df["window"] == HIGHLIGHT_WINDOW].iloc[0]
        print()
        print(
            f"{HIGHLIGHT_WINDOW}: n={int(highlight['n_phrases'])}, "
            f"slope={highlight['slope']:.4f} "
            f"± {highlight['slope_se']:.4f} "
            f"[95% CI {highlight['slope_ci_low']:.3f}, {highlight['slope_ci_high']:.3f}], "
            f"{_format_p_value(float(highlight['slope_pvalue']))}, "
            f"R²={highlight['r2']:.3f}, "
            f"dev={highlight['slope_deviation']:+.4f}, "
            f"ref_intercept={highlight['ref_intercept']:.4f}, "
            f"within_5x={highlight['within_5x']:.3f}"
        )


if __name__ == "__main__":
    main()
