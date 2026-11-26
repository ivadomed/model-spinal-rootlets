"""
Box plots of the rootlet segmentation metrics in the style of dice-hd95-graph.py (RootletSeg figures):
seaborn box plots with the spinal level on the x axis, one figure for Dice and one for the Hausdorff distance.

    plot_models(...)    : several models (datasets/folds) side by side, hue = model
                          (cf. plot_one_contrast_more_models: colors tab20c[17], Dark2[3], then Set2)
    plot_contrasts(...) : one model, hue = contrast (cf. plot_one_model_more_contrasts: colors Set2[2:6])

Figures are saved as dice_<tag>.png and hausdorff_<tag>.png (600 dpi). The data frame has one row per case and level
with the columns 'level' (C2, C3, ...), the metric columns and the hue column. C1 is removed beforehand
(sc_crop_rootlets_stats.py, -exclude-levels).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator

CONTRAST_ORDER = ["T1w-INV1", "T1w-INV2", "UNIT1", "T2w"]


def level_order(levels):
    return sorted(levels, key=lambda s: ("CTLS".find(s[0]) if s[:1] in "CTL" else 9,
                                         int(s[1:]) if s[1:].isdigit() else 0))


def metric_kind(col):
    """'dice', 'hausdorff' or None (other metric) from a MetricsReloaded column name."""
    c = col.lower()
    if "dice" in c or c == "dsc":
        return "dice"
    if "hausdorff" in c or c.startswith("hd"):
        return "hausdorff"
    return None


def model_palette(n):
    """Colors of dice-hd95-graph.py for the comparison of models, extended with Set2."""
    colors = [sns.color_palette("tab20c")[17], sns.color_palette("Dark2")[3]] + list(sns.color_palette("Set2"))
    return colors[:n]


def _axis(ax, kind, ylabel, data_max):
    ax.set_xlabel("Spinal level", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)
    if kind == "dice":
        ax.set_ylabel("Dice score", fontsize=22)
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylim(0.00, 0.90 if data_max <= 0.88 else 1.00)   # 1.00 if a box/outlier would touch 0.90
    elif kind == "hausdorff":
        ax.set_ylabel(ylabel, fontsize=22)
        top = 25 if data_max <= 24.5 else 5 * (int(data_max // 5) + 1)   # never cut data off
        ax.yaxis.set_major_locator(FixedLocator(range(0, top + 1, 5)))
        ax.set_ylim(0, top)
    else:
        ax.set_ylabel(ylabel, fontsize=22)


def _ylabel(col):
    c = col.lower()
    if metric_kind(col) == "hausdorff":
        return "Hausdorff distance [mm]"   # HD95 or HD, labelled as in the RootletSeg paper figures
    return col


def _fname(col):
    kind = metric_kind(col)
    return kind if kind else "".join(ch if ch.isalnum() else "_" for ch in col)


def boxplot(df, metric, hue, hue_order, palette, out_png, legend_outside):
    levels = level_order(df["level"].unique())
    kind = metric_kind(metric)
    plt.figure(figsize=(12, 6) if (kind == "dice" and legend_outside) else (8, 6))
    ax = sns.boxplot(x="level", y=metric, data=df, order=levels, hue=hue, hue_order=hue_order,
                     dodge=True, width=0.6, palette=palette)
    handles, labels = ax.get_legend_handles_labels()
    if kind == "dice" and legend_outside:
        ax.legend(handles, labels, title="Model", title_fontsize=18, fontsize=18, loc="upper left",
                  bbox_to_anchor=(1.02, 1.00), ncol=1, frameon=False)
    elif kind == "dice" and hue == "contrast":
        ax.legend(handles, labels, title="", fontsize=18, loc="lower right", ncol=2)
    elif kind == "dice":
        ax.legend(handles, labels, title="", fontsize=18, loc="lower left", ncol=1, frameon=True)
    else:
        ax.legend(handles, labels, title="", fontsize=18, loc="upper left",
                  ncol=2 if hue == "contrast" else 1, frameon=True)
    _axis(ax, kind, _ylabel(metric), df[metric].max())
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")


def plot_models(df, metric_cols, model_col, model_order, out_dir, tag="ALL"):
    """Models side by side (one contrast or all pooled): dice_<tag>.png, hausdorff_<tag>.png."""
    os.makedirs(out_dir, exist_ok=True)
    order = [m for m in model_order if m in set(df[model_col])]
    # 2 models: legend inside (lower left, as for MULTICON); more: legend next to the plot
    outside = len(order) > 2
    for m in metric_cols:
        boxplot(df, m, model_col, order, model_palette(len(order)),
                os.path.join(out_dir, f"{_fname(m)}_{tag}.png"), legend_outside=outside)


def plot_contrasts(df, metric_cols, out_dir, tag=""):
    """One model across contrasts: dice[_<tag>].png, hausdorff[_<tag>].png."""
    os.makedirs(out_dir, exist_ok=True)
    present = list(dict.fromkeys(df["contrast"]))
    order = [c for c in CONTRAST_ORDER if c in present] + [c for c in present if c not in CONTRAST_ORDER]
    palette = (list(sns.color_palette("Set2")[2:6]) + list(sns.color_palette("Set2")[:2]))[:len(order)]
    sfx = f"_{tag}" if tag else ""
    for m in metric_cols:
        boxplot(df, m, "contrast", order, palette, os.path.join(out_dir, f"{_fname(m)}{sfx}.png"),
                legend_outside=False)


def _panel_axis(ax, kind, ylabel, vmin, vmax):
    """Axes of the contrast panel figure: Dice 0.25-0.85 (ticks 0.10), Hausdorff 0-20 mm (ticks 2.5),
    widened in the same steps when data fall outside."""
    import math
    ax.set_xlabel("Spinal level", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5)
    if kind == "dice":
        lo = 0.25 if vmin >= 0.26 else max(0.05, 0.25 - 0.10 * math.ceil((0.26 - vmin) / 0.10))
        hi = 0.85 if vmax <= 0.84 else min(1.05, 0.85 + 0.10 * math.ceil((vmax - 0.84) / 0.10))
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator([round(lo + 0.10 * i, 2) for i in range(int(round((hi - lo) / 0.10)) + 1)]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_ylabel("Dice score", fontsize=22)
    else:
        hi = 20.0 if vmax <= 19.5 else 2.5 * math.ceil((vmax + 0.5) / 2.5)
        ax.set_ylim(0, hi)
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylabel(ylabel, fontsize=22)


def plot_contrasts_panel(df, metric_cols, out_png):
    """One model across contrasts, Dice (a) and Hausdorff (b) side by side, one box per contrast and
    spinal level (RootletSeg paper figure)."""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    cols = [m for m in metric_cols if metric_kind(m) == "dice"][:1] + \
           [m for m in metric_cols if metric_kind(m) == "hausdorff"][:1]
    if not cols:
        return
    present = list(dict.fromkeys(df["contrast"]))
    order = [c for c in CONTRAST_ORDER if c in present] + [c for c in present if c not in CONTRAST_ORDER]
    palette = (list(sns.color_palette("Set2")[2:6]) + list(sns.color_palette("Set2")[:2]))[:len(order)]
    levels = level_order(df["level"].unique())
    fig, axes = plt.subplots(1, len(cols), figsize=(9 * len(cols), 6.5), squeeze=False)
    for k, (ax, m) in enumerate(zip(axes[0], cols)):
        kind = metric_kind(m)
        sns.boxplot(x="level", y=m, data=df, order=levels, hue="contrast", hue_order=order, palette=palette,
                    dodge=True, width=0.6, linewidth=0.8, ax=ax,
                    flierprops=dict(marker="o", markersize=6, markerfacecolor="none",
                                    markeredgecolor="#9a9a9a", markeredgewidth=0.8))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="", fontsize=18, ncol=2, frameon=True,
                  loc="lower right" if kind == "dice" else "upper left")
        _panel_axis(ax, kind, _ylabel(m), df[m].min(), df[m].max())
        if len(cols) > 1:
            ax.text(-0.16, 1.02, "ab"[k] + ")", transform=ax.transAxes, fontsize=26, fontweight="bold",
                    va="bottom", ha="left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
