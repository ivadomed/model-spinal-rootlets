"""
Compare two rootlet segmentation models evaluated on the SAME test data, e.g. the model trained on images cropped with
sc-crop (predictions padded back with pad_predictions.py) vs. the existing model run on the original images.

Input: the metrics_per_case_level.csv files written by sc_crop_rootlets_stats.py for each model (one row per case
and level, MetricsReloaded metrics, e.g. Dice and HD), both computed against the same ORIGINAL ground truth.

Pairing: rows are matched by case and level; cases/levels present for only one model are listed and left out.

Statistics, for each metric (difference = model B - model A):
    - mean ± SD of each model and of the difference: overall (across levels and cases), per level, per contrast
    - paired test at the CASE level (one value per case = mean across its levels, so the levels of a case are not
      counted as independent samples); per level: paired test across cases, Holm-corrected across levels
    - the test depends on the normality of the paired differences (Shapiro-Wilk, alpha): normal -> paired t-test,
      otherwise -> Wilcoxon signed-rank test (same decision rule as statistics_cervical_rootlets_vs_PAM50.py in
      spinal-levels); both are saved
    For Dice higher is better, for HD lower is better.

Outputs (in --output):
    paired_metrics.csv                : paired rows (case, level, contrast, metric of model A and model B)
    comparison_overall.csv            : per metric, overall statistics and case-level paired test
    comparison_per_level.csv          : per metric and level, statistics and paired test (+ Holm correction)
    comparison_per_contrast.csv       : per metric and contrast (when more than one contrast)
    figures/dice_ALL.png, hausdorff_ALL.png : box plots per level, both models side by side
                                        (style of dice-hd95-graph.py; + per contrast if several)

Examples:
    # metrics computed with sc_crop_rootlets_stats.py -root (<ROOT>/metrics/<Dataset>/predict_fold_<fold>/...)
    python compare_models.py --metrics-root ~/results_rootlets/metrics \
        --dataset-a DatasetXX --dataset-b DatasetXY --fold all --output comparison_XX_vs_XY
    python compare_models.py \\
        --metrics-a metrics_existing/predict_fold_all/metrics_per_case_level.csv --name-a "RootletSeg" \\
        --metrics-b metrics_crop/predict_fold_all/metrics_per_case_level.csv --name-b "RootletSeg + sc-crop" \\
        --output comparison
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sc_crop_rootlets_stats import get_metric_cols, pretty_metric

KEYS = ["case", "level"]


def get_parser():
    p = argparse.ArgumentParser(description="Paired comparison of two models on the same test data.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--metrics-a", help="metrics_per_case_level.csv of model A (reference model).")
    p.add_argument("--metrics-b", help="metrics_per_case_level.csv of model B (new model).")
    p.add_argument("--metrics-root", help="Instead of --metrics-a/-b: output folder of sc_crop_rootlets_stats.py "
                                          "-root (contains <Dataset>/predict_fold_<fold>/metrics_per_case_level.csv).")
    p.add_argument("--dataset-a", help="With --metrics-root: dataset folder of model A, e.g. DatasetXX.")
    p.add_argument("--dataset-b", help="With --metrics-root: dataset folder of model B, e.g. DatasetXY.")
    p.add_argument("--fold", default="all", help="With --metrics-root: fold of both models (predict_fold_<fold>).")
    p.add_argument("--fold-b", default=None, help="With --metrics-root: fold of model B if it differs from --fold.")
    p.add_argument("--name-a", default=None, help="Name of model A (default: --dataset-a or 'existing model').")
    p.add_argument("--name-b", default=None, help="Name of model B (default: --dataset-b or 'sc-crop model').")
    p.add_argument("--output", default=None,
                   help="Output folder (with --metrics-root, default: <metrics-root>/comparison_<A>_vs_<B>).")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level (normality and paired tests).")
    p.add_argument("--overall-exclude-levels", nargs="*", default=["T1"],
                   help="Levels left out of the second overall row (default: T1 -> C2-C8, the overall "
                        "values of the RootletSeg paper); the first row keeps all levels.")
    p.add_argument("--no-plot", action="store_true", help="Skip the figures.")
    return p


def resolve_inputs(args):
    """Fill --metrics-a/-b, the names and the output folder from --metrics-root/--dataset-a/-b/--fold."""
    if args.metrics_root:
        if not (args.dataset_a and args.dataset_b):
            raise SystemExit("--metrics-root needs --dataset-a and --dataset-b")
        fold_b = args.fold_b or args.fold
        args.metrics_a = os.path.join(args.metrics_root, args.dataset_a, f"predict_fold_{args.fold}",
                                      "metrics_per_case_level.csv")
        args.metrics_b = os.path.join(args.metrics_root, args.dataset_b, f"predict_fold_{fold_b}",
                                      "metrics_per_case_level.csv")
        tag_a = args.dataset_a + (f"_fold{args.fold}" if args.fold_b else "")
        tag_b = args.dataset_b + (f"_fold{fold_b}" if args.fold_b else "")
        args.name_a = args.name_a or tag_a
        args.name_b = args.name_b or tag_b
        args.output = args.output or os.path.join(args.metrics_root, f"comparison_{tag_a}_vs_{tag_b}")
    if not (args.metrics_a and args.metrics_b and args.output):
        raise SystemExit("Give --metrics-a, --metrics-b and --output, or --metrics-root with --dataset-a/-b.")
    for f in (args.metrics_a, args.metrics_b):
        if not os.path.isfile(f):
            raise SystemExit(f"{f} not found")
    args.name_a = args.name_a or "existing model"
    args.name_b = args.name_b or "sc-crop model"
    return args


def level_order(levels):
    return sorted(levels, key=lambda s: ("CTLS".find(s[0]) if s[:1] in "CTL" else 9,
                                         int(s[1:]) if s[1:].isdigit() else 0))


def load_pair(fname_a, fname_b, name_a, name_b):
    """Paired table (rows in both files, matched by case and level) and the metric columns common to both."""
    df_a, df_b = pd.read_csv(fname_a), pd.read_csv(fname_b)
    for df, fname in ((df_a, fname_a), (df_b, fname_b)):
        missing = [c for c in KEYS if c not in df.columns]
        if missing:
            raise SystemExit(f"{fname} lacks the column(s) {missing}: use metrics_per_case_level.csv of "
                             f"sc_crop_rootlets_stats.py")
        df["case"] = df["case"].astype(str)
        df["level"] = df["level"].astype(str)
        if df.duplicated(KEYS).any():
            raise SystemExit(f"{fname} has several rows for the same case and level (several folds or labels?).")
    metrics = [m for m in get_metric_cols(df_a) if m in get_metric_cols(df_b)]
    if not metrics:
        raise SystemExit("No metric column common to both files.")

    keep_a = KEYS + (["contrast"] if "contrast" in df_a.columns else []) + metrics
    df = df_a[keep_a].merge(df_b[KEYS + metrics], on=KEYS, how="outer", suffixes=("_A", "_B"), indicator=True)
    for side, name in (("left_only", name_a), ("right_only", name_b)):
        only = df[df["_merge"] == side]
        if not only.empty:
            print(f"WARNING: {len(only)} case/level row(s) only in {name} (left out): "
                  f"{sorted(only['case'].unique())[:20]}{' ...' if only['case'].nunique() > 20 else ''}")
    cases_a, cases_b = set(df_a["case"]), set(df_b["case"])
    if cases_a != cases_b:
        print(f"WARNING: cases differ between the models ({len(cases_a)} vs {len(cases_b)}): "
              f"only in {name_a}: {sorted(cases_a - cases_b)}; only in {name_b}: {sorted(cases_b - cases_a)}")
    df = df[df["_merge"] == "both"].drop(columns="_merge").reset_index(drop=True)
    if "contrast" not in df.columns:
        df["contrast"] = "all"
    return df, metrics


def paired_test(a, b, alpha):
    """Paired comparison of b vs a (difference = b - a): descriptive statistics, Shapiro-Wilk of the differences,
    paired t-test and Wilcoxon signed-rank test, and the test selected by the normality of the differences."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    diff = b - a
    n = len(diff)
    res = {"n": n, "n_excluded_nan": int((~ok).sum()),
           "A_mean": a.mean() if n else np.nan, "A_sd": a.std(ddof=1) if n > 1 else np.nan,
           "B_mean": b.mean() if n else np.nan, "B_sd": b.std(ddof=1) if n > 1 else np.nan,
           "diff_mean": diff.mean() if n else np.nan, "diff_sd": diff.std(ddof=1) if n > 1 else np.nan,
           "diff_median": np.median(diff) if n else np.nan}
    if n < 3 or np.allclose(diff, 0):
        res.update({"shapiro_p": np.nan, "differences_normal": np.nan, "ttest_p": np.nan, "wilcoxon_p": np.nan,
                    "selected_test": "n < 3 or no difference", "selected_p": np.nan})
        return res
    ci = stats.t.interval(1 - alpha, n - 1, loc=diff.mean(), scale=stats.sem(diff))
    res["diff_95CI_low"], res["diff_95CI_high"] = ci
    res["shapiro_W"], res["shapiro_p"] = stats.shapiro(diff)
    res["differences_normal"] = bool(res["shapiro_p"] >= alpha)
    t = stats.ttest_rel(b, a)
    res["ttest_t"], res["ttest_p"] = t.statistic, t.pvalue
    w = stats.wilcoxon(b, a)
    res["wilcoxon_W"], res["wilcoxon_p"] = w.statistic, w.pvalue
    res["selected_test"] = "paired t-test" if res["differences_normal"] else "Wilcoxon signed-rank test"
    res["selected_p"] = res["ttest_p"] if res["differences_normal"] else res["wilcoxon_p"]
    return res


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaN kept)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def fmt(mean, sd):
    return "" if pd.isna(mean) else (f"{mean:.2f} ± {sd:.2f}" if pd.notna(sd) else f"{mean:.2f}")


def add_readable(row, name_a, name_b):
    row[name_a] = fmt(row["A_mean"], row["A_sd"])
    row[name_b] = fmt(row["B_mean"], row["B_sd"])
    row["difference (B - A)"] = fmt(row["diff_mean"], row["diff_sd"])
    return row


def compare(df, metrics, args):
    rows_overall, rows_level, rows_contrast = [], [], []
    levels = level_order(df["level"].unique())
    contrasts = list(dict.fromkeys(df["contrast"]))
    for m in metrics:
        col_a, col_b = f"{m}_A", f"{m}_B"
        pm = pretty_metric(m)
        # Overall: mean ± SD across all case/level rows; test on the per-case means (independent samples).
        # For all levels (C2-T1) and without --overall-exclude-levels (default T1 -> C2-C8, paper overall)
        level_sets = [(f"{levels[0]}-{levels[-1]} (all)", df)]
        sub = df[~df["level"].isin(args.overall_exclude_levels or [])]
        if args.overall_exclude_levels and len(sub) < len(df) and not sub.empty:
            lv = level_order(sub["level"].unique())
            level_sets.append((f"{lv[0]}-{lv[-1]} (paper overall)", sub))
        for name, d in level_sets:
            allrows = paired_test(d[col_a], d[col_b], args.alpha)
            per_case = d.groupby("case")[[col_a, col_b]].mean()
            case_test = paired_test(per_case[col_a], per_case[col_b], args.alpha)
            row = {"metric": pm, "levels": name, "n_rows": allrows["n"], "n_cases": case_test["n"],
                   **{k: allrows[k] for k in ("A_mean", "A_sd", "B_mean", "B_sd", "diff_mean", "diff_sd")},
                   **{f"case_level_{k}": v for k, v in case_test.items() if k not in ("n",)}}
            rows_overall.append(add_readable(row, args.name_a, args.name_b))

        level_rows = []
        for lvl in levels:
            sub = df[df["level"] == lvl]
            res = paired_test(sub[col_a], sub[col_b], args.alpha)
            level_rows.append(add_readable({"metric": pm, "level": lvl, **res}, args.name_a, args.name_b))
        for r, p_adj in zip(level_rows, holm([r["selected_p"] for r in level_rows])):
            r["selected_p_holm"] = p_adj
        rows_level += level_rows

        if len(contrasts) > 1:
            for c in contrasts:
                sub = df[df["contrast"] == c]
                pc = sub.groupby("case")[[col_a, col_b]].mean()
                res = paired_test(pc[col_a], pc[col_b], args.alpha)
                rows_contrast.append(add_readable({"metric": pm, "contrast": c, "n_rows": len(sub), **res},
                                                  args.name_a, args.name_b))
    return pd.DataFrame(rows_overall), pd.DataFrame(rows_level), pd.DataFrame(rows_contrast), levels


def p_str(p):
    return "" if pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def print_report(overall, per_level, args):
    print(f"\n=== {args.name_b} (B) vs {args.name_a} (A), difference = B - A ===")
    for metric in dict.fromkeys(overall["metric"]):
        print(f"\n{metric}")
        for _, r in overall[overall["metric"] == metric].iterrows():
            print(f"  {r['levels']}: {args.name_a} {r[args.name_a]}, {args.name_b} {r[args.name_b]}, "
                  f"difference {r['difference (B - A)']} (n = {r['n_rows']} case/level rows, {r['n_cases']} cases)")
            print(f"    case-level: Shapiro-Wilk p = {p_str(r['case_level_shapiro_p'])} -> "
                  f"{r['case_level_selected_test']} p = {p_str(r['case_level_selected_p'])}")
        sub = per_level[per_level["metric"] == metric]
        print(sub[["level", "n", args.name_a, args.name_b, "difference (B - A)", "selected_test"]]
              .assign(p=sub["selected_p"].map(p_str), p_holm=sub["selected_p_holm"].map(p_str))
              .to_string(index=False))


def main():
    args = resolve_inputs(get_parser().parse_args())
    print(f"A = {args.name_a}: {args.metrics_a}\nB = {args.name_b}: {args.metrics_b}")
    os.makedirs(args.output, exist_ok=True)
    df, metrics = load_pair(args.metrics_a, args.metrics_b, args.name_a, args.name_b)
    print(f"Paired {len(df)} case/level rows ({df['case'].nunique()} cases), metrics: {metrics}")
    df.to_csv(os.path.join(args.output, "paired_metrics.csv"), index=False)

    overall, per_level, per_contrast, levels = compare(df, metrics, args)
    overall.to_csv(os.path.join(args.output, "comparison_overall.csv"), index=False)
    per_level.to_csv(os.path.join(args.output, "comparison_per_level.csv"), index=False)
    if not per_contrast.empty:
        per_contrast.to_csv(os.path.join(args.output, "comparison_per_contrast.csv"), index=False)
    print_report(overall, per_level, args)
    print(f"\nTables saved to {args.output}/comparison_*.csv")

    if not args.no_plot:
        plot_comparison(df, metrics, args)


def plot_comparison(df, metrics, args):
    """Box plots of both models per level in the style of dice-hd95-graph.py: figures/dice_ALL.png,
    figures/hausdorff_ALL.png (all contrasts pooled), + per contrast if several."""
    from rootlet_graphs import plot_models
    keep = ["case", "level", "contrast"]
    long = pd.concat([df[keep + [f"{m}_A" for m in metrics]].rename(columns={f"{m}_A": m for m in metrics})
                        .assign(model=args.name_a),
                      df[keep + [f"{m}_B" for m in metrics]].rename(columns={f"{m}_B": m for m in metrics})
                        .assign(model=args.name_b)], ignore_index=True)
    fig_dir = os.path.join(args.output, "figures")
    order = [args.name_a, args.name_b]
    plot_models(long, metrics, "model", order, fig_dir, "ALL")
    contrasts = list(dict.fromkeys(long["contrast"]))
    if len(contrasts) > 1:
        for c in contrasts:
            plot_models(long[long["contrast"] == c], metrics, "model", order, fig_dir, c)


if __name__ == "__main__":
    main()
