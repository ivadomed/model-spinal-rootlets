#!/usr/bin/env python3
"""
Compute MetricsReloaded metrics for nnUNetv2 predictions and report them as in
RootletSeg (Sci Rep 2026, https://www.nature.com/articles/s41598-026-49164-0):

  * Same evaluation as the RootletSeg paper (defaults): Dice and HD95 (-metrics dsc hd_perc)
    computed with MetricsReloaded for all levels, levels empty in both reference and
    prediction kept (as get_statistics_dsc.py / dice-hd95-graph.py). Post-processing: level
    C1 is then removed from all statistics, tables and figures, T1 is kept (-exclude-levels;
    all levels kept in metrics_per_case_level_all_levels.csv).
    Mean ± SD overall = across levels, participants and contrasts, given for all levels
    (C2-T1) and, as in the paper, for C2-C8 (-overall-exclude-levels T1); per contrast =
    across levels C2-T1 and test scans; per level = across participants.
  * Dice and HD95 reported as mean ± SD
      - overall (across levels, participants and contrasts)
      - per contrast (across levels and test scans)
      - per level (across participants), overall and per contrast
  * Box plots in the style of dice-hd95-graph.py (rootlet_graphs.py): spinal level on the
    x axis, dice*.png and hausdorff*.png (600 dpi)
      - per dataset/fold: figures/dice_hausdorff_contrasts.png, a) Dice and b) Hausdorff,
        one box per contrast and spinal level (RootletSeg paper figure)
      - with -root: all datasets side by side, all contrasts pooled (dice_ALL.png) and
        per contrast (dice_<contrast>.png), legend names from -names
  * No significance tests (as in the paper)

Pairs each label in labelsTs with the prediction of the same name in every
predict_fold_XX folder and runs compute_metrics_reloaded.py per case.

Expected structure (nnUNetv2 naming, e.g. sub-001_T2w.nii.gz in both folders), either
  a) one model:  -labels <labels_dir> -pred-root <pred_root>
    <labels_dir>/                CASE.nii.gz
    <pred_root>/predict_fold_0/  CASE.nii.gz   (or point -pred-root at one fold)
  b) several models sharing one ground truth:  -root <ROOT>
    <ROOT>/groundtruthsTs/                  CASE.nii.gz   (-gt-dirname)
    <ROOT>/DatasetXX/predict_fold_XX/       CASE.nii.gz
    <ROOT>/DatasetXY/predict_fold_XX/       CASE.nii.gz
    -> <OUTPUT>/DatasetXX/predict_fold_XX/... for each dataset and fold, plus
       <OUTPUT>/metrics_all.csv and <OUTPUT>/summary_overall.csv (all datasets/folds)
       <OUTPUT>/figures/dice_ALL.png, hausdorff_ALL.png : all datasets side by side
       (+ dice_<contrast>.png, ... if several contrasts)

Contrast is parsed from the file name (T2w, UNIT1, INV1, INV2 by default; see
-contrasts) or read from a CSV with columns  case,contrast  (-contrast-csv).

Examples:
    python3 sc_crop_rootlets_stats.py -root ~/results_rootlets          # -> ~/results_rootlets/metrics
    python3 sc_crop_rootlets_stats.py -root ~/results_rootlets -datasets DatasetXX DatasetXY -folds all
    # per dataset: original model vs padded predictions of the sc-crop model
    python3 sc_crop_rootlets_stats.py -root ~/results_rootlets -folds DatasetXX:all DatasetXY:all_paded
    python3 sc_crop_rootlets_stats.py \
        -labels $nnUNet_raw/Dataset052_.../labelsTs \
        -pred-root $nnUNet_results/.../predict_fold_all \
        -output .../metrics
"""

import argparse
import glob
import os
import re
import shlex

import numpy as np
import pandas as pd

METRICS_SCRIPT = os.path.expanduser(
    "~/PycharmProjects/MetricsReloaded/compute_metrics_reloaded.py"
)

# Columns in the MetricsReloaded CSV that are not metrics
NON_METRIC_COLS = {"fold", "reference", "prediction", "label", "case", "contrast",
                   "level", "emptyref", "emptypred"}

# Default contrast patterns (regex, case-insensitive) -> display name, in plot order
DEFAULT_CONTRASTS = [
    ("T2w", r"t2w"),
    ("T1w-INV1", r"inv-?1"),
    ("T1w-INV2", r"inv-?2"),
    ("UNIT1", r"unit1"),
]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-root", "--root", default=None,
                   help="Folder with the ground truth folder (-gt-dirname) and one "
                        "Dataset* folder per model, each with predict_fold_XX folders. "
                        "Replaces -labels/-pred-root.")
    p.add_argument("-gt-dirname", "--gt-dirname", default="groundtruthsTs",
                   help="With -root: name of the ground truth folder (default: groundtruthsTs).")
    p.add_argument("-datasets", "--datasets", nargs="+", default=None,
                   help="With -root: dataset folders to evaluate (default: all Dataset* folders).")
    p.add_argument("-dataset-pattern", "--dataset-pattern", default="Dataset*",
                   help="With -root: glob pattern of the dataset folders (default: Dataset*).")
    p.add_argument("-names", "--names", nargs="+", default=None, metavar="DATASET[:FOLD]=LABEL",
                   help="With -root: short legend names, e.g. Dataset037_...=RootletSeg "
                        "Dataset052_...:all_paded='RootletSeg + sc-crop' (cf. DATASET_RENAMES of "
                        "dice-hd95-graph.py). Also sets the order of the models in the figures.")
    p.add_argument("-labels", "--labels", default=None,
                   help="Path to labelsTs folder (ground truth). Not needed with -root.")
    p.add_argument("-pred-root", "--pred-root", default=None,
                   help="predict_fold_XX folder, or a folder containing several. Not needed with -root.")
    p.add_argument("-folds", "--folds", nargs="+", default=None,
                   help="Folds to evaluate (predict_fold_<fold>), e.g. 0 1 2 or all. With -root, "
                        "DATASET:FOLD selects the fold(s) of one dataset, e.g. DatasetXX:all "
                        "DatasetXY:all_paded (plain values apply to the datasets not named). "
                        "Default: all found.")
    p.add_argument("-metrics", "--metrics", nargs="+", default=["dsc", "hd"],
                   help="Metrics passed to MetricsReloaded (default: dsc hd = Dice and "
                        "Hausdorff distance, as in the RootletSeg paper; "
                        "dsc hd for the maximum Hausdorff distance).")
    p.add_argument("-output", "--output", default=None,
                   help="Output folder for CSVs and figures (with -root, default: <ROOT>/metrics).")
    p.add_argument("-ext", "--ext", default=".nii.gz",
                   help="File extension (default: .nii.gz).")
    p.add_argument("-contrasts", "--contrasts", nargs="+", default=None, metavar="NAME=REGEX",
                   help="Contrast patterns matched in the file name, in plot order, "
                        "e.g. T2w=t2w UNIT1=unit1 T1w-INV1=inv-?1 T1w-INV2=inv-?2 "
                        "(this is the default).")
    p.add_argument("-default-contrast", "--default-contrast", default="T2w",
                   help="Contrast for cases whose file name matches no pattern "
                        "(default: T2w).")
    p.add_argument("-contrast-csv", "--contrast-csv", default=None,
                   help="Optional CSV with columns 'case,contrast' (overrides -contrasts).")
    p.add_argument("-exclude-levels", "--exclude-levels", nargs="*", default=["C1"],
                   help="Levels removed after the metric computation, before all statistics, "
                        "tables and figures (default: C1). All levels stay in "
                        "metrics_per_case_level_all_levels.csv. Pass '-exclude-levels' with no "
                        "value to keep all.")
    p.add_argument("-overall-exclude-levels", "--overall-exclude-levels", nargs="*", default=["T1"],
                   help="Levels left out of the extra 'paper overall' row of stats_overall.csv and "
                        "summary_overall.csv (default: T1 -> C2-C8, as the overall values of the "
                        "RootletSeg paper). Per-contrast and per-level tables keep all levels.")
    p.add_argument("-drop-empty", "--drop-empty", action="store_true",
                   help="Leave out rows where a level is empty in BOTH reference and "
                        "prediction (kept by default, as in get_statistics_dsc.py).")
    p.add_argument("-keep-empty", "--keep-empty", action="store_true",
                   help=argparse.SUPPRESS)   # old flag, now the default
    p.add_argument("-overwrite", "--overwrite", action="store_true",
                   help="Recompute cases whose CSV already exists.")
    p.add_argument("-no-plot", "--no-plot", action="store_true", help="Skip the figures.")
    return p


# ----------------------------------------------------------------------------
# Running MetricsReloaded
# ----------------------------------------------------------------------------
def folds_for(dataset, folds):
    """Folds of one dataset from -folds: DATASET:FOLD entries for this dataset if any, else the plain entries."""
    if not folds:
        return None
    own = [f.split(":", 1)[1] for f in folds if ":" in f and f.split(":", 1)[0] == dataset]
    plain = [f for f in folds if ":" not in f]
    named = {f.split(":", 1)[0] for f in folds if ":" in f}
    if own:
        return own
    if plain:
        return plain
    return [] if named else None   # only DATASET:FOLD entries, none for this dataset -> skip it


def normalize_folds(folds):
    """-folds entries given as paths of predict_fold_* folders -> DATASET:FOLD (dataset = parent folder name)."""
    if not folds:
        return folds
    out = []
    for f in folds:
        name = os.path.basename(os.path.normpath(f))
        if (os.sep in f or os.path.isdir(f)) and name.startswith("predict_fold_"):
            dataset = os.path.basename(os.path.dirname(os.path.normpath(os.path.expanduser(f))))
            out.append(f"{dataset}:{name[len('predict_fold_'):]}")
        else:
            out.append(f)
    return out


def find_fold_dirs(pred_root, folds, required=True):
    pred_root = os.path.normpath(pred_root)
    if os.path.basename(pred_root).startswith("predict_fold_"):
        return [pred_root]
    dirs = sorted(d for d in glob.glob(os.path.join(pred_root, "predict_fold_*"))
                  if os.path.isdir(d))
    if folds is not None:
        wanted = {f"predict_fold_{f}" for f in folds}
        dirs = [d for d in dirs if os.path.basename(d) in wanted]
    if not dirs and required:
        raise SystemExit(f"No predict_fold_* folders found in {pred_root}")
    return dirs


def case_id(path, ext):
    return os.path.basename(path)[: -len(ext)]


def run_fold(fold_dir, labels, args, out_root):
    fold_name = os.path.basename(fold_dir)
    case_out_dir = os.path.join(out_root, fold_name, "per_case")
    os.makedirs(case_out_dir, exist_ok=True)
    metrics_str = " ".join(args.metrics)
    # Per-case CSVs are reused only if they were computed with the same metrics
    stamp = os.path.join(case_out_dir, "_metrics.txt")
    previous = open(stamp).read().strip() if os.path.isfile(stamp) else None
    recompute = args.overwrite or previous != metrics_str
    if recompute and not args.overwrite and glob.glob(os.path.join(case_out_dir, "*.csv")):
        print(f"  metrics changed ({previous or 'unknown'} -> {metrics_str}): recomputing all cases")

    print(f"\n=== {fold_name} ===")

    tables = []
    for reference in labels:
        cid = case_id(reference, args.ext)
        pred = os.path.join(fold_dir, f"{cid}{args.ext}")
        if not os.path.isfile(pred):
            print(f"  [skip] no prediction for {cid}")
            continue

        output_file = os.path.join(case_out_dir, f"{cid}.csv")
        if os.path.isfile(output_file) and not recompute:
            print(f"  [exists] {cid}")
        else:
            print(f"  [run] {cid}")
            ret = os.system(
                f"python3 {shlex.quote(METRICS_SCRIPT)} "
                f"-reference {shlex.quote(reference)} "
                f"-prediction {shlex.quote(pred)} "
                f"-metrics {metrics_str} "
                f"-output {shlex.quote(output_file)}"
            )
            if ret != 0 or not os.path.isfile(output_file):
                print(f"  [error] MetricsReloaded failed for {cid}")
                continue
        df = pd.read_csv(output_file)
        df.insert(0, "case", cid)
        tables.append(df)

    with open(stamp, "w") as f:   # all cases now computed with these metrics
        f.write(metrics_str + "\n")
    if not tables:
        return None
    df_fold = pd.concat(tables, ignore_index=True)
    df_fold.insert(0, "fold", re.sub(r"^predict_fold_", "", fold_name))
    print(f"  {len(tables)} cases")
    return df_fold


# ----------------------------------------------------------------------------
# Annotation: contrast, level, metric names
# ----------------------------------------------------------------------------
def parse_contrasts(arg):
    if not arg:
        return DEFAULT_CONTRASTS
    out = []
    for item in arg:
        name, _, regex = item.partition("=")
        out.append((name, regex or re.escape(name)))
    return out


def assign_contrast(df, args):
    if args.contrast_csv:
        m = pd.read_csv(args.contrast_csv, dtype=str)
        mapping = dict(zip(m["case"], m["contrast"]))
        df["contrast"] = df["case"].map(mapping).fillna(args.default_contrast)
        order = list(dict.fromkeys(m["contrast"]))
    else:
        patterns = parse_contrasts(args.contrasts)
        order = [name for name, _ in patterns]

        def find(cid):
            for name, regex in patterns:
                if re.search(regex, cid, flags=re.IGNORECASE):
                    return name
            return None
        found = df["case"].apply(find)
        unk = sorted(df.loc[found.isna(), "case"].unique())
        if unk:
            print(f"Contrast not in file name for {len(unk)} case(s) -> "
                  f"'{args.default_contrast}': {unk}")
        df["contrast"] = found.fillna(args.default_contrast)
    if args.default_contrast not in order:
        order.append(args.default_contrast)
    return [c for c in order if c in set(df["contrast"])]


def level_name(label):
    """nnUNet rootlet label -> spinal level (2 -> C2, ..., 8 -> C8, 9 -> T1, ...)."""
    try:
        n = int(float(label))
    except (TypeError, ValueError):
        return str(label)
    if 1 <= n <= 8:
        return f"C{n}"
    if 9 <= n <= 20:
        return f"T{n - 8}"
    if 21 <= n <= 25:
        return f"L{n - 20}"
    return str(n)


def pretty_metric(col):
    c = col.lower()
    if "dice" in c or c == "dsc":
        return "Dice"
    if "hausdorff" in c or c.startswith("hd"):
        return "HD95 [mm]" if ("95" in c or "perc" in c) else "HD [mm]"
    if "normalizedsurface" in c or c == "nsd":
        return "NSD"
    if "relativevolume" in c:
        return "Relative volume error [%]"
    return col


def get_metric_cols(df):
    cols = []
    for c in df.columns:
        if c.lower() in NON_METRIC_COLS:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any():
            cols.append(c)
    return cols


def drop_both_empty(df):
    if {"EmptyRef", "EmptyPred"} <= set(df.columns):
        as_bool = lambda s: s.astype(str).str.lower().isin(["true", "1"])
        both = as_bool(df["EmptyRef"]) & as_bool(df["EmptyPred"])
        if both.any():
            print(f"Excluded {both.sum()} rows where the level is empty in both "
                  f"reference and prediction (-drop-empty).")
        return df[~both].copy()
    return df


# ----------------------------------------------------------------------------
# Statistics (mean ± SD, as in the paper)
# ----------------------------------------------------------------------------
def mean_sd_table(df, group_cols, metric_cols):
    g = df.groupby(group_cols, sort=False) if group_cols else None
    rows = []
    groups = g if g is not None else [((), df)]
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_scans"] = sub["case"].nunique()
        for m in metric_cols:
            v = sub[m].dropna()
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[pretty_metric(m)] = f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" \
                if len(v) > 1 else (f"{v.mean():.2f}" if len(v) else "")
        rows.append(row)
    return pd.DataFrame(rows)


def level_range(levels):
    """'C2-T1' style name of a set of levels."""
    lv = sorted(set(levels), key=lambda s: ("CTLS".find(s[0]) if s[:1] in "CTL" else 9,
                                            int(s[1:]) if s[1:].isdigit() else 0))
    return f"{lv[0]}-{lv[-1]}" if len(lv) > 1 else (lv[0] if lv else "none")


def overall_tables(df, group_cols, metric_cols, overall_exclude):
    """Overall mean ± SD for all levels and, as in the RootletSeg paper, without the levels of
    -overall-exclude-levels (default T1 -> C2-C8). Column 'levels' tells which."""
    rows = [mean_sd_table(df, group_cols, metric_cols).assign(levels=f"{level_range(df['level'])} (all)")]
    sub = df[~df["level"].isin(overall_exclude or [])]
    if overall_exclude and len(sub) < len(df) and not sub.empty:
        rows.append(mean_sd_table(sub, group_cols, metric_cols)
                    .assign(levels=f"{level_range(sub['level'])} (paper overall)"))
    out = pd.concat(rows, ignore_index=True)
    return out[group_cols + ["levels"] + [c for c in out.columns if c not in group_cols + ["levels"]]]


def report(df, metric_cols, contrasts, levels, out_dir, overall_exclude=("T1",)):
    pretty = [pretty_metric(m) for m in metric_cols]

    overall = overall_tables(df, [], metric_cols, overall_exclude)
    per_contrast = mean_sd_table(df, ["contrast"], metric_cols)
    per_contrast["contrast"] = pd.Categorical(per_contrast["contrast"], contrasts, ordered=True)
    per_contrast = per_contrast.sort_values("contrast")
    per_level = mean_sd_table(df, ["level"], metric_cols)
    per_level["level"] = pd.Categorical(per_level["level"], levels, ordered=True)
    per_level = per_level.sort_values("level")
    per_level_contrast = mean_sd_table(df, ["contrast", "level"], metric_cols)

    overall.to_csv(os.path.join(out_dir, "stats_overall.csv"), index=False)
    per_contrast.to_csv(os.path.join(out_dir, "stats_per_contrast.csv"), index=False)
    per_level.to_csv(os.path.join(out_dir, "stats_per_level.csv"), index=False)
    per_level_contrast.to_csv(os.path.join(out_dir, "stats_per_level_contrast.csv"),
                              index=False)

    # Level x contrast "Dice mean ± SD" grid (the numbers shown in Fig. 3)
    for m, pm in zip(metric_cols, pretty):
        grid = per_level_contrast.pivot(index="level", columns="contrast", values=pm)
        grid = grid.reindex(index=[l for l in levels if l in grid.index],
                            columns=[c for c in contrasts if c in grid.columns])
        safe = re.sub(r"[^A-Za-z0-9]+", "_", pm).strip("_")
        grid.to_csv(os.path.join(out_dir, f"stats_level_x_contrast_{safe}.csv"))

    print("\n=== Overall (mean ± SD across levels, participants, contrasts) ===")
    print(overall[["levels", "n_scans"] + pretty].to_string(index=False))
    print("\n=== Per contrast (mean ± SD across levels and test scans) ===")
    print(per_contrast[["contrast", "n_scans"] + pretty].to_string(index=False))
    print("\n=== Per level (mean ± SD across participants) ===")
    print(per_level[["level", "n_scans"] + pretty].to_string(index=False))
    print(f"\nTables saved to {out_dir}/stats_*.csv")


# ----------------------------------------------------------------------------
# Figures (style of dice-hd95-graph.py, see rootlet_graphs.py)
# ----------------------------------------------------------------------------
def parse_names(items):
    """-names DATASET=LABEL or DATASET:FOLD=LABEL -> dict (keys 'DATASET' or 'DATASET:FOLD')."""
    out = {}
    for item in items or []:
        key, sep, label = item.partition("=")
        if not sep:
            raise SystemExit(f"-names expects DATASET=LABEL or DATASET:FOLD=LABEL, got {item!r}")
        out[key] = label
    return out


def model_labels(df, names):
    """Legend label of each dataset/fold: -names if given (DATASET:FOLD, then DATASET), else the dataset
    name, + the fold when the dataset has several folds or the fold is not 'all'."""
    n_folds = df.groupby("dataset")["fold"].nunique()

    def label(r):
        ds, fold = r["dataset"], str(r["fold"])
        if f"{ds}:{fold}" in names:
            return names[f"{ds}:{fold}"]
        if ds in names and n_folds[ds] == 1:
            return names[ds]
        base = names.get(ds, ds)
        return base if n_folds[ds] == 1 and fold == "all" else f"{base} (fold {fold})"
    return df.apply(label, axis=1)


def plot_all_datasets(df_all, metric_cols, output, names, order_keys):
    """All datasets/folds side by side (style of dice-hd95-graph.py): figures/dice_ALL.png,
    figures/hausdorff_ALL.png (all contrasts pooled) and dice_<contrast>.png, ... per contrast."""
    from rootlet_graphs import plot_models
    df = df_all.copy()
    df["model"] = model_labels(df, names)
    # model order: order of -folds / -datasets / -names if given, else as found
    key = df["dataset"] + ":" + df["fold"].astype(str)
    rank = {k: i for i, k in enumerate(order_keys)}
    df["_rank"] = [rank.get(k, rank.get(k.split(":")[0], len(rank))) for k in key]
    order = list(dict.fromkeys(df.sort_values("_rank", kind="stable")["model"]))
    fig_dir = os.path.join(output, "figures")
    plot_models(df, metric_cols, "model", order, fig_dir, "ALL")
    contrasts = list(dict.fromkeys(df["contrast"]))
    if len(contrasts) > 1:
        for c in contrasts:
            plot_models(df[df["contrast"] == c], metric_cols, "model", order, fig_dir, c)


# ----------------------------------------------------------------------------
def find_jobs(args):
    """(dataset name, labels folder, prediction root, output folder) for each model to evaluate."""
    if args.root:
        root = os.path.normpath(os.path.expanduser(args.root))
        labels_dir = args.labels or os.path.join(root, args.gt_dirname)
        if not os.path.isdir(labels_dir):
            raise SystemExit(f"Ground truth folder {labels_dir} not found (see -gt-dirname)")
        if args.datasets:
            ds_dirs = [os.path.join(root, d) for d in args.datasets]
            missing = [d for d in ds_dirs if not os.path.isdir(d)]
            if missing:
                raise SystemExit(f"Dataset folder(s) not found: {missing}")
        else:
            ds_dirs = sorted(d for d in glob.glob(os.path.join(root, args.dataset_pattern))
                             if os.path.isdir(d))
        if not ds_dirs:
            raise SystemExit(f"No {args.dataset_pattern} folders in {root}")
        output = args.output or os.path.join(root, "metrics")
        return [(os.path.basename(d), labels_dir, d, os.path.join(output, os.path.basename(d)))
                for d in ds_dirs], output
    if not (args.labels and args.pred_root and args.output):
        raise SystemExit("Give -root, or -labels, -pred-root and -output.")
    return [(None, args.labels, args.pred_root, args.output)], args.output


def evaluate_fold(fold_dir, labels, args, out_root):
    """Metrics, tables and figures of one predict_fold_XX folder. Return the per case/level table."""
    df = run_fold(fold_dir, labels, args, out_root)
    if df is None:
        print("  No results.")
        return None
    fold_tag = os.path.basename(fold_dir)
    out_dir = os.path.join(out_root, fold_tag)

    contrasts = assign_contrast(df, args)
    df["level"] = df["label"].apply(level_name) if "label" in df.columns else "all"
    if args.drop_empty:
        df = drop_both_empty(df)
    # Post-processing: metrics are computed for all levels (saved with C1), then the excluded
    # levels (default C1) are removed before all statistics, tables and figures
    df.to_csv(os.path.join(out_dir, "metrics_per_case_level_all_levels.csv"), index=False)
    if args.exclude_levels:
        excl = df["level"].isin(args.exclude_levels)
        if excl.any():
            print(f"Removed {excl.sum()} rows of level(s) {args.exclude_levels} from the statistics "
                  f"(all levels kept in metrics_per_case_level_all_levels.csv)")
        df = df[~excl].copy()
    levels = sorted(df["level"].unique(),
                    key=lambda s: ("CTLS".find(s[0]) if s[:1] in "CTL" else 9,
                                   int(s[1:]) if s[1:].isdigit() else 0))
    df.to_csv(os.path.join(out_dir, "metrics_per_case_level.csv"), index=False)

    metric_cols = get_metric_cols(df)
    if not metric_cols:
        print(f"[warning] no numeric metric columns. Columns: {list(df.columns)}")
        return None
    print(f"\nMetrics: {metric_cols} | contrasts: {contrasts} | levels: {levels}")

    report(df, metric_cols, contrasts, levels, out_dir, args.overall_exclude_levels)
    if not args.no_plot:
        from rootlet_graphs import plot_contrasts_panel
        # one model across contrasts: a) Dice, b) Hausdorff, one box per contrast and level
        plot_contrasts_panel(df, metric_cols, os.path.join(out_dir, "figures", "dice_hausdorff_contrasts.png"))
    return df


def main():
    args = get_parser().parse_args()
    args.folds = normalize_folds(args.folds)
    if args.folds:
        print(f"Folds: {args.folds}")
    jobs, output = find_jobs(args)
    os.makedirs(output, exist_ok=True)

    all_tables = []
    for dataset, labels_dir, pred_root, out_root in jobs:
        labels = sorted(glob.glob(os.path.join(labels_dir, f"*{args.ext}")))
        if not labels:
            raise SystemExit(f"No *{args.ext} files found in {labels_dir}")
        if dataset:
            print(f"\n##### {dataset} #####")
        print(f"Found {len(labels)} reference labels in {labels_dir}")
        fold_dirs = find_fold_dirs(pred_root, folds_for(dataset, args.folds), required=dataset is None)
        if not fold_dirs:
            print(f"[warning] no predict_fold_* folders (or none of -folds) in {pred_root}, skipped")
            continue
        os.makedirs(out_root, exist_ok=True)
        for fold_dir in fold_dirs:
            df = evaluate_fold(fold_dir, labels, args, out_root)
            if df is not None and dataset:
                df.insert(0, "dataset", dataset)
                all_tables.append(df)

    # With -root: all datasets and folds in one table + one summary row per dataset/fold
    if all_tables:
        df_all = pd.concat(all_tables, ignore_index=True)
        df_all.to_csv(os.path.join(output, "metrics_all.csv"), index=False)
        metric_cols = get_metric_cols(df_all.drop(columns=["dataset"]))
        summary = overall_tables(df_all, ["dataset", "fold"], metric_cols, args.overall_exclude_levels)
        summary = summary.sort_values(["dataset", "fold"], kind="stable")   # all levels first, then C2-C8
        summary.to_csv(os.path.join(output, "summary_overall.csv"), index=False)
        print("\n=== All datasets (mean ± SD across levels, participants, contrasts) ===")
        print(summary[["dataset", "fold", "levels", "n_scans"] + [pretty_metric(m) for m in metric_cols]]
              .to_string(index=False))
        print(f"\nSaved {output}/metrics_all.csv and {output}/summary_overall.csv")
        if not args.no_plot:
            order_keys = list(args.folds or []) + list(args.datasets or []) + list(parse_names(args.names))
            plot_all_datasets(df_all, metric_cols, output, parse_names(args.names), order_keys)


if __name__ == "__main__":
    main()