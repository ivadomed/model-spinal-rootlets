"""
This script computes a lumbar spinal cord length from spinal level masks saved one level per file
(sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz, e.g. desc-L1 ... desc-S2), and plots:
    - the length per subject
    - the correlation between the length and subject height

Two metrics are available (-metric); both count slices along the superior-inferior (S-I) axis and multiply by the
slice thickness (S-I voxel size), i.e., the same logic as 03_compute_lumbar_midpoints_distance.py and
04_compute_lumbar_length_topL1_bottom_L5.py, assuming that the spinal cord is approximately straight:
    - top-bottom (default): length from the top of L1 to the bottom of L5 = (most superior slice of L1 - most inferior
      slice of L5 + 1) x slice thickness, i.e., both end slices included (as in 04)
    - midpoints: distance between the L1 and S1 midpoints = (center of mass slice of L1 - center of mass slice of S1)
      x slice thickness (as in 03)

Input:
- folder with the spinal level masks (one level per file)
- participants.tsv with the subject sex and height (default: participants.tsv one level above the input folder)

Output (saved to the current working directory; <metric> is e.g. length_L1-L5 or midpoints_distance_L1-S1):
    - lumbar_<metric>_masks.csv: length per subject, with sex and height
    - figure_lumbar_<metric>_masks.png: length per subject, colored by sex
    - figure_correlation_lumbar_<metric>_masks_height.png: length vs. height with a linear fit (visual guide), its 95%
      confidence band, and the Spearman correlation coefficient with its 95% CI

The script requires the SCT conda environment to be activated:
    source ${SCT_DIR}/python/etc/profile.d/conda.sh
    conda activate venv_sct

Example:
    python 07_lumbar_midpoints_distance_masks.py -i ~/data/balgrist/lumbar_mri_Sergio/spinalcord_masks
    python 07_lumbar_midpoints_distance_masks.py -i ~/data/balgrist/lumbar_mri_Sergio/spinalcord_masks -metric midpoints

Author: Jan Valosek
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats

from spinalcordtoolbox.image import Image

# Metrics: first and last spinal level, the point used on each level, and labels for the outputs and figures
METRICS = {
    # Both end slices included in the length (as in 04_compute_lumbar_length_topL1_bottom_L5.py)
    'top-bottom': {'first': 'L1', 'last': 'L5', 'point_first': 'top', 'point_last': 'bottom', 'add_slice': True,
                   'name': 'length_L1-L5', 'label': 'Length top L1 – bottom L5'},
    # Distance between the centers of mass (as in 03_compute_lumbar_midpoints_distance.py)
    'midpoints': {'first': 'L1', 'last': 'S1', 'point_first': 'center', 'point_last': 'center', 'add_slice': False,
                  'name': 'midpoints_distance_L1-S1', 'label': 'Distance L1–S1 midpoints'},
}

FONT_SIZE = 14

# Female and male colors and markers
SEX_STYLE = {'F': {'color': '#2a78d6', 'marker': 'o', 'label': 'Female'},
             'M': {'color': '#eb6834', 'marker': 's', 'label': 'Male'}}


def get_slice(fname, point):
    """
    Get a slice position of a binary mask along the superior-inferior (S-I) axis.
    :param fname: path to the mask
    :param point: 'top': most superior slice; 'bottom': most inferior slice; 'center': center of mass along the S-I axis
    :return: slice position (in slices) and slice thickness (S-I voxel size, in mm)
    """
    # Reorient to RPI so that the third axis is the S-I axis (slice 0 is the most inferior)
    img = Image(fname).change_orientation('RPI')
    z = np.where(img.data > 0)[2]   # S-I index of each voxel of the mask
    position = {'top': z.max(), 'bottom': z.min(), 'center': z.mean()}[point]
    return position, img.dim[6]


def compute_length(dir_path, metric):
    """
    Compute the length for each subject: number of slices between the two points x slice thickness.
    :param dir_path: folder with the spinal level masks (one level per file)
    :param metric: dict from METRICS
    :return: dataframe with 'subject' and '<metric name>_mm' columns
    """
    first, last = metric['first'], metric['last']
    rows = []
    for fname_first in sorted(glob.glob(os.path.join(dir_path, '**', f'*_desc-{first}_mask.nii.gz'), recursive=True)):
        fname_last = fname_first.replace(f'_desc-{first}_mask', f'_desc-{last}_mask')
        subject = os.path.basename(fname_first).split('_')[0]
        if not os.path.exists(fname_last):
            print(f'WARNING: {subject}: {last} mask not found; skipping.')
            continue
        slice_first, slice_thickness = get_slice(fname_first, metric['point_first'])
        slice_last, _ = get_slice(fname_last, metric['point_last'])
        n_slices = slice_first - slice_last + (1 if metric['add_slice'] else 0)
        length = n_slices * slice_thickness
        print(f"{subject}: {metric['label']} = {n_slices:.1f} slices x {slice_thickness:.2f} mm = {length:.1f} mm")
        rows.append({'subject': subject, f"{metric['name']}_mm": length})
    if not rows:
        raise FileNotFoundError(f'No *_desc-{first}_mask.nii.gz files found in {dir_path}. Check the -i path.')
    return pd.DataFrame(rows)


def spearman_ci(rho, n, alpha=0.05):
    """
    Confidence interval of the Spearman correlation coefficient using the Fisher z-transformation with the
    Bonett & Wright (2000) standard error, sqrt(1.06 / (n - 3)).
    """
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def linear_fit_ci_band(x, y, x_fit, alpha=0.05):
    """
    Linear fit and the confidence band of the fitted mean (ordinary least squares).
    :return: y_fit, lower bound, upper bound evaluated at x_fit
    """
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x_fit + intercept
    residuals = y - (slope * x + intercept)
    s = np.sqrt(np.sum(residuals ** 2) / (n - 2))   # residual standard error
    se_fit = s * np.sqrt(1 / n + (x_fit - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t_crit = stats.t.ppf(1 - alpha / 2, n - 2)
    return y_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit


def style_axes(ax):
    """Common axes style."""
    ax.tick_params(axis='both', labelsize=FONT_SIZE - 2)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Legend above the plot so it does not overlap the points
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1), ncol=3, frameon=False, fontsize=FONT_SIZE - 5,
              handletextpad=0.3, columnspacing=1)


def generate_length_figure(df, metric):
    """
    Figure with subjects on the x-axis and the length on the y-axis, colored by sex.
    """
    x = np.arange(1, len(df) + 1)
    y = df[f"{metric['name']}_mm"].to_numpy() / 10  # convert mm to cm for plotting

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.6), 5))
    for sex, style in SEX_STYLE.items():
        mask = (df['sex'] == sex).to_numpy()
        ax.scatter(x[mask], y[mask], color=style['color'], marker=style['marker'], s=50, zorder=3,
                   label=f"{style['label']} (n = {mask.sum()})")

    # Subject names as x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(df['subject'], rotation=45, ha='right')
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(0, 12)
    ax.set_ylabel(f"{metric['label']} [cm]", fontsize=FONT_SIZE)
    ax.grid(axis='y', alpha=0.2)
    style_axes(ax)
    plt.tight_layout()

    fname_figure = os.path.join(os.getcwd(), f"figure_lumbar_{metric['name']}_masks.png")
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')


def generate_correlation_figure(df, metric):
    """
    Scatter plot of the length vs. subject height, colored by sex, with a linear fit (visual guide) and its 95%
    confidence band, and the Spearman correlation coefficient with its 95% CI.
    """
    height = df['height'].astype(float)
    distance = df[f"{metric['name']}_mm"] / 10  # convert mm to cm

    # Normality test (Shapiro-Wilk); p < 0.05 means the data are not normally distributed
    print(f'Shapiro-Wilk normality test: height p = {stats.shapiro(height).pvalue:.3f}, '
          f'distance p = {stats.shapiro(distance).pvalue:.3f}')

    # Spearman correlation (rank-based, does not assume normality and is robust to outliers)
    rho, p = stats.spearmanr(height, distance)
    rho_lo, rho_hi = spearman_ci(rho, len(df))
    p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    print(f'n = {len(df)}, Spearman rho = {rho:.2f} [95% CI {rho_lo:.2f}, {rho_hi:.2f}], {p_str}')

    fig, ax = plt.subplots(figsize=(6, 5))
    for sex, style in SEX_STYLE.items():
        mask = df['sex'] == sex
        ax.scatter(height[mask], distance[mask], color=style['color'], marker=style['marker'], s=50, zorder=3,
                   edgecolors='white', linewidths=0.5, label=f"{style['label']} (n = {mask.sum()})")

    # Linear fit across all subjects (only as a visual guide) with its 95% confidence band
    x_fit = np.linspace(height.min(), height.max(), 100)
    y_fit, y_lo, y_hi = linear_fit_ci_band(height.to_numpy(), distance.to_numpy(), x_fit)
    ax.fill_between(x_fit, y_lo, y_hi, color='gray', alpha=0.2, linewidth=0, zorder=1, label='Linear fit, 95% CI')
    ax.plot(x_fit, y_fit, color='black', linewidth=2, zorder=2)

    # Spearman correlation in the upper left corner
    ax.text(0.03, 0.97, f'n = {len(df)}\nSpearman ρ = {rho:.2f} [95% CI {rho_lo:.2f}, {rho_hi:.2f}], {p_str}',
            transform=ax.transAxes, fontsize=FONT_SIZE - 4, va='top')

    ax.set_xlabel('Height [cm]', fontsize=FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))  # integer ticks for height
    ax.set_ylabel(f"{metric['label']} [cm]", fontsize=FONT_SIZE)
    ax.grid(alpha=0.2)
    # Add headroom above the data for the correlation text
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max + 0.2 * (y_max - y_min))
    style_axes(ax)
    plt.tight_layout()

    fname_figure = os.path.join(os.getcwd(), f"figure_correlation_lumbar_{metric['name']}_masks_height.png")
    fig.savefig(fname_figure, dpi=300)
    print(f'Figure saved to {fname_figure}')


def main():
    parser = argparse.ArgumentParser(description='Compute and plot a lumbar spinal cord length from spinal level masks '
                                                 'saved one level per file.')
    parser.add_argument('-i', required=True,
                        help='Folder with the spinal level masks '
                             '(sub-*/ses-*/anat/*_label-SC_desc-<level>_mask.nii.gz).')
    parser.add_argument('-metric', choices=list(METRICS), default='top-bottom',
                        help='top-bottom: length from the top of L1 to the bottom of L5; midpoints: distance between '
                             'the L1 and S1 midpoints. Default: top-bottom.')
    parser.add_argument('-participants', required=False,
                        help='participants.tsv with the subject sex and height. Default: participants.tsv one level '
                             'above the input folder.')
    args = parser.parse_args()
    dir_path = os.path.abspath(os.path.expanduser(args.i))
    fname_participants = args.participants or os.path.join(os.path.dirname(dir_path), 'participants.tsv')
    metric = METRICS[args.metric]

    # Compute the length for each subject
    df = compute_length(dir_path, metric)

    # Add the subject sex and height
    df_participants = pd.read_csv(fname_participants, sep='\t')
    df = df.merge(df_participants[['participant_id', 'sex', 'height']], how='left', left_on='subject',
                  right_on='participant_id').drop(columns='participant_id')
    df['sex'] = df['sex'].str.upper()

    # Save the lengths to a CSV file
    fname_csv = os.path.join(os.getcwd(), f"lumbar_{metric['name']}_masks.csv")
    df.to_csv(fname_csv, index=False)
    print(f'CSV file saved in {fname_csv}.')

    # Plot the distance per subject and its correlation with height (only subjects with known height)
    generate_length_figure(df, metric)
    generate_correlation_figure(df.dropna(subset=['height']).reset_index(drop=True), metric)
    plt.show()


if __name__ == '__main__':
    main()
