# =============================================================================
#  KI vs WT comparison per metabolite  -  Hippocampus, Female, APOE4
# =============================================================================
#
#  What this script does
#  ---------------------
#   1. Loads the PQN-normalised, log2-transformed data (CSV)
#   2. Keeps only Hippocampus + Female + APOE4 samples (change in STEP 0)
#   3. For every metabolite: compares KI vs WT (Welch t-test on log2 values)
#      and corrects for multiple testing (Benjamini-Hochberg FDR)
#   4. Saves a results table (CSV)
#   5. Saves a PDF with column plots (mean +/- SEM + individual samples + stars):
#        - first pages : all metabolites
#        - last page(s): only the significant metabolites
#
#  How to use
#  ----------
#   - Change the settings in STEP 0 (mainly DATA_FILE).
#   - Run the whole script (Spyder / VS Code / PyCharm "Run", or
#     `python compare_ki_vs_wt.py` in a terminal).
#   - Each "# %%" line starts a cell, so you can also run it step by step.
#
#  Packages needed:  pip install pandas numpy scipy statsmodels matplotlib
# =============================================================================

# %% Imports
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.stats.multitest import multipletests


# %% STEP 0 - Settings  (edit here)
DATA_FILE = "PQNregion_log2_samples_clean.csv"   # path to your data file
OUTPUT_FOLDER = "results"                       # where the CSV + PDF are saved

REGION = "Hippocmpus"     # value in column "Region of brain" (spelled like this in the file)
SEX = "F"                 # value in column "Gender"
APOE_LINE = "APOE4"       # value in column "line of APOE": "APOE2", "APOE3", "APOE4",
                          # or None to use all APOE lines together

ALPHA = 0.05              # significance threshold
USE_FDR = False           # False = use raw p-value, True = use FDR-corrected q-value
                          # (for the stars and for the "significant" page)

SHOW_PLOTS = True         # True = also show the column plots in the notebook / plot pane
                          # (they are always saved to the PDF)

WT_COLOR = "#9AA5B1"      # grey
KI_COLOR = "#C0504D"      # red


# %% STEP 1 - Load the data
data = pd.read_csv(DATA_FILE)
print(f"Loaded {DATA_FILE}: {data.shape[0]} samples x {data.shape[1]} columns")

# Metabolite columns are the ones starting with POS_ or NEG_
metabolites = [col for col in data.columns if col.startswith(("POS_", "NEG_"))]
print(f"Number of metabolites: {len(metabolites)}")


# %% STEP 2 - Keep only the chosen region + sex (+ APOE line), WT and KI
hip_f = data[
    (data["Region of brain"] == REGION)
    & (data["Gender"] == SEX)
    & (data["Genotype"].isin(["WT", "KI"]))
]
if APOE_LINE is not None:
    hip_f = hip_f[hip_f["line of APOE"] == APOE_LINE]

# Name used in titles and file names, e.g. "Hippocmpus_F_APOE4"
subset_name = f"{REGION}_{SEX}" + (f"_{APOE_LINE}" if APOE_LINE else "")

wt_samples = hip_f[hip_f["Genotype"] == "WT"]
ki_samples = hip_f[hip_f["Genotype"] == "KI"]
print(f"{subset_name}: WT = {len(wt_samples)} samples, KI = {len(ki_samples)} samples")


# %% STEP 3 - Statistics: KI vs WT for each metabolite
results = []
for met in metabolites:
    wt = wt_samples[met].dropna()
    ki = ki_samples[met].dropna()

    t_stat, p_value = stats.ttest_ind(ki, wt, equal_var=False)   # Welch t-test

    results.append({
        "metabolite": met,
        "n_WT": len(wt),
        "n_KI": len(ki),
        "mean_log2_WT": wt.mean(),
        "mean_log2_KI": ki.mean(),
        "log2FC_KI_vs_WT": ki.mean() - wt.mean(),
        "fold_change_KI_vs_WT": 2 ** (ki.mean() - wt.mean()),
        "t_stat": t_stat,
        "p_value": p_value,
    })

results = pd.DataFrame(results)

# Multiple-testing correction (Benjamini-Hochberg FDR)
results["q_value_FDR"] = multipletests(results["p_value"], method="fdr_bh")[1]

# Which value decides significance
sig_column = "q_value_FDR" if USE_FDR else "p_value"
results["significant"] = results[sig_column] < ALPHA
results["direction"] = np.where(results["log2FC_KI_vs_WT"] > 0, "up in KI", "down in KI")

results = results.sort_values("p_value").reset_index(drop=True)
significant = results[results["significant"]]

print(f"\nSignificant metabolites ({sig_column} < {ALPHA}): {len(significant)}")
if len(significant) > 0:
    print(significant[["metabolite", "log2FC_KI_vs_WT", "p_value", "q_value_FDR", "direction"]]
          .to_string(index=False))


# %% STEP 4 - Save the results table
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
table_file = os.path.join(OUTPUT_FOLDER, f"{subset_name}_KI_vs_WT_stats.csv")
results.to_csv(table_file, index=False)
print(f"\nSaved table: {table_file}")


# %% STEP 5 - Plot functions
def p_to_stars(p):
    """Convert a p-value into significance stars."""
    if p < 0.0001:
        return "****"
    elif p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def plot_one_metabolite(ax, row):
    """Column plot (mean +/- SEM + points) for one metabolite, with significance bracket.

    Values are shown as % of the WT mean (100 * 2^(log2 - mean WT)),
    because log2 values (~20) would make all bars look the same height.
    The statistics themselves are done on the log2 values.
    """
    met = row["metabolite"]
    wt_pct = 100 * 2 ** (wt_samples[met].dropna() - row["mean_log2_WT"])
    ki_pct = 100 * 2 ** (ki_samples[met].dropna() - row["mean_log2_WT"])

    groups = [("WT", wt_pct, WT_COLOR), ("KI", ki_pct, KI_COLOR)]
    rng = np.random.default_rng(0)   # fixed seed so the dot jitter is the same each run

    for x, (name, values, color) in enumerate(groups):
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values))
        ax.bar(x, mean, width=0.6, color=color, edgecolor="black", linewidth=0.6)
        ax.errorbar(x, mean, yerr=sem, color="black", capsize=4, linewidth=1)
        ax.scatter(x + rng.uniform(-0.15, 0.15, len(values)), values,
                   s=10, color="black", alpha=0.7, zorder=3)

    # Significance bracket + stars
    top = max(wt_pct.max(), ki_pct.max())
    h = 0.05 * top
    ax.plot([0, 0, 1, 1], [top + h, top + 2 * h, top + 2 * h, top + h], color="black", linewidth=0.8)
    ax.text(0.5, top + 2.2 * h, p_to_stars(row[sig_column]), ha="center", va="bottom", fontsize=9)

    # Labels
    mode, name = met.split("_", 1)            # e.g. "POS", "GLUTAMATE"
    ax.set_title(f"{name} [{mode}]\np = {row['p_value']:.3g}, q = {row['q_value_FDR']:.3g}", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"WT\n(n={len(wt_pct)})", f"KI\n(n={len(ki_pct)})"], fontsize=7)
    ax.set_ylabel("% of WT mean", fontsize=7)
    ax.set_ylim(0, top * 1.25)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def add_pages(pdf, table, title, n_rows, n_cols):
    """Add as many A4 pages as needed to plot every metabolite in `table`."""
    per_page = n_rows * n_cols
    n_pages = int(np.ceil(len(table) / per_page))

    for page in range(n_pages):
        page_rows = table.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.27, 11.69))   # A4
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, page_rows.iterrows()):
            plot_one_metabolite(ax, row)
        for ax in axes[len(page_rows):]:       # hide unused panels
            ax.axis("off")

        fig.suptitle(f"{title}  (page {page + 1}/{n_pages})", fontsize=11, fontweight="bold")
        fig.text(0.5, 0.01,
                 f"Mean ± SEM, dots = individual samples. Welch t-test on log2 data; "
                 f"stars based on {'FDR q' if USE_FDR else 'p-value'}: "
                 "* <0.05, ** <0.01, *** <0.001, **** <0.0001, ns = not significant",
                 ha="center", fontsize=6.5)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        pdf.savefig(fig)
        show_figure(fig)


def show_figure(fig):
    """Show the figure in Jupyter / VS Code / Spyder (if SHOW_PLOTS), then free memory."""
    if SHOW_PLOTS:
        try:
            from IPython import get_ipython
            from IPython.display import display
            in_notebook = get_ipython() is not None
        except ImportError:
            in_notebook = False
        if in_notebook:            # Jupyter / VS Code notebook / Spyder
            display(fig)
        else:                      # plain Python in a terminal: open a window
            plt.show()
    plt.close(fig)


# %% STEP 6 - Make the PDF
pdf_file = os.path.join(OUTPUT_FOLDER, f"{subset_name}_KI_vs_WT_plots.pdf")

with PdfPages(pdf_file) as pdf:
    # Pages 1..n: all metabolites (4 x 5 per page)
    add_pages(pdf, results, f"{subset_name}: KI vs WT - all metabolites", n_rows=5, n_cols=4)

    # Last page(s): significant metabolites only (3 x 4 per page, bigger)
    if len(significant) > 0:
        add_pages(pdf, significant,
                  f"{subset_name}: KI vs WT - significant ({sig_column} < {ALPHA})",
                  n_rows=4, n_cols=3)
    else:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.5, f"No significant metabolites ({sig_column} < {ALPHA})",
                 ha="center", fontsize=14)
        pdf.savefig(fig)
        show_figure(fig)

print(f"Saved plots: {pdf_file}")
