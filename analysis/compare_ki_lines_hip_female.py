"""
Compare the KI effects between APOE lines
Female hippocampus

Input : the table made by log2fc_hip_female.py (log2FC, p, FDR per KI line)
Output: 1) correlation matrix of the log2FC profiles (Spearman, all metabolites)
        2) scatter plots of log2FC, one dot per metabolite, for each pair of KI lines
           points on the diagonal   = same effect in both lines
           points off the diagonal  = effect specific to one line
"""

import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

# ---------------------------------------------------------------- settings
INPUT_FILE = "log2FC_Hippocampus_Female.csv"
CORR_FILE = "KI_lines_correlation_Hippocampus_Female.csv"
SCATTER_FILE = "KI_lines_scatter_Hippocampus_Female.png"
APOE_LINES = ["APOE2", "APOE3", "APOE4"]
P_CUTOFF = 0.05
LABEL_MIN_FC = 0.4          # label significant metabolites only if |log2FC| >= this in either line

# colours: grey = not significant, blue / orange = significant in one line, black = both
GREY, BLUE, ORANGE, BLACK = "#c3c2b7", "#2a78d6", "#eb6834", "#0b0b0b"

# ---------------------------------------------------------------- load results
results = pd.read_csv(INPUT_FILE)
names = results["Metabolite"].str.title()
names = names.where(names.str.len() <= 22, names.str[:20] + "..")   # shorten long names
dup = names.duplicated(keep=False)                                    # measured in both modes
names[dup] = names[dup] + " (" + results.loc[dup, "Mode"] + ")"
fc = results[[f"log2FC KI-{apoe}" for apoe in APOE_LINES]]
fc.columns = [f"KI-{apoe}" for apoe in APOE_LINES]

# ---------------------------------------------------------------- correlation matrix
corr = fc.corr(method="spearman").round(2)
corr.to_csv(CORR_FILE)
print("Spearman correlation of log2FC (all metabolites):")
print(corr)

# ---------------------------------------------------------------- scatter plots
pairs = list(combinations(APOE_LINES, 2))
fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5.2))
lim = 1.08 * fc.abs().max().max()                  # same axis range in every panel

for ax, (a, b) in zip(axes, pairs):
    x, y = fc[f"KI-{a}"], fc[f"KI-{b}"]
    sig_x = results[f"p KI-{a}"] < P_CUTOFF
    sig_y = results[f"p KI-{b}"] < P_CUTOFF

    groups = [
        (~sig_x & ~sig_y, GREY, 14, "not significant"),
        (sig_x & ~sig_y, BLUE, 30, f"p < {P_CUTOFF} in KI-{a} only"),
        (~sig_x & sig_y, ORANGE, 30, f"p < {P_CUTOFF} in KI-{b} only"),
        (sig_x & sig_y, BLACK, 40, f"p < {P_CUTOFF} in both"),
    ]
    for mask, colour, size, label in groups:
        ax.scatter(x[mask], y[mask], s=size, color=colour, edgecolor="white",
                   linewidth=0.6, label=f"{label} (n = {mask.sum()})", zorder=3)

    # label significant metabolites with a clear effect (and all significant in both)
    big = (x.abs() >= LABEL_MIN_FC) | (y.abs() >= LABEL_MIN_FC)
    for i in ((sig_x | sig_y) & big | (sig_x & sig_y))[lambda s: s].index:
        ax.annotate(names[i], (x[i], y[i]), xytext=(4, 3), textcoords="offset points",
                    fontsize=6, color="#52514e")

    ax.axhline(0, color="#d9d8d3", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#d9d8d3", linewidth=0.8, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], color="#8a8983", linewidth=0.8,
            linestyle="--", zorder=2)              # diagonal: same effect in both
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"log2FC KI-{a} vs pooled WT", fontsize=9)
    ax.set_ylabel(f"log2FC KI-{b} vs pooled WT", fontsize=9)
    ax.set_title(f"KI-{a} vs KI-{b}\nSpearman rho = {corr.loc[f'KI-{a}', f'KI-{b}']:.2f}",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

fig.suptitle("Female hippocampus: KI effects compared between APOE lines", fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(SCATTER_FILE, dpi=300, bbox_inches="tight")
print(f"\nSaved {CORR_FILE} and {SCATTER_FILE}")
