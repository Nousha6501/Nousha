"""
Compare the KI effects between APOE lines
Female hippocampus

Input : the table made by log2fc_hip_female.py (log2FC, p, FDR per KI line)
Output: 1) correlation matrix of the log2FC profiles (Spearman, all metabolites)
        2) correlation per pair of KI lines with its p-value
        3) table of all metabolites: log2FC and p of the three KI lines side by side,
           in which lines each is significant, and the pattern
           (shared same direction / shared opposite direction / one line only / not significant)
        4) scatter plots of log2FC, one dot per metabolite, for each pair of KI lines
           points on the diagonal   = same effect in both lines
           points off the diagonal  = effect specific to one line
"""

import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from scipy import stats

# ---------------------------------------------------------------- settings
INPUT_FILE = "log2FC_Hippocampus_Female.csv"
CORR_FILE = "KI_lines_correlation_Hippocampus_Female.csv"
SCATTER_FILE = "KI_lines_scatter_Hippocampus_Female.png"
PAIRS_FILE = "KI_lines_pairs_Hippocampus_Female.csv"            # rho and p per pair
METABOLITE_FILE = "KI_lines_metabolites_Hippocampus_Female.csv"  # all metabolites, 3 KI lines
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

# ---------------------------------------------------------------- correlation per pair
pairs = list(combinations(APOE_LINES, 2))
pair_rows = []
for a, b in pairs:
    rho, p_rho = stats.spearmanr(fc[f"KI-{a}"], fc[f"KI-{b}"])
    pair_rows.append({"Pair": f"KI-{a} vs KI-{b}", "Spearman rho": round(rho, 2),
                      "p-value": p_rho, "n metabolites": len(fc)})
pair_table = pd.DataFrame(pair_rows)
pair_table.to_csv(PAIRS_FILE, index=False)
print("\nCorrelation per pair:")
print(pair_table.to_string(index=False))

# ---------------------------------------------------------------- metabolite table
# one row per metabolite, the three KI lines side by side, and in which lines it is significant
table = results[["Metabolite", "Mode"]].copy()
for apoe in APOE_LINES:
    table[f"log2FC KI-{apoe}"] = results[f"log2FC KI-{apoe}"]
    table[f"p KI-{apoe}"] = results[f"p KI-{apoe}"]

def significant_in(row):
    hits = []
    for apoe in APOE_LINES:
        if row[f"p KI-{apoe}"] < P_CUTOFF:
            direction = "down" if row[f"log2FC KI-{apoe}"] < 0 else "up"
            hits.append(f"KI-{apoe} {direction}")
    return ", ".join(hits) if hits else "none"

def pattern(row):
    hits = [apoe for apoe in APOE_LINES if row[f"p KI-{apoe}"] < P_CUTOFF]
    if not hits:
        return "not significant"
    if len(hits) == 1:
        return f"KI-{hits[0]} only"
    signs = {row[f"log2FC KI-{apoe}"] > 0 for apoe in hits}
    return "shared, same direction" if len(signs) == 1 else "shared, opposite direction"

table["Significant in"] = table.apply(significant_in, axis=1)
table["Pattern"] = table.apply(pattern, axis=1)
table.to_csv(METABOLITE_FILE, index=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 250)
sig = table[table["Pattern"] != "not significant"].sort_values(["Pattern", "Metabolite"])
print(f"\nMetabolites with p < {P_CUTOFF} in at least one KI line ({len(sig)}):")
print(sig.drop(columns="Mode").round(3).to_string(index=False))

# ---------------------------------------------------------------- scatter plots
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
print(f"\nSaved {CORR_FILE}, {PAIRS_FILE}, {METABOLITE_FILE} and {SCATTER_FILE}")
