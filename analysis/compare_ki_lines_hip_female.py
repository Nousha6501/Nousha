"""
Compare the KI effects between APOE lines
Female hippocampus

Input : the table made by log2fc_hip_female.py (log2FC, p, FDR per KI line)
Output: 1) correlation matrix of the log2FC profiles (Spearman, all metabolites)
        2) correlation per pair of KI lines with its p-value
        3) table of metabolites per pair, with the group each one falls in
           (shared / opposite / one line only / not significant)
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
METABOLITE_FILE = "KI_lines_metabolites_Hippocampus_Female.csv"  # metabolites per pair
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

# ---------------------------------------------------------------- metabolites per pair
def group_name(fc_a, fc_b, sig_a, sig_b, a, b):
    if sig_a and sig_b:
        return "shared, same direction" if fc_a * fc_b > 0 else "shared, opposite direction"
    if sig_a:
        return f"KI-{a} only"
    if sig_b:
        return f"KI-{b} only"
    return "not significant"

rows = []
for a, b in pairs:
    for i in results.index:
        fc_a, fc_b = fc.loc[i, f"KI-{a}"], fc.loc[i, f"KI-{b}"]
        p_a, p_b = results.loc[i, f"p KI-{a}"], results.loc[i, f"p KI-{b}"]
        rows.append({
            "Pair": f"KI-{a} vs KI-{b}",
            "Metabolite": results.loc[i, "Metabolite"],
            "Mode": results.loc[i, "Mode"],
            f"log2FC first": fc_a, f"p first": p_a,
            f"log2FC second": fc_b, f"p second": p_b,
            "Group": group_name(fc_a, fc_b, p_a < P_CUTOFF, p_b < P_CUTOFF, a, b),
        })
met_table = pd.DataFrame(rows)
met_table.to_csv(METABOLITE_FILE, index=False)

print(f"\nMetabolites with p < {P_CUTOFF} in at least one line of the pair:")
for pair, part in met_table.groupby("Pair", sort=False):
    sig = part[part["Group"] != "not significant"].sort_values("Group")
    print(f"\n{pair}  (first = {pair.split(' vs ')[0]}, second = {pair.split(' vs ')[1]})")
    print(sig[["Metabolite", "Mode", "log2FC first", "p first", "log2FC second",
               "p second", "Group"]].round(3).to_string(index=False))

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
