"""
Compare the KI effects between APOE lines
Female hippocampus

Input : 1) the table made by log2fc_hip_female.py (log2FC, p, FDR per KI line)
        2) the sample data (drift + PQN, log2) for the permutation test and direct comparisons
Output: 1) correlation matrix of the log2FC profiles (Spearman, all metabolites)
        2) correlation per pair of KI lines, with the range expected if the KI lines
           were identical (the shared pooled WT makes the lines correlate on its own)
        3) table of all metabolites: log2FC and p of the three KI lines side by side,
           in which lines each is significant, and the pattern
           (shared same direction / shared opposite direction / one line only / not significant)
        4) scatter plots of log2FC, one dot per metabolite, for each pair of KI lines
           points on the diagonal   = same effect in both lines
           points off the diagonal  = effect specific to one line
        5) permutation test: is one KI line less similar to the others than chance?
        6) direct comparisons between KI lines, per metabolite (Welch t-test + FDR)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
INPUT_FILE = "log2FC_Hippocampus_Female.csv"
DATA_FILE = "DriftCor_PQNregion_log2_samples_clean.csv"         # same data as the heatmap script
REGION = "Hippocmpus"                                           # spelled as in the data file
SEX = "F"
EXCLUDE_SAMPLES = ["34H_AP3_F_WT_163"]                          # same as the heatmap script
CORR_FILE = "KI_lines_correlation_Hippocampus_Female.csv"
SCATTER_FILE = "KI_lines_scatter_Hippocampus_Female.png"
PAIRS_FILE = "KI_lines_pairs_Hippocampus_Female.csv"            # rho and p per pair
METABOLITE_FILE = "KI_lines_metabolites_Hippocampus_Female.csv"  # all metabolites, 3 KI lines
PERMUTATION_FILE = "KI_lines_permutation_Hippocampus_Female.csv" # odd-one-out test
DIRECT_FILE = "KI_lines_direct_Hippocampus_Female.csv"           # KI line vs KI line, per metabolite
N_PERMUTATIONS = 5000
SEED = 1                                                         # same seed = same p-values
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

# sample data (needed for the permutation test and the direct comparisons)
data = pd.read_csv(DATA_FILE)
samples = data[(data["Region of brain"] == REGION) & (data["Gender"] == SEX)]
samples = samples[~samples["Sample ID"].isin(EXCLUDE_SAMPLES)]
metabolites = [col for col in data.columns if col.startswith(("POS_", "NEG_"))]
X = samples[metabolites].to_numpy()
group = np.where(samples["Genotype"] == "WT", "WT", "KI-" + samples["line of APOE"]).astype(object)
LINES = [f"KI-{apoe}" for apoe in APOE_LINES]
print(f"Samples: WT n = {(group == 'WT').sum()}, "
      + ", ".join(f"{line} n = {(group == line).sum()}" for line in LINES))

# same log2FC as the heatmap script? (checks that both scripts use the same data and samples)
check = X[group == LINES[0]].mean(0) - X[group == "WT"].mean(0)
assert np.allclose(check, fc[LINES[0]].to_numpy(), atol=1e-3), \
    "log2FC table and DATA_FILE / EXCLUDE_SAMPLES do not match: rerun the heatmap script"

# ---------------------------------------------------------------- correlation matrix
corr = fc.corr(method="spearman").round(2)
corr.to_csv(CORR_FILE)
print("Spearman correlation of log2FC (all metabolites):")
print(corr)

# ---------------------------------------------------------------- permutation test
# All KI lines are compared with the same pooled WT, so their log2FC profiles correlate even
# if the lines had identical effects. To know what "chance" looks like, the KI labels are
# shuffled among the KI mice (WT kept as is, group sizes kept), and everything is recomputed.
pairs = list(combinations(APOE_LINES, 2))

def correlations(labels):
    wt_mean = X[labels == "WT"].mean(0)
    profile = {line: X[labels == line].mean(0) - wt_mean for line in LINES}
    return np.array([stats.spearmanr(profile[f"KI-{a}"], profile[f"KI-{b}"])[0]
                     for a, b in pairs])

def odd_one_out(rhos):
    # D for each line: (correlation of the other two lines) - (mean correlation with this line)
    d = []
    for apoe in APOE_LINES:
        with_line = [r for (a, b), r in zip(pairs, rhos) if apoe in (a, b)]
        without_line = [r for (a, b), r in zip(pairs, rhos) if apoe not in (a, b)]
        d.append(np.mean(without_line) - np.mean(with_line))
    return np.array(d)

rng = np.random.default_rng(SEED)
ki_rows = np.where(group != "WT")[0]
observed_rho = correlations(group)
observed_d = odd_one_out(observed_rho)
null_rho, null_d = [], []
for _ in range(N_PERMUTATIONS):
    shuffled = group.copy()
    shuffled[ki_rows] = rng.permutation(group[ki_rows])
    r = correlations(shuffled)
    null_rho.append(r)
    null_d.append(odd_one_out(r))
null_rho, null_d = np.array(null_rho), np.array(null_d)

pair_table = pd.DataFrame({
    "Pair": [f"KI-{a} vs KI-{b}" for a, b in pairs],
    "Spearman rho": observed_rho.round(2),
    "expected rho if lines identical": null_rho.mean(0).round(2),
    "chance range low (2.5%)": np.percentile(null_rho, 2.5, axis=0).round(2),
    "chance range high (97.5%)": np.percentile(null_rho, 97.5, axis=0).round(2),
})
pair_table.to_csv(PAIRS_FILE, index=False)
print("\nCorrelation per pair (compare with the range expected by chance):")
print(pair_table.to_string(index=False))

def perm_p(null, observed):
    return (np.sum(null >= observed) + 1) / (len(null) + 1)

perm_table = pd.DataFrame({
    "Odd one out": LINES,
    "D (others' rho - rho with this line)": observed_d.round(2),
    "p (this line chosen in advance)": [perm_p(null_d[:, k], observed_d[k]) for k in range(3)],
})
p_any = perm_p(null_d.max(1), observed_d.max())
perm_table["p (any line, fair if not chosen in advance)"] = p_any
perm_table.to_csv(PERMUTATION_FILE, index=False)
print(f"\nPermutation test, {N_PERMUTATIONS} shuffles:")
print(perm_table.round(3).to_string(index=False))
print(f"Most different line: {LINES[observed_d.argmax()]} (D = {observed_d.max():.2f}, "
      f"fair p = {p_any:.3f})")

# ---------------------------------------------------------------- direct comparisons
# KI line vs KI line, per metabolite: Welch t-test, FDR across metabolites per comparison
direct = pd.DataFrame({"Metabolite": results["Metabolite"], "Mode": results["Mode"]})
print("\nDirect comparisons between KI lines:")
for a, b in [("APOE4", "APOE2"), ("APOE4", "APOE3"), ("APOE2", "APOE3")]:
    xa, xb = X[group == f"KI-{a}"], X[group == f"KI-{b}"]
    p = stats.ttest_ind(xa, xb, equal_var=False).pvalue
    q = multipletests(p, method="fdr_bh")[1]
    name = f"KI-{a} vs KI-{b}"
    direct[f"log2FC {name}"] = xa.mean(0) - xb.mean(0)
    direct[f"p {name}"] = p
    direct[f"FDR {name}"] = q
    hits = direct.loc[q < 0.05, "Metabolite"].tolist()
    print(f"  {name}: {(p < P_CUTOFF).sum()} p < {P_CUTOFF}, {(q < 0.05).sum()} FDR < 0.05 {hits}")
direct.round(4).to_csv(DIRECT_FILE, index=False)

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
print(f"\nSaved: {CORR_FILE}, {PAIRS_FILE}, {PERMUTATION_FILE}, {DIRECT_FILE}, "
      f"{METABOLITE_FILE}, {SCATTER_FILE}")
