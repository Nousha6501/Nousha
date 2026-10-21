"""
log2 fold change: KI vs pooled WT, per APOE genotype
Female hippocampus

Input : drift-corrected, PQN-normalised, log2-transformed data (one row per sample)
Output: 1) table with, for each metabolite and each KI line,
           log2FC, p-value (Dunnett vs pooled WT) and FDR (Benjamini-Hochberg)
        2) heatmap of the log2FC (dot = p < 0.05, star = FDR < 0.05)
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
INPUT_FILE = "DriftCor_PQNregion_log2_samples_clean.csv"
OUTPUT_FILE = "log2FC_Hippocampus_Female.csv"
HEATMAP_FILE = "log2FC_Hippocampus_Female_heatmap.png"
ONLY_SIGNIFICANT = False       # True: heatmap shows only metabolites with p < 0.05 in any KI line
COLOR_LIMIT = 0.6              # log2FC at which the colour is darkest
REGION = "Hippocmpus"          # spelled as in the data file
SEX = "F"
APOE_LINES = ["APOE2", "APOE3", "APOE4"]

# ---------------------------------------------------------------- load data
data = pd.read_csv(INPUT_FILE)
samples = data[(data["Region of brain"] == REGION) & (data["Gender"] == SEX)]
metabolites = [col for col in data.columns if col.startswith(("POS_", "NEG_"))]

wt = samples[samples["Genotype"] == "WT"]                     # pooled WT
ki = {apoe: samples[(samples["Genotype"] == "KI") & (samples["line of APOE"] == apoe)]
      for apoe in APOE_LINES}

print(f"WT (pooled): n = {len(wt)}")
for apoe, group in ki.items():
    print(f"KI-{apoe}: n = {len(group)}")

# ---------------------------------------------------------------- statistics
results = pd.DataFrame({
    "Metabolite": [m.split("_", 1)[1] for m in metabolites],
    "Mode": [m.split("_", 1)[0] for m in metabolites],
})

# Dunnett test: each KI group vs pooled WT, one metabolite at a time
p_values = {apoe: [] for apoe in APOE_LINES}
for m in metabolites:
    ki_groups = [ki[apoe][m].dropna() for apoe in APOE_LINES]
    p = stats.dunnett(*ki_groups, control=wt[m].dropna()).pvalue
    for apoe, p_apoe in zip(APOE_LINES, p):
        p_values[apoe].append(p_apoe)

for apoe in APOE_LINES:
    # data are log2, so the fold change is a difference of means
    results[f"log2FC KI-{apoe}"] = (ki[apoe][metabolites].mean()
                                    - wt[metabolites].mean()).values
    results[f"p KI-{apoe}"] = p_values[apoe]
    results[f"FDR KI-{apoe}"] = multipletests(p_values[apoe], method="fdr_bh")[1]

# ---------------------------------------------------------------- save
results = results.round(4)
results.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved {len(results)} metabolites to {OUTPUT_FILE}")

for apoe in APOE_LINES:
    n_p = (results[f"p KI-{apoe}"] < 0.05).sum()
    n_fdr = (results[f"FDR KI-{apoe}"] < 0.05).sum()
    print(f"KI-{apoe}: {n_p} metabolites p < 0.05, {n_fdr} FDR < 0.05")

# ---------------------------------------------------------------- heatmap
fc = results[[f"log2FC KI-{apoe}" for apoe in APOE_LINES]].to_numpy()
p = results[[f"p KI-{apoe}" for apoe in APOE_LINES]].to_numpy()
fdr = results[[f"FDR KI-{apoe}" for apoe in APOE_LINES]].to_numpy()
labels = (results["Metabolite"] + " (" + results["Mode"] + ")").to_numpy()

if ONLY_SIGNIFICANT:
    keep = (p < 0.05).any(axis=1)
    fc, p, fdr, labels = fc[keep], p[keep], fdr[keep], labels[keep]

# blue = lower in KI, grey = no change, red = higher in KI
cmap = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])

fig, ax = plt.subplots(figsize=(5, 0.18 * len(labels) + 1.8))
im = ax.imshow(fc, cmap=cmap, vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT, aspect="auto")

for i in range(len(labels)):
    for j in range(len(APOE_LINES)):
        if fdr[i, j] < 0.05:
            ax.text(j, i, "★", ha="center", va="center", fontsize=8)
        elif p[i, j] < 0.05:
            ax.text(j, i, "•", ha="center", va="center", fontsize=9)

ax.set_xticks(range(len(APOE_LINES)), [f"KI\n{apoe}" for apoe in APOE_LINES], fontsize=8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(labels)), labels, fontsize=6)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)

bar = fig.colorbar(im, ax=ax, shrink=0.3, extend="both")
bar.set_label("log2FC vs pooled WT", fontsize=8)
ax.set_title("Female hippocampus: KI vs pooled WT\n• p < 0.05   ★ FDR < 0.05", fontsize=9, pad=12)

fig.tight_layout()
fig.savefig(HEATMAP_FILE, dpi=300, bbox_inches="tight")
print(f"Saved heatmap to {HEATMAP_FILE}")
