"""
log2 fold change: KI vs pooled WT, per APOE genotype
Female hippocampus

Input : drift-corrected, PQN-normalised, log2-transformed data (one row per sample)
Output: one table with, for each metabolite and each KI line,
        log2FC, p-value (Dunnett vs pooled WT) and FDR (Benjamini-Hochberg)
"""

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
INPUT_FILE = "DriftCor_PQNregion_log2_samples_clean.csv"
OUTPUT_FILE = "log2FC_Hippocampus_Female.csv"
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
