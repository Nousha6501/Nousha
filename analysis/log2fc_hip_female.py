# log2FC of KI vs pooled WT in female hippocampus, per APOE genotype.
# Data are already log2, so log2FC = mean(KI) - mean(WT).
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

data = pd.read_csv("DriftCor_PQNregion_log2_samples_clean.csv")
hip_f = data[(data["Region of brain"] == "Hippocmpus") & (data["Gender"] == "F")]
metabolites = [c for c in data.columns if c.startswith(("POS_", "NEG_"))]

wt = hip_f[hip_f["Genotype"] == "WT"]
ki = {apoe: hip_f[(hip_f["Genotype"] == "KI") & (hip_f["line of APOE"] == apoe)]
      for apoe in ["APOE2", "APOE3", "APOE4"]}

table = pd.DataFrame(index=metabolites)
for apoe, group in ki.items():
    table[f"log2FC_KI-{apoe}"] = group[metabolites].mean() - wt[metabolites].mean()

# p-values: Dunnett test, each KI group vs pooled WT
p = pd.DataFrame([stats.dunnett(*[g[m].dropna() for g in ki.values()],
                                control=wt[m].dropna()).pvalue for m in metabolites],
                 index=metabolites, columns=list(ki))
for apoe in ki:
    table[f"p_KI-{apoe}"] = p[apoe]
    table[f"FDR_KI-{apoe}"] = multipletests(p[apoe], method="fdr_bh")[1]

table.round(4).to_csv("log2FC_Hippocampus_Female.csv", index_label="Metabolite")
print(table.round(3))
