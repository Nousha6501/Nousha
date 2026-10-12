# =============================================================================
#  CHECK: ARE DRIFT CORRECTION AND PQN GOOD ENOUGH?
#
#  Fair comparison of two versions that differ ONLY in the drift correction:
#     A = raw data            -> PQN -> log2
#     B = drift-corrected     -> PQN -> log2
#  Same PQN (per mode, per region, WT median as reference), same scaling.
#  Everything is done separately for POS and NEG mode.
#
#  TEST 1  Drift per metabolite        rho with injection order, A vs B
#  TEST 2  PCA vs injection order      |rho| of PC1-PC5 with injection #, A vs B
#  TEST 3  What each PC is             % of each PC explained by region, genotype,
#                                      sex, APOE and injection order
#  TEST 4  PCA plots                   same PCA coloured by injection, region,
#                                      genotype, APOE, sex (A and B, per mode)
#  TEST 5  PQN is sound                PQN factors: KI vs WT, POS vs NEG
#  TEST 6  Precision                   QC RSD (QC_1-2 excluded), spread within groups
#  TEST 7  POS-NEG agreement           same metabolite measured in both modes
#
#  How to use: change the paths in STEP 0, run the whole script.
#  Plots are shown in the notebook and saved as PNG in OUTPUT_DIR.
#  Needs: pandas, numpy, scipy, matplotlib, openpyxl
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# STEP 0 - YOUR FILES  (edit only this part)
# =============================================================================

RAW_FILE    = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\master_with_QC3_real_clean_23H_9,10,11,12H_C.csv"
DRIFT_FILE  = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\Drift_corrected_only.xlsx"
OUTPUT_DIR  = r"O:\metabolom\Result\new version result_09.2026\Check_drift_PQN"

COL_ID      = "Sample ID"
COL_INJ     = "injection #"
COL_TYPE    = "Type"
COL_REGION  = "Region of brain"
COL_GENO    = "Genotype"
COL_SEX     = "Gender"
COL_APOE    = "line of APOE"
WT_LABEL    = "WT"
QC_EXCLUDE  = ["QC_1", "QC_2"]                     # conditioning injections
INJECTION_FIXES = {"40H_AP4_M_KI_131": 178, "39H_AP2_M_KI_104": 180}
DUAL_MODE   = {"POS_CYCLIC AMP":            "NEG_CYCLIC-AMP",
               "POS_AMINOADIPATE":          "NEG_AMINOADIPIC ACID",
               "POS_INOSINE-MONOPHOSPHATE": "NEG_INOSINE MONOPHOSPHATE"}
N_PC = 5
SHOW_PLOTS = True          # True: show plots in the notebook (and save them); False: only save


# =============================================================================
# STEP 1 - LOAD BOTH VERSIONS IN THE SAME ROW ORDER
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

drift = pd.read_excel(DRIFT_FILE, sheet_name=0)
raw = pd.read_csv(RAW_FILE)
for sid, inj in INJECTION_FIXES.items():
    raw.loc[raw[COL_ID] == sid, COL_INJ] = inj
raw = raw.set_index(COL_ID).loc[drift[COL_ID]].reset_index().copy()   # same order as drift file

info = drift[[c for c in drift.columns if not c.startswith(("POS_", "NEG_"))]].copy()
for c in (COL_REGION, COL_GENO, COL_SEX, COL_APOE):
    info[c] = info[c].astype(str).str.strip()

metabolites = [c for c in drift.columns if c.startswith(("POS_", "NEG_")) and "13C" not in c]
is_sample = (info[COL_TYPE] == "Sample").to_numpy()
is_qc     = ((info[COL_TYPE] == "Quality control") & ~info[COL_ID].isin(QC_EXCLUDE)).to_numpy()
is_wt     = is_sample & (info[COL_GENO] == WT_LABEL).to_numpy()
injection = info[COL_INJ].to_numpy(float)
regions   = sorted(info.loc[is_sample, COL_REGION].unique())

print(f"{is_sample.sum()} samples, {is_qc.sum()} QCs used, {len(metabolites)} metabolites")


# =============================================================================
# STEP 2 - SAME PQN + log2 FOR BOTH VERSIONS
#   per mode; samples: reference = WT median of the same region;
#   QCs: reference = median of all samples
# =============================================================================

def pqn_log2(table):
    log = np.log2(table[metabolites].astype(float))
    out = log.copy()
    factors = pd.DataFrame(index=table.index, columns=["POS", "NEG"], dtype=float)
    for mode in ["POS", "NEG"]:
        cols = [c for c in metabolites if c.startswith(mode + "_")]
        groups = [(is_sample & (info[COL_REGION] == r).to_numpy(),
                   is_wt & (info[COL_REGION] == r).to_numpy()) for r in regions]
        groups.append((is_qc, is_sample))
        for rows, ref_rows in groups:
            ref = log.loc[ref_rows, cols].median()
            f = (log.loc[rows, cols] - ref).median(axis=1)           # log2 PQN factor
            out.loc[rows, cols] = log.loc[rows, cols].sub(f, axis=0)
            factors.loc[rows, mode] = 2 ** f
    return out, factors

A, factors_A = pqn_log2(raw)        # A = no drift correction
B, factors_B = pqn_log2(drift)      # B = drift correction
versions = {"A_no_drift_corr": A, "B_drift_corr": B}
summary = []


# =============================================================================
# TEST 1 - DRIFT PER METABOLITE (samples)
# =============================================================================

def rho_inj(values):
    return stats.spearmanr(injection[is_sample], values[is_sample])[0]

test1 = pd.DataFrame({"Metabolite": metabolites,
                      "rho A (no drift corr)": [rho_inj(A[c].to_numpy()) for c in metabolites],
                      "rho B (drift corr)":    [rho_inj(B[c].to_numpy()) for c in metabolites]})
for mode in ["POS", "NEG"]:
    m = test1["Metabolite"].str.startswith(mode)
    for label in ["rho A (no drift corr)", "rho B (drift corr)"]:
        summary.append(["1 Drift per metabolite", mode, f"|rho| >= 0.3: {label}",
                        int((test1.loc[m, label].abs() >= 0.3).sum())])


# =============================================================================
# TEST 2 + 3 - PCA PER MODE: RELATION OF EACH PC WITH INJECTION ORDER AND BIOLOGY
#   PCA on samples, autoscaled (each metabolite mean 0, SD 1), per mode.
#   Explained share (eta^2) = how much of a PC's spread is explained by a factor.
# =============================================================================

def pca(table, cols, rows):
    Z = table.loc[rows, cols]
    Z = (Z - Z.mean()) / Z.std(ddof=1)
    U, S, _ = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    return U[:, :N_PC] * S[:N_PC], S**2 / (S**2).sum() * 100

def eta2(score, labels):
    groups = [score[labels == g] for g in np.unique(labels)]
    total = ((score - score.mean())**2).sum()
    between = sum(len(g) * (g.mean() - score.mean())**2 for g in groups)
    return 100 * between / total

pca_store = {}
test23 = []
for mode in ["POS", "NEG"]:
    cols = [c for c in metabolites if c.startswith(mode + "_")]
    for vname, table in versions.items():
        scores, var = pca(table, cols, is_sample)
        pca_store[(mode, vname)] = (scores, var)
        for k in range(N_PC):
            sc = scores[:, k]
            row = {"Mode": mode, "Version": vname, "PC": f"PC{k+1}",
                   "% variance": var[k],
                   "|rho| with injection #": abs(stats.spearmanr(injection[is_sample], sc)[0])}
            for name, col in [("region", COL_REGION), ("genotype", COL_GENO),
                              ("sex", COL_SEX), ("APOE", COL_APOE)]:
                row[f"% explained by {name}"] = eta2(sc, info.loc[is_sample, col].to_numpy())
            test23.append(row)
test23 = pd.DataFrame(test23)

for mode in ["POS", "NEG"]:
    for vname in versions:
        part = test23[(test23["Mode"] == mode) & (test23["Version"] == vname)]
        summary.append(["2 PCA vs injection", mode, f"max |rho| PC1-{N_PC} with injection, {vname}",
                        round(part["|rho| with injection #"].max(), 2)])


# =============================================================================
# TEST 4 - PCA PLOTS: same PCA, five colourings, per mode, A and B
# =============================================================================

colourings = [("Injection #", None), ("Region", COL_REGION), ("Genotype", COL_GENO),
              ("APOE", COL_APOE), ("Sex", COL_SEX)]
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

for mode in ["POS", "NEG"]:
    fig, axes = plt.subplots(2, len(colourings), figsize=(4.2 * len(colourings), 8))
    for i, vname in enumerate(versions):
        scores, var = pca_store[(mode, vname)]
        for j, (title, col) in enumerate(colourings):
            ax = axes[i, j]
            if col is None:
                pts = ax.scatter(scores[:, 0], scores[:, 1], c=injection[is_sample],
                                 cmap="viridis", s=12)
                if j == 0 and i == 1:
                    fig.colorbar(pts, ax=axes[:, 0], label="Injection #", shrink=0.6)
            else:
                labels = info.loc[is_sample, col].to_numpy()
                for k, g in enumerate(sorted(np.unique(labels))):
                    m = labels == g
                    ax.scatter(scores[m, 0], scores[m, 1], s=12, color=palette[k % 5], label=g)
                ax.legend(fontsize=7, frameon=False)
            ax.set_title(f"{vname} - {title}", fontsize=9)
            ax.set_xlabel(f"PC1 ({var[0]:.0f}%)", fontsize=8)
            ax.set_ylabel(f"PC2 ({var[1]:.0f}%)", fontsize=8)
    fig.suptitle(f"{mode} mode: PCA without (top) and with (bottom) drift correction, both PQN + log2")
    fig.savefig(os.path.join(OUTPUT_DIR, f"PCA_{mode}_A_vs_B.png"), dpi=130, bbox_inches="tight")
    plt.show() if SHOW_PLOTS else plt.close(fig)

# PC scores vs injection order (where drift would show up)
for mode in ["POS", "NEG"]:
    fig, axes = plt.subplots(2, N_PC, figsize=(3.4 * N_PC, 6), sharex=True)
    for i, vname in enumerate(versions):
        scores, var = pca_store[(mode, vname)]
        for k in range(N_PC):
            ax = axes[i, k]
            r = stats.spearmanr(injection[is_sample], scores[:, k])[0]
            ax.scatter(injection[is_sample], scores[:, k], s=8, color="#2a78d6")
            ax.set_title(f"{vname[:1]}: PC{k+1} ({var[k]:.0f}%)  rho={r:+.2f}", fontsize=8)
            if i == 1:
                ax.set_xlabel("Injection #", fontsize=8)
    fig.suptitle(f"{mode} mode: PC scores vs injection order (A top, B bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"PCscores_vs_injection_{mode}.png"), dpi=130)
    plt.show() if SHOW_PLOTS else plt.close(fig)


# =============================================================================
# TEST 5 - PQN IS SOUND (version B)
# =============================================================================

for mode in ["POS", "NEG"]:
    for r in regions:
        m = is_sample & (info[COL_REGION] == r).to_numpy()
        wt = factors_B.loc[m & is_wt, mode]
        ki = factors_B.loc[m & ~is_wt, mode]
        p = stats.mannwhitneyu(wt, ki).pvalue
        summary.append(["5 PQN sound", mode, f"{r}: PQN factor WT {wt.median():.2f} vs KI "
                        f"{ki.median():.2f}, p (want > 0.05)", round(p, 3)])
summary.append(["5 PQN sound", "both", "POS vs NEG PQN factor correlation (want > 0.7)",
                round(np.corrcoef(factors_B.loc[is_sample, "POS"],
                                  factors_B.loc[is_sample, "NEG"])[0, 1], 2)])


# =============================================================================
# TEST 6 - PRECISION
#   QC RSD: QCs (QC_1-2 excluded), after PQN, from log2 SD
#   within-group spread: SD between animals of the same region x genotype x sex x APOE
# =============================================================================

def rsd_from_log2(values):
    sd = np.std(values, ddof=1) * np.log(2)
    return 100 * np.sqrt(np.exp(sd**2) - 1)

group = info.loc[is_sample, [COL_REGION, COL_GENO, COL_SEX, COL_APOE]].agg("|".join, axis=1).to_numpy()
for mode in ["POS", "NEG"]:
    cols = [c for c in metabolites if c.startswith(mode + "_")]
    for vname, table in versions.items():
        qc_rsd = np.median([rsd_from_log2(table.loc[is_qc, c]) for c in cols])
        within = np.median([rsd_from_log2(np.concatenate(
            [v - v.mean() for _, v in pd.Series(table.loc[is_sample, c].to_numpy()).groupby(group)]))
            for c in cols])
        summary.append(["6 Precision", mode, f"median QC RSD %, {vname}", round(qc_rsd, 1)])
        summary.append(["6 Precision", mode, f"median within-group spread %, {vname}", round(within, 1)])


# =============================================================================
# TEST 7 - POS-NEG AGREEMENT (samples)
# =============================================================================

for pos, neg in DUAL_MODE.items():
    if pos in metabolites and neg in metabolites:
        for vname, table in versions.items():
            r = np.corrcoef(table.loc[is_sample, pos], table.loc[is_sample, neg])[0, 1]
            summary.append(["7 POS-NEG agreement", "both", f"{pos[4:]}: r, {vname}", round(r, 3)])


# =============================================================================
# STEP 8 - SAVE AND PRINT
# =============================================================================

summary = pd.DataFrame(summary, columns=["Test", "Mode", "Measure", "Value"])
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "Check_drift_PQN.xlsx")) as excel:
    summary.to_excel(excel, sheet_name="Summary", index=False)
    test1.to_excel(excel, sheet_name="1_Drift_per_metabolite", index=False)
    test23.round(3).to_excel(excel, sheet_name="2_3_PCA_components", index=False)
    pd.concat([info, factors_B.add_prefix("PQN_factor_")], axis=1).to_excel(
        excel, sheet_name="5_PQN_factors", index=False)

pd.set_option("display.width", 200, "display.max_colwidth", 80)
print(summary.to_string(index=False))
print("\nPCA components (what each PC is):")
print(test23.round(1).to_string(index=False))
print(f"\nSaved table and figures in {OUTPUT_DIR}")
