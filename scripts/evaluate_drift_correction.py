# =============================================================================
#  EVALUATE THE DRIFT CORRECTION
#
#  Compares the raw master data with the drift-corrected data and answers:
#    TEST 1  Is the drift gone?                 (trend with injection order)
#    TEST 2  Is the biology kept?               (region, sex, genotype, APOE)
#    TEST 3  Is the data less noisy?            (RSD of samples and QCs)
#    TEST 4  Do POS and NEG agree better?       (metabolites measured in both modes)
#    TEST 5  Is drift gone from the big picture? (PCA vs injection order)
#    TEST 6  Accuracy with a KNOWN drift        (add fake drift, check it is recovered)
#    TEST 7  Over-fitting check                 (shuffled injection order)
#
#  How to use:
#    1. Run drift_correction.py first.
#    2. Change the paths in STEP 0 below and run this whole script.
#  Needs: pandas, numpy, scipy, matplotlib, openpyxl
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import BSpline


# =============================================================================
# STEP 0 - YOUR FILES  (edit only this part)
# =============================================================================

RAW_FILE       = r"C:\Users\YourName\Documents\master_with_QC3_real_clean_23H_9101112H_C.csv"
CORRECTED_FILE = r"C:\Users\YourName\Documents\Drift_corrected_master.xlsx"   # from drift_correction.py
OUTPUT_FOLDER  = r"C:\Users\YourName\Documents\Drift_evaluation"

COL_ID      = "Sample ID"
COL_INJ     = "injection #"
COL_TYPE    = "Type"
COL_SEGMENT = "Batch segment( before jump=A, after jump=B)"
COL_BIOLOGY = ["Group_Region", "Gender", "Genotype", "line of APOE"]
INJECTION_FIXES = {"40H_AP4_M_KI_131": 178, "39H_AP2_M_KI_104": 180}

# Metabolites measured in both modes (same compound, two measurements)
DUAL_MODE = {"POS_CYCLIC AMP":            "NEG_CYCLIC-AMP",
             "POS_AMINOADIPATE":          "NEG_AMINOADIPIC ACID",
             "POS_INOSINE-MONOPHOSPHATE": "NEG_INOSINE MONOPHOSPHATE"}

RANDOM_SEED = 1


# =============================================================================
# STEP 1 - LOAD RAW AND CORRECTED DATA (same order of rows)
# =============================================================================

import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

raw_data = pd.read_csv(RAW_FILE)
for sample_id, inj in INJECTION_FIXES.items():
    raw_data.loc[raw_data[COL_ID] == sample_id, COL_INJ] = inj
empty = raw_data[COL_SEGMENT].isna()
raw_data.loc[empty, COL_SEGMENT] = np.where(raw_data.loc[empty, COL_INJ] >= 124, "B", "A")
raw_data = raw_data.sort_values(COL_INJ).reset_index(drop=True)

corr_data = pd.read_excel(CORRECTED_FILE, sheet_name="Corrected_raw")
corr_data = corr_data.set_index(COL_ID).loc[raw_data[COL_ID]].reset_index().copy()

features    = [c for c in raw_data.columns if c.startswith(("POS_", "NEG_"))]
metabolites = [c for c in features if "13C" not in c]

raw  = raw_data[features].astype(float)
corr = corr_data[features].astype(float)

injection = raw_data[COL_INJ].to_numpy(float)
segment   = raw_data[COL_SEGMENT].to_numpy()
is_sample = (raw_data[COL_TYPE] == "Sample").to_numpy()
is_qc     = ((raw_data[COL_TYPE] == "Quality control")
             & ~raw_data[COL_ID].isin(["QC_1", "QC_2"])).to_numpy()

print(f"Loaded {len(raw_data)} injections, {len(features)} features")

summary = []          # one line per test result -> "Summary" sheet


# =============================================================================
# TEST 1 - IS THE DRIFT GONE?
#   Spearman rho of each metabolite with injection order (samples only).
#   Good result: |rho| close to 0 after correction for every metabolite.
# =============================================================================

def rho_with_injection(table):
    return np.array([stats.spearmanr(injection[is_sample], table[f][is_sample])[0]
                     for f in features])

rho_before = rho_with_injection(raw)
rho_after  = rho_with_injection(corr)

test1 = pd.DataFrame({"Metabolite": features,
                      "rho before": rho_before, "rho after": rho_after})

summary += [
    ["1 Drift gone", "Features with clear drift (|rho| >= 0.3)",
     (abs(rho_before) >= 0.3).sum(), (abs(rho_after) >= 0.3).sum(), "after should be 0"],
    ["1 Drift gone", "Features with any drift (|rho| >= 0.16, p < 0.05)",
     (abs(rho_before) >= 0.16).sum(), (abs(rho_after) >= 0.16).sum(), "after should be ~0"],
    ["1 Drift gone", "Largest |rho|",
     abs(rho_before).max(), abs(rho_after).max(), "after should be < 0.16"],
]

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.linspace(-1, 1, 41)
ax.hist(rho_before, bins, alpha=.6, label="before", color="#eb6834")
ax.hist(rho_after,  bins, alpha=.6, label="after",  color="#2a78d6")
ax.axvspan(-0.16, 0.16, color="#eeeeea", zorder=0)
ax.set_xlabel("Spearman rho with injection order"); ax.set_ylabel("Number of features")
ax.set_title("TEST 1: drift before vs after (grey = not significant)")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER, "test1_drift_rho.png"), dpi=150); plt.close(fig)


# =============================================================================
# TEST 2 - IS THE BIOLOGY KEPT?
#   For each biological factor: log2 fold change between groups, before and after.
#   Good result: fold changes almost unchanged (small median |change|), and the
#   number of significant metabolites similar. Note: when a difference is tiny
#   (e.g. KI vs WT), the correlation is low even for small changes - look at the
#   median |change| instead. Changes are expected where groups were unbalanced
#   over the run (e.g. APOE2 is 28% of segment A but 43% of segment B); there
#   the corrected fold change is the more trustworthy one.
#   SE = bootstrap uncertainty of each fold change: a change smaller than ~2 SE
#   is within noise.
# =============================================================================

comparisons = [("Group_Region", "Hippocmpus", "Cortex"),
               ("Genotype", "KI", "WT"),
               ("Gender", "M", "F"),
               ("line of APOE", "APOE4", "APOE3"),
               ("line of APOE", "APOE2", "APOE3")]

def bootstrap_se(values, g1, g2, n_boot=200):
    """Standard error of the median log2 fold change (all metabolites at once)."""
    a, b = np.log2(values[g1]), np.log2(values[g2])
    fcs = [np.median(a[rng_boot.integers(0, len(a), len(a))], axis=0)
           - np.median(b[rng_boot.integers(0, len(b), len(b))], axis=0) for _ in range(n_boot)]
    return np.std(fcs, axis=0)

rng_boot = np.random.default_rng(RANDOM_SEED)
test2_rows = []
for column, group1, group2 in comparisons:
    g1 = is_sample & (raw_data[column] == group1).to_numpy()
    g2 = is_sample & (raw_data[column] == group2).to_numpy()
    fc_b, fc_a, p_b, p_a = [], [], [], []
    for f in metabolites:
        b, a = raw[f].to_numpy(), corr[f].to_numpy()
        fc_b.append(np.log2(np.median(b[g1]) / np.median(b[g2])))
        fc_a.append(np.log2(np.median(a[g1]) / np.median(a[g2])))
        p_b.append(stats.mannwhitneyu(b[g1], b[g2]).pvalue)
        p_a.append(stats.mannwhitneyu(a[g1], a[g2]).pvalue)
        test2_rows.append([f"{group1} vs {group2}", f, fc_b[-1], fc_a[-1], p_b[-1], p_a[-1]])
    fc_b, fc_a = np.array(fc_b), np.array(fc_a)
    se = bootstrap_se(raw[metabolites].to_numpy(), g1, g2)
    summary += [
        ["2 Biology kept", f"{group1} vs {group2}: median uncertainty (SE) of log2 FC",
         "", np.median(se), "changes smaller than this are just noise"],
        ["2 Biology kept", f"{group1} vs {group2}: % of metabolites whose FC changed by more than 2 SE",
         "", 100 * np.mean(np.abs(fc_a - fc_b) > 2 * se), "low = biology kept"],
        ["2 Biology kept", f"{group1} vs {group2}: median |log2 FC|",
         np.median(np.abs(fc_b)), np.median(np.abs(fc_a)), "size of the biology"],
        ["2 Biology kept", f"{group1} vs {group2}: median |change in log2 FC|",
         "", np.median(np.abs(fc_a - fc_b)), "small compared with the line above"],
        ["2 Biology kept", f"{group1} vs {group2}: correlation of log2 FC before/after",
         "", np.corrcoef(fc_b, fc_a)[0, 1], "close to 1 when the biology is large"],
        ["2 Biology kept", f"{group1} vs {group2}: metabolites with p < 0.05",
         (np.array(p_b) < 0.05).sum(), (np.array(p_a) < 0.05).sum(), "should be similar"],
    ]
test2 = pd.DataFrame(test2_rows, columns=["Comparison", "Metabolite",
                                          "log2 FC before", "log2 FC after",
                                          "p before", "p after"])

fig, axes = plt.subplots(1, len(comparisons), figsize=(4 * len(comparisons), 4))
for ax, (column, group1, group2) in zip(axes, comparisons):
    part = test2[test2["Comparison"] == f"{group1} vs {group2}"]
    lim = np.abs(part[["log2 FC before", "log2 FC after"]].to_numpy()).max() * 1.1
    ax.scatter(part["log2 FC before"], part["log2 FC after"], s=10, color="#2a78d6")
    ax.plot([-lim, lim], [-lim, lim], color="#c3c2b7", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(f"{group1} vs {group2}", fontsize=9)
    ax.set_xlabel("log2 FC before"); ax.set_ylabel("log2 FC after")
fig.suptitle("TEST 2: biological differences before vs after (on the line = unchanged)")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_FOLDER, "test2_biology_kept.png"), dpi=150); plt.close(fig)


# =============================================================================
# TEST 3 - IS THE DATA LESS NOISY?
#   RSD% = SD / mean x 100.  Sample RSD should drop for drifting metabolites.
#   QC RSD is reported, but the QCs jump at segment B while the samples do not,
#   so QC RSD is NOT a fair judge of this correction.
# =============================================================================

def rsd(values):
    return np.std(values, ddof=1) / np.mean(values) * 100

test3 = pd.DataFrame({
    "Metabolite": features,
    "sample RSD% before": [rsd(raw[f][is_sample]) for f in features],
    "sample RSD% after":  [rsd(corr[f][is_sample]) for f in features],
    "QC RSD% before":     [rsd(raw[f][is_qc]) for f in features],
    "QC RSD% after":      [rsd(corr[f][is_qc]) for f in features],
})
drifting = np.abs(rho_before) >= 0.3
summary += [
    ["3 Less noise", "Median sample RSD%, all features",
     test3["sample RSD% before"].median(), test3["sample RSD% after"].median(), "should drop a little"],
    ["3 Less noise", "Median sample RSD%, features that drifted",
     test3["sample RSD% before"][drifting].median(), test3["sample RSD% after"][drifting].median(),
     "should drop clearly"],
    ["3 Less noise", "Median QC RSD% (for information only)",
     test3["QC RSD% before"].median(), test3["QC RSD% after"].median(),
     "QCs do not follow the samples"],
]


# =============================================================================
# TEST 4 - DO POS AND NEG AGREE BETTER?
#   The same compound measured in both modes should correlate across samples.
#   Drift differs between modes, so removing it should raise the correlation.
# =============================================================================

for pos, neg in DUAL_MODE.items():
    r_b = np.corrcoef(np.log2(raw[pos][is_sample]),  np.log2(raw[neg][is_sample]))[0, 1]
    r_a = np.corrcoef(np.log2(corr[pos][is_sample]), np.log2(corr[neg][is_sample]))[0, 1]
    summary.append(["4 POS-NEG agreement", f"{pos[4:]}: r between modes", r_b, r_a, "should rise"])


# =============================================================================
# TEST 5 - DRIFT IN THE BIG PICTURE (PCA)
#   PCA on log2, autoscaled metabolites (samples). Good result: no principal
#   component is related to injection order after correction.
# =============================================================================

def pca_scores(table):
    z = np.log2(table.loc[is_sample, metabolites].to_numpy())
    z = (z - z.mean(0)) / z.std(0)
    u, s, _ = np.linalg.svd(z, full_matrices=False)
    return u[:, :3] * s[:3], s ** 2 / (s ** 2).sum() * 100

scores_b, var_b = pca_scores(raw)
scores_a, var_a = pca_scores(corr)
for i in range(3):
    summary.append(["5 PCA", f"PC{i+1}: |rho| with injection order",
                    abs(stats.spearmanr(injection[is_sample], scores_b[:, i])[0]),
                    abs(stats.spearmanr(injection[is_sample], scores_a[:, i])[0]),
                    "should be < 0.16"])

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, sc, var, title in ((axes[0], scores_b, var_b, "before"), (axes[1], scores_a, var_a, "after")):
    points = ax.scatter(sc[:, 0], sc[:, 1], c=injection[is_sample], cmap="viridis", s=14)
    ax.set_xlabel(f"PC1 ({var[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({var[1]:.0f}%)")
    ax.set_title(f"PCA {title} (colour = injection #)")
fig.colorbar(points, ax=axes, label="Injection #")
fig.savefig(os.path.join(OUTPUT_FOLDER, "test5_pca.png"), dpi=150); plt.close(fig)


# =============================================================================
# The correction method (same as drift_correction.py), needed for TEST 6 and 7
# =============================================================================

def spline_columns(x, start, end):
    knots = np.r_[[start] * 4, np.linspace(start, end, 4)[1:-1], [end] * 4]
    return BSpline.design_matrix(np.clip(x, start, end), knots, 3).toarray()

def make_drift_columns(inj, seg_labels):
    blocks = []
    for seg in ["A", "B"]:
        in_seg = seg_labels == seg
        block = np.zeros((len(inj), 6))
        block[in_seg] = spline_columns(inj[in_seg], inj[in_seg].min(), inj[in_seg].max())
        blocks.append(block)
    return np.hstack(blocks)

BIOLOGY = pd.get_dummies(raw_data[COL_BIOLOGY].fillna("QC")).astype(float).to_numpy()

def run_correction(table, inj, seg_labels):
    DRIFT = make_drift_columns(inj, seg_labels)
    X = np.hstack([DRIFT, BIOLOGY])[is_sample]
    out = table.copy()
    for name in table.columns:
        y = np.log2(table[name].to_numpy())
        coef = np.linalg.lstsq(X, y[is_sample], rcond=None)[0]
        resid = y[is_sample] - X @ coef
        mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        keep = np.abs(resid) < 4 * mad
        coef = np.linalg.lstsq(X[keep], y[is_sample][keep], rcond=None)[0]
        drift = DRIFT @ coef[:DRIFT.shape[1]]
        out[name] = 2 ** (y - (drift - drift[is_sample].mean()))
    return out


# =============================================================================
# TEST 6 - ACCURACY WITH A KNOWN DRIFT (simulation)
#   Start from the corrected data (no drift left), ADD a drift we invent and
#   therefore know exactly, run the correction, and compare with the start.
#   Three kinds of fake drift: straight decline, curved decline + jump at B,
#   a rising drift, and a sudden drop in the middle of segment A (a hard case:
#   the smooth curve cannot follow a sharp step exactly).
#   Accuracy = how much of the fake drift is removed.
# =============================================================================

rng = np.random.default_rng(RANDOM_SEED)
pos = (injection - injection.min()) / (injection.max() - injection.min())   # 0 ... 1
fake_drifts = {
    "straight 40% decline":      np.log2(1.2 - 0.4 * pos),
    "curved decline + jump at B": np.log2(1.3 - 0.5 * pos ** 0.5 + 0.2 * (segment == "B")),
    "rising 30%":                np.log2(0.85 + 0.3 * pos),
    "sudden 25% drop at inj 60 (hard case)": np.log2(np.where(injection < 60, 1.12, 0.88)),
}

test6_rows = []
for label, log_drift in fake_drifts.items():
    # give each metabolite its own size of drift (between 0.5x and 1.5x)
    strength = rng.uniform(0.5, 1.5, len(metabolites))
    truth    = corr[metabolites]
    drifted  = truth * 2 ** (np.outer(log_drift, strength))
    fixed    = run_correction(drifted, injection, segment)

    for j, f in enumerate(metabolites):
        t, d, c = (np.log2(x[f].to_numpy()[is_sample]) for x in (truth, drifted, fixed))
        added   = d - t
        left    = c - t
        test6_rows.append([label, f,
                           np.std(added - added.mean()),                       # drift added
                           np.std(left - left.mean()),                         # drift left
                           stats.spearmanr(injection[is_sample], 2 ** d)[0],
                           stats.spearmanr(injection[is_sample], 2 ** c)[0]])
test6 = pd.DataFrame(test6_rows, columns=["Fake drift", "Metabolite",
                                          "SD of added drift (log2)", "SD of error left (log2)",
                                          "rho with drift", "rho after correction"])
test6["% of drift removed"] = 100 * (1 - test6["SD of error left (log2)"]
                                     / test6["SD of added drift (log2)"])
for label in fake_drifts:
    part = test6[test6["Fake drift"] == label]
    summary.append(["6 Known-drift accuracy", f"{label}: median % of drift removed",
                    "", part["% of drift removed"].median(), "higher = better (> 80% good)"])
    summary.append(["6 Known-drift accuracy", f"{label}: features still |rho| >= 0.3",
                    (part["rho with drift"].abs() >= 0.3).sum(),
                    (part["rho after correction"].abs() >= 0.3).sum(), "after should be 0"])


# =============================================================================
# TEST 7 - OVER-FITTING CHECK (shuffled injection order)
#   Give the samples a random position in the run (injection # and segment
#   shuffled together) so there is no real drift to find, run the correction,
#   and repeat 20 times.
#   A good method then:
#     - removes almost nothing (RSD barely changes, much less than with the
#       real order, especially for the metabolites that really drift), and
#     - leaves the real drift in place (features with |rho| >= 0.3 stay).
#   A small RSD drop with shuffled order is normal: 12 curve parameters fitted
#   to 149 samples always absorb a little noise.
# =============================================================================

n_shuffles = 20
rsd_raw  = np.array([rsd(raw[f][is_sample]) for f in metabolites])
rsd_real = np.array([rsd(corr[f][is_sample]) for f in metabolites])
drift_met = np.array([abs(stats.spearmanr(injection[is_sample], raw[f][is_sample])[0]) >= 0.3
                      for f in metabolites])

drop_all, drop_drift, still_drifting = [], [], []
for _ in range(n_shuffles):
    order = rng.permutation(len(injection))
    fake_fix = run_correction(raw[metabolites], injection[order], segment[order])
    rsd_fake = np.array([rsd(fake_fix[f][is_sample]) for f in metabolites])
    drop_all.append(np.median(rsd_raw - rsd_fake))
    drop_drift.append(np.median((rsd_raw - rsd_fake)[drift_met]))
    still_drifting.append(sum(abs(stats.spearmanr(injection[is_sample], fake_fix[f][is_sample])[0]) >= 0.3
                              for f in metabolites))

summary += [
    ["7 Over-fitting", "Median RSD drop, all metabolites (%-points): real vs shuffled",
     np.median(rsd_raw - rsd_real), np.mean(drop_all), "Before = real order, After = shuffled"],
    ["7 Over-fitting", "Median RSD drop, drifting metabolites (%-points): real vs shuffled",
     np.median((rsd_raw - rsd_real)[drift_met]), np.mean(drop_drift), "shuffled should be much smaller"],
    ["7 Over-fitting", "Metabolites still |rho| >= 0.3 after correction: real vs shuffled",
     sum(abs(test1.set_index("Metabolite").loc[metabolites, "rho after"]) >= 0.3),
     np.mean(still_drifting), "shuffled should keep the real drift"],
]


# =============================================================================
# STEP 8 - SAVE AND PRINT
# =============================================================================

summary = pd.DataFrame(summary, columns=["Test", "Measure", "Before", "After", "Good result"])
with pd.ExcelWriter(os.path.join(OUTPUT_FOLDER, "Drift_evaluation.xlsx")) as excel:
    summary.to_excel(excel, sheet_name="Summary", index=False)
    test1.to_excel(excel, sheet_name="1_Drift_rho", index=False)
    test2.to_excel(excel, sheet_name="2_Biology", index=False)
    test3.to_excel(excel, sheet_name="3_RSD", index=False)
    test6.to_excel(excel, sheet_name="6_Known_drift", index=False)

pd.set_option("display.width", 200, "display.max_colwidth", 60)
print(summary.round(3).to_string(index=False))
print(f"\nSaved results and figures in {OUTPUT_FOLDER}")
