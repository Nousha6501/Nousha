# =============================================================================
#  DRIFT CORRECTION FOR LC-MS METABOLOMICS  (one metabolite at a time)
#  Order: raw data  ->  drift correction  ->  PQN  ->  log2
#
#  How to use:
#    1. Change the two paths in STEP 0 below.
#    2. Run the whole script (Spyder: F5, Jupyter: copy into one cell,
#       terminal: python drift_correction.py).
#  Needs: pandas, numpy, scipy, openpyxl
# =============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline


# =============================================================================
# STEP 0 - YOUR FILES  (edit only this part)
# =============================================================================

INPUT_FILE  = r"C:\Users\YourName\Documents\master_with_QC3_real_clean_23H_9101112H_C.csv"
OUTPUT_FILE = r"C:\Users\YourName\Documents\Drift_corrected_master.xlsx"

# Column names in your master file
COL_ID      = "Sample ID"
COL_INJ     = "injection #"
COL_TYPE    = "Type"                                         # "Sample" / "Quality control"
COL_SEGMENT = "Batch segment( before jump=A, after jump=B)"
COL_BIOLOGY = ["Group_Region", "Gender", "Genotype", "line of APOE"]

# Injection numbers that were wrong in the master file (confirmed)
INJECTION_FIXES = {"40H_AP4_M_KI_131": 178,
                   "39H_AP2_M_KI_104": 180}


# =============================================================================
# STEP 1 - LOAD AND CLEAN THE DATA
# =============================================================================

data = pd.read_csv(INPUT_FILE)

# fix injection numbers
for sample_id, injection in INJECTION_FIXES.items():
    data.loc[data[COL_ID] == sample_id, COL_INJ] = injection

# empty segment -> A or B by position (segment B starts at injection 124)
empty = data[COL_SEGMENT].isna()
data.loc[empty, COL_SEGMENT] = np.where(data.loc[empty, COL_INJ] >= 124, "B", "A")

# sort by injection order
data = data.sort_values(COL_INJ).reset_index(drop=True)

# which columns are what
features      = [c for c in data.columns if c.startswith(("POS_", "NEG_"))]   # all 122
internal_std  = [c for c in features if "13C" in c]                          # 11 IS
metabolites   = [c for c in features if "13C" not in c]                      # 111
info_columns  = [c for c in data.columns if c not in features]

is_sample = (data[COL_TYPE] == "Sample").to_numpy()
injection = data[COL_INJ].to_numpy(dtype=float)
segment   = data[COL_SEGMENT].to_numpy()

print(f"Loaded {len(data)} injections: {is_sample.sum()} samples, "
      f"{(~is_sample).sum()} QC; {len(features)} features")


# =============================================================================
# STEP 2 - BUILD THE MODEL COLUMNS
#
# For every metabolite we will fit:
#     log2(value) = DRIFT CURVE (injection #)  +  BIOLOGY (region, sex, genotype, APOE)
#
# DRIFT CURVE: a smooth curve over injection number, made of 6 small "bumps"
#              (cubic B-splines) in segment A and 6 in segment B, so it can follow
#              a slow decline, a curve and the jump between segments.
# BIOLOGY:     one 0/1 column per group level. It is in the model so that group
#              differences are not mistaken for drift - and it is NOT removed.
# =============================================================================

def spline_columns(x, start, end):
    """6 smooth bump columns covering injection numbers start..end."""
    knots = np.r_[[start] * 4, np.linspace(start, end, 4)[1:-1], [end] * 4]
    return BSpline.design_matrix(np.clip(x, start, end), knots, 3).toarray()

drift_columns = []
for seg in ["A", "B"]:
    in_seg = segment == seg
    block = np.zeros((len(data), 6))
    block[in_seg] = spline_columns(injection[in_seg],
                                   injection[in_seg].min(), injection[in_seg].max())
    drift_columns.append(block)
DRIFT = np.hstack(drift_columns)                                   # 12 columns

BIOLOGY = pd.get_dummies(data[COL_BIOLOGY].fillna("QC")).astype(float).to_numpy()

# the model is fitted on study samples only
X = np.hstack([DRIFT, BIOLOGY])[is_sample]
n_drift = DRIFT.shape[1]


# =============================================================================
# STEP 3 - DRIFT CORRECTION, ONE METABOLITE AT A TIME
# =============================================================================

raw         = data[features].astype(float)
log_raw     = np.log2(raw)
corrected   = raw.copy()
drift_factor = raw.copy()

for name in features:

    y = log_raw[name].to_numpy()

    # 3a. fit drift curve + biology together (least squares, samples only)
    coef = np.linalg.lstsq(X, y[is_sample], rcond=None)[0]

    # 3b. refit without strong outliers (> 4 MAD), so one odd sample
    #     cannot bend the curve (the sample itself stays in the data)
    resid = y[is_sample] - X @ coef
    mad   = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    keep  = np.abs(resid) < 4 * mad
    coef  = np.linalg.lstsq(X[keep], y[is_sample][keep], rcond=None)[0]

    # 3c. keep ONLY the drift part -> one drift value per injection (QCs too)
    drift = DRIFT @ coef[:n_drift]

    # 3d. center it, so the average sample does not change
    drift = drift - drift[is_sample].mean()

    # 3e. remove the drift   (log scale: subtract  =  raw scale: divide)
    corrected[name]    = 2 ** (y - drift)
    drift_factor[name] = 2 ** drift

print("Drift correction done for all features")


# =============================================================================
# STEP 4 - PQN NORMALIZATION, THEN log2
#   reference   = median of each metabolite over the study samples
#   PQN factor  = median of (sample / reference) over the 111 metabolites
#                 (internal standards are not used for the factor)
# =============================================================================

reference  = corrected.loc[is_sample, metabolites].median()
pqn_factor = (corrected[metabolites] / reference).median(axis=1)
pqn_log2   = np.log2(corrected.div(pqn_factor, axis=0))

print(f"PQN factors (samples): {pqn_factor[is_sample].min():.2f} - "
      f"{pqn_factor[is_sample].max():.2f}")


# =============================================================================
# STEP 5 - CHECK THE RESULT
#   rho           : Spearman correlation with injection order (drift; ~0 = none)
#   RSD           : variation between samples (should drop for drifting compounds)
#   Hip vs Cortex : biology must stay about the same
# =============================================================================

qc_for_check = (~is_sample) & ~data[COL_ID].isin(["QC_1", "QC_2"]).to_numpy()
hip = is_sample & (data["Group_Region"] == "Hippocmpus").to_numpy()
ctx = is_sample & (data["Group_Region"] == "Cortex").to_numpy()

def rsd(values):
    return np.std(values, ddof=1) / np.mean(values) * 100

rows = []
for name in features:
    before = raw[name].to_numpy()
    after  = corrected[name].to_numpy()
    rows.append({
        "Metabolite":            name,
        "rho before":            stats.spearmanr(injection[is_sample], before[is_sample])[0],
        "rho after":             stats.spearmanr(injection[is_sample], after[is_sample])[0],
        "sample RSD% before":    rsd(before[is_sample]),
        "sample RSD% after":     rsd(after[is_sample]),
        "QC RSD% before":        rsd(before[qc_for_check]),
        "QC RSD% after":         rsd(after[qc_for_check]),
        "log2 Hip/Cortex before": np.log2(np.median(before[hip]) / np.median(before[ctx])),
        "log2 Hip/Cortex after":  np.log2(np.median(after[hip]) / np.median(after[ctx])),
    })
validation = pd.DataFrame(rows)

print(f"Features with clear drift (|rho| >= 0.3): "
      f"{(validation['rho before'].abs() >= 0.3).sum()} before -> "
      f"{(validation['rho after'].abs() >= 0.3).sum()} after")


# =============================================================================
# STEP 6 - SAVE
# =============================================================================

info = data[info_columns]
with pd.ExcelWriter(OUTPUT_FILE) as excel:
    pd.concat([info, corrected], axis=1).to_excel(excel, sheet_name="Corrected_raw", index=False)
    pd.concat([info, pqn_factor.rename("PQN_factor"), pqn_log2], axis=1).to_excel(
        excel, sheet_name="Corrected_PQN_log2", index=False)
    pd.concat([info, drift_factor], axis=1).to_excel(excel, sheet_name="Drift_factor", index=False)
    validation.to_excel(excel, sheet_name="Validation", index=False)

print(f"Saved: {OUTPUT_FILE}")
