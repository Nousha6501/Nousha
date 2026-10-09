"""
Per-metabolite drift correction for targeted LC-MS metabolomics
================================================================

Idea
----
In a randomized run, biology is not related to injection order. So if a
metabolite steadily rises or falls with injection number, that trend is
technical (instrument, column, sample stability in the autosampler).

For EACH metabolite separately we fit, on the study samples:

    log2(value) = drift(injection #)               <- smooth curve, one per segment (A, B)
                + region + sex + genotype + APOE   <- biology, kept in the data
                + noise

and then subtract only the drift curve. The biological terms are in the
model so that the drift estimate is not confused with group differences
(the groups are not perfectly balanced between segment A and B).

Order of processing:  raw data -> drift correction -> PQN -> log2

Usage
-----
    python drift_correction.py master.csv output_folder/

Requirements: numpy, pandas, scipy (>= 1.8)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline

# ----------------------------------------------------------------------------
# Settings: column names in the master file
# ----------------------------------------------------------------------------
ID_COL = "Sample ID"
INJ_COL = "injection #"
TYPE_COL = "Type"                       # "Sample" or "Quality control"
SEG_COL = "Batch segment( before jump=A, after jump=B)"
BIO_COLS = ["Group_Region", "Gender", "Genotype", "line of APOE"]
FEATURE_PREFIXES = ("POS_", "NEG_")
IS_TAG = "13C"                          # internal standards contain "13C"

SPLINE_INNER_KNOTS = 2                  # flexibility of the drift curve per segment
OUTLIER_MAD = 4                         # points > 4 MAD from the fit are ignored in a refit
EXCLUDE_QC = ["QC_1", "QC_2"]           # conditioning injections, not used for checks

# Corrections confirmed by the user (wrong injection numbers in the master file)
INJECTION_FIXES = {"40H_AP4_M_KI_131": 178, "39H_AP2_M_KI_104": 180}


# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
def load_master(path):
    df = pd.read_csv(path)

    for sample_id, inj in INJECTION_FIXES.items():
        df.loc[df[ID_COL] == sample_id, INJ_COL] = inj

    # Empty segment -> decide from position (segment B starts at injection 124)
    missing = df[SEG_COL].isna()
    df.loc[missing, SEG_COL] = np.where(df.loc[missing, INJ_COL] >= 124, "B", "A")

    if not df[INJ_COL].is_unique:
        raise ValueError("Injection numbers are not unique - fix the master file first.")

    return df.sort_values(INJ_COL).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Design matrix
# ----------------------------------------------------------------------------
def spline_basis(x, lo, hi, n_inner=SPLINE_INNER_KNOTS, degree=3):
    """Cubic B-spline basis on [lo, hi]: a set of smooth 'bumps' whose weighted
    sum can follow a gradual or curved trend over injection order."""
    inner = np.linspace(lo, hi, n_inner + 2)[1:-1]
    knots = np.r_[[lo] * (degree + 1), inner, [hi] * (degree + 1)]
    return BSpline.design_matrix(np.clip(x, lo, hi), knots, degree).toarray()


def drift_design(injection, segment):
    """Drift part of the model: a separate smooth curve for segment A and for
    segment B, so the jump between segments is also captured."""
    blocks = []
    for seg in sorted(pd.unique(segment)):
        in_seg = segment == seg
        lo, hi = injection[in_seg].min(), injection[in_seg].max()
        n_basis = SPLINE_INNER_KNOTS + 4
        block = np.zeros((len(injection), n_basis))
        block[in_seg] = spline_basis(injection[in_seg], lo, hi)
        blocks.append(block)
    return np.hstack(blocks)


def biology_design(df):
    """Biological part of the model: one indicator column per level of
    region, sex, genotype and APOE."""
    return pd.get_dummies(df[BIO_COLS].fillna("QC")).astype(float).values


# ----------------------------------------------------------------------------
# 3. Fit and remove drift, one metabolite at a time
# ----------------------------------------------------------------------------
def fit_drift_one_metabolite(log_values, D_drift, D_bio, is_sample):
    """Return the drift curve (log2 scale) for every injection, samples and QCs.

    log_values : log2 intensities of ONE metabolite for all injections
    D_drift    : spline columns (injection order)
    D_bio      : biology columns
    is_sample  : True for study samples (only these are used to fit)
    """
    X = np.hstack([D_drift[is_sample], D_bio[is_sample]])
    y = log_values[is_sample]
    n_drift = D_drift.shape[1]

    # Least-squares fit of drift + biology together
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Robust step: refit without gross outliers (they stay in the data, they
    # are just not allowed to pull the curve)
    resid = y - X @ beta
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    keep = np.abs(resid) < OUTLIER_MAD * mad
    beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]

    # Keep only the drift coefficients -> drift curve for ALL injections
    drift = D_drift @ beta[:n_drift]

    # Center so the average sample is unchanged (correction only removes the trend)
    return drift - drift[is_sample].mean()


def correct_drift(df, features):
    injection = df[INJ_COL].to_numpy(float)
    segment = df[SEG_COL].to_numpy()
    is_sample = (df[TYPE_COL] == "Sample").to_numpy()

    D_drift = drift_design(injection, segment)
    D_bio = biology_design(df)

    log_raw = np.log2(df[features].to_numpy(float))
    drift = np.column_stack([
        fit_drift_one_metabolite(log_raw[:, j], D_drift, D_bio, is_sample)
        for j in range(len(features))
    ])

    corrected = pd.DataFrame(2 ** (log_raw - drift), columns=features, index=df.index)
    drift_factor = pd.DataFrame(2 ** drift, columns=features, index=df.index)
    return corrected, drift_factor        # corrected = raw / drift_factor


# ----------------------------------------------------------------------------
# 4. PQN normalization and log2 (after drift correction)
# ----------------------------------------------------------------------------
def pqn_log2(corrected, metabolites, is_sample):
    """PQN: reference = median study sample; each sample's factor = median of
    its metabolite/reference ratios. Internal standards are not used for the
    factor. Returns log2(corrected / factor) and the factors."""
    reference = corrected.loc[is_sample, metabolites].median()
    factor = (corrected[metabolites] / reference).median(axis=1)
    return np.log2(corrected.div(factor, axis=0)), factor


# ----------------------------------------------------------------------------
# 5. Checks
# ----------------------------------------------------------------------------
def rsd(x):
    return np.std(x, ddof=1) / np.mean(x) * 100


def validation_table(df, raw, corrected, features):
    injection = df[INJ_COL].to_numpy(float)
    is_sample = (df[TYPE_COL] == "Sample").to_numpy()
    is_qc = ((df[TYPE_COL] == "Quality control") & ~df[ID_COL].isin(EXCLUDE_QC)).to_numpy()
    region = df["Group_Region"].to_numpy()
    hip = is_sample & (region == "Hippocmpus")
    ctx = is_sample & (region == "Cortex")

    rows = []
    for f in features:
        b, a = raw[f].to_numpy(float), corrected[f].to_numpy(float)
        rows.append({
            "Metabolite": f,
            "rho_vs_injection_before": stats.spearmanr(injection[is_sample], b[is_sample])[0],
            "rho_vs_injection_after": stats.spearmanr(injection[is_sample], a[is_sample])[0],
            "sample_RSD_before": rsd(b[is_sample]),
            "sample_RSD_after": rsd(a[is_sample]),
            "QC_RSD_before": rsd(b[is_qc]),
            "QC_RSD_after": rsd(a[is_qc]),
            # biology must survive: hippocampus vs cortex difference
            "log2_Hip_vs_Cortex_before": np.log2(np.median(b[hip]) / np.median(b[ctx])),
            "log2_Hip_vs_Cortex_after": np.log2(np.median(a[hip]) / np.median(a[ctx])),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(master_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_master(master_path)
    features = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    metabolites = [c for c in features if IS_TAG not in c]
    meta = [c for c in df.columns if c not in features]
    is_sample = (df[TYPE_COL] == "Sample").to_numpy()

    corrected, drift_factor = correct_drift(df, features)
    pqn, pqn_factor = pqn_log2(corrected, metabolites, is_sample)
    checks = validation_table(df, df[features], corrected, features)

    pd.concat([df[meta], corrected], axis=1).to_csv(out / "corrected_raw.csv", index=False)
    pd.concat([df[meta], pqn_factor.rename("PQN_factor"), pqn], axis=1).to_csv(
        out / "corrected_PQN_log2.csv", index=False)
    pd.concat([df[meta], drift_factor], axis=1).to_csv(out / "drift_factor.csv", index=False)
    checks.to_csv(out / "validation.csv", index=False)

    n_before = (checks["rho_vs_injection_before"].abs() >= 0.3).sum()
    n_after = (checks["rho_vs_injection_after"].abs() >= 0.3).sum()
    print(f"Features with clear drift (|rho| >= 0.3): {n_before} before, {n_after} after")
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python drift_correction.py master.csv output_folder/")
    main(sys.argv[1], sys.argv[2])
