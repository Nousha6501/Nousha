# ============================================================
# FINAL NORMALIZATION
#   - no IS normalization, no protein normalization
#   - PQN per ionization mode (POS, NEG), with a separate reference per brain region
#       reference = median profile of the WT samples of that region (same reference for KI and WT)
#       PQN factors are computed from reproducible metabolites only (raw QC RSD <= 30%)
#   - QC: normalized against the median profile of all samples
#   - QC RSD: flag metabolites (all are kept), log2, save
#
# What PQN within region does and does NOT do:
#   * removes the dilution differences BETWEEN SAMPLES OF THE SAME REGION
#     (tissue 90-150 mg CTX / 30-40 mg Hip in a fixed solvent volume)
#   * keeps each region at its raw overall level, which still includes the
#     ~1.31x higher tissue concentration of CTX extracts
#     (pilot weights n = 6: CTX 121/1800 uL vs Hip 38/750 uL; 95% CI 1.10-1.56)
#   -> KI vs WT within a region, and log2(KI/WT) Hip vs CTX: valid on the main file
#   -> absolute CTX vs Hip: use the CTX-adjusted files (main 1.31, sensitivity 1.10 / 1.56)
# ============================================================
import os
import numpy as np
import pandas as pd

input_path = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\master_with_QC3.csv"
output_dir = r"O:\metabolom\Result\new version result_09.2026"

REGION_COL = 'Region of brain'
CTX_LABEL  = 'Cortex'        # exact label of cortex in REGION_COL
GENO_COL   = 'Genotype'      # column with KI/KI vs WT/WT
WT_LABEL   = 'WT/WT'         # exact label of WT in GENO_COL (PQN reference group)
QC_RSD_MAX = 30              # %: metabolites above this are flagged and not used for PQN factors
CTX_FACTOR = {'main': 1.31, 'low': 1.10, 'high': 1.56}

# ---------- load ----------
df = pd.read_csv(input_path)
metabolite_cols = [c for c in df.columns if c.startswith(('POS_', 'NEG_')) and '13C' not in c]
is_cols = [c for c in df.columns if '13C' in c]
qc_mask = df['Type'] == 'Quality control'
s_mask  = df['Type'] == 'Sample'

assert df.index.is_unique
assert df.loc[s_mask, REGION_COL].notna().all(), "some samples have no region"
assert CTX_LABEL in set(df.loc[s_mask, REGION_COL]), f"'{CTX_LABEL}' not found in {REGION_COL}"
assert GENO_COL in df.columns, f"'{GENO_COL}' column not found"
assert WT_LABEL in set(df.loc[s_mask, GENO_COL]), f"'{WT_LABEL}' not found in {GENO_COL}"
wt_mask = s_mask & (df[GENO_COL] == WT_LABEL)
print(f"{s_mask.sum()} samples, {qc_mask.sum()} QCs, {len(metabolite_cols)} metabolites")
print("Samples per region:", df.loc[s_mask, REGION_COL].value_counts().to_dict())


def rsd(s):
    return 100 * s.std(ddof=1) / s.mean()


def pqn(data, cols, ref, factor_cols):
    """Divide each row by the median quotient to ref, using only factor_cols (> 0 in ref)."""
    use = [c for c in factor_cols if pd.notna(ref[c]) and ref[c] > 0]
    q = data[use].where(data[use] > 0).div(ref[use])          # zeros/negatives ignored
    factor = q.median(axis=1, skipna=True)
    return data[cols].div(factor, axis=0), factor


# ---------- PQN ----------
raw_qc_rsd = df.loc[qc_mask, metabolite_cols].apply(rsd)
df_norm = df.copy()

for mode in ['POS', 'NEG']:
    cols = [c for c in metabolite_cols if c.startswith(mode + '_')]
    factor_cols = [c for c in cols if raw_qc_rsd[c] <= QC_RSD_MAX]
    fcol = f'PQN_factor_{mode}'
    print(f"\n{mode}: {len(cols)} metabolites, {len(factor_cols)} used for PQN factors (raw QC RSD <= {QC_RSD_MAX}%)")

    # samples: one reference per region = median of the WT samples of that region,
    # applied to KI and WT alike
    for region, idx in df[s_mask].groupby(REGION_COL).groups.items():
        ref = df.loc[wt_mask & (df[REGION_COL] == region), cols].median()
        normed, factor = pqn(df.loc[idx], cols, ref, factor_cols)
        df_norm.loc[idx, cols] = normed
        df_norm.loc[idx, fcol] = factor

    # QCs (pooled, no region): reference = median of all samples
    ref_all = df.loc[s_mask, cols].median()
    normed, factor = pqn(df.loc[qc_mask], cols, ref_all, factor_cols)
    df_norm.loc[qc_mask, cols] = normed
    df_norm.loc[qc_mask, fcol] = factor

    f = df_norm.loc[s_mask, fcol]
    print(f"{mode}: PQN factor in samples {f.min():.2f}-{f.max():.2f}; "
          f"median by region (~1 by design): {f.groupby(df.loc[s_mask, REGION_COL]).median().round(2).to_dict()}")
    print(f"{mode}: median factor of WT by region (~1 by design): "
          f"{df_norm.loc[wt_mask, fcol].groupby(df.loc[wt_mask, REGION_COL]).median().round(2).to_dict()}")
    print(f"{mode}: QC median RSD raw {raw_qc_rsd[cols].median():.1f}% -> "
          f"after PQN {df_norm.loc[qc_mask, cols].apply(rsd).median():.1f}%")

# ---------- checks ----------
# 1) PQN factors should not differ between genotypes within a region
#    (if they do, PQN may have removed part of a global KI effect)
#    WT median factor is ~1 by design, so a KI median far from 1 = global KI shift
from scipy.stats import mannwhitneyu
print()
for mode in ['POS', 'NEG']:
    for region, d in df_norm[s_mask].groupby(REGION_COL):
        groups = {g: v[f'PQN_factor_{mode}'].dropna().values for g, v in d.groupby(GENO_COL)}
        p = mannwhitneyu(*groups.values()).pvalue if len(groups) == 2 else np.nan
        med = {g: round(float(np.median(v)), 2) for g, v in groups.items()}
        print(f"{mode} {region}: median PQN factor by genotype {med}  p = {p:.3f}  (want p > 0.05)")

# 2) POS and NEG come from the same extract -> their factors should correlate
r = df_norm.loc[s_mask, ['PQN_factor_POS', 'PQN_factor_NEG']].corr().iloc[0, 1]
print(f"POS vs NEG PQN factor correlation r = {r:.2f}  (expect high, e.g. > 0.7)")

# ---------- QC RSD after PQN: flag, do NOT remove ----------
qc_rsd = df_norm.loc[qc_mask, metabolite_cols].apply(rsd)
qc_table = pd.DataFrame({'Metabolite': metabolite_cols,
                         'QC_RSD_raw_%': raw_qc_rsd.round(1).values,
                         'QC_RSD_%': qc_rsd.round(1).values,
                         'Reliability': np.where(qc_rsd > QC_RSD_MAX,
                                                 f'low (QC RSD > {QC_RSD_MAX}%)', 'good')})
flagged = qc_table[qc_table['Reliability'] != 'good'].sort_values('QC_RSD_%', ascending=False)
print(f"\nKept all {len(metabolite_cols)} metabolites; {len(flagged)} flagged as low reliability:")
print(flagged.to_string(index=False))
qc_table.to_csv(os.path.join(output_dir, 'metabolite_QC_RSD_flags.csv'), index=False)

# ---------- log2 + save (samples only, no IS columns, all metabolites) ----------
meta_cols = [c for c in df.columns if c not in metabolite_cols + is_cols]
out = df_norm.loc[s_mask, meta_cols + ['PQN_factor_POS', 'PQN_factor_NEG'] + metabolite_cols].copy()

n_bad = int((out[metabolite_cols] <= 0).sum().sum())
if n_bad:
    print(f"\nWARNING: {n_bad} values <= 0 set to NaN before log2 (no imputation)")
out[metabolite_cols] = np.log2(out[metabolite_cols].where(out[metabolite_cols] > 0))

out_path = os.path.join(output_dir, 'PQNregion_log2_samples.csv')
out.to_csv(out_path, index=False)
print(f"\nSaved {out.shape[0]} samples x {len(metabolite_cols)} metabolites (log2, PQN within region) to {out_path}")
print("  -> use for KI vs WT within region and for log2(KI/WT) Hip vs CTX")

# ---------- CTX-adjusted copies, ONLY for absolute CTX vs Hip comparison ----------
is_ctx = out[REGION_COL] == CTX_LABEL
for k, fct in CTX_FACTOR.items():
    adj = out.copy()
    adj.loc[is_ctx, metabolite_cols] -= np.log2(fct)
    p = os.path.join(output_dir, f'PQNregion_log2_CTXadj_{k}.csv')
    adj.to_csv(p, index=False)
    print(f"Saved CTX / {fct} ({k}) to {os.path.basename(p)}  -> absolute CTX vs Hip only")
print("Saved QC RSD + reliability flag per metabolite to metabolite_QC_RSD_flags.csv")
