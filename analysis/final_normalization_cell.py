# ============================================================
# FINAL NORMALIZATION
#   - no IS normalization, no protein normalization
#   - PQN per ionization mode (POS, NEG), with a separate reference per brain region
#       reference = median profile of the WT samples of that region (same reference for KI and WT)
#       PQN factors are computed from metabolites detected in >= 90% of samples
#       (QCs are NOT used to choose them: QCs drifted and the first vial was poor)
#   - QC: normalized against the median profile of all samples
#   - reliability flag per metabolite (all metabolites are kept):
#       robust QC RSD within batch segment (A/B jump removed, MAD-based,
#       bad QC vials can be excluded) + D-ratio (QC spread vs sample spread)
#   - log2, save
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
CTX_LABEL  = 'Cortex'        # label of cortex in REGION_COL (only cortex gets the 1.31 factor;
                             # all regions are found automatically for PQN)
GENO_COL   = 'Genotype'      # column with KI vs WT
WT_LABEL   = 'WT'            # label of WT in GENO_COL (PQN reference group)
ID_COL     = 'Sample ID'
QC_EXCLUDE = []              # Sample IDs of bad QC vials to leave out of reliability stats,
                             # e.g. the first vial -> fill in after looking at the QC table below
MIN_DETECT = 0.90            # PQN factors use metabolites detected in >= 90% of samples
QC_RSD_MAX = 30              # %: robust within-batch QC RSD above this ...
D_RATIO_MAX = 50             # %: ... AND D-ratio above this -> flagged 'low' (still kept)
CTX_FACTOR = {'main': 1.31, 'low': 1.10, 'high': 1.56}

# ---------- load ----------
df = pd.read_csv(input_path)
metabolite_cols = [c for c in df.columns if c.startswith(('POS_', 'NEG_')) and '13C' not in c]
is_cols = [c for c in df.columns if '13C' in c]
qc_mask = df['Type'] == 'Quality control'
s_mask  = df['Type'] == 'Sample'

# strip stray spaces in the label columns ('Cortex ' -> 'Cortex')
for c in (REGION_COL, GENO_COL):
    assert c in df.columns, f"'{c}' column not found. Columns: {list(df.columns[:12])} ..."
    df[c] = df[c].astype(str).str.strip().where(df[c].notna())

regions  = sorted(df.loc[s_mask, REGION_COL].dropna().unique())
genotypes = sorted(df.loc[s_mask, GENO_COL].dropna().unique())
print("Regions found:  ", regions)
print("Genotypes found:", genotypes)

assert df.index.is_unique
assert df.loc[s_mask, REGION_COL].notna().all(), "some samples have no region"
assert len(regions) == 2, f"expected 2 regions, found {regions}"
assert CTX_LABEL in regions, f"CTX_LABEL = '{CTX_LABEL}' not in {regions}: copy the exact label"
assert WT_LABEL in genotypes, f"WT_LABEL = '{WT_LABEL}' not in {genotypes}: copy the exact label"
wt_mask = s_mask & (df[GENO_COL] == WT_LABEL)

batch_cols = [c for c in df.columns if str(c).strip().lower().startswith('batch segment')]
if batch_cols and df.loc[qc_mask, batch_cols[0]].notna().all():
    batch = df[batch_cols[0]].astype(str).str.strip()
    print(f"Batch column: '{batch_cols[0]}' -> {batch[qc_mask].value_counts().to_dict()} QCs per batch")
else:
    batch = pd.Series('all', index=df.index)
    print("No batch segment found for all QCs: QC stats use one batch")
qc_use = qc_mask & ~df[ID_COL].astype(str).str.strip().isin([str(x).strip() for x in QC_EXCLUDE])
print(f"QCs used for reliability stats: {qc_use.sum()} of {qc_mask.sum()} (excluded: {QC_EXCLUDE})")
print(f"{s_mask.sum()} samples, {qc_mask.sum()} QCs, {len(metabolite_cols)} metabolites")
print("Samples per region:", df.loc[s_mask, REGION_COL].value_counts().to_dict())


def rsd(s):
    return 100 * s.std(ddof=1) / s.mean()


def robust_sd_log2(x):
    """MAD-based SD of log2 values (insensitive to a few bad points)."""
    x = np.log2(x.where(x > 0)).dropna()
    return 1.4826 * (x - x.median()).abs().median()


def centered(data, groups):
    """Divide each value by the median of its group (removes batch / region level)."""
    return data / data.groupby(groups).transform('median')


def pqn(data, cols, ref, factor_cols):
    """Divide each row by the median quotient to ref, using only factor_cols (> 0 in ref)."""
    use = [c for c in factor_cols if pd.notna(ref[c]) and ref[c] > 0]
    q = data[use].where(data[use] > 0).div(ref[use])          # zeros/negatives ignored
    factor = q.median(axis=1, skipna=True)
    return data[cols].div(factor, axis=0), factor


# ---------- PQN ----------
raw_qc_rsd = df.loc[qc_mask, metabolite_cols].apply(rsd)
detect = (df.loc[s_mask, metabolite_cols] > 0).mean()
df_norm = df.copy()

for mode in ['POS', 'NEG']:
    cols = [c for c in metabolite_cols if c.startswith(mode + '_')]
    factor_cols = [c for c in cols if detect[c] >= MIN_DETECT]
    fcol = f'PQN_factor_{mode}'
    print(f"\n{mode}: {len(cols)} metabolites, {len(factor_cols)} used for PQN factors "
          f"(detected in >= {MIN_DETECT:.0%} of samples)")

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

# ---------- QC vials: find bad ones (fill QC_EXCLUDE above, then re-run) ----------
# A/B jump per metabolite, estimated from the many samples (median per region, then across regions)
s_vals = df_norm.loc[s_mask, metabolite_cols]
s_reg, s_bat = df.loc[s_mask, REGION_COL], batch[s_mask]
shift = pd.DataFrame({
    b: pd.concat([s_vals[(s_reg == r) & (s_bat == b)].median() / s_vals[s_reg == r].median()
                  for r in regions if ((s_reg == r) & (s_bat == b)).any()], axis=1).median(axis=1)
    for b in sorted(batch[qc_mask].unique())}).T            # rows = batch, cols = metabolites
shift = shift.reindex(sorted(batch[qc_mask].unique())).fillna(1.0)

qcn = df_norm.loc[qc_mask, metabolite_cols]
qc_corr = qcn / shift.loc[batch[qc_mask]].values           # QCs with the A/B jump removed
qc_corr = qc_corr.where(qc_corr > 0)
dev = np.log2(qc_corr / qc_corr.median()).abs().median(axis=1)
inj_col = next((c for c in df.columns if str(c).strip().lower().startswith('injection')), None)
qc_view = pd.DataFrame({'Sample ID': df.loc[qc_mask, ID_COL],
                        'injection': df.loc[qc_mask, inj_col] if inj_col else np.nan,
                        'batch': batch[qc_mask],
                        'median |log2 dev| from QC median': dev.round(3)})
print("\nQC vials (a vial far above the others is a candidate for QC_EXCLUDE):")
print(qc_view.sort_values('injection').to_string(index=False))

# ---------- reliability per metabolite: flag, do NOT remove ----------
#   QC_RSD_robust_%: MAD-based RSD of QCs after removing the A/B jump (estimated from samples)
#                    (insensitive to the jump and to one or two bad vials)
#   D_ratio_%:       QC spread / sample spread (samples centered per region);
#                    < 50% = technical noise is small compared with the biological signal
qc_c = qc_corr.loc[qc_use[qc_mask].values]                 # jump-corrected, bad vials excluded
s_c  = centered(df_norm.loc[s_mask, metabolite_cols], df.loc[s_mask, REGION_COL])
sd_qc = qc_c.apply(robust_sd_log2)
sd_s  = s_c.apply(robust_sd_log2)
qc_rsd_robust = 100 * np.sqrt(np.exp((sd_qc * np.log(2)) ** 2) - 1)   # log-SD -> RSD
d_ratio = 100 * sd_qc / sd_s
qc_rsd = df_norm.loc[qc_mask, metabolite_cols].apply(rsd)

reliability = np.select(
    [qc_rsd_robust <= QC_RSD_MAX, d_ratio <= D_RATIO_MAX],
    ['good', f'acceptable (QC RSD > {QC_RSD_MAX}% but D-ratio <= {D_RATIO_MAX}%)'],
    default='low')
qc_table = pd.DataFrame({'Metabolite': metabolite_cols,
                         'QC_RSD_raw_%': raw_qc_rsd.round(1).values,
                         'QC_RSD_afterPQN_%': qc_rsd.round(1).values,
                         'QC_RSD_robust_%': qc_rsd_robust.round(1).values,
                         'D_ratio_%': d_ratio.round(1).values,
                         'Reliability': reliability})
not_good = qc_table[qc_table['Reliability'] != 'good'].sort_values('QC_RSD_robust_%', ascending=False)
print(f"\nKept all {len(metabolite_cols)} metabolites; "
      f"{(qc_table['Reliability'] == 'good').sum()} good, "
      f"{qc_table['Reliability'].str.startswith('acceptable').sum()} acceptable, "
      f"{(qc_table['Reliability'] == 'low').sum()} low:")
print(not_good.to_string(index=False))
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
print("Saved QC RSD, robust QC RSD, D-ratio and reliability per metabolite to metabolite_QC_RSD_flags.csv")
