# ══════════════════════════════════════════════════════════════════
# PROTEIN NORMALIZATION ONLY (Pr_Con)
#   value_norm = raw value / protein concentration of that sample  (mg/ml)
#   then multiplied by the median protein of all samples, so the numbers stay in the
#   original intensity range (ratios between samples are not changed by this)
#   - no PQN, no IS normalization, no 1.31 factor
#   - samples only (QCs have no protein value)
#   - exclusions: 9-12 H/C (homogenization issue) and 23H_AP4_F_KI_162
#   - saves: linear values + log2 values + per-sample protein table
# Note: Pr_Con did not correlate with the PQN dilution factors within region
# (rho ~0.07), so compare the before/after CV printed below before relying on this file.
# ══════════════════════════════════════════════════════════════════
import os
import numpy as np
import pandas as pd

input_path = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\master_with_QC3.csv"
output_dir = r"O:\metabolom\Result\new version result_09.2026"

PROT_COL   = 'Pr_Con'
REGION_COL = 'Region of brain'
EXCL_PREFIXES = ('9H_', '10H_', '11H_', '12H_', '9C_', '10C_', '11C_', '12C_')
EXCL_SAMPLES  = ['23H_AP4_F_KI_162']

# ---------- load, samples only ----------
df = pd.read_csv(input_path)
df.columns = [' '.join(str(c).split()) for c in df.columns]          # remove line breaks in names
metabolite_cols = [c for c in df.columns if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
is_cols = [c for c in df.columns if '13C' in c]
assert metabolite_cols, "no POS_/NEG_ metabolite columns found"
assert PROT_COL in df.columns, f"'{PROT_COL}' column not found"

df = df[df['Type'].astype(str).str.strip() == 'Sample'].copy()
df[REGION_COL] = df[REGION_COL].astype(str).str.strip()

# ---------- exclusions ----------
sid = df['Sample ID'].astype(str).str.strip()
drop = sid.str.startswith(EXCL_PREFIXES) | sid.isin(EXCL_SAMPLES)
missing = [s for s in EXCL_SAMPLES if s not in set(sid)]
if missing:
    print(f"WARNING: not found (check spelling): {missing}")
print(f"Samples in file: {len(df)} | excluded {drop.sum()}: {sorted(sid[drop])}")
df = df[~drop].reset_index(drop=True)

# ---------- parse protein ('1.8 mg/ml', '3.45mg/ml', '1,8 mg/ml' -> 1.8) ----------
prot = pd.to_numeric(df[PROT_COL].astype(str).str.replace(',', '.')
                     .str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
bad = prot.isna() | (prot <= 0)
if bad.any():
    print(f"WARNING: no usable protein value for {bad.sum()} samples (left as NaN in the output):")
    print(df.loc[bad, ['Sample ID', PROT_COL]].to_string(index=False))
df['Protein_mg_ml'] = prot.where(~bad)
print("Protein (mg/ml) by region:")
print(df.groupby(REGION_COL)['Protein_mg_ml'].describe()[['count', 'mean', 'min', 'max']].round(2).to_string())

# ---------- normalize ----------
raw = df[metabolite_cols].apply(pd.to_numeric, errors='coerce')
scale = df['Protein_mg_ml'].median()
norm = raw.div(df['Protein_mg_ml'], axis=0) * scale

# ---------- quality check: does protein reduce the sample-to-sample spread? ----------
# total-signal CV within region: lower after normalization = protein captured real dilution
def cv(x):
    x = x.dropna()
    return 100 * x.std() / x.mean()
ok = df['Protein_mg_ml'].notna()                       # same samples before and after
print("\nCV (%) of the summed signal per sample, within region (lower = better):")
for r, idx in df[ok].groupby(REGION_COL).groups.items():
    print(f"  {r}: raw {cv(raw.loc[idx].sum(axis=1, min_count=1)):.1f}%  ->  protein-normalized "
          f"{cv(norm.loc[idx].sum(axis=1, min_count=1)):.1f}%")

# ---------- save ----------
meta_cols = [c for c in df.columns if c not in metabolite_cols + is_cols]
out_lin = pd.concat([df[meta_cols], norm], axis=1)
out_log = out_lin.copy()
n_bad = int((norm <= 0).sum().sum())
if n_bad:
    print(f"\nWARNING: {n_bad} values <= 0 set to NaN before log2 (no imputation)")
out_log[metabolite_cols] = np.log2(norm.where(norm > 0))

p_lin = os.path.join(output_dir, 'ProteinNorm_samples.csv')
p_log = os.path.join(output_dir, 'ProteinNorm_log2_samples.csv')
p_prot = os.path.join(output_dir, 'ProteinNorm_protein_per_sample.csv')
try:
    out_lin.to_csv(p_lin, index=False)
    out_log.to_csv(p_log, index=False)
    df[['Sample ID', REGION_COL, PROT_COL, 'Protein_mg_ml']].to_csv(p_prot, index=False)
except PermissionError:
    raise PermissionError("Cannot write the output: close the file if it is open in Excel, "
                          "or check that you can write to output_dir") from None

print(f"\nSaved {len(out_lin)} samples x {len(metabolite_cols)} metabolites")
print(f"  linear (raw / protein x {scale:.2f}): {p_lin}")
print(f"  log2:                               {p_log}")
print(f"  protein per sample:                 {p_prot}")
