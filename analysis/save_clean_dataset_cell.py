# ══════════════════════════════════════════════════════════════════
# Save cleaned dataset with exclusions applied:
#   - documented homogenization-issue samples: 9, 10, 11, 12  (both H and C)
#   - outlier-flagged sample: 23H_AP4_F_KI_162  (hippocampus only; 23C is kept)
# KEPT (not excluded): 72H_AP4_M_KI_64, 21H_AP2_F_KI_192 / 21C_AP2_F_KI_192
# QC rows and all other columns are kept unchanged.
# Fully self-contained
# ══════════════════════════════════════════════════════════════════
import os
import pandas as pd

# ── 1. Load source data ──
file_path = r"C:\Users\nnekooiemarnany\OneDrive - UTHealth Houston\Desktop\Metabolom_analys\CSV2_Revised\RevisedScript\lasVersion_Pipline\Norm_Data_IS\ISnorm_clean_imputed_Data.csv"
output_path = r"C:\Users\nnekooiemarnany\OneDrive - UTHealth Houston\Desktop\Metabolom_analys\CSV2_Revised\RevisedScript\lasVersion_Pipline\Norm_Data_IS\ISnorm_clean_excl_homog9-12HC_23H.csv"

excl_prefixes = ('9H_', '10H_', '11H_', '12H_',      # homogenization issue, hippocampus
                 '9C_', '10C_', '11C_', '12C_')      # homogenization issue, cortex
excl_samples  = ['23H_AP4_F_KI_162']                 # outlier-flagged

df_raw = pd.read_csv(file_path)
sid = df_raw['Sample ID'].astype(str).str.strip()
n_start = len(df_raw)

# ── 2. Find the rows to exclude ──
by_prefix = sid.str.startswith(excl_prefixes)
by_name   = sid.isin([s.strip() for s in excl_samples])

missing = [s for s in excl_samples if s.strip() not in set(sid)]
if missing:
    print(f"WARNING: not found in the file (check spelling): {missing}")
for p in excl_prefixes:
    if not sid.str.startswith(p).any():
        print(f"NOTE: no sample starts with '{p}'")

print(f"Starting rows: {n_start}")
print(f"Homogenization-issue samples removed ({by_prefix.sum()}): {sorted(sid[by_prefix])}")
print(f"Outlier samples removed ({(by_name & ~by_prefix).sum()}): {sorted(sid[by_name & ~by_prefix])}")

# ── 3. Exclude and save ──
df_clean = df_raw[~(by_prefix | by_name)].reset_index(drop=True)
assert len(df_clean) == n_start - (by_prefix | by_name).sum()

if os.path.isdir(output_path) or not output_path.lower().endswith('.csv'):   # folder given -> add a name
    output_path = os.path.join(output_path, 'clean_excl_homog9-12HC_23H.csv')
try:
    df_clean.to_csv(output_path, index=False)
except PermissionError:
    raise PermissionError(f"Cannot write {output_path}: close it if it is open in Excel, "
                          f"or check that you can write to this folder") from None

# small log of what was removed, next to the clean file
log_path = os.path.splitext(output_path)[0] + '_excluded_samples.csv'
pd.DataFrame({'Sample ID': sid[by_prefix | by_name].values,
              'Reason': ['homogenization issue' if p else 'outlier-flagged'
                         for p in by_prefix[by_prefix | by_name]]}).to_csv(log_path, index=False)

print(f"\nRemoved {n_start - len(df_clean)} rows -> {len(df_clean)} rows remain")
print(f"Final shape: {df_clean.shape}")
print(f"Saved clean file: {output_path}")
print(f"Saved exclusion log: {log_path}")
