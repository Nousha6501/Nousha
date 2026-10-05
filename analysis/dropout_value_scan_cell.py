# ══════════════════════════════════════════════════════════════════
# Scan the RAW file for single "dropout" values: one metabolite in one sample
# far below the rest of its region (missed / badly integrated peak), like
# NEG_ATP/dGTP in 35H_AP4_F_WT_147 (19,964 vs median ~94,000,000).
# Lists them and writes a table; nothing is changed in the data here.
# ══════════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np

raw_path = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\master_with_QC3.csv"
out_csv  = r"O:\metabolom\Result\new version result_09.2026\dropout_values.csv"
FOLD = 100            # flag values more than 100x below the region median of that metabolite

raw = pd.read_csv(raw_path)
raw.columns = [' '.join(str(c).split()) for c in raw.columns]
s = raw[raw['Type'].astype(str).str.strip() == 'Sample'].copy()
metab = [c for c in s.columns if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
vals = s[metab].apply(pd.to_numeric, errors='coerce')
region = s['Region of brain'].astype(str).str.strip()

med = vals.groupby(region).transform('median')
ratio = vals / med
hits = (ratio < 1 / FOLD) | (vals <= 0) | vals.isna()

rows = []
for i, c in zip(*np.where(hits.values)):
    rows.append({'Sample ID': s['Sample ID'].iloc[i], 'Region': region.iloc[i], 'Metabolite': metab[c],
                 'value': vals.iat[i, c], 'region_median': med.iat[i, c],
                 'fold_below_median': (med.iat[i, c] / vals.iat[i, c]) if vals.iat[i, c] > 0 else np.inf})
res = pd.DataFrame(rows).sort_values('fold_below_median', ascending=False)
print(f"{len(res)} dropout / missing values (> {FOLD}x below region median, <= 0, or empty):")
print(res.to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
print("\nPer metabolite:", res['Metabolite'].value_counts().to_dict())
res.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
