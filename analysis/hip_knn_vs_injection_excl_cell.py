# ══════════════════════════════════════════════════════════════════
# Hippocampus - KNN distance vs. injection order  [SENSITIVITY RE-RUN]
# Leaves out 72H_AP4_M_KI_64, 23H_AP4_F_KI_162 (outlier-flagged) and 9H-12H
# (homogenization issue) from THIS screen only; no data file is changed.
# Data: IS-normalized, imputed file (not raw) -> not directly comparable with the raw screen
# colored by consensus flag count (n_methods_flagged)
# Fully self-contained.
# Default = FLAG ONLY: no sample is removed; 9H-12H (homogenization issue) are marked
# with a black ring. For a sensitivity re-run (do other samples become outliers once the
# strongest ones are left out?) fill EXCLUDE_SAMPLES and/or set EXCLUDE_HOMOG = True.
# This only leaves them out of THIS screen; no data file is changed.
# Works with the new file (text labels, 'Region of brain', 'line of APOE', POS_/NEG_)
# and the old one (numeric codes, 'Brain_Region', 'APOE_Genotype', Pos_/Neg_).
# ══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM

# ── 0a. LOAD DATA ──
file_path = r"C:\Users\nnekooiemarnany\OneDrive - UTHealth Houston\Desktop\Metabolom_analys\CSV2_Revised\RevisedScript\lasVersion_Pipline\Norm_Data_IS\ISnorm_clean_imputed_Data.csv"
REGION_KEY  = 'hip'                                  # text label match ('Hippocmpus', ...)
REGION_CODE = 0                                      # numeric code of hippocampus in old files
homog_prefixes = ('9H_', '10H_', '11H_', '12H_')     # documented issues: FLAGGED, not removed
EXCLUDE_SAMPLES = ['72H_AP4_M_KI_64', '23H_AP4_F_KI_162']
EXCLUDE_HOMOG   = True                               # also leave out 9H-12H
QC_OUTLIER_ZONE = (0, 10)                            # injections of the poor first QC vials (6-7)
CONTAMINATION = 0.05

df_raw = pd.read_csv(file_path)
df_raw.columns = [' '.join(str(c).split()) for c in df_raw.columns]   # remove line breaks in names

def find_col(*names):
    """First column whose name matches one of names (case/space-insensitive)."""
    norm = {c.lower().replace(' ', '').replace('_', ''): c for c in df_raw.columns}
    for n in names:
        key = n.lower().replace(' ', '').replace('_', '')
        if key in norm:
            return norm[key]
    raise KeyError(f"none of {names} found. Columns: {list(df_raw.columns[:15])} ...")

REGION_COL = find_col('Region of brain', 'Brain_Region')
APOE_COL   = find_col('line of APOE', 'APOE_Genotype')
SEX_COL    = find_col('Gender', 'Sex')
GENO_COL   = find_col('Genotype')
INJ_COL    = next(c for c in df_raw.columns if c.lower().startswith('injection'))
BATCH_COL  = next((c for c in df_raw.columns if c.lower().startswith('batch segment')), None)

if 'Type' in df_raw.columns:                                         # samples only (no QCs)
    df_raw = df_raw[df_raw['Type'].astype(str).str.strip().isin(['Sample', '1'])]

# ── 0b. metabolite columns: POS_/NEG_ (any case), excluding 13C internal standards ──
all_metab = [c for c in df_raw.columns
             if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
assert all_metab, f"no POS_/NEG_ columns found. First columns: {list(df_raw.columns[:15])}"

# ── 0c. subset to Hippocampus (text or numeric region labels) ──
reg = df_raw[REGION_COL]
if pd.api.types.is_numeric_dtype(reg):
    hip_mask = reg == REGION_CODE
else:
    hip_mask = reg.astype(str).str.strip().str.lower().str.contains(REGION_KEY)
df_target = df_raw[hip_mask].reset_index(drop=True)
assert len(df_target), f"no Hippocampus rows. Values in '{REGION_COL}': {reg.unique().tolist()}"

# ── optional exclusion (sensitivity re-run only) ──
sid = df_target['Sample ID'].astype(str).str.strip()
missing = [x for x in EXCLUDE_SAMPLES if x not in set(sid)]
if missing:
    print(f"WARNING: not found in Hippocampus samples (check spelling): {missing}")
drop = sid.isin([x.strip() for x in EXCLUDE_SAMPLES])
if EXCLUDE_HOMOG:
    drop |= sid.str.startswith(homog_prefixes)
excluded = sid[drop].tolist()
df_target = df_target[~drop].reset_index(drop=True)
print(f"Left out of this screen: {len(excluded)} {excluded}" if excluded else "Nothing left out")

# ── 0d. readable labels (numeric codes are translated, text is kept) ──
def label(col, codes):
    s = df_target[col]
    return s.map(codes) if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.strip()

df_target['Sex_label']      = label(SEX_COL, {0: 'Male', 1: 'Female'}).replace({'M': 'Male', 'F': 'Female'})
df_target['Genotype_label'] = label(GENO_COL, {0: 'KI', 1: 'WT'})
df_target['APOE_label']     = label(APOE_COL, {0: 'APOE2', 1: 'APOE3', 2: 'APOE4'})
df_target['Injection_number'] = pd.to_numeric(df_target[INJ_COL], errors='coerce')
df_target['homog_issue'] = df_target['Sample ID'].astype(str).str.strip().str.startswith(homog_prefixes)

# ── 1. log1p (raw only), fill missing, scale — feeds ALL 5 methods ──
vals = df_target[all_metab].apply(pd.to_numeric, errors='coerce')
already_log = vals.max().max() < 60
log_metab = vals if already_log else np.log1p(vals.clip(lower=0))
log_metab = log_metab.fillna(log_metab.min())        # missing / <= 0 -> lowest value of that metabolite
log_metab = log_metab.loc[:, log_metab.std() > 0]    # drop constant columns
x_scaled = StandardScaler().fit_transform(log_metab)

# ── 2. stratum z-score (Sex x Genotype x APOE; strata < 3 samples not flagged) ──
strata = ['Sex_label', 'Genotype_label', 'APOE_label']
df_target['mean_log_intensity'] = log_metab.mean(axis=1)
df_target['stratum_z'] = (df_target.groupby(strata)['mean_log_intensity']
                          .transform(lambda x: (x - x.mean()) / x.std() if len(x) >= 3 else np.nan))
df_target['stratum_z_flag'] = (df_target['stratum_z'].abs() > 2).astype(int)

# ── 3. KNN distance (k=5, mean + 2 SD) ──
k = 5
distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(x_scaled).kneighbors(x_scaled)
df_target['mean_knn_distance'] = distances[:, 1:].mean(axis=1)
knn_threshold = df_target['mean_knn_distance'].mean() + 2 * df_target['mean_knn_distance'].std()
df_target['knn_flag'] = (df_target['mean_knn_distance'] > knn_threshold).astype(int)

# ── 4. PyOD models ──
df_target['iforest_flag'] = IForest(contamination=CONTAMINATION, random_state=42).fit(x_scaled).labels_
df_target['abod_flag']    = ABOD(contamination=CONTAMINATION).fit(x_scaled).labels_
df_target['ocsvm_flag']   = OCSVM(contamination=CONTAMINATION).fit(x_scaled).labels_

# ── 5. consensus count ──
flag_cols = ['stratum_z_flag', 'knn_flag', 'iforest_flag', 'abod_flag', 'ocsvm_flag']
df_target['n_methods_flagged'] = df_target[flag_cols].sum(axis=1)

print(f"Hippocampus samples in screen: {len(df_target)} | metabolites: {log_metab.shape[1]} | "
      f"data {'already log2' if already_log else 'raw -> log1p'}")
print("Flagged by >= 2 methods:")
print(df_target.loc[df_target['n_methods_flagged'] >= 2,
                    ['Sample ID', 'Injection_number', 'homog_issue'] + flag_cols + ['n_methods_flagged']]
      .sort_values('n_methods_flagged', ascending=False).to_string(index=False))

# A/B cutoff = first injection of batch B (from the data; fallback 124)
if BATCH_COL is not None:
    b = df_raw.loc[df_raw[BATCH_COL].astype(str).str.strip() == 'B', INJ_COL]
    batch_cut = pd.to_numeric(b, errors='coerce').min() if len(b) else 124
else:
    batch_cut = 124

# ── 6. plot: KNN distance vs. injection order ──
counts = sorted(df_target['n_methods_flagged'].unique())
palette = dict(zip(counts, sns.color_palette('rocket_r', len(counts) + 1)[1:]))
plt.figure(figsize=(11, 5))
sns.scatterplot(data=df_target, x='Injection_number', y='mean_knn_distance',
                hue='n_methods_flagged', palette=palette, s=60)
h = df_target[df_target['homog_issue']]
plt.scatter(h['Injection_number'], h['mean_knn_distance'], s=180, facecolors='none',
            edgecolors='black', linewidths=1.5, label='homog. issue (9H-12H)')
plt.axhline(knn_threshold, color='red', linestyle=':', label=f'KNN threshold ({knn_threshold:.2f})')
plt.axvline(batch_cut, color='gray', linestyle='--', label=f'Batch A/B cutoff (inj {batch_cut:g})')
plt.axvspan(*QC_OUTLIER_ZONE, color='red', alpha=0.1, label='known QC outlier zone (inj 6-7)')
plt.legend(title='n methods flagged', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xlabel('Injection number')
plt.ylabel(f'Mean distance to {k} nearest neighbors (scaled)')
plt.title('Hippocampus - KNN distance vs. injection order'
          + (f'\n(left out: {", ".join(excluded)})' if excluded else ''))
plt.tight_layout()
plt.show()
