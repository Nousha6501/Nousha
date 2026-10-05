# ══════════════════════════════════════════════════════════════════
# Hippocampus outlier detection — 5 methods (stratum-z, KNN, IForest, ABOD, OCSVM)
# FLAG ONLY: no sample is removed. Documented homogenization-issue samples are
# kept and marked in the summary table (column 'homog_issue').
#
# Data: RAW file recommended (outlier detection BEFORE normalization).
#   raw data      -> log1p
#   already log2  -> used as is (detected automatically)
# Labels: works with text labels ('Hippocmpus', 'KI', 'M', 'APOE4')
#         and with numeric codes (0/1/2) of the older files.
# Run once in a separate cell if needed:   %pip install pyod
# ══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pyod.models.iforest import IForest
from pyod.models.abod import ABOD
from pyod.models.ocsvm import OCSVM

# ── 0a. LOAD DATA — update path if needed ──
file_path = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\master_with_QC3.csv"
output_csv = r"O:\metabolom\Result\new version result_09.2026\Hippocampus_outlier_summary.csv"
REGION_KEY = 'hip'                                   # text label match ('Hippocmpus', ...)
REGION_CODE = 0                                      # numeric code of hippocampus in old files
homog_prefixes = ('9H_', '10H_', '11H_', '12H_')     # documented issues: FLAGGED, not removed
CONTAMINATION = 0.05

df_raw = pd.read_csv(file_path)
df_raw.columns = [' '.join(str(c).split()) for c in df_raw.columns]   # remove line breaks in names
if 'Type' in df_raw.columns:                                         # samples only (no QCs)
    df_raw = df_raw[df_raw['Type'].astype(str).str.strip().isin(['Sample', '1'])]

# ── 0b. metabolite columns: POS_/NEG_ (any case), excluding 13C internal standards ──
all_metab = [c for c in df_raw.columns
             if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
assert all_metab, f"no POS_/NEG_ columns found. First columns: {list(df_raw.columns[:15])}"

# ── 0c. subset to Hippocampus (text or numeric region labels) ──
reg = df_raw['Region of brain']
if pd.api.types.is_numeric_dtype(reg):
    hip_mask = reg == REGION_CODE
else:
    hip_mask = reg.astype(str).str.strip().str.lower().str.contains(REGION_KEY)
df_target = df_raw[hip_mask].reset_index(drop=True)
assert len(df_target), f"no Hippocampus rows. Values in 'Region of brain': {reg.unique().tolist()}"

# ── 0d. readable labels (numeric codes are translated, text is kept) ──
def label(col, codes):
    s = df_target[col]
    return s.map(codes) if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.strip()

df_target['Sex_label']      = label('Gender', {0: 'Male', 1: 'Female'}).replace({'M': 'Male', 'F': 'Female'})
df_target['Genotype_label'] = label('Genotype', {0: 'KI', 1: 'WT'})
df_target['APOE_label']     = label('line of APOE', {0: 'APOE2', 1: 'APOE3', 2: 'APOE4'})
inj_col = next(c for c in df_target.columns if c.lower().startswith('injection'))
df_target['Injection_number'] = pd.to_numeric(df_target[inj_col], errors='coerce')
df_target['homog_issue'] = df_target['Sample ID'].astype(str).str.strip().str.startswith(homog_prefixes)

# ── 1. log transform (raw only), fill missing, scale — feeds ALL 5 methods ──
vals = df_target[all_metab].apply(pd.to_numeric, errors='coerce')
already_log = vals.max().max() < 60
log_metab = vals if already_log else np.log1p(vals.clip(lower=0))
n_missing = int(log_metab.isna().sum().sum())
log_metab = log_metab.fillna(log_metab.min())        # missing / <= 0 -> lowest value of that metabolite
log_metab = log_metab.loc[:, log_metab.std() > 0]    # drop constant columns (cannot be scaled)
x_scaled = StandardScaler().fit_transform(log_metab)

# ── 2. PCA — for visualization only, not used in flagging ──
pca_model = PCA(n_components=2)
x_pca = pca_model.fit_transform(x_scaled)
df_target['pca1'], df_target['pca2'] = x_pca[:, 0], x_pca[:, 1]

# ── 3. stratum z-score (mean log intensity, z-scored within Sex x Genotype x APOE) ──
df_target['mean_log_intensity'] = log_metab.mean(axis=1)
strata = ['Sex_label', 'Genotype_label', 'APOE_label']
df_target['stratum_n'] = df_target.groupby(strata)['mean_log_intensity'].transform('size')
df_target['stratum_z'] = (df_target.groupby(strata)['mean_log_intensity']
                          .transform(lambda x: (x - x.mean()) / x.std() if len(x) >= 3 else np.nan))
stratum_flag = (df_target['stratum_z'].abs() > 2).astype(int)        # strata < 3 samples: not flagged

# ── 4. KNN distance (k=5, mean + 2 SD threshold) ──
k = 5
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x_scaled)
distances, indices = nbrs.kneighbors(x_scaled)
df_target['mean_knn_distance'] = distances[:, 1:].mean(axis=1)
df_target['nearest_neighbor_ids'] = [df_target['Sample ID'].iloc[r].tolist() for r in indices[:, 1:]]
knn_threshold = df_target['mean_knn_distance'].mean() + 2 * df_target['mean_knn_distance'].std()
knn_flag = (df_target['mean_knn_distance'] > knn_threshold).astype(int)

# ── 5. PyOD models — all on the SAME scaled data ──
iforest_model = IForest(contamination=CONTAMINATION, random_state=42).fit(x_scaled)
abod_model    = ABOD(contamination=CONTAMINATION).fit(x_scaled)
ocsvm_model   = OCSVM(contamination=CONTAMINATION).fit(x_scaled)

# ── 6. master consensus table (flags only, nothing removed) ──
df_outlier_summary = pd.DataFrame({
    'Sample ID':         df_target['Sample ID'],
    'Sex':               df_target['Sex_label'],
    'Genotype':          df_target['Genotype_label'],
    'APOE':              df_target['APOE_label'],
    'injection_num':     df_target['Injection_number'],
    'homog_issue':       df_target['homog_issue'],
    'pca1':              x_pca[:, 0],
    'pca2':              x_pca[:, 1],
    'stratum_z':         df_target['stratum_z'].round(2),
    'mean_knn_distance': df_target['mean_knn_distance'].round(2),
    'stratum_z_flag':    stratum_flag,
    'knn_flag':          knn_flag,
    'iforest_flag':      iforest_model.labels_,
    'abod_flag':         abod_model.labels_,
    'ocsvm_flag':        ocsvm_model.labels_,
})
flag_cols = ['stratum_z_flag', 'knn_flag', 'iforest_flag', 'abod_flag', 'ocsvm_flag']
df_outlier_summary['n_methods_flagged'] = df_outlier_summary[flag_cols].sum(axis=1)
df_target['n_methods_flagged'] = df_outlier_summary['n_methods_flagged'].values
df_outlier_summary = df_outlier_summary.sort_values('n_methods_flagged', ascending=False)

print(f"Total Hippocampus samples: {len(df_target)}  (nothing removed; "
      f"{df_target['homog_issue'].sum()} marked homog_issue)")
print(f"Metabolites used: {log_metab.shape[1]} of {len(all_metab)} | "
      f"data {'already log2' if already_log else 'raw -> log1p'} | missing values filled: {n_missing}")
print(f"PCA explained variance: PC1 {pca_model.explained_variance_ratio_[0]:.1%}, "
      f"PC2 {pca_model.explained_variance_ratio_[1]:.1%}")
small = df_target.groupby(strata).size()
if (small < 3).any():
    print("Strata with < 3 samples (no stratum-z):", small[small < 3].to_dict())
print("\nSamples flagged by 2 or more methods:")
show = ['Sample ID', 'Sex', 'Genotype', 'APOE', 'injection_num', 'homog_issue'] + flag_cols + ['n_methods_flagged']
print(df_outlier_summary.loc[df_outlier_summary['n_methods_flagged'] >= 2, show].to_string(index=False))
import os
if os.path.isdir(output_csv) or not output_csv.lower().endswith('.csv'):   # folder given -> add a file name
    output_csv = os.path.join(output_csv, 'Hippocampus_outlier_summary.csv')
try:
    df_outlier_summary.to_csv(output_csv, index=False)
except PermissionError:
    raise PermissionError(f"Cannot write {output_csv}: close it if it is open in Excel, "
                          f"or check that you can write to this folder") from None
print(f"\nSaved full table to {output_csv}")

# ── 7. visual screens ──
plt.figure(figsize=(14, 5))
plt.plot(df_target.index, df_target['mean_knn_distance'], color='steelblue', linewidth=1)
plt.scatter(df_target.index, df_target['mean_knn_distance'],
            c=np.where(df_target['mean_knn_distance'] > knn_threshold, 'red', 'steelblue'), s=25, zorder=3)
plt.axhline(knn_threshold, color='red', linestyle='--', label=f'mean + 2 SD ({knn_threshold:.2f})')
plt.xticks(df_target.index, df_target['Sample ID'], rotation=90, fontsize=6)
plt.ylabel(f'Mean distance to {k} nearest neighbors (scaled)')
plt.title('Hippocampus - KNN distance profile')
plt.legend()
plt.tight_layout()
plt.show()

flags_by_row = df_outlier_summary.set_index('Sample ID').loc[df_target['Sample ID']]
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, flag_col, name in zip(axes, ['iforest_flag', 'abod_flag', 'ocsvm_flag'], ['IForest', 'ABOD', 'OCSVM']):
    sns.scatterplot(x=df_target['pca1'], y=df_target['pca2'],
                    hue=flags_by_row[flag_col].map({0: 'normal', 1: 'outlier'}).values,
                    palette={'normal': 'blue', 'outlier': 'red'}, alpha=0.7, s=60, ax=ax)
    ax.set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.0%})')
    ax.set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.0%})')
    ax.set_title(f'Hippocampus - {name}')
plt.tight_layout()
plt.show()
