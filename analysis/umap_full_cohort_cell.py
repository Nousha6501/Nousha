# ══════════════════════════════════════════════════════════════════
# Full Cohort Metabolomic Landscape (UMAP) — Male vs Female panels
# Input: PQN-within-region, log2, CTX / 1.31 file (both regions on a comparable scale)
# Works with text labels ('Region of brain', 'Hippocmpus', 'KI', 'M', 'APOE4', POS_/NEG_)
# and with the old numeric-coded files ('Brain_Region', 'APOE_Genotype', 0/1/2, Pos_/Neg_).
# ══════════════════════════════════════════════════════════════════
import os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import umap

# ---- Settings ----
FILE_PATH = r"O:\metabolom\Result\new version result_09.2026\PQNregion_log2_CTXadj_clean_main.csv"
OUT_PATH  = r"O:\metabolom\Result\new version result_09.2026"
EXCL_PREFIXES = ('9H_', '10H_', '11H_', '12H_', '9C_', '10C_', '11C_', '12C_')   # homogenization issue
EXCL_SAMPLES  = ['23H_AP4_F_KI_162']                                             # outlier-flagged

# ---- Load ----
df_raw_clean = pd.read_csv(FILE_PATH)
df_raw_clean.columns = [' '.join(str(c).split()) for c in df_raw_clean.columns]

def find_col(*names):
    """First column whose name matches one of names (case/space/underscore-insensitive)."""
    norm = {c.lower().replace(' ', '').replace('_', ''): c for c in df_raw_clean.columns}
    for n in names:
        key = n.lower().replace(' ', '').replace('_', '')
        if key in norm:
            return norm[key]
    raise KeyError(f"none of {names} found. Columns: {list(df_raw_clean.columns[:15])} ...")

REGION_COL = find_col('Region of brain', 'Brain_Region')
APOE_COL   = find_col('line of APOE', 'APOE_Genotype')
SEX_COL    = find_col('Gender', 'Sex')
GENO_COL   = find_col('Genotype')

# metabolites: POS_/NEG_ in any case; 13C internal standards excluded
all_metab = [c for c in df_raw_clean.columns
             if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
assert all_metab, f"no POS_/NEG_ columns found. Columns: {list(df_raw_clean.columns[:15])} ..."
df_clean = df_raw_clean.drop(columns=[c for c in df_raw_clean.columns if '13C' in c])

# ---- Exclusions (QCs, homogenization issue, outlier) — prints what was removed ----
sid = df_clean['Sample ID'].astype(str).str.strip()
is_qc = sid.str.upper().str.startswith('QC')
if 'Type' in df_clean.columns:
    is_qc |= df_clean['Type'].astype(str).str.strip().str.lower().str.startswith('quality')
is_homog = sid.str.startswith(EXCL_PREFIXES)
is_excl  = sid.isin(EXCL_SAMPLES)
print(f"Rows in file: {len(df_clean)} | QCs removed: {is_qc.sum()} | "
      f"homogenization removed: {is_homog.sum()} {sorted(sid[is_homog])} | "
      f"outliers removed: {is_excl.sum()} {sorted(sid[is_excl])}")
df_clean = df_clean[~(is_qc | is_homog | is_excl)].reset_index(drop=True)

# ---- Labels (numeric codes translated, text kept) ----
def label(col, codes, extra=None):
    s = df_clean[col]
    out = s.map(codes) if pd.api.types.is_numeric_dtype(s) else s.astype(str).str.strip()
    return out.replace(extra) if extra else out

df_full = df_clean.copy()
df_full['Sex_label']      = label(SEX_COL, {0: 'Male', 1: 'Female'}, {'M': 'Male', 'F': 'Female'})
df_full['Region_label']   = label(REGION_COL, {0: 'Hippocampus', 1: 'Cortex'})
df_full['Region_label']   = np.where(df_full['Region_label'].str.lower().str.startswith('hip'),
                                     'Hippocampus', 'Cortex')
df_full['Genotype_label'] = label(GENO_COL, {0: 'KI', 1: 'WT'})
df_full['APOE_label']     = label(APOE_COL, {0: 'APOE2', 1: 'APOE3', 2: 'APOE4'})
df_full['Sample ID']      = df_full['Sample ID'].astype(str)
for c in ['Sex_label', 'Region_label', 'Genotype_label', 'APOE_label']:
    print(f"{c}: {df_full[c].value_counts(dropna=False).to_dict()}")

# ---- Log (raw only) + impute ----
vals = df_full[all_metab].apply(pd.to_numeric, errors='coerce')
already_log = vals.max().max() < 60          # log2 data never gets this high; raw data does
if already_log:
    # half-minimum on the log2 scale = minimum - 1
    log_metab_full = vals.fillna(vals.min() - 1)
else:
    vals = vals.where(vals > 0)
    log_metab_full = np.log1p(vals.fillna(vals.min() / 2))
log_metab_full = log_metab_full.loc[:, log_metab_full.std() > 0]      # drop constant columns
print(f"Data {'already log2 -> used as is' if already_log else 'raw -> log1p'} | "
      f"missing values imputed: {int(vals.isna().sum().sum())} | metabolites used: {log_metab_full.shape[1]}")

# ---- UMAP ----
x_scaled_full = StandardScaler().fit_transform(log_metab_full)
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
embedding = reducer.fit_transform(x_scaled_full)
df_full['UMAP1'] = embedding[:, 0]
df_full['UMAP2'] = embedding[:, 1]
print("df_full shape:", df_full.shape)

# ---- Plot ----
apoe_colors        = {'APOE2': '#2E8B57', 'APOE3': '#E8A33D', 'APOE4': '#7B3F9E'}
genotype_markers   = {'KI': 'o', 'WT': 's'}
region_kde_colors  = {'Hippocampus': '#4C72B0', 'Cortex': '#C44E52'}
for a in df_full['APOE_label'].dropna().unique():          # any APOE label not in the dict
    apoe_colors.setdefault(a, '#888888')

sexes = [s for s in ['Male', 'Female'] if (df_full['Sex_label'] == s).any()]
fig, axes = plt.subplots(1, len(sexes), figsize=(9 * len(sexes), 8), facecolor='white', squeeze=False)

for ax, sex in zip(axes[0], sexes):
    sub = df_full[df_full['Sex_label'] == sex]

    for region, color in region_kde_colors.items():
        region_pts = sub[sub['Region_label'] == region]
        if len(region_pts) < 5:                            # too few points for a density
            continue
        sns.kdeplot(data=region_pts, x='UMAP1', y='UMAP2', ax=ax,
                    color=color, fill=True, alpha=0.15, levels=2, thresh=0.15, zorder=0)
        sns.kdeplot(data=region_pts, x='UMAP1', y='UMAP2', ax=ax,
                    color=color, fill=False, alpha=0.6, levels=1, thresh=0.15,
                    linewidths=1.5, zorder=1)

    for apoe, color in apoe_colors.items():
        for geno, marker in genotype_markers.items():
            pts = sub[(sub['APOE_label'] == apoe) & (sub['Genotype_label'] == geno)]
            ax.scatter(pts['UMAP1'], pts['UMAP2'], c=color, marker=marker, s=90,
                       alpha=0.9, edgecolor='white', linewidth=0.8, zorder=2)

    ax.set_title(f'{sex} (n = {len(sub)})', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)

region_legend = [Patch(facecolor=color, alpha=0.3, edgecolor=color, label=region)
                 for region, color in region_kde_colors.items()]
fig.legend(handles=region_legend, bbox_to_anchor=(1.0, 0.98), loc='upper left',
           fontsize=10, title='Brain Region\n(shaded density)', frameon=False)

geno_shape_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=9, label='KI'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=9, label='WT')
]
fig.legend(handles=geno_shape_legend, bbox_to_anchor=(1.0, 0.80), loc='upper left',
           fontsize=10, title='Genotype', frameon=False)

present_apoe = [a for a in apoe_colors if (df_full['APOE_label'] == a).any()]
apoe_color_legend = [Patch(facecolor=apoe_colors[a], label=a) for a in present_apoe]
fig.legend(handles=apoe_color_legend, bbox_to_anchor=(1.0, 0.62), loc='upper left',
           fontsize=10, title='APOE Genotype', frameon=False)

fig.suptitle('Full Cohort Metabolomic Landscape (UMAP)', fontsize=17, fontweight='bold', y=1.03)
plt.tight_layout()

# save with bbox_inches='tight' so external legends aren't clipped
save_path = os.path.join(OUT_PATH, 'UMAP_full_cohort_landscape.png')
try:
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved to: {save_path}")
except PermissionError:
    print(f"Could not save {save_path}: close it if it is open, or check folder permissions")

plt.show()
