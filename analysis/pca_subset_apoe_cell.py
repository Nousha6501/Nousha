# ══════════════════════════════════════════════════════════════════
# PCA of one subset (default: WT, Hippocampus, Female), colored by APOE genotype
# Input: PQNregion_log2_CTXadj_main.csv (PQN within region, log2, CTX / 1.31)
#   - data are already log2 -> NO sqrt / log transform here; Pareto scaling only
#   - within ONE region the 1.31 factor does not matter (PCA centers each metabolite)
#   - exclusions: 9-12 H/C (homogenization issue) and 23H_AP4_F_KI_162
# Change GENOTYPE / REGION_KEY / SEX to look at other subsets (SEX = None -> both sexes).
# ══════════════════════════════════════════════════════════════════
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn.decomposition import PCA

# ── 0. Settings ──────────────────────────────────────────────────────────
file_path = r"O:\metabolom\Result\new version result_09.2026\PQNregion_log2_CTXadj_main.csv"
fig_dir   = r"C:\Users\nnekooiemarnany\OneDrive - UTHealth Houston\Desktop\metabolomic_projects\figures_distribution"

GENOTYPE   = 'WT'          # 'WT' or 'KI'
REGION_KEY = 'hip'         # 'hip' or 'cortex' (case does not matter)
SEX        = 'Female'      # 'Female', 'Male' or None (both)

EXCL_PREFIXES = ('9H_', '10H_', '11H_', '12H_', '9C_', '10C_', '11C_', '12C_')
EXCL_SAMPLES  = ['23H_AP4_F_KI_162']

# ── 1. Load data ─────────────────────────────────────────────────────────
df = pd.read_csv(file_path)
df.columns = [' '.join(str(c).split()) for c in df.columns]
if 'Type' in df.columns:
    df = df[df['Type'].astype(str).str.strip() == 'Sample'].copy()
for c in ['Sample ID', 'Region of brain', 'Genotype', 'Gender', 'line of APOE']:
    df[c] = df[c].astype(str).str.strip()

sid = df['Sample ID']
drop = sid.str.startswith(EXCL_PREFIXES) | sid.isin(EXCL_SAMPLES)
print(f"Excluded {drop.sum()} samples: {sorted(sid[drop])}")
df = df[~drop]

metab_cols = [c for c in df.columns if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
assert metab_cols, "no POS_/NEG_ metabolite columns found"

df['Sex_label']    = df['Gender'].replace({'M': 'Male', 'F': 'Female'})
df['Region_label'] = np.where(df['Region of brain'].str.lower().str.startswith('hip'), 'Hippocampus', 'Cortex')
df['APOE_label']   = df['line of APOE']

# ── 2. Subset ────────────────────────────────────────────────────────────
m = (df['Genotype'] == GENOTYPE) & df['Region of brain'].str.lower().str.contains(REGION_KEY.lower())
if SEX:
    m &= df['Sex_label'] == SEX
sub = df[m].copy()
region_name = sub['Region_label'].iloc[0] if len(sub) else REGION_KEY
subset_txt = f"{GENOTYPE}, {region_name}, {SEX or 'both sexes'}"
assert len(sub) >= 3, (f"only {len(sub)} samples for {subset_txt}. Values in file: "
                       f"Genotype {sorted(df['Genotype'].unique())}, Region {sorted(df['Region of brain'].unique())}, "
                       f"Gender {sorted(df['Gender'].unique())}")
print(f"n per APOE group ({subset_txt}):")
print(sub['APOE_label'].value_counts().sort_index().to_string())

# ── 3. Missing values + Pareto scaling (data already log2) ───────────────
X = sub[metab_cols].apply(pd.to_numeric, errors='coerce')
n_missing = int(X.isna().sum().sum())
X = X.fillna(X.min() - 1)                 # half-minimum on the log2 scale
X = X.loc[:, X.std() > 0]                 # drop constant columns
X_pareto = (X - X.mean()) / np.sqrt(X.std())
print(f"Metabolites used: {X.shape[1]} | missing values filled: {n_missing}")

# ── 4. PCA ───────────────────────────────────────────────────────────────
n_comp = min(5, len(sub) - 1, X.shape[1])
pca = PCA(n_components=n_comp)
scores = pca.fit_transform(X_pareto)
explained = pca.explained_variance_ratio_ * 100
print("Variance explained: " + ", ".join(f"PC{i+1}={e:.1f}%" for i, e in enumerate(explained[:3])))

pca_df = pd.DataFrame(scores[:, :2], columns=['PC1', 'PC2'])
pca_df['APOE_label'] = sub['APOE_label'].values
pca_df['Sample ID']  = sub['Sample ID'].values

# top loadings: which metabolites drive PC1 / PC2
load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=['PC1', 'PC2'])
for pc in ['PC1', 'PC2']:
    top = load[pc].abs().sort_values(ascending=False).index[:8]
    print(f"\nTop {pc} loadings:\n" + load.loc[top, pc].round(3).to_string())

# ── 5. Plot PC1 vs PC2, colored by APOE genotype, with 95% ellipses ──────
palette_apoe = {'APOE2': '#4C72B0', 'APOE3': '#55A868', 'APOE4': '#C44E52'}
for a in pca_df['APOE_label'].unique():
    palette_apoe.setdefault(a, '#888888')

def confidence_ellipse(x, y, ax, color, level=0.95, **kwargs):
    """95% data ellipse (covers ~95% of the group's points; chi-square, 2 df)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    k = np.sqrt(chi2.ppf(level, 2))                       # 2.45 for 95% in 2D
    width, height = 2 * k * np.sqrt(eigvals)
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), width, height, angle=angle,
                         edgecolor=color, facecolor=color, alpha=0.12, lw=1.5, **kwargs))

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
for apoe_label, color in palette_apoe.items():
    grp = pca_df[pca_df['APOE_label'] == apoe_label]
    if grp.empty:
        continue
    ax.scatter(grp['PC1'], grp['PC2'], color=color, label=f"{apoe_label} (n={len(grp)})",
               s=70, edgecolor='black', linewidth=0.5, alpha=0.85)
    confidence_ellipse(grp['PC1'].values, grp['PC2'].values, ax, color)

ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
ax.set_title(f"PCA — {subset_txt}\ncolored by APOE genotype (95% ellipses)")
ax.axhline(0, color='grey', lw=0.5)
ax.axvline(0, color='grey', lw=0.5)
ax.legend(title='APOE genotype')
plt.tight_layout()

os.makedirs(fig_dir, exist_ok=True)
fname = f"pca_{GENOTYPE}_{region_name}_{SEX or 'bothSexes'}_APOE.png"
save_path = os.path.join(fig_dir, fname)
try:
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
except PermissionError:
    print(f"\nCould not save {save_path}: close it if it is open, or check folder permissions")
plt.show()
