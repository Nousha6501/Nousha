# ══════════════════════════════════════════════════════════════════
# Volcano plot: KI of ONE APOE line vs POOLED WT of several APOE lines
#   default: KI = APOE4 KI  vs  WT = WT of APOE2 + APOE3 + APOE4 pooled
#            Hippocampus, Female
#   input = PQNregion_log2_samples_clean.csv (PQN within region, WT reference, log2)
#   - KI and WT are DIFFERENT mice -> unpaired Welch t-test (does not assume equal variances)
#   - test and fold change on the SAME scale (log2):
#       log2FC = mean log2(KI) - mean log2(WT pooled) = log2 of the ratio of geometric means
#   - Benjamini-Hochberg FDR across all metabolites
#   - point colours (see legend):
#       strong red / blue = significant AND >= fold-change cut-off (higher / lower in KI)
#       light red / blue  = significant but below the fold-change cut-off
#       grey              = not significant
#   - saves: figure (PNG + PDF) + full results table
# Change the SUBGROUP block to use other APOE lines / region / sex.
# ══════════════════════════════════════════════════════════════════
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

input_path = r"O:\metabolom\Result\new version result_09.2026\PQNregion_log2_samples_clean.csv"
output_dir = r"O:\metabolom\Result\new version result_09.2026\Volcano_KI_vs_WT"

# ---------- subgroup ----------
KI_APOE = ['APOE4']                        # APOE line(s) for the KI group
WT_APOE = ['APOE2', 'APOE3', 'APOE4']      # APOE line(s) pooled for the WT group
REGION  = 'Hippocmpus'                     # exact label in the file (spelling as in the metadata)
SEX     = 'F'                              # 'F' or 'M'

# ---------- cut-offs ----------
P_COL     = 'p_value'          # 'p_value' (nominal) or 'FDR' -> which value decides "significant"
ALPHA     = 0.05
FC_CUTOFF = np.log2(1.5)       # |log2FC| >= log2(1.5) = at least 1.5-fold
N_LABELS  = 15                 # label at most this many significant points (smallest p first)
N_TOP     = 5                  # also label the N_TOP smallest p and every |log2FC| >= 1, even if n.s.

REGION_COL, SEX_COL, GENO_COL, APOE_COL = 'Region of brain', 'Gender', 'Genotype', 'line of APOE'
WT, KI = 'WT', 'KI'
SEX_NAME = {'F': 'Female', 'M': 'Male'}

# short names for titles / files, e.g. "KI-APOE4" and "WT-APOE2+3+4"
KI_NAME = 'KI-APOE' + '+'.join(a.replace('APOE', '') for a in KI_APOE)
WT_NAME = 'WT-APOE' + '+'.join(a.replace('APOE', '') for a in WT_APOE)

# diverging pair (blue = lower in KI, red = higher in KI); light tints = significant but small change;
# neutral gray = not significant
UP, DOWN, NS = '#e34948', '#2a78d6', '#c9c8c3'
UP_SMALL, DOWN_SMALL = '#f4a3a2', '#94bbea'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

os.makedirs(output_dir, exist_ok=True)

# ---------- load + subset ----------
df = pd.read_csv(input_path)
df = df[df['Type'].astype(str).str.strip() == 'Sample'].copy()
for c in (REGION_COL, SEX_COL, GENO_COL, APOE_COL):
    df[c] = df[c].astype(str).str.strip()
metabolite_cols = [c for c in df.columns if c.startswith(('POS_', 'NEG_')) and '13C' not in c]
X = df[metabolite_cols].apply(pd.to_numeric, errors='coerce')
assert X.stack().median() < 50, "input does not look like log2 data (use the *_log2_* file)"

in_region_sex = (df[REGION_COL] == REGION) & (df[SEX_COL] == SEX)
ki_mask = in_region_sex & (df[GENO_COL] == KI) & df[APOE_COL].isin(KI_APOE)
wt_mask = in_region_sex & (df[GENO_COL] == WT) & df[APOE_COL].isin(WT_APOE)
sub = df[ki_mask | wt_mask]
Xs = X[ki_mask | wt_mask]
ki = ki_mask[ki_mask | wt_mask].values
wt = wt_mask[ki_mask | wt_mask].values

print(f"{REGION} / {SEX_NAME.get(SEX, SEX)}: {KI_NAME} n = {ki.sum()}, {WT_NAME} n = {wt.sum()}")
assert ki.sum() >= 3 and wt.sum() >= 3, "fewer than 3 mice in a group: check the labels in SUBGROUP"
print("  WT per APOE line:", ', '.join(f"{a} n={(sub.loc[wt, APOE_COL] == a).sum()}" for a in WT_APOE))
print("  WT:", ', '.join(sub.loc[wt, 'Sample ID']))
print("  KI:", ', '.join(sub.loc[ki, 'Sample ID']))

# ---------- per-metabolite test ----------
rows = []
for c in metabolite_cols:
    a, b = Xs.loc[ki, c].dropna(), Xs.loc[wt, c].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    t, p = ttest_ind(a, b, equal_var=False)                 # Welch
    rows.append({'Metabolite': c, 'log2FC_KI_vs_WT': a.mean() - b.mean(),
                 't': t, 'p_value': p, 'n_KI': len(a), 'n_WT': len(b)})
res = pd.DataFrame(rows)
res['FDR'] = multipletests(res['p_value'], method='fdr_bh')[1]
is_sig = res[P_COL] < ALPHA                                  # passes the p (or FDR) threshold
is_big = res['log2FC_KI_vs_WT'].abs() >= FC_CUTOFF           # passes the fold-change cut-off
is_up = res['log2FC_KI_vs_WT'] > 0
res['Direction'] = np.select(
    [is_sig & is_big & is_up, is_sig & is_big & ~is_up, is_sig & ~is_big & is_up, is_sig & ~is_big & ~is_up],
    ['higher in KI', 'lower in KI', 'higher in KI (small change)', 'lower in KI (small change)'],
    default='n.s.')
res = res.sort_values('p_value').reset_index(drop=True)

# ---------- plot ----------
fig, ax = plt.subplots(figsize=(7.5, 7.0))
x, y = res['log2FC_KI_vs_WT'], -np.log10(res['p_value'])
color = res['Direction'].map({'higher in KI': UP, 'lower in KI': DOWN,
                              'higher in KI (small change)': UP_SMALL,
                              'lower in KI (small change)': DOWN_SMALL, 'n.s.': NS})
ax.scatter(x, y, c=color, s=40, edgecolor='white', linewidth=0.8, zorder=3)

lim = max(abs(x).max(), FC_CUTOFF) * 1.15                      # symmetric x-axis
ax.set_xlim(-lim, lim)                                         # set before placing the line labels

# threshold lines: nominal p, and the p that corresponds to FDR = ALPHA (if any metabolite reaches it)
ax.axhline(-np.log10(ALPHA), color=INK2, linestyle='--', linewidth=0.8)
ax.text(-lim, -np.log10(ALPHA), f' p = {ALPHA}', color=INK2, fontsize=7, va='bottom')
if (res['FDR'] < ALPHA).any():
    p_fdr = res.loc[res['FDR'] < ALPHA, 'p_value'].max()
    ax.axhline(-np.log10(p_fdr), color=INK2, linestyle='-.', linewidth=0.8)
    ax.text(-lim, -np.log10(p_fdr), f' FDR = {ALPHA}', color=INK2, fontsize=7, va='bottom')
for v in (-FC_CUTOFF, FC_CUTOFF):
    ax.axvline(v, color=INK2, linestyle=':', linewidth=0.8)
ax.axvline(0, color=GRID, linewidth=1, zorder=0)

lab_idx = (res.index[res['Direction'] != 'n.s.'][:N_LABELS]
           .union(res.index[:N_TOP]).union(res.index[res['log2FC_KI_vs_WT'].abs() >= 1]))
lab = res.loc[lab_idx]
for _, r in lab.iterrows():
    name = r['Metabolite'].split('_', 1)[1]          # name as in the data file (keeps ATP, GDP, ...)
    ax.annotate(name, (r['log2FC_KI_vs_WT'], -np.log10(r['p_value'])), fontsize=7, color=INK,
                xytext=(4 if r['log2FC_KI_vs_WT'] >= 0 else -4, 3), textcoords='offset points',
                ha='left' if r['log2FC_KI_vs_WT'] >= 0 else 'right')

ax.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.set_xlabel(f'log2 fold change ({KI_NAME} / {WT_NAME})')
ax.set_ylabel('-log10(p), Welch t-test')
n_up, n_down = (res['Direction'] == 'higher in KI').sum(), (res['Direction'] == 'lower in KI').sum()
n_small = res['Direction'].str.contains('small change').sum()
ax.set_title(f"{KI_NAME} vs pooled {WT_NAME} – {REGION}, {SEX_NAME.get(SEX, SEX)} "
             f"(WT {wt.sum()}, KI {ki.sum()})\n"
             f"{n_up} higher, {n_down} lower in KI ({'FDR' if P_COL == 'FDR' else 'p'} < {ALPHA}, "
             f"≥ {2 ** FC_CUTOFF:.1f}-fold) | {n_small} more with smaller change | FDR < {ALPHA}: {(res['FDR'] < ALPHA).sum()}",
             color=INK, fontsize=10)
sig_txt = f"{'FDR' if P_COL == 'FDR' else 'p'} < {ALPHA}"
handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor=col,
                      markeredgecolor='white', label=l)
           for col, l in [
               (UP, f'higher in KI: {sig_txt} and ≥ {2 ** FC_CUTOFF:.1f}-fold'),
               (DOWN, f'lower in KI: {sig_txt} and ≥ {2 ** FC_CUTOFF:.1f}-fold'),
               (UP_SMALL, f'higher in KI: {sig_txt}, < {2 ** FC_CUTOFF:.1f}-fold'),
               (DOWN_SMALL, f'lower in KI: {sig_txt}, < {2 ** FC_CUTOFF:.1f}-fold'),
               (NS, f'not significant ({"FDR" if P_COL == "FDR" else "p"} ≥ {ALPHA})')]]
# legend below the plot, so it never covers points or labels
ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
          loc='upper center', bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()

tag = f"{KI_NAME}_vs_{WT_NAME}_{REGION}_{SEX}"
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(output_dir, f'Volcano_{tag}.{ext}'), dpi=300, bbox_inches='tight')
plt.show()

# ---------- table ----------
res.to_csv(os.path.join(output_dir, f'Volcano_{tag}.csv'), index=False)
print(f"\n{len(res)} metabolites tested | nominal p < {ALPHA}: {(res['p_value'] < ALPHA).sum()} "
      f"| FDR < {ALPHA}: {(res['FDR'] < ALPHA).sum()} | ≥ {2 ** FC_CUTOFF:.1f}-fold: {n_up} up, {n_down} down "
      f"| smaller change: {n_small}")
print(res.head(20).round(4).to_string(index=False))
print(f"\nSaved figure and table to {output_dir}")
