# ══════════════════════════════════════════════════════════════════
# Volcano plot: is the KI effect (KI / pooled WT) different in APOE4 vs APOE3?
#   default: Hippocampus, Female
#     ratio APOE4 = KI-APOE4 / pooled WT (APOE2 + APOE3 + APOE4)
#     ratio APOE3 = KI-APOE3 / pooled WT (same pooled WT)
#   input = PQNregion_log2_samples_clean.csv (PQN within region, WT reference, log2)
#   - every KI mouse is expressed relative to the pooled WT mean:
#       log2(KI / WT) = log2(KI mouse) - mean log2(pooled WT)
#   - difference of the two ratios (log2 of the "ratio of ratios"):
#       delta log2FC = log2FC(APOE4 KI / WT) - log2FC(APOE3 KI / WT)
#     NOTE: because both lines use the SAME pooled WT, the WT term cancels out,
#     so this is equal to mean log2(KI-APOE4) - mean log2(KI-APOE3).
#   - KI-APOE4 and KI-APOE3 are DIFFERENT mice -> unpaired Welch t-test
#   - Benjamini-Hochberg FDR across all metabolites
#   - point colours (see legend):
#       strong red / blue = significant AND >= fold-change cut-off (KI effect larger / smaller in APOE4)
#       light red / blue  = significant but below the fold-change cut-off
#       grey              = not significant
#   - Figure 1: volcano (delta log2FC vs -log10 p)
#   - Figure 2: log2FC(KI/WT) in APOE3 (x) vs APOE4 (y), coloured as in Figure 1
#   - saves: figures (PNG + PDF) + full results table
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
LINE_A  = 'APOE4'                          # KI line A (numerator of the comparison)
LINE_B  = 'APOE3'                          # KI line B (reference of the comparison)
WT_APOE = ['APOE2', 'APOE3', 'APOE4']      # APOE line(s) pooled for the WT reference
REGION  = 'Hippocmpus'                     # exact label in the file (spelling as in the metadata)
SEX     = 'F'                              # 'F' or 'M'

# ---------- cut-offs ----------
P_COL     = 'p_value'          # 'p_value' (nominal) or 'FDR' -> which value decides "significant"
ALPHA     = 0.05
FC_CUTOFF = np.log2(1.5)       # |delta log2FC| >= log2(1.5) = ratios differ at least 1.5-fold
N_LABELS  = 15                 # label at most this many significant points (smallest p first)
N_TOP     = 5                  # also label the N_TOP smallest p and every |delta log2FC| >= 1, even if n.s.

REGION_COL, SEX_COL, GENO_COL, APOE_COL = 'Region of brain', 'Gender', 'Genotype', 'line of APOE'
WT, KI = 'WT', 'KI'
SEX_NAME = {'F': 'Female', 'M': 'Male'}

# short names for titles / files, e.g. "APOE4", "APOE3", "WT-APOE2+3+4"
A, B = LINE_A, LINE_B
WT_NAME = 'WT-APOE' + '+'.join(a.replace('APOE', '') for a in WT_APOE)
UP_LAB, DOWN_LAB = f'KI effect higher in {A}', f'KI effect lower in {A}'

# diverging pair (red = KI/WT higher in line A, blue = lower in line A); light tints = significant
# but small change; neutral gray = not significant
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
a_mask  = in_region_sex & (df[GENO_COL] == KI) & (df[APOE_COL] == A)
b_mask  = in_region_sex & (df[GENO_COL] == KI) & (df[APOE_COL] == B)
wt_mask = in_region_sex & (df[GENO_COL] == WT) & df[APOE_COL].isin(WT_APOE)

print(f"{REGION} / {SEX_NAME.get(SEX, SEX)}: KI-{A} n = {a_mask.sum()}, KI-{B} n = {b_mask.sum()}, "
      f"{WT_NAME} n = {wt_mask.sum()}")
assert min(a_mask.sum(), b_mask.sum(), wt_mask.sum()) >= 3, \
    "fewer than 3 mice in a group: check the labels in SUBGROUP"
print("  WT per APOE line:", ', '.join(f"{a} n={(df.loc[wt_mask, APOE_COL] == a).sum()}" for a in WT_APOE))
print(f"  KI-{A}:", ', '.join(df.loc[a_mask, 'Sample ID']))
print(f"  KI-{B}:", ', '.join(df.loc[b_mask, 'Sample ID']))
print("  WT:", ', '.join(df.loc[wt_mask, 'Sample ID']))

# ---------- each KI mouse relative to the pooled WT mean: log2(KI / WT) ----------
wt_mean = X[wt_mask].mean()                    # mean log2 of pooled WT, per metabolite
rel_a = X[a_mask] - wt_mean                    # log2(KI-A mouse / pooled WT)
rel_b = X[b_mask] - wt_mean                    # log2(KI-B mouse / pooled WT)

# ---------- per-metabolite test ----------
rows = []
for c in metabolite_cols:
    a, b = rel_a[c].dropna(), rel_b[c].dropna()
    if len(a) < 3 or len(b) < 3 or X.loc[wt_mask, c].notna().sum() < 3:
        continue
    t, p = ttest_ind(a, b, equal_var=False)                 # Welch
    rows.append({'Metabolite': c,
                 f'log2FC_KI_vs_WT_{A}': a.mean(), f'log2FC_KI_vs_WT_{B}': b.mean(),
                 f'delta_log2FC_{A}_minus_{B}': a.mean() - b.mean(),
                 't': t, 'p_value': p, f'n_KI_{A}': len(a), f'n_KI_{B}': len(b),
                 'n_WT': int(X.loc[wt_mask, c].notna().sum())})
res = pd.DataFrame(rows)
DCOL = f'delta_log2FC_{A}_minus_{B}'
res['FDR'] = multipletests(res['p_value'], method='fdr_bh')[1]
is_sig = res[P_COL] < ALPHA                                  # passes the p (or FDR) threshold
is_big = res[DCOL].abs() >= FC_CUTOFF                        # passes the fold-change cut-off
is_up = res[DCOL] > 0
res['Direction'] = np.select(
    [is_sig & is_big & is_up, is_sig & is_big & ~is_up, is_sig & ~is_big & is_up, is_sig & ~is_big & ~is_up],
    [UP_LAB, DOWN_LAB, f'{UP_LAB} (small change)', f'{DOWN_LAB} (small change)'],
    default='n.s.')
res = res.sort_values('p_value').reset_index(drop=True)

color = res['Direction'].map({UP_LAB: UP, DOWN_LAB: DOWN,
                              f'{UP_LAB} (small change)': UP_SMALL,
                              f'{DOWN_LAB} (small change)': DOWN_SMALL, 'n.s.': NS})
n_up, n_down = (res['Direction'] == UP_LAB).sum(), (res['Direction'] == DOWN_LAB).sum()
n_small = res['Direction'].str.contains('small change').sum()
sig_txt = f"{'FDR' if P_COL == 'FDR' else 'p'} < {ALPHA}"
fold_txt = f"{2 ** FC_CUTOFF:.1f}-fold"
handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7, markerfacecolor=col,
                      markeredgecolor='white', label=l)
           for col, l in [
               (UP, f'{UP_LAB}: {sig_txt} and ≥ {fold_txt}'),
               (DOWN, f'{DOWN_LAB}: {sig_txt} and ≥ {fold_txt}'),
               (UP_SMALL, f'{UP_LAB}: {sig_txt}, < {fold_txt}'),
               (DOWN_SMALL, f'{DOWN_LAB}: {sig_txt}, < {fold_txt}'),
               (NS, f'not significant ({"FDR" if P_COL == "FDR" else "p"} ≥ {ALPHA})')]]

lab_idx = (res.index[res['Direction'] != 'n.s.'][:N_LABELS]
           .union(res.index[:N_TOP]).union(res.index[res[DCOL].abs() >= 1]))
lab = res.loc[lab_idx]
tag = f"KIvsWT_{A}_vs_{B}_{WT_NAME}_{REGION}_{SEX}"


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ---------- Figure 1: volcano ----------
fig, ax = plt.subplots(figsize=(7.5, 7.0))
x, y = res[DCOL], -np.log10(res['p_value'])
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

for _, r in lab.iterrows():
    name = r['Metabolite'].split('_', 1)[1]          # name as in the data file (keeps ATP, GDP, ...)
    ax.annotate(name, (r[DCOL], -np.log10(r['p_value'])), fontsize=7, color=INK,
                xytext=(4 if r[DCOL] >= 0 else -4, 3), textcoords='offset points',
                ha='left' if r[DCOL] >= 0 else 'right')

style_axes(ax)
ax.set_xlabel(f'Δ log2 fold change: log2(KI-{A} / WT) − log2(KI-{B} / WT)')
ax.set_ylabel(f'-log10(p), Welch t-test (KI-{A} vs KI-{B}, each / pooled WT)')
ax.set_title(f"KI / pooled {WT_NAME}: {A} vs {B} – {REGION}, {SEX_NAME.get(SEX, SEX)} "
             f"(KI-{A} {a_mask.sum()}, KI-{B} {b_mask.sum()}, WT {wt_mask.sum()})\n"
             f"{n_up} higher, {n_down} lower in {A} ({sig_txt}, ≥ {fold_txt}) | "
             f"{n_small} more with smaller change | FDR < {ALPHA}: {(res['FDR'] < ALPHA).sum()}",
             color=INK, fontsize=9.5)
ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
          loc='upper center', bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(output_dir, f'Volcano_{tag}.{ext}'), dpi=300, bbox_inches='tight')
plt.show()

# ---------- Figure 2: log2FC(KI/WT) in line B vs line A ----------
fig, ax = plt.subplots(figsize=(7.0, 7.4))
fa, fb = res[f'log2FC_KI_vs_WT_{A}'], res[f'log2FC_KI_vs_WT_{B}']
ax.scatter(fb, fa, c=color, s=40, edgecolor='white', linewidth=0.8, zorder=3)
lim2 = max(abs(fa).max(), abs(fb).max()) * 1.15
ax.set_xlim(-lim2, lim2)
ax.set_ylim(-lim2, lim2)
ax.plot([-lim2, lim2], [-lim2, lim2], color=INK2, linestyle='--', linewidth=0.8)   # same effect in both
ax.text(lim2, lim2, 'same KI effect\nin both lines ', color=INK2, fontsize=7, ha='right', va='top')
ax.axhline(0, color=GRID, linewidth=1, zorder=0)
ax.axvline(0, color=GRID, linewidth=1, zorder=0)
for _, r in lab.iterrows():
    name = r['Metabolite'].split('_', 1)[1]
    ax.annotate(name, (r[f'log2FC_KI_vs_WT_{B}'], r[f'log2FC_KI_vs_WT_{A}']), fontsize=7, color=INK,
                xytext=(4, 3), textcoords='offset points')
style_axes(ax)
ax.set_aspect('equal')
ax.set_xlabel(f'log2 fold change KI-{B} / pooled WT')
ax.set_ylabel(f'log2 fold change KI-{A} / pooled WT')
ax.set_title(f"KI / pooled {WT_NAME} – {A} vs {B}, {REGION}, {SEX_NAME.get(SEX, SEX)}\n"
             f"points off the diagonal: KI effect differs between {A} and {B}",
             color=INK, fontsize=9.5)
ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2,
          loc='upper center', bbox_to_anchor=(0.5, -0.10))
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(output_dir, f'FCvsFC_{tag}.{ext}'), dpi=300, bbox_inches='tight')
plt.show()

# ---------- table ----------
res.to_csv(os.path.join(output_dir, f'Volcano_{tag}.csv'), index=False)
print(f"\n{len(res)} metabolites tested | nominal p < {ALPHA}: {(res['p_value'] < ALPHA).sum()} "
      f"| FDR < {ALPHA}: {(res['FDR'] < ALPHA).sum()} | ≥ {fold_txt}: {n_up} higher in {A}, "
      f"{n_down} lower in {A} | smaller change: {n_small}")
print(res.head(20).round(4).to_string(index=False))
print(f"\nSaved figures and table to {output_dir}")
