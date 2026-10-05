# ══════════════════════════════════════════════════════════════════
# Outliers of the PCA subset — run AFTER pca_subset_apoe_cell
# (uses sub, X, X_pareto, pca, scores, pca_df from that cell)
#   1) Hotelling T²  : sample far from the others INSIDE the PCA model (95% / 99% limits)
#   2) Q residual    : sample whose profile the model cannot reproduce
#   3) for the most extreme sample: metadata + metabolites that make it different
# ══════════════════════════════════════════════════════════════════
from scipy.stats import f

GROUP = 'APOE4'                              # look closer at this APOE group (None = whole subset)

n, A = scores.shape
# 1) Hotelling T² with F-distribution limits
t2 = (scores ** 2 / scores.var(axis=0, ddof=1)).sum(axis=1)
t2_lim = {lvl: A * (n - 1) * (n + 1) / (n * (n - A)) * f.ppf(lvl, A, n - A) for lvl in (0.95, 0.99)}

# 2) Q residual (squared reconstruction error); limit = mean + 3 SD of the subset
recon = scores @ pca.components_ + pca.mean_
q = ((X_pareto.values - recon) ** 2).sum(axis=1)
q_lim = q.mean() + 3 * q.std(ddof=1)

meta = [c for c in ['Injection_number', 'injection #', 'PQN_factor_POS', 'PQN_factor_NEG']
        if c in sub.columns]
batch_col = next((c for c in sub.columns if c.lower().startswith('batch segment')), None)
res = pd.DataFrame({'Sample ID': sub['Sample ID'].values,
                    'APOE': sub['APOE_label'].values,
                    'PC1': scores[:, 0].round(2), 'PC2': scores[:, 1].round(2),
                    'T2': t2.round(1), 'Q': q.round(1)})
for c in meta + ([batch_col] if batch_col else []):
    res[c] = sub[c].values
res['T2>95%'] = t2 > t2_lim[0.95]
res['T2>99%'] = t2 > t2_lim[0.99]
res['Q>mean+3SD'] = q > q_lim

print(f"Hotelling T² limits: 95% = {t2_lim[0.95]:.1f}, 99% = {t2_lim[0.99]:.1f} | "
      f"Q limit (mean+3SD) = {q_lim:.1f}   (n = {n}, {A} PCs)")
show = res if GROUP is None else res[res['APOE'] == GROUP]
print(show.sort_values('T2', ascending=False).to_string(index=False))

# 3) the most extreme sample of the group
worst = show.sort_values('T2', ascending=False).iloc[0]['Sample ID']
row = (sub['Sample ID'] == worst).values
z = (X - X[~row].mean()) / X[~row].std()          # vs the other samples of the subset
zs = z[row].iloc[0].sort_values(key=abs, ascending=False)
print(f"\nMost extreme {GROUP or ''} sample: {worst}")
print("Metabolites that differ most (z vs the other samples of this subset):")
print(zs.head(15).round(2).to_string())
