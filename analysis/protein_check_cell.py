# ============================================================
# PROTEIN (Pr_Con) CHECK + DATA-DRIVEN CTX/Hip FACTOR
#   run AFTER the FINAL NORMALIZATION cell (uses df, df_norm, out, s_mask, ...)
#
#   1) parse Pr_Con ('1.8 mg/ml', '3.45mg/ml', ...) to numbers
#   2) within each region: does protein follow the PQN factor?
#        PQN factor = how concentrated each extract is (from the metabolites)
#        protein    = how concentrated each extract is (from a separate assay)
#        -> if they correlate, protein measures the tissue amount and can be trusted
#   3) CTX/Hip factor from protein of ALL samples (replaces the pilot 1.31, n = 6)
#   4) save PQN-within-region data with CTX divided by that factor
#      -> for ABSOLUTE CTX vs Hip only; KI vs WT keeps using PQNregion_log2_samples.csv
# ============================================================
import re
from scipy.stats import spearmanr

PROT_COL = 'Pr_Con'
N_BOOT = 2000
rng = np.random.default_rng(0)

# ---------- 1) parse ----------
prot = pd.to_numeric(df[PROT_COL].astype(str).str.replace(',', '.')
                     .str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
bad = s_mask & prot.isna()
print(f"Protein parsed for {(s_mask & prot.notna()).sum()} of {s_mask.sum()} samples")
if bad.any():
    print("Could not parse:", df.loc[bad, [ID_COL, PROT_COL]].to_string(index=False))
reg = df[REGION_COL]
print("Median protein (mg/ml) by region:",
      prot[s_mask].groupby(reg[s_mask]).median().round(2).to_dict())

# ---------- 2) protein vs PQN factor, within region ----------
# geometric mean of the POS and NEG factors = one dilution estimate per sample
pqn_f = np.sqrt(df_norm['PQN_factor_POS'] * df_norm['PQN_factor_NEG'])
print("\nProtein vs PQN factor within region (Spearman; r > ~0.4 = protein tracks tissue amount):")
for r in regions:
    m = s_mask & (reg == r) & prot.notna()
    rho, p = spearmanr(prot[m], pqn_f[m])
    print(f"  {r}: rho = {rho:.2f}, p = {p:.3g}, n = {m.sum()}")

# ---------- 3) CTX / Hip factor from protein ----------
hip_label = [r for r in regions if r != CTX_LABEL][0]
ctx_p = prot[s_mask & (reg == CTX_LABEL)].dropna().values
hip_p = prot[s_mask & (reg == hip_label)].dropna().values

# pair CTX and Hip of the same mouse: '16C_AP4_M_KI_122' and '16H_AP4_M_KI_122' -> '16_AP4_M_KI_122'
mouse = df[ID_COL].astype(str).str.strip().str.replace(r'^(\d+)[CH]_', r'\1_', regex=True)
pairs = pd.DataFrame({'mouse': mouse, 'reg': reg, 'prot': prot})[s_mask].pivot_table(
    index='mouse', columns='reg', values='prot', aggfunc='first').dropna()

if len(pairs) >= 10:
    lr = np.log(pairs[CTX_LABEL] / pairs[hip_label]).values
    est = np.exp(lr.mean())
    boot = [np.exp(rng.choice(lr, len(lr)).mean()) for _ in range(N_BOOT)]
    how = f"paired, {len(pairs)} mice (geometric mean of CTX/Hip per mouse)"
else:
    est = np.median(ctx_p) / np.median(hip_p)
    boot = [np.median(rng.choice(ctx_p, len(ctx_p))) / np.median(rng.choice(hip_p, len(hip_p)))
            for _ in range(N_BOOT)]
    how = f"unpaired, {len(ctx_p)} CTX vs {len(hip_p)} Hip (ratio of medians)"
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nCTX/Hip protein factor = {est:.2f}  (95% CI {lo:.2f}-{hi:.2f}; {how})")
print(f"Pilot-weight factor was 1.31 (95% CI 1.10-1.56)")

# ---------- 4) save ----------
PROT_FACTOR = {'protein': est, 'protein_low': lo, 'protein_high': hi}
is_ctx = out[REGION_COL] == CTX_LABEL
for k, fct in PROT_FACTOR.items():
    adj = out.copy()
    adj.loc[is_ctx, metabolite_cols] -= np.log2(fct)
    p = os.path.join(output_dir, f'PQNregion_log2_CTXadj_{k}.csv')
    adj.to_csv(p, index=False)
    print(f"Saved CTX / {fct:.2f} ({k}) to {os.path.basename(p)}  -> absolute CTX vs Hip only")
