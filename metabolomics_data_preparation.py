# ==============================================================================
# Metabolomics data preparation (25-M113) - one cell, all parameters on top
# Paste this whole file into ONE Jupyter cell (replaces cells 4 - 47).
# ==============================================================================
import os
import re
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 1. PARAMETERS - change only this block
# ------------------------------------------------------------------------------
META_FILE = r"O:\metabolom\Result\Core_results\25-M-113\metaData_NN.xlsx"
POS_FILE  = r"O:\metabolom\Result\Core_results\25-M-113\raw data\new version_09.2026\25-M113-pos-export matrix-area.xlsx"
NEG_FILE  = r"O:\metabolom\Result\Core_results\25-M-113\raw data\new version_09.2026\25-M113-neg-matrics-area-revised.xlsx"
OUT_DIR   = r"C:\Users\nnekooiemarnany\Desktop\Metabolom_analys\Data_analysis"

META_ID_COL    = 'Sample ID'     # sample column in metadata
METAB_NAME_COL = 'Sample Name'   # metabolite-name column in pos/neg files

QC_PATTERN      = r'QC'                   # columns that are pooled QC samples
EXCLUDE_PATTERN = r'HB|Blank|blank'       # columns/rows to drop (standards mixes, blanks)

# sample names that differ between files -> fixed everywhere (after '-' -> '_')
RENAME_SAMPLES = {
    '55H_AaspartateP4_M_KI_52': '55H_AP4_M_KI_52',  # typo in an older pos export
    '86H_AP4_M_KI_9':           '86H_AP4_M_KI_98',  # metab file; mouse 86 is ..._98 (see 86C) - check!
}
DROP_SAMPLES = []   # e.g. ['55H_AP4_M_KI_52'] - pos file labels it 'Standard', decide yourself

MAX_MISSING = 0.80   # drop a metabolite if missing in > 80 % of biological samples
QC_CV_MAX   = None   # e.g. 0.30 -> drop metabolites with CV > 30 % in QCs; None = no QC filter
IMPUTE      = 'half_min'   # 'half_min' (1/2 of the metabolite's minimum), 'min', or None

RENAME_META_COLS = {'Pr_Con.': 'Pr_Con'}
GROUP_COLS       = {'Group_Region': 'Region of brain', 'Group_APOE': 'line of APOE'}

# ------------------------------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------------------------------
def std_name(s):
    """Unify sample names: '-' -> '_', strip spaces, apply RENAME_SAMPLES."""
    s = str(s).strip().replace('-', '_')
    return RENAME_SAMPLES.get(s, s)

def load_matrix(path, mode):
    """Read an export matrix -> numeric DataFrame (rows = metabolites, cols = samples)."""
    df = pd.read_excel(path)
    df.columns = [c if c == METAB_NAME_COL else std_name(c) for c in df.columns]

    # 'Sample Type' row: report non-'Unknown' samples, then remove the row
    types = df.loc[df[METAB_NAME_COL] == 'Sample Type'].iloc[:1, 1:].squeeze(axis=0)
    odd = types[types != 'Unknown'] if len(types) else types
    if len(odd):
        print(f"[{mode}] Sample Type not 'Unknown': {odd.to_dict()}")
    df = df[df[METAB_NAME_COL] != 'Sample Type']

    df = df.dropna(subset=[METAB_NAME_COL]).set_index(METAB_NAME_COL)
    df.index = df.index.str.strip()
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace(0, np.nan)                                    # 0 area = not detected
    df = df.drop(columns=df.columns[df.columns.str.contains(EXCLUDE_PATTERN)])
    df = df.drop(columns=[c for c in DROP_SAMPLES if c in df.columns])
    return df

def filter_and_impute(df, mode):
    qc_cols  = df.columns[df.columns.str.contains(QC_PATTERN)]
    bio_cols = df.columns.difference(qc_cols, sort=False)
    n0 = len(df)

    # a) missingness judged on biological samples only
    miss = df[bio_cols].isna().mean(axis=1)
    df = df[miss <= MAX_MISSING]
    n_miss = len(df)

    # b) optional QC reproducibility filter
    if QC_CV_MAX is not None and len(qc_cols):
        qc = df[qc_cols]
        cv = qc.std(axis=1) / qc.mean(axis=1)
        df = df[cv <= QC_CV_MAX]

    # c) imputation of the remaining (partial) NaN
    still = df[bio_cols].isna().sum(axis=1)
    still = still[still > 0]
    if IMPUTE in ('half_min', 'min'):
        factor = 0.5 if IMPUTE == 'half_min' else 1.0
        fill = df[bio_cols].min(axis=1) * factor                   # per metabolite
        df = df.apply(lambda row: row.fillna(fill[row.name]), axis=1)

    print(f"[{mode}] metabolites: {n0} raw -> {n_miss} after <= {MAX_MISSING:.0%} missing"
          f" -> {len(df)} after QC-CV filter | imputed ({IMPUTE}): {still.to_dict()}")
    df.index = f'{mode}_' + df.index
    return df[bio_cols], df[qc_cols]

# ------------------------------------------------------------------------------
# 3. LOAD + CLEAN
# ------------------------------------------------------------------------------
meta = pd.read_excel(META_FILE).rename(columns=RENAME_META_COLS)
meta = meta.dropna(subset=[META_ID_COL])
meta[META_ID_COL] = meta[META_ID_COL].map(std_name)
meta = meta[~meta[META_ID_COL].str.contains(QC_PATTERN + '|' + EXCLUDE_PATTERN)]
meta = meta[~meta[META_ID_COL].isin(DROP_SAMPLES)]

pos_bio, pos_qc = filter_and_impute(load_matrix(POS_FILE, 'POS'), 'POS')
neg_bio, neg_qc = filter_and_impute(load_matrix(NEG_FILE, 'NEG'), 'NEG')

# ------------------------------------------------------------------------------
# 4. COMBINE POS + NEG, MATCH WITH METADATA, MERGE
# ------------------------------------------------------------------------------
features = pd.concat([pos_bio, neg_bio], axis=0).T          # rows = samples
features.index.name = META_ID_COL
qc_features = pd.concat([pos_qc, neg_qc], axis=0).T

only_meta  = set(meta[META_ID_COL]) - set(features.index)
only_metab = set(features.index) - set(meta[META_ID_COL])
print(f"\nOnly in metadata    : {sorted(only_meta)  or 'none'}")
print(f"Only in metabolomics: {sorted(only_metab) or 'none'}")

master = meta.merge(features.reset_index(), on=META_ID_COL, how='inner')
for new, old in GROUP_COLS.items():
    master[new] = master[old].astype(str).str.strip()

feature_cols = list(features.columns)

# ------------------------------------------------------------------------------
# 5. FINAL CHECKS + SAVE
# ------------------------------------------------------------------------------
print(f"\nMaster: {master.shape[0]} samples x {len(feature_cols)} metabolites "
      f"({pos_bio.shape[0]} POS + {neg_bio.shape[0]} NEG), {qc_features.shape[0]} QC samples")
print("NaN left in features:", int(master[feature_cols].isna().sum().sum()))
for new in GROUP_COLS:
    print(f"{new}: {master[new].value_counts().to_dict()}")

master.to_csv(os.path.join(OUT_DIR, 'master_analysis_file.csv'), index=False)
qc_features.to_csv(os.path.join(OUT_DIR, 'qc_features.csv'))
print(f"\nSaved master_analysis_file.csv and qc_features.csv to {OUT_DIR}")
