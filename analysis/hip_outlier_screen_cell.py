# ══════════════════════════════════════════════════════════════════
# Hippocampus - global outlier screen scatter plots
# Nothing is removed: samples 9H-12H (homogenization issue) are marked with a black ring
# Fully self-contained: rebuilds df_target from scratch
#
# Which file?
#   RAW (master_with_QC3.csv)        -> shows dilution / homogenization problems
#                                      (a sample with little tissue sits low)
#   PQN (PQNregion_log2_samples.csv) -> dilution already removed, so the mean level is
#                                      ~flat by design; points far off = other problems
# The cell detects whether the file is already log2 and only log-transforms raw data.
# ══════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 0a. LOAD DATA — update path if needed ──
file_path = r"O:\metabolom\Result\Core_results_original_data\25-M-113\raw data\new version_09.2026\last_raw data\master_with_QC3.csv"
REGION_KEY = 'hip'                       # matches 'Hippocmpus', 'Hippocampus', ...
df_raw = pd.read_csv(file_path)
df_raw.columns = [' '.join(str(c).split()) for c in df_raw.columns]   # remove line breaks in names

# ── 0b. samples only; documented homogenization-issue samples are KEPT and marked ──
homog_prefixes = ('9H_', '10H_', '11H_', '12H_')
if 'Type' in df_raw.columns:
    df_raw = df_raw[df_raw['Type'].astype(str).str.strip() == 'Sample']
df_raw_clean = df_raw.reset_index(drop=True)
df_raw_clean['homog_issue'] = (df_raw_clean['Sample ID'].astype(str).str.strip()
                               .str.startswith(homog_prefixes))

# ── 0c. metabolite columns: POS_/NEG_ (any case), excluding 13C internal standards ──
all_metab = [c for c in df_raw_clean.columns
             if c.upper().startswith(('POS_', 'NEG_')) and '13C' not in c]
assert all_metab, "no POS_/NEG_ metabolite columns found"

# ── 0d. subset to Hippocampus only (text labels, spaces stripped) ──
for c in ['Region of brain', 'Gender', 'Genotype', 'line of APOE']:
    if c in df_raw_clean.columns:
        df_raw_clean[c] = df_raw_clean[c].astype(str).str.strip()
regions = df_raw_clean['Region of brain'].unique()
df_target = df_raw_clean[df_raw_clean['Region of brain'].str.lower().str.contains(REGION_KEY)
                         ].reset_index(drop=True)
assert len(df_target), f"no rows match '{REGION_KEY}'. Regions in file: {list(regions)}"

# readable labels: use the text already in the file
df_target['Sex_label']      = df_target['Gender'].replace({'M': 'Male', 'F': 'Female'})
df_target['Genotype_label'] = df_target['Genotype']
df_target['APOE_label']     = df_target['line of APOE']
inj_col = next(c for c in df_target.columns if c.lower().startswith('injection'))
df_target['Injection_number'] = pd.to_numeric(df_target[inj_col], errors='coerce')
print(f"Nothing removed; marked homog_issue (black ring in plots): "
      f"{df_target.loc[df_target['homog_issue'], 'Sample ID'].tolist()}")

# ── 0e. mean log2 intensity per sample (y-axis of both plots) ──
vals = df_target[all_metab].apply(pd.to_numeric, errors='coerce')
already_log = vals.max().max() < 60          # log2 data never gets this high; raw data does
log_metab = vals if already_log else np.log2(vals.where(vals > 0))
df_target['mean_log_intensity'] = log_metab.mean(axis=1, skipna=True)

mean_val = df_target['mean_log_intensity'].mean()
sd_val   = df_target['mean_log_intensity'].std()
out_hi, out_lo = mean_val + 3 * sd_val, mean_val - 3 * sd_val

print(f"Hippocampus samples: {len(df_target)} | metabolites: {len(all_metab)} | "
      f"data {'already log2' if already_log else 'raw -> log2'}")
print("Sex:", df_target['Sex_label'].value_counts().to_dict(),
      "| Genotype:", df_target['Genotype_label'].value_counts().to_dict(),
      "| APOE:", df_target['APOE_label'].value_counts().to_dict())
flag = df_target[(df_target['mean_log_intensity'] > out_hi) | (df_target['mean_log_intensity'] < out_lo)]
print(f"Outside mean ± 3 SD: {len(flag)}",
      flag[['Sample ID', 'mean_log_intensity']].round(2).to_string(index=False) if len(flag) else "")

# ── 1. single-panel global scatter, x-axis = Sample ID ──
plt.figure(figsize=(14, 6))
sns.scatterplot(data=df_target, x=df_target.index, y='mean_log_intensity',
                hue='Genotype_label', style='APOE_label')
h = df_target[df_target['homog_issue']]
plt.scatter(h.index, h['mean_log_intensity'], s=160, facecolors='none',
            edgecolors='black', linewidths=1.5, label='homog. issue')
plt.axhline(out_hi, color='red', linestyle='--')
plt.axhline(out_lo, color='red', linestyle='--')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
plt.xticks(df_target.index, df_target['Sample ID'], rotation=90, fontsize=6)
plt.xlabel('Sample ID')
plt.ylabel('mean log2 intensity')
plt.title('Hippocampus - global outlier screen, colored by Genotype / APOE')
plt.tight_layout()
plt.show()

# ── 2. faceted by Sex, x-axis = injection order ──
g = sns.FacetGrid(df_target, col='Sex_label', height=5, aspect=1.3)
g.map_dataframe(sns.scatterplot, x='Injection_number', y='mean_log_intensity',
                hue='Genotype_label', style='APOE_label')
for (sex,), ax in zip([(n,) for n in g.col_names], g.axes.flat):
    h = df_target[df_target['homog_issue'] & (df_target['Sex_label'] == sex)]
    ax.scatter(h['Injection_number'], h['mean_log_intensity'], s=160, facecolors='none',
               edgecolors='black', linewidths=1.5)
    ax.axhline(out_hi, color='red', linestyle='--')
    ax.axhline(out_lo, color='red', linestyle='--')
g.add_legend()
g.set_axis_labels('Injection number', 'mean log2 intensity')
plt.tight_layout()
plt.show()
