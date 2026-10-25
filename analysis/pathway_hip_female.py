"""
Pathway (metabolite-set) analysis: each KI line vs pooled WT
Female hippocampus

Input : drift-corrected, PQN-normalised, log2-transformed data (one row per sample)
Method: for every pathway and KI line
          1) per metabolite: Welch t statistic, KI vs pooled WT
          2) pathway score:
               direction = mean t   (are the pathway's metabolites shifted together up or down?)
               change    = mean |t| (are they changed, in any direction?)
          3) p-value by permutation: KI and WT labels are shuffled between the mice,
             everything is recomputed, N_PERMUTATIONS times (keeps the correlation between
             metabolites, so pathways that share metabolites are handled correctly)
          4) FDR (Benjamini-Hochberg) across pathways, per KI line
        This is a self-contained test, the same idea as MetaboAnalyst's QEA / global test.
Pathways: KEGG mouse metabolic pathways (file made by build_kegg_sets.py)
Output: table of all pathways x KI lines, and a dot plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------- settings
INPUT_FILE = "DriftCor_PQNregion_log2_samples_clean.csv"
OUTPUT_FILE = "pathways_Hippocampus_Female.csv"
PLOT_FILE = "pathways_Hippocampus_Female.png"
REGION = "Hippocmpus"          # spelled as in the data file
SEX = "F"
APOE_LINES = ["APOE2", "APOE3", "APOE4"]
EXCLUDE_SAMPLES = ["34H_AP3_F_WT_163"]
N_PERMUTATIONS = 5000
SEED = 1
MIN_SIZE = 3                   # pathways with fewer measured metabolites are skipped

# Pathway sets: official KEGG mouse metabolic pathways (downloaded from rest.kegg.jp with
# build_kegg_sets.py), restricted to the measured metabolites, at least 3 per pathway.
PATHWAYS = {
    "Purine metabolism": [
        "ADENINE", "ADENOSINE", "ADENOSINE 3',5'-DIPHOSPHATE", "ADENOSINE DIPHOSPHATE RIBOSE",
        "ADENOSINE-MONOPHOSPHATE", "ADENYLSUCCINIC ACID", "ATP/dGTP", "CYCLIC AMP",
        "D-RIBOSE 5-PHOSPHATE", "DEOXYGUANOSINE-MONOPHOSPHATE", "GDP", "GLUTAMINE", "GLYCINE",
        "GTP", "GUANINE", "GUANOSINE", "GUANOSINE MONOPHOSPHATE", "HYPOXANTHINE", "INOSINE",
        "INOSINE-MONOPHOSPHATE", "URIC ACID", "XANTHINE"
    ],
    "Aminoacyl-tRNA biosynthesis": [
        "ALLOTHREONINE/THREONINE", "ARGININE", "ASPARTATE", "GLUTAMATE", "GLUTAMINE",
        "GLYCINE", "HISTIDINE", "L-ALANINE", "LEUCINE/ISOLEUCINE", "LYSINE", "METHIONINE",
        "PHENYLALANINE", "PROLINE", "SERINE", "TRYPTOPHAN", "TYROSINE"
    ],
    "D-Amino acid metabolism": [
        "ALLOTHREONINE/THREONINE", "ARGININE", "ASPARTATE", "GLUTAMATE", "GLUTAMINE",
        "GLYCINE", "HISTIDINE", "L-ALANINE", "LYSINE", "METHIONINE", "N-ACETYLGLUTAMATE",
        "PHENYLALANINE", "PROLINE", "SERINE"
    ],
    "Glycine, serine and threonine metabolism": [
        "2-PHOSPHOGLYCERATE", "ALLOTHREONINE/THREONINE", "ASPARTATE", "BETAINE", "CREATINE",
        "GLYCINE", "GUANIDINOACETATE", "HOMOSERINE", "SARCOSINE", "SERINE", "TRYPTOPHAN"
    ],
    "Arginine and proline metabolism": [
        "4-GUANIDINOBUTANOATE", "ARGININE", "CREATINE", "CREATININE", "GAMMA-AMINOBUTYRATE",
        "GLUTAMATE", "GUANIDINOACETATE", "PROLINE", "S-ADENOSYLMETHIONINE", "SARCOSINE"
    ],
    "Alanine, aspartate and glutamate metabolism": [
        "ADENYLSUCCINIC ACID", "ASPARTATE", "FUMARATE", "GAMMA-AMINOBUTYRATE", "GLUTAMATE",
        "GLUTAMINE", "L-ALANINE", "N-ACETYLASPARTATE", "SUCCINATE"
    ],
    "Cysteine and methionine metabolism": [
        "ASPARTATE", "GLUTATHIONE REDUCED", "HOMOSERINE", "L-ALANINE", "METHIONINE",
        "METHYLTHIOADENOSINE", "S-ADENOSYLHOMOCYSTEINE", "S-ADENOSYLMETHIONINE", "SERINE"
    ],
    "Glyoxylate and dicarboxylate metabolism": [
        "2-PHOSPHOGLYCERATE", "ACETYL-COA", "ASPARTATE", "GLUTAMATE", "GLUTAMINE", "GLYCINE",
        "MALATE", "SERINE", "SUCCINATE"
    ],
    "One carbon pool by folate": [
        "ADENOSINE", "ATP/dGTP", "BETAINE", "GLYCINE", "METHIONINE", "S-ADENOSYLHOMOCYSTEINE",
        "S-ADENOSYLMETHIONINE", "SARCOSINE", "SERINE"
    ],
    "Pyrimidine metabolism": [
        "CDP", "CYTIDINE", "CYTIDINE MONOPHOSPHATE", "CYTOSINE", "GLUTAMINE", "URACIL",
        "URIDINE 5'-DIPHOSPHATE", "URIDINE MONOPHOSPHATE", "UTP"
    ],
    "Glutathione metabolism": [
        "ACETYL-COA", "GLUTAMATE", "GLUTATHIONE DISULFIDE", "GLUTATHIONE REDUCED", "GLYCINE",
        "NADP", "NADPH", "PYROGLUTAMIC ACID/OXOPROLINE"
    ],
    "Lysine degradation": [
        "ACETYL-COA", "AMINOADIPATE", "DEOXYCARNITINE", "GLYCINE", "LYSINE",
        "N,N,N-TRIMETHYLLYSINE", "PIPECOLATE", "SUCCINATE"
    ],
    "Nicotinate and nicotinamide metabolism": [
        "ASPARTATE", "FUMARATE", "GAMMA-AMINOBUTYRATE", "NAD+", "NADP", "NICOTINAMIDE",
        "SUCCINATE"
    ],
    "beta-Alanine metabolism": [
        "ACETYL-COA", "ASPARTATE", "CARNOSINE", "GAMMA-AMINOBUTYRATE", "HISTIDINE",
        "PANTOTHENATE", "URACIL"
    ],
    "Amino sugar and nucleotide sugar metabolism": [
        "ACETYLGLUCOSAMINE/N-ACETYLGALACTOSAMIN", "ADP-GLUCOSE",
        "GUANOSINE DIPHOSPHATE MANNOSE", "UDP-N-ACETYL-GLUCOSAMINE",
        "URIDINE DIPHOSPHATE GLUCURONIC ACID",
        "URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE"
    ],
    "Arginine biosynthesis": [
        "ARGININE", "ASPARTATE", "FUMARATE", "GLUTAMATE", "GLUTAMINE", "N-ACETYLGLUTAMATE"
    ],
    "Butanoate metabolism": [
        "2-HYDROXYGLUTERATE", "ACETYL-COA", "FUMARATE", "GAMMA-AMINOBUTYRATE", "GLUTAMATE",
        "SUCCINATE"
    ],
    "Glycerophospholipid metabolism": [
        "CDP-ETHANOLAMINE", "CITICOLINE", "DIETHANOLAMINE", "ETHANOLAMINE PHOSPHATE",
        "PHOSPHORYLCHOLINE", "SERINE"
    ],
    "Sulfur metabolism": [
        "ADENOSINE 3',5'-DIPHOSPHATE", "HOMOSERINE", "METHIONINE", "SERINE", "SUCCINATE",
        "TAURINE"
    ],
    "Biosynthesis of various nucleotide sugars": [
        "GUANOSINE DIPHOSPHATE MANNOSE", "N-ACETYLNEURAMINATE", "UDP-N-ACETYL-GLUCOSAMINE",
        "URIDINE DIPHOSPHATE GLUCURONIC ACID",
        "URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE"
    ],
    "Galactose metabolism": [
        "ACETYLGLUCOSAMINE/N-ACETYLGALACTOSAMIN", "GALACTITOL", "MYO-INOSITOL", "SORBITOL",
        "URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE"
    ],
    "Pantothenate and CoA biosynthesis": [
        "ADENOSINE 3',5'-DIPHOSPHATE", "ASPARTATE", "COENZYME A", "PANTOTHENATE", "URACIL"
    ],
    "Phenylalanine metabolism": [
        "ACETYL-COA", "FUMARATE", "PHENYLALANINE", "SUCCINATE", "TYROSINE"
    ],
    "Pyruvate metabolism": [
        "ACETYL-COA", "FUMARATE", "LACTATE", "MALATE", "SUCCINATE"
    ],
    "Ascorbate and aldarate metabolism": [
        "GUANOSINE DIPHOSPHATE MANNOSE", "MYO-INOSITOL",
        "URIDINE DIPHOSPHATE GLUCURONIC ACID",
        "URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE"
    ],
    "Citrate cycle (TCA cycle)": [
        "ACETYL-COA", "FUMARATE", "MALATE", "SUCCINATE"
    ],
    "Fructose and mannose metabolism": [
        "FRUCTOSE-1,6-BISPHOSPHATE", "GUANOSINE DIPHOSPHATE MANNOSE", "LACTATE", "SORBITOL"
    ],
    "Glycolysis / Gluconeogenesis": [
        "2-PHOSPHOGLYCERATE", "ACETYL-COA", "FRUCTOSE-1,6-BISPHOSPHATE", "LACTATE"
    ],
    "Histidine metabolism": [
        "ASPARTATE", "CARNOSINE", "GLUTAMATE", "HISTIDINE"
    ],
    "Oxidative phosphorylation": [
        "ATP/dGTP", "FUMARATE", "NAD+", "SUCCINATE"
    ],
    "Pentose phosphate pathway": [
        "2-PHOSPHOGLYCERATE", "6-PHOSPHOGLUCONATE", "D-RIBOSE 5-PHOSPHATE",
        "FRUCTOSE-1,6-BISPHOSPHATE"
    ],
    "Phenylalanine, tyrosine and tryptophan biosynthesis": [
        "FRUCTOSE-1,6-BISPHOSPHATE", "PHENYLALANINE", "TRYPTOPHAN", "TYROSINE"
    ],
    "Taurine and hypotaurine metabolism": [
        "ACETYL-COA", "GLUTAMATE", "L-ALANINE", "TAURINE"
    ],
    "C5-Branched dibasic acid metabolism": [
        "2-HYDROXYGLUTERATE", "ACETYL-COA", "GLUTAMATE"
    ],
    "Fatty acid degradation": [
        "ACETYL-COA", "COENZYME A", "PALMITOYLCARNITINE"
    ],
    "Neomycin, kanamycin and gentamicin biosynthesis": [
        "GLUTAMATE", "UDP-N-ACETYL-GLUCOSAMINE",
        "URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE"
    ],
    "Porphyrin metabolism": [
        "ALLOTHREONINE/THREONINE", "GLUTAMATE", "GLYCINE"
    ],
    "Propanoate metabolism": [
        "ACETYL-COA", "LACTATE", "SUCCINATE"
    ],
    "Riboflavin metabolism": [
        "FAD", "GTP", "RIBOFLAVIN"
    ],
    "Thiamine metabolism": [
        "GLYCINE", "NAD+", "TYROSINE"
    ],
    "Tyrosine metabolism": [
        "FUMARATE", "SUCCINATE", "TYROSINE"
    ],
    "Valine, leucine and isoleucine biosynthesis": [
        "ACETYL-COA", "ALLOTHREONINE/THREONINE", "LEUCINE/ISOLEUCINE"
    ],
}

# compounds measured twice: keep one name, drop the other
DUPLICATES = {"CYCLIC-AMP": "CYCLIC AMP", "INOSINE MONOPHOSPHATE": "INOSINE-MONOPHOSPHATE",
              "AMINOADIPIC ACID": "AMINOADIPATE", "DL-PIPECOLIC ACID": "PIPECOLATE"}

# ---------------------------------------------------------------- load data
data = pd.read_csv(INPUT_FILE)
samples = data[(data["Region of brain"] == REGION) & (data["Gender"] == SEX)]
samples = samples[~samples["Sample ID"].isin(EXCLUDE_SAMPLES)]
columns = {c.split("_", 1)[1]: c for c in data.columns if c.startswith(("POS_", "NEG_"))}
for dup in DUPLICATES:
    columns.pop(dup, None)
names = list(columns)

missing = sorted({m for mets in PATHWAYS.values() for m in mets} - set(names))
assert not missing, f"pathway metabolites not found in the data: {missing}"

X = samples[[columns[n] for n in names]].to_numpy()
group = np.where(samples["Genotype"] == "WT", "WT",
                 "KI-" + samples["line of APOE"]).astype(object)
membership = {p: np.array([n in mets for n in names]) for p, mets in PATHWAYS.items()
              if len(mets) >= MIN_SIZE}
print(f"Samples: WT n = {(group == 'WT').sum()}, "
      + ", ".join(f"KI-{a} n = {(group == f'KI-{a}').sum()}" for a in APOE_LINES))
print(f"{len(names)} metabolites, {len(membership)} pathways")

# ---------------------------------------------------------------- statistics
def welch_t(a, b):
    """Welch t statistic per metabolite (columns), group a vs group b."""
    va, vb = a.var(0, ddof=1) / len(a), b.var(0, ddof=1) / len(b)
    return (a.mean(0) - b.mean(0)) / np.sqrt(va + vb)

def scores(t):
    direction = np.array([t[m].mean() for m in membership.values()])
    change = np.array([np.abs(t[m]).mean() for m in membership.values()])
    return direction, change

rng = np.random.default_rng(SEED)
rows = []
for apoe in APOE_LINES:
    line = f"KI-{apoe}"
    keep = (group == line) | (group == "WT")
    Xs, labels = X[keep], group[keep]
    is_ki = labels == line

    obs_dir, obs_chg = scores(welch_t(Xs[is_ki], Xs[~is_ki]))
    null_dir, null_chg = [], []
    for _ in range(N_PERMUTATIONS):
        shuffled = rng.permutation(is_ki)
        d, c = scores(welch_t(Xs[shuffled], Xs[~shuffled]))
        null_dir.append(d)
        null_chg.append(c)
    null_dir, null_chg = np.array(null_dir), np.array(null_chg)

    # two-sided for direction, one-sided (larger = more change) for change
    p_dir = (np.sum(np.abs(null_dir) >= np.abs(obs_dir), axis=0) + 1) / (N_PERMUTATIONS + 1)
    p_chg = (np.sum(null_chg >= obs_chg, axis=0) + 1) / (N_PERMUTATIONS + 1)
    q_dir = multipletests(p_dir, method="fdr_bh")[1]
    q_chg = multipletests(p_chg, method="fdr_bh")[1]

    for k, pathway in enumerate(membership):
        members = [n for n, m in zip(names, membership[pathway]) if m]
        lfc = Xs[is_ki][:, membership[pathway]].mean(0) - Xs[~is_ki][:, membership[pathway]].mean(0)
        rows.append({
            "KI line": line, "Pathway": pathway, "n metabolites": len(members),
            "mean log2FC": lfc.mean(),
            "direction (mean t)": obs_dir[k], "p direction": p_dir[k], "FDR direction": q_dir[k],
            "change (mean |t|)": obs_chg[k], "p change": p_chg[k], "FDR change": q_chg[k],
            "Metabolites": "; ".join(members),
        })

table = pd.DataFrame(rows)
table.round(4).to_csv(OUTPUT_FILE, index=False)

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", None)
show = ["Pathway", "n metabolites", "mean log2FC", "direction (mean t)", "p direction",
        "FDR direction", "p change", "FDR change"]
for line, part in table.groupby("KI line", sort=False):
    print(f"\n{line} vs pooled WT")
    print(part.sort_values("p direction")[show].round(3).to_string(index=False))

# ---------------------------------------------------------------- dot plot
# rows = pathways, columns = KI lines; colour = direction, size = -log10 p (direction)
order = table.groupby("Pathway")["p direction"].min().sort_values(ascending=False).index
lines = [f"KI-{a}" for a in APOE_LINES]
cmap = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])
lim = np.ceil(table["direction (mean t)"].abs().max())

fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(order) + 1.6))
for j, line in enumerate(lines):
    part = table[table["KI line"] == line].set_index("Pathway").loc[order]
    y = np.arange(len(order))
    size = 40 + 140 * -np.log10(part["p direction"])
    dots = ax.scatter(np.full(len(order), j), y, s=size, c=part["direction (mean t)"],
                      cmap=cmap, vmin=-lim, vmax=lim, edgecolor="#52514e", linewidth=0.4)
    fdr = (part["FDR direction"] < 0.05).to_numpy()
    nominal = (part["p direction"] < 0.05).to_numpy() & ~fdr
    ax.scatter(np.full(fdr.sum(), j + 0.28), y[fdr], marker="*", s=50, color="black")
    ax.scatter(np.full(nominal.sum(), j + 0.28), y[nominal], marker="o", s=10, color="black")

ax.set_xticks(range(len(lines)), [l.replace("-", "\n") for l in lines], fontsize=8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(order)), order, fontsize=8)
ax.set_xlim(-0.6, len(lines) - 0.3)
ax.set_ylim(-0.7, len(order) - 0.3)
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.grid(axis="y", color="#ecebe7", linewidth=0.6)
ax.set_axisbelow(True)

bar = fig.colorbar(dots, ax=ax, shrink=0.35, pad=0.02)
bar.set_label("direction (mean t)\nblue = down in KI, red = up", fontsize=7)
for p_ref in [0.5, 0.05, 0.005]:
    ax.scatter([], [], s=40 + 140 * -np.log10(p_ref), color="#c3c2b7", edgecolor="#52514e",
               linewidth=0.4, label=f"p = {p_ref}")
ax.scatter([], [], marker="o", s=10, color="black", label="p < 0.05")
ax.scatter([], [], marker="*", s=50, color="black", label="FDR < 0.05")
ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1), fontsize=7, frameon=False,
          labelspacing=1.2)
ax.set_title("Female hippocampus: pathway shifts, KI vs pooled WT", fontsize=9, pad=12)
fig.tight_layout()
fig.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
print(f"\nSaved {OUTPUT_FILE} and {PLOT_FILE}")
