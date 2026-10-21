"""KI vs pooled WT per region x sex: results table, log2FC heatmap, MetaboAnalyst inputs.

Input: drift-corrected, PQN-normalised, log2-transformed sample table (one row per sample).
Usage: python ki_vs_wt_results.py <input.csv> <output_dir>

Statistics per region x sex x metabolite:
  - one-way design WT (pooled across colonies), KI-APOE2, KI-APOE3, KI-APOE4
  - Dunnett test of each KI line vs pooled WT
  - Benjamini-Hochberg FDR across metabolites, within each comparison
  - sensitivity: log2FC and Welch p vs colony-matched WT
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from statsmodels.stats.multitest import multipletests

from metabolite_annotation import ANNOT, CLASSES

KI_LINES = ["APOE2", "APOE3", "APOE4"]
REGIONS = [("Hippocmpus", "Hippocampus"), ("Cortex", "Cortex")]
SEXES = [("F", "Female"), ("M", "Male")]


def load(path):
    d = pd.read_csv(path)
    d = d[d["Type"] == "Sample"].rename(
        columns={"line of APOE": "APOE", "Region of brain": "Region", "Gender": "Sex"})
    mets = [c for c in d.columns if c.startswith(("POS_", "NEG_"))]
    missing = [m for m in mets if m not in ANNOT]
    if missing:
        raise SystemExit(f"No annotation for: {missing}")
    return d, mets


def results_table(d, mets):
    rows = []
    for reg, reg_label in REGIONS:
        for sex, sex_label in SEXES:
            s = d[(d.Region == reg) & (d.Sex == sex)]
            wt = s[s.Genotype == "WT"]
            for m in mets:
                g_wt = wt[m].dropna()
                g_ki = [s[(s.Genotype == "KI") & (s.APOE == a)][m].dropna() for a in KI_LINES]
                p_dun = stats.dunnett(*g_ki, control=g_wt).pvalue
                for a, g, p in zip(KI_LINES, g_ki, p_dun):
                    g_match = wt[wt.APOE == a][m].dropna()
                    name, kegg, hmdb, cls, note = ANNOT[m]
                    rows.append(dict(
                        region=reg_label, sex=sex_label, KI_line=f"KI-{a}",
                        metabolite=name, column=m, mode=m[:3], KEGG=kegg, HMDB=hmdb,
                        pathway_class=cls, note=note,
                        n_KI=len(g), n_WT=len(g_wt),
                        mean_KI=g.mean(), mean_WT=g_wt.mean(),
                        log2FC=g.mean() - g_wt.mean(),
                        p_dunnett=p,
                        log2FC_vs_matched_WT=g.mean() - g_match.mean(),
                        p_welch_vs_matched_WT=stats.ttest_ind(g, g_match, equal_var=False).pvalue,
                        n_matched_WT=len(g_match)))
    r = pd.DataFrame(rows)
    r["q_FDR"] = r.groupby(["region", "sex", "KI_line"])["p_dunnett"].transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    r["same_direction_matched_WT"] = np.sign(r.log2FC) == np.sign(r.log2FC_vs_matched_WT)
    r["call"] = np.select([r.q_FDR < 0.05, r.p_dunnett < 0.05],
                          ["FDR<0.05", "nominal p<0.05"], "ns")
    cols = ["region", "sex", "KI_line", "metabolite", "mode", "pathway_class", "KEGG", "HMDB",
            "n_KI", "n_WT", "mean_KI", "mean_WT", "log2FC", "p_dunnett", "q_FDR", "call",
            "log2FC_vs_matched_WT", "p_welch_vs_matched_WT", "n_matched_WT",
            "same_direction_matched_WT", "note", "column"]
    return r[cols]


def heatmap(r, mets, path):
    comps = [(rl, sl, f"KI-{a}") for _, rl in REGIONS for _, sl in SEXES for a in KI_LINES]
    order = sorted(mets, key=lambda m: (CLASSES.index(ANNOT[m][3]), ANNOT[m][0].lower()))
    key = r.set_index(["column", "region", "sex", "KI_line"])
    fc = np.array([[key.loc[(m, *c), "log2FC"] for c in comps] for m in order])
    p = np.array([[key.loc[(m, *c), "p_dunnett"] for c in comps] for m in order])
    q = np.array([[key.loc[(m, *c), "q_FDR"] for c in comps] for m in order])

    # diverging blue <-> red with neutral gray midpoint
    cmap = LinearSegmentedColormap.from_list(
        "div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1d1c"])
    lim = 0.6
    ink, muted = "#0b0b0b", "#52514e"

    def label(m):
        name, note = ANNOT[m][0], ANNOT[m][4]
        if note.startswith("also"):
            return f"{name} ({m[:3]})"
        if note.startswith("duplicate"):
            return f"{name} ({m[4:].title()})"
        return name
    labels = [label(m) for m in order]
    fig_h = 0.17 * len(order) + 2.6
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    fig.patch.set_facecolor("#fcfcfb")
    im = ax.imshow(np.clip(fc, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto",
                   interpolation="nearest")
    # 2px-style surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(comps)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(order)):
        for j in range(len(comps)):
            if q[i, j] < 0.05:
                ax.text(j, i, "★", ha="center", va="center", fontsize=7, color=ink)
            elif p[i, j] < 0.05:
                ax.text(j, i, "•", ha="center", va="center", fontsize=8, color=ink)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=6.5, color=ink)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([c[2].replace("KI-APOE", "E") for c in comps], fontsize=7.5, color=ink)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # group separators: region x sex blocks (columns), pathway classes (rows)
    for j in (3, 6, 9):
        ax.axvline(j - 0.5, color=muted, linewidth=0.9)
    for k, (_, rl) in enumerate(REGIONS):
        for l, (_, sl) in enumerate(SEXES):
            ax.text(k * 6 + l * 3 + 1, -1.5, f"{rl}\n{sl}", ha="center", va="bottom",
                    fontsize=6.5, color=ink, fontweight="bold")
    classes = [ANNOT[m][3] for m in order]
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or classes[i] != classes[start]:
            if i < len(order):
                ax.axhline(i - 0.5, color=muted, linewidth=0.9)
            ax.text(len(comps) - 0.3, (start + i - 1) / 2, classes[start], ha="left",
                    va="center", fontsize=7, color=muted)
            start = i

    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.02, pad=0.01,
                      aspect=30, shrink=0.5, extend="both")
    cb.set_label("log2 fold change, KI vs pooled WT (clipped at ±0.6)", fontsize=7.5, color=ink)
    cb.ax.tick_params(labelsize=7, colors=ink)
    cb.outline.set_visible(False)
    fig.text(0.5, 1 - 0.25 / fig_h, "KI vs pooled WT: log2 fold change per region × sex",
             ha="center", va="top", fontsize=10, color=ink, fontweight="bold")
    fig.text(0.5, 1 - 0.5 / fig_h, "Dunnett test vs pooled WT   •  p < 0.05   "
             "★  FDR q < 0.05 (BH across 111 metabolites)", ha="center", va="top",
             fontsize=7.5, color=muted)
    fig.subplots_adjust(left=0.27, right=0.72, top=1 - 1.35 / fig_h, bottom=0.9 / fig_h)
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def metaboanalyst_inputs(d, mets, outdir):
    """One file per region x sex: samples as rows, Label column, one column per compound.
    For compounds measured twice, keep the measurement with the lower WT CV (log scale SD)."""
    keep = {}
    for m in mets:
        name = ANNOT[m][0]
        sd = d[d.Genotype == "WT"].groupby(["Region", "Sex"])[m].std().mean()
        if name not in keep or sd < keep[name][1]:
            keep[name] = (m, sd)
    chosen = [v[0] for v in keep.values()]
    paths = []
    for reg, rl in REGIONS:
        for sex, sl in SEXES:
            s = d[(d.Region == reg) & (d.Sex == sex)]
            lab = np.where(s.Genotype == "WT", "WT", "KI_" + s.APOE)
            cols = {"Sample": s["Sample ID"].values, "Label": lab}
            cols.update({ANNOT[m][2] or ANNOT[m][0]: s[m].values for m in chosen})
            out = pd.DataFrame(cols)
            p = os.path.join(outdir, f"metaboanalyst_{rl}_{sl}.csv")
            out.to_csv(p, index=False)
            paths.append(p)
    dropped = sorted(set(mets) - set(chosen))
    return paths, dropped


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    d, mets = load(src)
    r = results_table(d, mets)
    r.to_csv(os.path.join(outdir, "KI_vs_pooledWT_results.csv"), index=False)
    heatmap(r, mets, os.path.join(outdir, "KI_vs_pooledWT_log2FC_heatmap.png"))
    _, dropped = metaboanalyst_inputs(d, mets, outdir)

    print("Dropped duplicate measurements for MetaboAnalyst:", dropped)
    summ = r.groupby(["region", "sex", "KI_line"]).agg(
        nominal=("p_dunnett", lambda p: int((p < 0.05).sum())),
        FDR=("q_FDR", lambda q: int((q < 0.05).sum())))
    print(summ.to_string())
    hits = r[r.p_dunnett < 0.05].sort_values("p_dunnett")
    print(hits[["region", "sex", "KI_line", "metabolite", "log2FC", "p_dunnett", "q_FDR",
                "log2FC_vs_matched_WT", "same_direction_matched_WT"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
