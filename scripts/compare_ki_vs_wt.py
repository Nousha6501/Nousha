#!/usr/bin/env python3
"""Compare KI vs WT per metabolite within one region/sex subset (default: Hippocampus, female).

Input: the PQN-normalised, log2-transformed sample table (one row per sample,
metadata columns first, then one column per metabolite prefixed POS_/NEG_).

Statistics are run on the log2 values:
  * Welch's t-test (default) or Mann-Whitney U (--test mwu), KI vs WT
  * Benjamini-Hochberg FDR across all metabolites in the subset

Outputs (in --outdir):
  * <prefix>_stats.csv  - n, means, log2 fold change (KI - WT), p, FDR q per metabolite
  * <prefix>_plots.pdf  - column plots (mean +/- SEM with individual points) for all
                          metabolites, followed by separate page(s) showing only the
                          significant metabolites (if any)

Plots show each sample as % of the WT mean (100 * 2^(log2 - mean_WT)), so bars start
at zero and WT = 100 %. Use --scale log2 to plot the raw log2 values instead.

Example:
  python scripts/compare_ki_vs_wt.py PQNregion_log2_samples_clean.csv \
      --region Hippocmpus --sex F --outdir results
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.stats.multitest import multipletests

GROUP_COL = "Genotype"
REGION_COL = "Region of brain"
SEX_COL = "Gender"
GROUPS = ["WT", "KI"]
COLORS = {"WT": "#9AA5B1", "KI": "#C0504D"}


def stars(p):
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def metabolite_columns(df):
    return [c for c in df.columns if c.startswith(("POS_", "NEG_"))]


def run_stats(sub, mets, test):
    rows = []
    for m in mets:
        wt = sub.loc[sub[GROUP_COL] == "WT", m].dropna()
        ki = sub.loc[sub[GROUP_COL] == "KI", m].dropna()
        if len(wt) < 2 or len(ki) < 2:
            p = np.nan
        elif test == "mwu":
            p = stats.mannwhitneyu(ki, wt, alternative="two-sided").pvalue
        else:
            p = stats.ttest_ind(ki, wt, equal_var=False).pvalue
        rows.append({
            "metabolite": m,
            "n_WT": len(wt), "n_KI": len(ki),
            "mean_log2_WT": wt.mean(), "mean_log2_KI": ki.mean(),
            "log2FC_KI_vs_WT": ki.mean() - wt.mean(),
            "FC_KI_vs_WT": 2 ** (ki.mean() - wt.mean()),
            "p_value": p,
        })
    res = pd.DataFrame(rows)
    ok = res["p_value"].notna()
    res["q_value_BH"] = np.nan
    res.loc[ok, "q_value_BH"] = multipletests(res.loc[ok, "p_value"], method="fdr_bh")[1]
    return res


def draw_panel(ax, sub, row, scale, sig_on, rng):
    m = row["metabolite"]
    wt_mean = row["mean_log2_WT"]
    data = {}
    for g in GROUPS:
        v = sub.loc[sub[GROUP_COL] == g, m].dropna().to_numpy()
        data[g] = 100 * 2 ** (v - wt_mean) if scale == "pct" else v

    tops = []
    for i, g in enumerate(GROUPS):
        v = data[g]
        mean = v.mean()
        sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        ax.bar(i, mean, width=0.6, color=COLORS[g], alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.errorbar(i, mean, yerr=sem, color="black", capsize=4, linewidth=1)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color="black", alpha=0.7, zorder=3)
        tops.append(max(v.max(), mean + sem))

    ymax = max(tops)
    if scale == "log2":
        lo = min(data["WT"].min(), data["KI"].min())
        span = ymax - lo
        ax.set_ylim(lo - 0.3 * span, ymax + 0.35 * span)
        h = 0.08 * span
    else:
        ax.set_ylim(0, ymax * 1.25)
        h = 0.05 * ymax

    pval = row["q_value_BH"] if sig_on == "fdr" else row["p_value"]
    y = ymax + h
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], color="black", linewidth=0.8)
    ax.text(0.5, y + 1.2 * h, stars(pval) if not np.isnan(pval) else "n/a",
            ha="center", va="bottom", fontsize=9)

    name = m.split("_", 1)[1] if "_" in m else m
    ax.set_title(f"{name}\n[{m.split('_', 1)[0]}]  p={row['p_value']:.3g}, q={row['q_value_BH']:.3g}",
                 fontsize=7)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f"{g}\n(n={len(data[g])})" for g in GROUPS], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("% of WT mean" if scale == "pct" else "log2 abundance", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_pages(pdf, sub, res, title, scale, sig_on, ncols, nrows, rng):
    per_page = ncols * nrows
    n_pages = max(1, math.ceil(len(res) / per_page))
    for page in range(n_pages):
        chunk = res.iloc[page * per_page:(page + 1) * per_page]
        fig, axes = plt.subplots(nrows, ncols, figsize=(8.27, 11.69))  # A4 portrait
        axes = np.atleast_1d(axes).ravel()
        for ax, (_, row) in zip(axes, chunk.iterrows()):
            draw_panel(ax, sub, row, scale, sig_on, rng)
        for ax in axes[len(chunk):]:
            ax.axis("off")
        suffix = f" (page {page + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(title + suffix, fontsize=11, fontweight="bold")
        fig.text(0.5, 0.01,
                 f"Mean ± SEM, dots = samples. Significance by {'BH-FDR q' if sig_on == 'fdr' else 'p-value'}: "
                 "* <0.05, ** <0.01, *** <0.001, **** <0.0001, ns = not significant",
                 ha="center", fontsize=7)
        fig.tight_layout(rect=[0, 0.02, 1, 0.97])
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="PQN-normalised log2 sample table")
    ap.add_argument("--region", default="Hippocmpus", help="value of 'Region of brain' (default: Hippocmpus)")
    ap.add_argument("--sex", default="F", help="value of 'Gender' (default: F)")
    ap.add_argument("--test", choices=["welch", "mwu"], default="welch",
                    help="welch = Welch t-test (default), mwu = Mann-Whitney U")
    ap.add_argument("--sig-on", choices=["p", "fdr"], default="p",
                    help="use raw p (default) or BH-FDR q for stars and the significant page")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--scale", choices=["pct", "log2"], default="pct",
                    help="y-axis: %% of WT mean (default) or raw log2 values")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--prefix", default=None, help="output file prefix (default: <region>_<sex>_KI_vs_WT)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sub = df[(df[REGION_COL] == args.region) & (df[SEX_COL] == args.sex) & df[GROUP_COL].isin(GROUPS)]
    if sub.empty:
        raise SystemExit(f"No samples for region={args.region!r}, sex={args.sex!r}")
    mets = metabolite_columns(df)

    res = run_stats(sub, mets, args.test)
    sig_col = "q_value_BH" if args.sig_on == "fdr" else "p_value"
    res["significant"] = res[sig_col] < args.alpha
    res["direction"] = np.where(res["log2FC_KI_vs_WT"] > 0, "up in KI", "down in KI")

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix or f"{args.region}_{args.sex}_KI_vs_WT"
    stats_path = os.path.join(args.outdir, f"{prefix}_stats.csv")
    res.sort_values("p_value").to_csv(stats_path, index=False)

    test_name = "Welch t-test" if args.test == "welch" else "Mann-Whitney U"
    label = f"{args.region} {args.sex}: KI vs WT ({test_name})"
    sig = res[res["significant"]].sort_values("p_value")
    rng = np.random.default_rng(0)

    pdf_path = os.path.join(args.outdir, f"{prefix}_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        plot_pages(pdf, sub, res, f"All metabolites — {label}", args.scale, args.sig_on, 4, 5, rng)
        if len(sig):
            plot_pages(pdf, sub, sig,
                       f"Significant metabolites ({'q' if args.sig_on == 'fdr' else 'p'} < {args.alpha}) — {label}",
                       args.scale, args.sig_on, 3, 4, rng)
        else:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.5, f"No significant metabolites ({args.sig_on} < {args.alpha})",
                     ha="center", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Samples: " + ", ".join(f"{g}={int((sub[GROUP_COL] == g).sum())}" for g in GROUPS))
    print(f"Metabolites tested: {len(res)}")
    print(f"Significant ({args.sig_on} < {args.alpha}): {len(sig)}")
    if len(sig):
        print(sig[["metabolite", "log2FC_KI_vs_WT", "p_value", "q_value_BH", "direction"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"Wrote {stats_path}\nWrote {pdf_path}")


if __name__ == "__main__":
    main()
