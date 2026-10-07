#!/usr/bin/env python3
"""KI vs WT comparison of metabolites within one region/sex/APOE subgroup.

Input: PQN-normalised, log2-transformed sample table (one row per sample,
metadata columns first, then one column per metabolite prefixed POS_/NEG_).

For every metabolite:
  * Welch's t-test on log2 values (KI vs WT) + Mann-Whitney U as a check
  * log2 fold change = mean(KI) - mean(WT)
  * Benjamini-Hochberg FDR across all metabolites
  * a column plot (mean +/- SEM with individual samples) and significance stars

Outputs (in --outdir):
  stats_<group>.csv           full results table, sorted by p-value
  significant_<group>.csv     metabolites passing the significance cut-off
  barplots_<group>.pdf        one page per metabolite
  barplots_<group>/*.png      one PNG per metabolite (skip with --no-png)
  volcano_<group>.png         volcano plot
  significant_grid_<group>.png  all significant metabolites on one figure

Example:
  python scripts/ki_vs_wt.py PQNregion_log2_samples_clean.csv \
      --region Hip --sex F --apoe APOE4 --outdir results
"""
import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from statsmodels.stats.multitest import multipletests

WT_COLOR = "#8C8C8C"
KI_COLOR = "#C0392B"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="PQN log2 sample table")
    p.add_argument("--region", default="Hip", help="Region prefix, matched case-insensitively (default: Hip)")
    p.add_argument("--sex", default="F", help="Gender value (default: F)")
    p.add_argument("--apoe", default="APOE4", help="'line of APOE' value (default: APOE4)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance cut-off (default: 0.05)")
    p.add_argument("--use-raw-p", action="store_true",
                   help="Call significance on raw Welch p instead of BH-FDR q")
    p.add_argument("--min-abs-log2fc", type=float, default=0.0,
                   help="Also require |log2FC| >= this to call a hit (default: 0)")
    p.add_argument("--scale", choices=["relative", "log2"], default="relative",
                   help="Y axis: 'relative' = linear abundance relative to WT mean (WT=1), "
                        "'log2' = raw log2 intensity (default: relative)")
    p.add_argument("--outdir", default="results", help="Output directory (default: results)")
    p.add_argument("--no-png", action="store_true", help="Skip individual PNG files")
    return p.parse_args()


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def select_subgroup(df, region, sex, apoe):
    mask = (
        df["Region of brain"].astype(str).str.lower().str.startswith(region.lower())
        & (df["Gender"].astype(str).str.upper() == sex.upper())
        & (df["line of APOE"].astype(str).str.upper() == apoe.upper())
    )
    if "Type" in df.columns:
        mask &= df["Type"].astype(str).str.lower() == "sample"
    sub = df[mask].copy()
    sub["Genotype"] = sub["Genotype"].astype(str).str.upper()
    return sub[sub["Genotype"].isin(["KI", "WT"])]


def metabolite_columns(df):
    return [c for c in df.columns if c.startswith(("POS_", "NEG_"))]


def run_stats(sub, mets):
    rows = []
    for m in mets:
        ki = sub.loc[sub["Genotype"] == "KI", m].dropna().to_numpy(float)
        wt = sub.loc[sub["Genotype"] == "WT", m].dropna().to_numpy(float)
        row = {
            "metabolite": m,
            "mode": m.split("_", 1)[0],
            "name": m.split("_", 1)[1],
            "n_KI": len(ki),
            "n_WT": len(wt),
            "mean_log2_KI": ki.mean() if len(ki) else np.nan,
            "mean_log2_WT": wt.mean() if len(wt) else np.nan,
            "sd_log2_KI": ki.std(ddof=1) if len(ki) > 1 else np.nan,
            "sd_log2_WT": wt.std(ddof=1) if len(wt) > 1 else np.nan,
        }
        row["log2FC_KI_vs_WT"] = row["mean_log2_KI"] - row["mean_log2_WT"]
        row["FC_KI_vs_WT"] = 2 ** row["log2FC_KI_vs_WT"]
        if len(ki) >= 2 and len(wt) >= 2 and (np.ptp(ki) > 0 or np.ptp(wt) > 0):
            t = stats.ttest_ind(ki, wt, equal_var=False)
            row["t_welch"], row["p_welch"] = t.statistic, t.pvalue
            row["p_mannwhitney"] = stats.mannwhitneyu(ki, wt, alternative="two-sided").pvalue
            pooled = np.sqrt(((len(ki) - 1) * ki.var(ddof=1) + (len(wt) - 1) * wt.var(ddof=1))
                             / (len(ki) + len(wt) - 2))
            row["cohens_d"] = row["log2FC_KI_vs_WT"] / pooled if pooled > 0 else np.nan
        else:
            row["t_welch"] = row["p_welch"] = row["p_mannwhitney"] = row["cohens_d"] = np.nan
        rows.append(row)

    res = pd.DataFrame(rows)
    ok = res["p_welch"].notna()
    res["q_BH"] = np.nan
    res.loc[ok, "q_BH"] = multipletests(res.loc[ok, "p_welch"], method="fdr_bh")[1]
    return res


def plot_values(sub, m, ref_mean, scale):
    v = sub[m].astype(float)
    return 2 ** (v - ref_mean) if scale == "relative" else v


def draw_column(ax, sub, row, scale, test_col):
    m = row["metabolite"]
    vals = plot_values(sub, m, row["mean_log2_WT"], scale)
    groups = [("WT", WT_COLOR), ("KI", KI_COLOR)]
    rng = np.random.default_rng(0)
    tops = []
    for i, (g, color) in enumerate(groups):
        y = vals[sub["Genotype"] == g].dropna().to_numpy()
        mean = y.mean()
        sem = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0
        ax.bar(i, mean, width=0.6, color=color, alpha=0.35, edgecolor=color, linewidth=1.5)
        ax.errorbar(i, mean, yerr=sem, color="black", capsize=6, linewidth=1.2, zorder=3)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=22, color=color,
                   edgecolor="black", linewidth=0.5, zorder=4)
        tops.append(max(y.max(), mean + sem))

    p = row[test_col]
    y_top = max(tops)
    y_rng = y_top - (0 if scale == "relative" else vals.min())
    bar_y = y_top + 0.06 * y_rng
    ax.plot([0, 0, 1, 1], [bar_y, bar_y + 0.02 * y_rng, bar_y + 0.02 * y_rng, bar_y],
            color="black", linewidth=1)
    label = p_to_stars(p) if pd.notna(p) else "n/a"
    ax.text(0.5, bar_y + 0.03 * y_rng, label, ha="center", va="bottom", fontsize=11)

    ax.set_xticks([0, 1], [f"WT\n(n={row['n_WT']})", f"KI\n(n={row['n_KI']})"])
    ax.set_xlim(-0.6, 1.6)
    if scale == "relative":
        ax.set_ylim(0, bar_y + 0.15 * y_rng)
        ax.axhline(1, color="grey", linestyle=":", linewidth=0.8)
        ax.set_ylabel("Relative abundance (WT mean = 1)")
    else:
        lo = vals.min()
        ax.set_ylim(lo - 0.1 * y_rng, bar_y + 0.15 * y_rng)
        ax.set_ylabel("log2 intensity (PQN)")
    ax.set_title(row["name"], fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def stat_caption(row):
    return (f"log2FC={row['log2FC_KI_vs_WT']:.2f}  p={row['p_welch']:.3g}  "
            f"q={row['q_BH']:.3g}  MW p={row['p_mannwhitney']:.3g}")


def volcano(res, sig_mask, alpha_col, alpha, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    x = res["log2FC_KI_vs_WT"]
    y = -np.log10(res["p_welch"])
    ax.scatter(x[~sig_mask], y[~sig_mask], s=18, color="#BBBBBB")
    up = sig_mask & (x > 0)
    down = sig_mask & (x < 0)
    ax.scatter(x[up], y[up], s=26, color=KI_COLOR, label=f"Up in KI ({up.sum()})")
    ax.scatter(x[down], y[down], s=26, color="#2E86C1", label=f"Down in KI ({down.sum()})")
    for _, r in res[sig_mask].iterrows():
        ax.annotate(r["name"], (r["log2FC_KI_vs_WT"], -np.log10(r["p_welch"])),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("log2 fold change (KI / WT)")
    ax.set_ylabel("-log10 p (Welch)")
    ax.set_title(f"{title}\nhits: {alpha_col} < {alpha}")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    a = parse_args()
    df = pd.read_csv(a.csv)
    sub = select_subgroup(df, a.region, a.sex, a.apoe)
    tag = f"{a.region}_{a.sex}_{a.apoe}"
    title = f"{tag}: KI vs WT"
    n = sub["Genotype"].value_counts()
    print(f"[{tag}] samples: KI={n.get('KI', 0)}, WT={n.get('WT', 0)}")
    if n.get("KI", 0) < 2 or n.get("WT", 0) < 2:
        raise SystemExit("Need at least 2 samples per genotype - check --region/--sex/--apoe")

    mets = metabolite_columns(sub)
    res = run_stats(sub, mets)
    alpha_col = "p_welch" if a.use_raw_p else "q_BH"
    res["significant"] = (res[alpha_col] < a.alpha) & (res["log2FC_KI_vs_WT"].abs() >= a.min_abs_log2fc)
    res["direction"] = np.where(res["log2FC_KI_vs_WT"] > 0, "up_in_KI", "down_in_KI")
    res = res.sort_values("p_welch").reset_index(drop=True)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / f"stats_{tag}.csv", index=False)
    sig = res[res["significant"]]
    sig.to_csv(out / f"significant_{tag}.csv", index=False)

    # Per-metabolite column plots: PDF (all) + optional PNGs. Stars use the
    # same p/q that defines significance.
    png_dir = out / f"barplots_{tag}"
    if not a.no_png:
        png_dir.mkdir(exist_ok=True)
    with PdfPages(out / f"barplots_{tag}.pdf") as pdf:
        for _, row in res.iterrows():
            fig, ax = plt.subplots(figsize=(3.6, 4.4))
            draw_column(ax, sub, row, a.scale, alpha_col)
            fig.text(0.5, 0.01, stat_caption(row), ha="center", fontsize=6)
            fig.suptitle(title, fontsize=8, y=0.99)
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
            pdf.savefig(fig)
            if not a.no_png:
                fig.savefig(png_dir / f"{safe_name(row['metabolite'])}.png", dpi=200)
            plt.close(fig)

    volcano(res, res["significant"].to_numpy(), alpha_col, a.alpha, out / f"volcano_{tag}.png", title)

    if len(sig):
        ncol = min(5, len(sig))
        nrow = int(np.ceil(len(sig) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.6 * nrow), squeeze=False)
        for ax, (_, row) in zip(axes.flat, sig.iterrows()):
            draw_column(ax, sub, row, a.scale, alpha_col)
        for ax in axes.flat[len(sig):]:
            ax.axis("off")
        fig.suptitle(f"{title} - significant ({alpha_col} < {a.alpha})")
        fig.tight_layout()
        fig.savefig(out / f"significant_grid_{tag}.png", dpi=200)
        plt.close(fig)

    print(f"[{tag}] metabolites tested: {res['p_welch'].notna().sum()}")
    print(f"[{tag}] raw p<0.05: {(res['p_welch'] < 0.05).sum()},  "
          f"BH q<0.05: {(res['q_BH'] < 0.05).sum()},  "
          f"called significant ({alpha_col}<{a.alpha}): {len(sig)}")
    cols = ["name", "log2FC_KI_vs_WT", "p_welch", "q_BH", "p_mannwhitney", "direction"]
    print(res.head(20)[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"Results written to {out.resolve()}")


if __name__ == "__main__":
    main()
