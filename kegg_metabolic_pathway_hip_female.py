"""
KEGG metabolic pathway analysis — Hip, Female
=============================================

Input : the differential results table you used for the heatmap / log2FC
        scatter plot (one row per feature with an ID, log2FC and p-value).
Output: KEGG metabolic pathway enrichment
          1. ORA  (over-representation) for all / up / down significant features
          2. GSEA (pre-ranked by log2FC) on every measured feature
        as CSV tables and PDF plots.

Works for two kinds of data — set FEATURE_TYPE below:
  "gene"       -> transcriptomics / proteomics (gene symbols, Ensembl or Entrez IDs)
  "metabolite" -> metabolomics (KEGG compound IDs, e.g. C00031)

Pathways are taken live from the KEGG REST API (https://rest.kegg.jp) and
restricted to the KEGG "Metabolism" section (pathway IDs 00xxx / 01xxx).

Install once:
    pip install pandas numpy scipy matplotlib requests gseapy openpyxl
Run:
    python kegg_metabolic_pathway_hip_female.py
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

# ---- 1. Settings: edit these -------------------------------------------------
INPUT_FILE = "hip_female_results.csv"  # .csv, .tsv/.txt or .xlsx
FEATURE_TYPE = "gene"                  # "gene" or "metabolite"

ID_COL = "gene"     # column with feature IDs
LFC_COL = "log2FC"  # column with log2 fold change
P_COL = "padj"      # column with adjusted p-value

# gene data only
ID_TYPE = "SYMBOL"  # "SYMBOL", "ENSEMBL" (e.g. ENSMUSG00000000001) or "ENTREZID"
ORGANISM = "mmu"    # KEGG organism code: mmu mouse, rno rat, hsa human

P_CUT = 0.05    # significance cut-off for features
LFC_CUT = 1.0   # |log2FC| cut-off for features (0 = use p only)
Q_CUT = 0.05    # cut-off for enriched pathways (BH-adjusted p)
MIN_SIZE = 3    # smallest pathway (measured features) to test
TOP_N = 20      # pathways shown in plots

OUT_DIR = "pathway_hip_female"
KEGG = "https://rest.kegg.jp"


# ---- 2. KEGG download (cached in OUT_DIR/kegg_cache) -------------------------
def kegg_get(query):
    cache_dir = os.path.join(OUT_DIR, "kegg_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, query.replace("/", "_") + ".txt")
    if not os.path.exists(path):
        print(f"Downloading {KEGG}/{query}")
        r = requests.get(f"{KEGG}/{query}", timeout=120)
        r.raise_for_status()
        with open(path, "w") as f:
            f.write(r.text)
    with open(path) as f:
        return [line.split("\t") for line in f.read().splitlines() if line]


def is_metabolic(pathway_id):
    """KEGG pathways 00xxx and 01xxx form the 'Metabolism' section."""
    return re.search(r"0[01]\d{3}$", pathway_id) is not None


def load_kegg_pathways():
    """Return {pathway_id: set(feature keys)} and {pathway_id: name}."""
    if FEATURE_TYPE == "gene":
        links = kegg_get(f"link/pathway/{ORGANISM}")      # mmu:11287  path:mmu00010
        names = kegg_get(f"list/pathway/{ORGANISM}")      # mmu00010   Glycolysis ... - Mus musculus
        strip = lambda x: x.split(":")[1]                  # -> 11287 (Entrez ID)
        clean_name = lambda n: re.sub(r" - [^-]+$", "", n)  # drop " - Mus musculus (house mouse)"
    else:
        links = kegg_get("link/pathway/compound")          # cpd:C00022 path:map00010
        names = kegg_get("list/pathway")                   # map00010   Glycolysis / Gluconeogenesis
        strip = lambda x: x.split(":")[1]                  # -> C00022
        clean_name = lambda n: n

    sets = {}
    for feat, pw in links:
        pw = pw.replace("path:", "")
        if FEATURE_TYPE == "metabolite" and not pw.startswith("map"):
            continue
        if is_metabolic(pw):
            sets.setdefault(pw, set()).add(strip(feat))
    pw_names = {row[0].replace("path:", ""): clean_name(row[-1]) for row in names}
    return sets, pw_names


def symbol_to_entrez():
    """Map gene symbols to Entrez IDs using KEGG's gene list for the organism."""
    mapping = {}
    for row in kegg_get(f"list/{ORGANISM}"):
        entrez = row[0].split(":")[1]
        # last column looks like "Pzp, A1m, A2m; PZP alpha-2-macroglobulin like"
        symbols = row[-1].split(";")[0]
        for i, sym in enumerate(s.strip() for s in symbols.split(",")):
            # first symbol is the official one; aliases only fill gaps
            if sym and (i == 0 or sym.upper() not in mapping):
                mapping[sym.upper()] = entrez
    return mapping


def ensembl_to_entrez(ids):
    """Map Ensembl gene IDs to Entrez IDs with the mygene.info API (cached)."""
    species = {"mmu": "mouse", "rno": "rat", "hsa": "human"}.get(ORGANISM)
    if species is None:
        sys.exit(f"Ensembl mapping: add a mygene species name for '{ORGANISM}'")
    cache = os.path.join(OUT_DIR, "kegg_cache", f"ensembl_entrez_{ORGANISM}.csv")
    mapping = {}
    if os.path.exists(cache):
        mapping = pd.read_csv(cache, dtype=str).set_index("ensembl")["entrez"].to_dict()
    todo = sorted(set(ids) - set(mapping))
    for i in range(0, len(todo), 1000):
        print(f"Mapping Ensembl -> Entrez via mygene.info ({i + 1}-{min(i + 1000, len(todo))} of {len(todo)})")
        r = requests.post("https://mygene.info/v3/query", timeout=120, data={
            "q": ",".join(todo[i:i + 1000]), "scopes": "ensembl.gene",
            "fields": "entrezgene", "species": species})
        r.raise_for_status()
        for hit in r.json():
            if "entrezgene" in hit and hit["query"] not in mapping:
                mapping[hit["query"]] = str(hit["entrezgene"])
    for q in todo:
        mapping.setdefault(q, None)  # remember misses so they aren't re-queried
    pd.DataFrame({"ensembl": list(mapping), "entrez": list(mapping.values())}).to_csv(cache, index=False)
    return {k: v for k, v in mapping.items() if isinstance(v, str)}


# ---- 3. Read results ---------------------------------------------------------
def read_results():
    if INPUT_FILE.endswith((".xlsx", ".xls")):
        df = pd.read_excel(INPUT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE, sep=None, engine="python")
    missing = [c for c in (ID_COL, LFC_COL, P_COL) if c not in df.columns]
    if missing:
        sys.exit(f"Columns {missing} not in file. Available: {list(df.columns)}")
    df = pd.DataFrame({
        "id": df[ID_COL].astype(str).str.strip(),
        "log2FC": pd.to_numeric(df[LFC_COL], errors="coerce"),
        "padj": pd.to_numeric(df[P_COL], errors="coerce"),
    }).dropna(subset=["log2FC"])
    return df.drop_duplicates("id")


# ---- 4. Statistics -----------------------------------------------------------
def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    return out


def run_ora(query, universe, sets, pw_names, label_of):
    """Hypergeometric test of `query` against each pathway, background = universe."""
    query = set(query) & universe
    N, n = len(universe), len(query)
    rows = []
    for pw, members in sets.items():
        members = members & universe
        K = len(members)
        if K < MIN_SIZE:
            continue
        hits = members & query
        k = len(hits)
        if k == 0:
            continue
        rows.append({
            "pathway_id": pw,
            "pathway": pw_names.get(pw, pw),
            "hits": k,
            "measured_in_pathway": K,
            "pathway_size_KEGG": len(sets[pw]),
            "rich_factor": k / K,
            "fold_enrichment": (k / n) / (K / N),
            "pvalue": hypergeom.sf(k - 1, N, K, n),
            "features": ";".join(sorted(label_of[h] for h in hits)),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["padj"] = bh_adjust(res["pvalue"])
    return res.sort_values("pvalue").reset_index(drop=True)


def run_gsea(ranks, sets, pw_names, label_of):
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed -> skipping GSEA (pip install gseapy)")
        return pd.DataFrame()
    universe = set(ranks.index)
    gene_sets = {pw: sorted(m & universe) for pw, m in sets.items()
                 if len(m & universe) >= MIN_SIZE}
    if not gene_sets:
        return pd.DataFrame()
    pre = gp.prerank(rnk=ranks, gene_sets=gene_sets, min_size=MIN_SIZE,
                     max_size=1000, permutation_num=1000, seed=42,
                     outdir=None, verbose=False)
    res = pre.res2d.rename(columns={"Term": "pathway_id", "NOM p-val": "pvalue",
                                    "FDR q-val": "padj", "Lead_genes": "leading_edge"})
    res["pathway"] = res["pathway_id"].map(lambda x: pw_names.get(x, x))
    res["leading_edge"] = res["leading_edge"].map(
        lambda s: ";".join(label_of.get(g, g) for g in str(s).split(";")))
    for c in ("ES", "NES", "pvalue", "padj"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    cols = ["pathway_id", "pathway", "ES", "NES", "pvalue", "padj", "leading_edge"]
    return res[cols].sort_values("pvalue").reset_index(drop=True)


# ---- 5. Plots ----------------------------------------------------------------
def dotplot(res, title, path):
    top = res.head(TOP_N).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(top) + 1.5)))
    sc = ax.scatter(top["rich_factor"], top["pathway"], s=top["hits"] * 25,
                    c=-np.log10(top["padj"]), cmap="viridis_r", edgecolor="k")
    fig.colorbar(sc, ax=ax, label="-log10(padj)")
    for k in sorted(set(top["hits"]))[:: max(1, len(set(top["hits"])) // 4)]:
        ax.scatter([], [], s=k * 25, c="grey", edgecolor="k", label=str(k))
    ax.legend(title="Hits", loc="lower right", frameon=True, labelspacing=1.2)
    ax.set_xlabel("Rich factor (hits / measured in pathway)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def nes_barplot(res, path):
    top = res.head(TOP_N).sort_values("NES")
    colors = ["#d6604d" if v > 0 else "#4393c3" for v in top["NES"]]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(top) + 1.5)))
    ax.barh(top["pathway"], top["NES"], color=colors)
    for y, (nes, q) in enumerate(zip(top["NES"], top["padj"])):
        if q < Q_CUT:
            ax.text(nes + (0.05 if nes > 0 else -0.05), y, "*",
                    ha="left" if nes > 0 else "right", va="center", fontsize=14)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Normalized enrichment score (red = up, blue = down)")
    ax.set_title(f"Hip, female — GSEA of KEGG metabolic pathways (* padj < {Q_CUT})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---- 6. Main -----------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = read_results()
    print(f"Features in table: {len(df)}")

    sets, pw_names = load_kegg_pathways()
    print(f"KEGG metabolic pathways loaded: {len(sets)}")

    # map IDs to KEGG keys (Entrez for genes, Cxxxxx for compounds)
    if FEATURE_TYPE == "gene":
        if ID_TYPE == "SYMBOL":
            m = symbol_to_entrez()
            df["key"] = df["id"].str.upper().map(m)
        elif ID_TYPE == "ENSEMBL":
            ens = df["id"].str.replace(r"\.\d+$", "", regex=True)  # drop version (.12)
            m = ensembl_to_entrez(ens.unique().tolist())
            df["key"] = ens.map(m)
        else:
            df["key"] = df["id"].str.replace(r"\.0$", "", regex=True)
    elif FEATURE_TYPE == "metabolite":
        df["key"] = df["id"].str.replace("cpd:", "", regex=False).str.upper()
    else:
        sys.exit("FEATURE_TYPE must be 'gene' or 'metabolite'")

    unmapped = df[df["key"].isna()]
    if len(unmapped):
        unmapped.to_csv(os.path.join(OUT_DIR, "unmapped_features.csv"), index=False)
    df = df.dropna(subset=["key"]).drop_duplicates("key")
    in_kegg = set().union(*sets.values())
    print(f"Mapped to KEGG IDs: {len(df)}  |  in a metabolic pathway: "
          f"{df['key'].isin(in_kegg).sum()}")

    label_of = dict(zip(df["key"], df["id"]))  # report original names in tables
    universe = set(df["key"])
    sig = df[(df["padj"] < P_CUT) & (df["log2FC"].abs() >= LFC_CUT)]
    groups = {"all": sig["key"], "up": sig.loc[sig["log2FC"] > 0, "key"],
              "down": sig.loc[sig["log2FC"] < 0, "key"]}
    print(f"Significant: {len(sig)}  (up {len(groups['up'])}, down {len(groups['down'])})")

    # ORA
    for name, keys in groups.items():
        res = run_ora(keys, universe, sets, pw_names, label_of)
        if res.empty:
            print(f"ORA {name}: no pathway hits")
            continue
        res.to_csv(os.path.join(OUT_DIR, f"ORA_{name}_KEGG_metabolic.csv"), index=False)
        print(f"ORA {name}: {(res['padj'] < Q_CUT).sum()} pathways with padj < {Q_CUT}")
        dotplot(res, f"Hip, female — KEGG metabolic pathways ({name})",
                os.path.join(OUT_DIR, f"ORA_{name}_dotplot.pdf"))

    # GSEA
    ranks = df.set_index("key")["log2FC"].sort_values(ascending=False)
    gsea = run_gsea(ranks, sets, pw_names, label_of)
    if not gsea.empty:
        gsea.to_csv(os.path.join(OUT_DIR, "GSEA_KEGG_metabolic.csv"), index=False)
        print(f"GSEA: {(gsea['padj'] < Q_CUT).sum()} pathways with padj < {Q_CUT}")
        nes_barplot(gsea, os.path.join(OUT_DIR, "GSEA_NES_barplot.pdf"))

    print(f"\nDone. Results in: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
