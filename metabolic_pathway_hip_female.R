## =============================================================================
## Metabolic pathway analysis — Hip, Female
##
## Input : the differential results table you already used for the heatmap /
##         log2FC scatter plot (one row per feature, with log2FC and p-values).
## Output: KEGG metabolic pathway enrichment (ORA for up / down / all
##         significant features + GSEA on the full ranked list), tables and plots.
##
## Works for two kinds of data — set `feature_type` below:
##   "gene"       -> transcriptomics / proteomics (gene symbols, Ensembl or Entrez)
##   "metabolite" -> metabolomics (KEGG compound IDs, e.g. C00031)
## =============================================================================

## ---- 1. Settings: edit these ------------------------------------------------
input_file   <- "hip_female_results.csv"  # your hip / female results table
feature_type <- "gene"                    # "gene" or "metabolite"

id_col  <- "gene"     # column with feature IDs
lfc_col <- "log2FC"   # column with log2 fold change
p_col   <- "padj"     # column with adjusted p-value (use raw p if you must)

# gene data only
id_type  <- "SYMBOL"        # "SYMBOL", "ENSEMBL" or "ENTREZID"
orgdb    <- "org.Mm.eg.db"  # org.Mm.eg.db mouse, org.Rn.eg.db rat, org.Hs.eg.db human
organism <- "mmu"           # KEGG code: mmu mouse, rno rat, hsa human

p_cut   <- 0.05  # significance cut-off for features
lfc_cut <- 1     # |log2FC| cut-off for features (set 0 to use p only)
q_cut   <- 0.05  # cut-off for enriched pathways (padj)

out_dir <- "pathway_hip_female"

## ---- 2. Packages ------------------------------------------------------------
## Install once if needed:
## install.packages(c("BiocManager", "ggplot2", "dplyr", "readr"))
## BiocManager::install(c("clusterProfiler", "enrichplot", "KEGGREST", orgdb))
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(enrichplot)
  library(KEGGREST)
  library(ggplot2)
  library(dplyr)
  library(readr)
})

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## KEGG pathway IDs 00xxx and 01xxx are the "Metabolism" section of KEGG
is_metabolic <- function(ids) grepl("^[a-z]*0[01][0-9]{3}$", sub("^path:", "", ids))

## ---- 3. Read results --------------------------------------------------------
res <- read_csv(input_file, show_col_types = FALSE) %>%
  transmute(id = as.character(.data[[id_col]]),
            log2FC = as.numeric(.data[[lfc_col]]),
            padj = as.numeric(.data[[p_col]])) %>%
  filter(!is.na(id), !is.na(log2FC)) %>%
  distinct(id, .keep_all = TRUE)

cat("Features in table:", nrow(res), "\n")

## ---- 4. Map IDs and build the pathway sets ----------------------------------
if (feature_type == "gene") {
  library(orgdb, character.only = TRUE)
  if (id_type != "ENTREZID") {
    map <- bitr(res$id, fromType = id_type, toType = "ENTREZID", OrgDb = get(orgdb))
    res <- inner_join(res, map, by = c("id" = id_type)) %>%
      distinct(ENTREZID, .keep_all = TRUE)
  } else {
    res$ENTREZID <- res$id
  }
  res$key <- res$ENTREZID
  cat("Genes mapped to Entrez:", nrow(res), "\n")

} else if (feature_type == "metabolite") {
  # compound -> pathway links straight from KEGG (needs internet)
  links <- keggLink("pathway", "compound")
  term2gene <- data.frame(term = sub("path:", "", links),
                          gene = sub("cpd:", "", names(links))) %>%
    filter(grepl("^map", term), is_metabolic(term))
  pw_names <- keggList("pathway")
  term2name <- data.frame(term = sub("path:", "", names(pw_names)),
                          name = unname(pw_names)) %>%
    filter(term %in% term2gene$term)
  res$key <- sub("^cpd:", "", res$id)
  cat("Compounds found in KEGG metabolic pathways:",
      sum(res$key %in% term2gene$gene), "of", nrow(res), "\n")

} else {
  stop("feature_type must be 'gene' or 'metabolite'")
}

## ---- 5. Significant features ------------------------------------------------
sig  <- res %>% filter(!is.na(padj), padj < p_cut, abs(log2FC) >= lfc_cut)
up   <- sig$key[sig$log2FC > 0]
down <- sig$key[sig$log2FC < 0]
universe <- res$key
cat("Significant:", nrow(sig), " up:", length(up), " down:", length(down), "\n")

## ---- 6. Over-representation analysis (ORA) ----------------------------------
run_ora <- function(keys) {
  if (length(keys) < 3) return(NULL)
  if (feature_type == "gene") {
    e <- enrichKEGG(gene = keys, organism = organism, universe = universe,
                    pvalueCutoff = 1, qvalueCutoff = 1, minGSSize = 3)
    if (is.null(e)) return(NULL)
    e@result <- e@result[is_metabolic(e@result$ID), ]   # keep metabolic pathways
    e <- setReadable(e, OrgDb = get(orgdb), keyType = "ENTREZID")
  } else {
    e <- enricher(gene = keys, universe = universe, TERM2GENE = term2gene,
                  TERM2NAME = term2name, pvalueCutoff = 1, qvalueCutoff = 1,
                  minGSSize = 2)
  }
  if (is.null(e) || nrow(e@result) == 0) return(NULL)
  e@result$p.adjust <- p.adjust(e@result$pvalue, method = "BH")  # re-adjust after filtering
  e
}

ora <- list(all = run_ora(sig$key), up = run_ora(up), down = run_ora(down))

for (nm in names(ora)) {
  e <- ora[[nm]]
  if (is.null(e)) { cat("ORA", nm, ": too few features / no hits\n"); next }
  write_csv(e@result, file.path(out_dir, paste0("ORA_", nm, "_metabolic_pathways.csv")))
  n_sig <- sum(e@result$p.adjust < q_cut)
  cat("ORA", nm, ":", n_sig, "pathways with padj <", q_cut, "\n")
  e@result <- e@result[order(e@result$pvalue), ]
  p <- dotplot(e, showCategory = 15) +
    ggtitle(paste0("Hip, female — metabolic pathways (", nm, ")"))
  ggsave(file.path(out_dir, paste0("ORA_", nm, "_dotplot.pdf")), p, width = 8, height = 7)
}

## ---- 7. GSEA on all features ranked by log2FC -------------------------------
## Uses every measured feature, so it can find pathways shifted as a whole
## even when few single features pass the cut-off.
ranks <- sort(setNames(res$log2FC, res$key), decreasing = TRUE)
ranks <- ranks[!duplicated(names(ranks))]

if (feature_type == "gene") {
  gsea <- gseKEGG(geneList = ranks, organism = organism, minGSSize = 10,
                  pvalueCutoff = 1, eps = 0, seed = TRUE)
  gsea@result <- gsea@result[is_metabolic(gsea@result$ID), ]
} else {
  gsea <- GSEA(geneList = ranks, TERM2GENE = term2gene, TERM2NAME = term2name,
               minGSSize = 3, pvalueCutoff = 1, eps = 0, seed = TRUE)
}
gsea@result$p.adjust <- p.adjust(gsea@result$pvalue, method = "BH")
gsea@result <- gsea@result[order(gsea@result$pvalue), ]
write_csv(gsea@result, file.path(out_dir, "GSEA_metabolic_pathways.csv"))
cat("GSEA:", sum(gsea@result$p.adjust < q_cut), "pathways with padj <", q_cut, "\n")

if (nrow(gsea@result) > 0) {
  # NES bar plot: positive = pathway up in this group, negative = down
  top <- head(gsea@result, 20) %>%
    mutate(Description = reorder(Description, NES),
           direction = ifelse(NES > 0, "Up", "Down"))
  p <- ggplot(top, aes(NES, Description, fill = direction)) +
    geom_col() +
    geom_text(aes(label = ifelse(p.adjust < q_cut, "*", "")),
              hjust = ifelse(top$NES > 0, -0.3, 1.3), size = 5) +
    scale_fill_manual(values = c(Up = "#d6604d", Down = "#4393c3")) +
    labs(title = "Hip, female — GSEA of KEGG metabolic pathways",
         subtitle = paste("* padj <", q_cut), x = "Normalized enrichment score", y = NULL) +
    theme_bw()
  ggsave(file.path(out_dir, "GSEA_NES_barplot.pdf"), p, width = 9, height = 7)

  # running-score plots for the top 3 pathways
  pdf(file.path(out_dir, "GSEA_top3_running_score.pdf"), width = 10, height = 7)
  print(gseaplot2(gsea, geneSetID = head(gsea@result$ID, 3), pvalue_table = TRUE))
  dev.off()
}

## ---- 8. Optional: colour a KEGG map by log2FC -------------------------------
## BiocManager::install("pathview"); then pick an ID from the tables above, e.g.
## library(pathview)
## pathview(gene.data = ranks, pathway.id = "00190", species = organism)  # genes
## pathview(cpd.data = ranks, pathway.id = "00020", species = "ko")       # metabolites

cat("\nDone. Results in:", normalizePath(out_dir), "\n")
