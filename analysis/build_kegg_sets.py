"""
Build KEGG mouse metabolic pathway sets for the measured metabolites.

Downloads (needs internet access to rest.kegg.jp):
  - the list of mouse (mmu) pathways
  - the compound -> pathway links
Keeps metabolic pathways (KEGG map00010 - map00999; overview maps like 01100 are left out)
that exist in mouse, and writes, for every pathway, which of the measured metabolites belong to it.

Usage: python build_kegg_sets.py <data.csv> <output.csv>
Output columns: Pathway ID, Pathway, Metabolites (names as in the data, without POS_/NEG_,
separated by " | "), KEGG IDs
"""
import sys
import urllib.request

import pandas as pd

from metabolite_annotation import ANNOT

# one column per compound when it was measured twice
DUPLICATES = ["NEG_CYCLIC-AMP", "NEG_INOSINE MONOPHOSPHATE", "NEG_AMINOADIPIC ACID",
              "POS_DL-PIPECOLIC ACID"]
# isobaric features: also count the alternative compound
ALTERNATIVES = {"POS_LEUCINE/ISOLEUCINE": ["C00407"],
                "NEG_URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE": ["C00052"],
                "POS_ACETYLGLUCOSAMINE/N-ACETYLGALACTOSAMIN": ["C01132"]}


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode()


def main(data_file, output_file):
    columns = [c for c in pd.read_csv(data_file, nrows=0).columns
               if c.startswith(("POS_", "NEG_")) and c not in DUPLICATES]

    mouse = {}
    for line in get("https://rest.kegg.jp/list/pathway/mmu").strip().splitlines():
        pid, name = line.split("\t")
        mouse["map" + pid[3:]] = name.replace(" - Mus musculus (house mouse)", "")

    pathways_of = {}
    for line in get("https://rest.kegg.jp/link/pathway/compound").strip().splitlines():
        cpd, path = line.split("\t")
        path = path.replace("path:", "")
        if path in mouse and int(path[3:]) < 1000:          # metabolic maps only
            pathways_of.setdefault(cpd.replace("cpd:", ""), set()).add(path)

    members = {}
    for col in columns:
        ids = [ANNOT[col][1]] + ALTERNATIVES.get(col, [])
        for kegg_id in filter(None, ids):
            for path in pathways_of.get(kegg_id, ()):
                members.setdefault(path, {})[col.split("_", 1)[1]] = kegg_id

    rows = [{"Pathway ID": path, "Pathway": mouse[path], "n": len(mets),
             "Metabolites": " | ".join(sorted(mets)),
             "KEGG IDs": " | ".join(mets[m] for m in sorted(mets))}
            for path, mets in members.items()]
    table = pd.DataFrame(rows).sort_values(["n", "Pathway"], ascending=[False, True])
    table.to_csv(output_file, index=False)
    no_id = [c for c in columns if not ANNOT[c][1]]
    print(f"{len(table)} mouse metabolic pathways contain at least one measured metabolite")
    print(f"metabolites without a KEGG ID (not in any pathway): {no_id}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
