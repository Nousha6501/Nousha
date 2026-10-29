"""
Build mouse-specific KEGG metabolite sets for the pathway analysis.

KEGG's compound list for a pathway is the same for every organism (it includes bacterial
and plant steps). Here a measured compound is kept in a mouse pathway only if it takes part in
a reaction of that pathway that is catalysed by a mouse gene:
    mouse genes in mmuXXXXX -> their KEGG orthologs (KO) -> reactions of those KOs
    -> keep reactions drawn on mapXXXXX -> compounds of those reactions
    -> keep compounds that are on mapXXXXX -> keep the ones you measured
Maps drawn without reactions (e.g. oxidative phosphorylation) keep all mouse-enzyme reactions.

Output: prints the PATHWAYS dictionary to paste into the pathway cell, and saves it as JSON.
Needs internet access to rest.kegg.jp. KEGG is updated over time, so report the download date.
"""
import collections
import datetime
import json
import urllib.request

MIN_SIZE = 3
OUTPUT_FILE = "kegg_mouse_sets.json"

# measured metabolite name -> KEGG compound IDs (several IDs when the method cannot tell
# isomers apart, e.g. leucine/isoleucine, R/S forms)
IDS = {
 '2-HYDROXYGLUTERATE':['C02630','C01087','C03196'], '2-PHOSPHOGLYCERATE':['C00631'],
 '4-GUANIDINOBUTANOATE':['C01035'], '6-PHOSPHOGLUCONATE':['C00345'], 'ACETYL-COA':['C00024'],
 'ACETYLGLUCOSAMINE/N-ACETYLGALACTOSAMIN':['C00140','C01132','C03136','C03878'],
 'ADENINE':['C00147'], 'ADENOSINE':['C00212'], "ADENOSINE 3',5'-DIPHOSPHATE":['C00054'],
 'ADENOSINE DIPHOSPHATE RIBOSE':['C00301'], 'ADENOSINE-MONOPHOSPHATE':['C00020'],
 'ADENYLSUCCINIC ACID':['C03794'], 'ADP-GLUCOSE':['C00498'], 'ALLOTHREONINE/THREONINE':['C00188','C05519'],
 'AMINOADIPATE':['C00956'], 'ARGININE':['C00062'], 'ASPARTATE':['C00049'], 'ATP/dGTP':['C00002','C00286'],
 'BETAINE':['C00719'], 'CARNOSINE':['C00386'], 'CDP':['C00112'], 'CDP-ETHANOLAMINE':['C00570'],
 'CITICOLINE':['C00307'], 'COENZYME A':['C00010'], 'CREATINE':['C00300'], 'CREATININE':['C00791'],
 'CYCLIC AMP':['C00575'], 'CYTIDINE':['C00475'], 'CYTIDINE MONOPHOSPHATE':['C00055'], 'CYTOSINE':['C00380'],
 'D-RIBOSE 5-PHOSPHATE':['C00117'], 'DEOXYCARNITINE':['C01181'], 'DEOXYGUANOSINE-MONOPHOSPHATE':['C00362'],
 'DIETHANOLAMINE':['C06772'], 'ETHANOLAMINE PHOSPHATE':['C00346'], 'FAD':['C00016'],
 'FRUCTOSE-1,6-BISPHOSPHATE':['C00354','C05378'], 'FUMARATE':['C00122'], 'GALACTITOL':['C01697'],
 'GAMMA-AMINOBUTYRATE':['C00334'], 'GDP':['C00035'], 'GLUTAMATE':['C00025'], 'GLUTAMINE':['C00064'],
 'GLUTATHIONE DISULFIDE':['C00127'], 'GLUTATHIONE REDUCED':['C00051'], 'GLYCINE':['C00037'], 'GTP':['C00044'],
 'GUANIDINOACETATE':['C00581'], 'GUANINE':['C00242'], 'GUANOSINE':['C00387'],
 'GUANOSINE DIPHOSPHATE MANNOSE':['C00096'], 'GUANOSINE MONOPHOSPHATE':['C00144'], 'HISTIDINE':['C00135'],
 'HOMOSERINE':['C00263'], 'HYPOXANTHINE':['C00262'], 'INOSINE':['C00294'], 'INOSINE-MONOPHOSPHATE':['C00130'],
 'L-ALANINE':['C00041'], 'LACTATE':['C00186','C00256'], 'LEUCINE/ISOLEUCINE':['C00123','C00407'],
 'LYSINE':['C00047'], 'MALATE':['C00149','C00497'], 'METHIONINE':['C00073'], 'METHYLTHIOADENOSINE':['C00170'],
 'MYO-INOSITOL':['C00137'], 'N,N,N-TRIMETHYLLYSINE':['C03793'], 'N-ACETYLASPARTATE':['C01042'],
 'N-ACETYLGLUTAMATE':['C00624'], 'N-ACETYLNEURAMINATE':['C00270','C19910'], 'NAD+':['C00003'], 'NADP':['C00006'],
 'NADPH':['C00005'], 'NICOTINAMIDE':['C00153'], 'PALMITOYLCARNITINE':['C02990'], 'PANTOTHENATE':['C00864'],
 'PHENYLALANINE':['C00079'], 'PHOSPHORYLCHOLINE':['C00588'], 'PIPECOLATE':['C00408'],
 'PROLINE':['C00148'], 'PYROGLUTAMIC ACID/OXOPROLINE':['C01879'], 'RIBOFLAVIN':['C00255'],
 'S-ADENOSYLHOMOCYSTEINE':['C00021'], 'S-ADENOSYLMETHIONINE':['C00019'], 'SARCOSINE':['C00213'],
 'SERINE':['C00065','C00740'], 'SORBITOL':['C00794'], 'SUCCINATE':['C00042'], 'TAURINE':['C00245'],
 'TRYPTOPHAN':['C00078'], 'TYROSINE':['C00082'], 'UDP-N-ACETYL-GLUCOSAMINE':['C00043'], 'URACIL':['C00106'],
 'URIC ACID':['C00366'], "URIDINE 5'-DIPHOSPHATE":['C00015'], 'URIDINE DIPHOSPHATE GLUCURONIC ACID':['C00167'],
 'URIDINE DIPHOSPHATEGALACTOSE/URIDINE DIPHOSPHATE GLUCOSE':['C00052','C00029'],
 'URIDINE MONOPHOSPHATE':['C00105'], 'UTP':['C00075'], 'XANTHINE':['C00385'],
}


def download(path):
    with urllib.request.urlopen(f"https://rest.kegg.jp/{path}", timeout=300) as r:
        return [line.split("\t") for line in r.read().decode().strip().splitlines()]


def grouped(rows, key=0, value=1, strip=""):
    out = collections.defaultdict(set)
    for row in rows:
        out[row[key].replace(strip, "")].add(row[value])
    return out


print("Downloading from KEGG ...")
names = {pid: name.split(" - Mus musculus")[0].strip()
         for pid, name in download("list/pathway/mmu")}
gene_ko = grouped(download("link/ko/mmu"))
ko_rn = grouped(download("link/reaction/ko"))
rn_cpd = grouped(download("link/compound/reaction"))
map_rn = grouped(download("link/reaction/pathway"), strip="path:")
map_cpd = grouped(download("link/compound/pathway"), strip="path:")
path_genes = grouped(download("link/pathway/mmu"), key=1, value=0, strip="path:")

measured = collections.defaultdict(set)
for metabolite, cpds in IDS.items():
    for c in cpds:
        measured["cpd:" + c].add(metabolite)

sets = {}
for pid, name in names.items():
    if not pid[3:].startswith("00"):          # metabolic maps only, no overview maps (011xx)
        continue
    mp = "map" + pid[3:]
    mouse_rn = {r for g in path_genes[pid] for k in gene_ko[g] for r in ko_rn[k]}
    if map_rn[mp]:
        mouse_rn &= map_rn[mp]
    mouse_cpd = {c for r in mouse_rn for c in rn_cpd[r]} & map_cpd[mp]
    members = sorted({m for c in mouse_cpd for m in measured.get(c, ())})
    if len(members) >= MIN_SIZE:
        sets[name] = members

sets = dict(sorted(sets.items(), key=lambda kv: (-len(kv[1]), kv[0])))
json.dump({"date": str(datetime.date.today()), "sets": sets}, open(OUTPUT_FILE, "w"), indent=1)
print(f"{len(sets)} pathways with >= {MIN_SIZE} measured metabolites, saved to {OUTPUT_FILE}\n")
print("PATHWAYS = {")
for name, members in sets.items():
    print(f"    {name!r}: {members!r},")
print("}")
