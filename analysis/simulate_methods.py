"""
Simulation: which is more reliable for Hip vs CTX comparison?
  Method A: PQN per region, then divide CTX by a pilot-based factor (1.31)
  Method B: PQN per region, then log2(KI/WT) per region, compare Hip vs CTX
For each simulated dataset we know the TRUE answer and measure
  r      = correlation between estimated and true values (across metabolites)
  bias   = mean error (log2 units)
  spread = SD of error (log2 units)
Pure standard library (no numpy needed).
"""
import math, random, statistics as st

random.seed(1)
N_SIM = 1000
N_MET = 50            # metabolites
N_PER_GENO = 6        # mice per genotype (each mouse gives CTX + Hip)
SOLV = {"CTX": 1800.0, "HIP": 750.0}
WEIGHT = {"CTX": (121.0, 19.0), "HIP": (38.0, 4.0)}   # pilot mean, SD (mg)
PILOT_FACTOR = 1.31
BIO_SD = 0.20         # biological between-mouse SD (log2)
TECH_SD = 0.07        # extraction + instrument SD (log2)
PILOT_N = 6

def pqn(samples, ref):
    out = []
    for s in samples:
        q = sorted(s[j] / ref[j] for j in range(len(ref)))
        d = st.median(q)
        out.append([x / d for x in s])
    return out

def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))

res = {"A": {"r": [], "bias": [], "sd": []}, "B": {"r": [], "bias": [], "sd": []}}
for _ in range(N_SIM):
    # true biology (log2 scale)
    base = [random.gauss(0, 1.5) for _ in range(N_MET)]
    region_eff = [random.gauss(0, 0.6) for _ in range(N_MET)]           # true CTX vs HIP
    geno_ctx = [random.gauss(0, 0.4) if random.random() < 0.3 else 0 for _ in range(N_MET)]
    geno_hip = [g + (random.gauss(0, 0.4) if random.random() < 0.3 else 0) for g in geno_ctx]
    # the real cohort's mean tissue weight differs from the pilot (sampling of the pilot)
    cohort_mean = {r: random.gauss(m, s / math.sqrt(PILOT_N)) for r, (m, s) in WEIGHT.items()}

    data = {"CTX": {"KI": [], "WT": []}, "HIP": {"KI": [], "WT": []}}
    for geno in ("WT", "KI"):
        for _m in range(N_PER_GENO):
            mouse = [random.gauss(0, BIO_SD) for _ in range(N_MET)]
            for reg in ("CTX", "HIP"):
                w = max(5.0, random.gauss(cohort_mean[reg], WEIGHT[reg][1]))
                conc = w / SOLV[reg]
                vals = []
                for j in range(N_MET):
                    lv = base[j] + (region_eff[j] if reg == "CTX" else 0) + mouse[j]
                    if geno == "KI":
                        lv += geno_ctx[j] if reg == "CTX" else geno_hip[j]
                    lv += random.gauss(0, TECH_SD)
                    vals.append(2 ** lv * conc)            # signal per uL extract
                data[reg][geno].append(vals)
    # PQN per region, common reference = median of WT in that region
    for reg in data:
        ref = [st.median(s[j] for s in data[reg]["WT"]) for j in range(N_MET)]
        for g in ("WT", "KI"):
            data[reg][g] = pqn(data[reg][g], ref)

    def gmean_log(samples, j):
        return st.mean(math.log2(s[j]) for s in samples)

    # Method A: absolute CTX vs HIP (WT only), CTX / 1.31
    estA, trueA = [], []
    for j in range(N_MET):
        c = gmean_log(data["CTX"]["WT"], j) - math.log2(PILOT_FACTOR)
        h = gmean_log(data["HIP"]["WT"], j)
        estA.append(c - h); trueA.append(region_eff[j])
    # Method B: log2(KI/WT) in HIP minus in CTX
    estB, trueB = [], []
    for j in range(N_MET):
        fc_c = gmean_log(data["CTX"]["KI"], j) - gmean_log(data["CTX"]["WT"], j)
        fc_h = gmean_log(data["HIP"]["KI"], j) - gmean_log(data["HIP"]["WT"], j)
        estB.append(fc_h - fc_c); trueB.append(geno_hip[j] - geno_ctx[j])
    for k, est, tru in (("A", estA, trueA), ("B", estB, trueB)):
        err = [e - t for e, t in zip(est, tru)]
        res[k]["r"].append(pearson(est, tru))
        res[k]["bias"].append(st.mean(err))
        res[k]["sd"].append(st.stdev(err))

def q(v, p):
    v = sorted(v); return v[int(p * (len(v) - 1))]

for k, name in (("A", "Method A: CTX / 1.31, compare absolute levels"),
                ("B", "Method B: log2(KI/WT), compare Hip vs CTX")):
    r, b, s = res[k]["r"], res[k]["bias"], res[k]["sd"]
    print(name)
    print(f"  r with truth          median {st.median(r):.3f}  (5-95%: {q(r,.05):.3f}-{q(r,.95):.3f})")
    print(f"  bias (log2)           median {st.median(b):+.3f}  (5-95%: {q(b,.05):+.3f} to {q(b,.95):+.3f})")
    print(f"  |bias| as fold        95th pct {2**q([abs(x) for x in b],.95):.2f}x")
    print(f"  spread of error (SD)  median {st.median(s):.3f} log2")
    print()
