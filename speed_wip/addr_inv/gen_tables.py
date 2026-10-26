import json, numpy as np
CK = [("init", "res_0.json"), ("old s1000*", "res_old1000.json"), ("s6000", "res_6000.json"), ("s7000", "res_b.json"),
      ("s6000 thoughts + init addr", "res_6000_initaddr.json"), ("s7000 thoughts + init addr", "res_b_initaddr.json")]
R = {n: json.load(open(f)) for n, f in CK}
mn = lambda xs: float(np.mean(xs))
P = print
G4 = ("dead", "low", "mid", "busy")


def lv(r, l, key):
    return [x for x in r[key] if x["level"] == l and x["pass_"] != "w0"]


P("### T1. Headline per level (means over passes p1-p5 and 3 heads; fixed batch of 16,384 positions)\n")
P("| point | level | distinct rows /1k pos, per head (ideal 5932) | pooled over heads /1k (ideal 17396) | rows: top-1% share / Gini (uniform draws: 0.07 / 0.73) | sub-key ent/log S | sub-key dead in batch | a PR / erank | â PR / erank | q PR / erank (of 64) |")
P("|---|---|---|---|---|---|---|---|---|---|")
for n, _ in CK:
    for l in range(4):
        u, g, G = lv(R[n], l, "usage"), lv(R[n], l, "g2"), lv(R[n], l, "geom")
        P(f"| {n} | {l} ({[16, 32, 64, 128][l]}) | {mn([x['per_head'] for x in g]):.0f} | {mn([x['pooled'] for x in g]):.0f} | {mn([x['r_top1'] for x in u]):.2f} / {mn([x['r_gini'] for x in u]):.3f} | "
          f"{mn([(x['h1_ent'] + x['h2_ent']) / 2 for x in u]):.3f} | {mn([(x['h1_dead'] + x['h2_dead']) / 2 for x in u]):.3f} | {mn([x['a']['pr'] for x in G]):.1f} / {mn([x['a']['erank'] for x in G]):.0f} | "
          f"{mn([x['a_hat']['pr'] for x in G]):.1f} / {mn([x['a_hat']['erank'] for x in G]):.0f} | {mn([x['q']['pr'] for x in G]):.1f} / {mn([x['q']['erank'] for x in G]):.0f} |")

P("\n### T2. Usage per pass (mean over levels and heads): distinct rows /1k pos per head / sub-key dead / q resultant R (cone)\n")
P("| point | w0 | p1 | p2 | p3 | p4 | p5 |")
P("|---|---|---|---|---|---|---|")
for n, _ in CK[:4]:
    cells = []
    for p in ["w0", "p1", "p2", "p3", "p4", "p5"]:
        g = [x for x in R[n]["g2"] if x["pass_"] == p]
        u = [x for x in R[n]["usage"] if x["pass_"] == p]
        G = [x for x in R[n]["geom"] if x["pass_"] == p]
        cells.append("-" if not g else f"{mn([x['per_head'] for x in g]):.0f} / {mn([(x['h1_dead'] + x['h2_dead']) / 2 for x in u]):.2f} / {mn([x['q']['R'] for x in G]):.2f}")
    P(f"| {n} | " + " | ".join(cells) + " |")

P("\n### T3. Pass x level, distinct rows /1k pos per head (12k run)\n")
for n in ("s6000", "s7000"):
    P(f"| {n} | p1 | p2 | p3 | p4 | p5 |")
    P("|---|---|---|---|---|---|")
    for l in range(4):
        P(f"| L{l} | " + " | ".join(f"{x['per_head']:.0f}" for x in lv(R[n], l, "g2")) + " |")
    P("")

P("### T4. Thought / query geometry (mean over passes). top1/5/20 = variance share of the top PCA directions; cos = mean pairwise cosine of random positions (uncentred); R = length of the mean unit vector\n")
P("| point | level | a: top1/5/20, cos | â: top1/5/20, cos | q: top1/5/20, cos, R | keys: PR (of 64), mean cos, R |")
P("|---|---|---|---|---|---|")
for n, _ in CK[:4]:
    for l in range(4):
        G = lv(R[n], l, "geom")
        k = R[n]["geom_keys"][l]
        f = lambda t: f"{mn([x[t]['top1'] for x in G]):.2f}/{mn([x[t]['top5'] for x in G]):.2f}/{mn([x[t]['top20'] for x in G]):.2f}, {mn([x[t]['cos'] for x in G]):.3f}"
        P(f"| {n} | {l} | {f('a')} | {f('a_hat')} | {f('q')}, {mn([x['q']['R'] for x in G]):.2f} | {k['pr']:.1f}, {k['cos']:.3f}, {k['R']:.2f} |")

P("\n### T5. Key geometry by batch usage group (mean over heads and halves; the pass-p3 queries). dead = 0 selections in the batch, low <= median, busy = top 10%\n")
P("| point | level | n dead/low/mid/busy (of 3072) | cos(key, mean query dir) | max cos to any query | median 8th-best score | drift abs(k_online - k_EMA) | Spearman(usage, max cos) | Spearman(usage, drift) |")
P("|---|---|---|---|---|---|---|---|---|")
for n, _ in CK[:4]:
    for r in R[n]["keygeom"]:
        j = lambda nm, fmt: "/".join(format(r[f"{g}_{nm}"], fmt) for g in G4)
        P(f"| {n} | {r['level']} | {'/'.join(str(r['n_' + g]) for g in G4)} | {j('cmean', '.2f')} | {j('cmax', '.2f')} | {r['sp_mth']:.2f} | {j('drift', '.3f')} | {r['sp_cmax']:.2f} | {r['sp_drift']:.2f} |")

kinds = ["lag1", "lag8", "lag64", "lag256", "sameDoc>64", "diffSeq", "sameTok_diffSeq", "samePos<4_diffSeq", "samePos>=256_diffSeq"]
P("\n### T6. Content dependence (mean over passes p1-p5 and heads): row-set overlap (shared rows / 6) by pair type; MI (nats, shuffle-corrected) of the top-1 sub-key (half 1) with the document / token (top-50 tokens) / position bucket (32). H(doc) = 3.57, H(token | top-50) = 3.35\n")
P("| point | level | " + " | ".join(kinds) + " | MI doc | MI token | MI pos |")
P("|---" * (len(kinds) + 5) + "|")
for n, _ in CK:
    for l in range(4):
        c = lv(R[n], l, "content")
        P(f"| {n} | {l} | " + " | ".join(f"{mn([x[k] for x in c]):.3f}" for k in kinds) + f" | {mn([x['mi_doc'] for x in c]):.2f} | {mn([x['mi_tok50'] for x in c]):.2f} | {mn([x['mi_pos32'] for x in c]):.2f} |")
w0 = [x for x in R["old s1000*"]["content"] if x["pass_"] == "w0"]
P("| old s1000 w0 (word-keyed read0, mean over levels) | all | " + " | ".join(f"{mn([x[k] for x in w0]):.3f}" for k in kinds)
  + f" | {mn([x['mi_doc'] for x in w0]):.2f} | {mn([x['mi_tok50'] for x in w0]):.2f} | {mn([x['mi_pos32'] for x in w0]):.2f} |")

P("\n### T7. Product-key halves (H4): over the selected pairs (i, j), H(i), H(j) and the shuffle-corrected MI(i; j), in nats (log 512 = 6.24). exp(MI) = the factor lost in row perplexity relative to independent halves\n")
P("| point | level | H(i) | H(j) | MI(i;j) | exp(MI) |")
P("|---|---|---|---|---|---|")
for n, _ in CK:
    for l in range(4):
        u = lv(R[n], l, "usage")
        P(f"| {n} | {l} | {mn([x['Hi'] for x in u]):.2f} | {mn([x['Hj'] for x in u]):.2f} | {mn([x['mi'] for x in u]):.2f} | {np.exp(mn([x['mi'] for x in u])):.1f} |")

V = ["trained", "random_keys", "reseed_dead", "centred_q", "centred_q+random_keys", "whitened_ahat", "whitened_ahat+random_keys", "raw_a(no_norm)"]
P("\n### T8. Counterfactual re-addressing of the SAME thoughts (pass p4; p1 is in the JSON): distinct rows per head over 16,384 positions (uniform random would give about 82,000) / sub-key dead in the batch\n")
P("| point | level | " + " | ".join(V) + " |")
P("|---" * (len(V) + 2) + "|")
for n, _ in CK[:4]:
    for x in [x for x in R[n]["counterfactual"] if x["pass_"] == "p4"]:
        P(f"| {n} | {x['level']} | " + " | ".join(f"{x[v]['row_dist']:.0f} / {x[v]['sub_dead']:.2f}" for v in V) + " |")

P("\n### T9. Who uses the top-50 rows by write count (s6000; all heads and passes of the batch)\n")
P("| level | wcount: top-50 mean / median / q90 | share of all row draws (uniform: 0.0002) | positions touching one | token entropy (bits) / distinct tokens | median freq rank (batch: 310) | share pos<16 (base 0.031) | top tokens (share / batch share) | pass p1..p5 | batch top-50 ∩ wcount top-50 |")
P("|---|---|---|---|---|---|---|---|---|---|")
for d in R["s6000"]["popular"]:
    tt = ", ".join(f"{t.replace(chr(0x120), '_').replace(chr(0x10a), 'NL')} {s:.3f}/{b:.3f}" for (t, s), b in list(zip(d["top_tokens"], d["top_tokens_batch_share"]))[:5])
    P(f"| {d['level']} | {d['wc_top50_mean']:.0f} / {d['wc_median']:.0f} / {d['wc_q90']:.0f} | {d['share_of_selections']:.3f} | {d['share_positions']:.2f} | {d['tok_ent_bits']:.1f} / {d['tok_distinct']} | {d['rank_median']:.0f} | {d['share_pos<16']:.3f} | {tt} | "
      + "/".join(f"{v:.2f}" for v in d["pass_hist"].values()) + f" | {d['overlap_batch_top50']} |")

P("\n### T10. Gate vs row popularity: Spearman of the read gate with the mean log(1 + wcount) of its 6 rows (mean gate in brackets)\n")
P("| point | level | p1 | p2 | p3 | p4 | p5 |")
P("|---|---|---|---|---|---|---|")
for n in ("s6000", "s7000"):
    for l in range(4):
        P(f"| {n} | {l} | " + " | ".join(f"{x['sp']:+.2f} ({x['gate_mean']:.2f})" for x in R[n]["gate"] if x["level"] == l) + " |")

P("\n### T11. g1: 20 diverse contexts (sequences 0-19, position 400, pass p3). Per head: distinct rows of the 120 / rows shared by 2 or more contexts / max multiplicity / contexts touching a shared row\n")
P("| point | level | head 0; head 1; head 2 | pooled 3 heads: distinct of 360 |")
P("|---|---|---|---|")
for n, _ in CK[:4]:
    for x in R[n]["g1"]:
        P(f"| {n} | {x['level']} | " + "; ".join(f"{h['distinct']}/{h['shared_rows']}/{h['max_mult']}/{h['contexts_with_shared']}" for h in x["per_head"]) + f" | {x['pooled_distinct']} |")
