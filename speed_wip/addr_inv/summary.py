import json, numpy as np, sys
F = [("init (step 0)", "res_0.json"), ("old run step 1000 (read0 on)", "res_old1000.json"), ("12k run step 6000", "res_6000.json"),
     ("step-6000 thoughts, INIT W_q/norm/keys", "res_6000_initaddr.json")] + [tuple(a.split("=")) for a in sys.argv[1:]]
for name, f in F:
    try: R = json.load(open(f))
    except Exception: continue
    print(f"\n### {name}  (verify {R.get('verify')})")
    print("lvl | rows top1% | rows top10% | rows gini | distinct/1k pos per head (ideal 5932) | pooled level distinct/1k (ideal 17396) | subkey ent | subkey dead | a PR / erank | a_hat PR / erank | q PR / erank (of 64) | q mean cos | q R")
    for l in range(4):
        u = [r for r in R["usage"] if r["level"] == l and r["pass_"] != "w0"]
        g = [r for r in R["g2"] if r["level"] == l and r["pass_"] != "w0"]
        G = [r for r in R["geom"] if r["level"] == l and r["pass_"] != "w0"]
        mn = lambda xs: float(np.mean(xs))
        print(f"L{l} | {mn([r['r_top1'] for r in u]):.3f} | {mn([r['r_top10'] for r in u]):.3f} | {mn([r['r_gini'] for r in u]):.3f} | "
              f"{mn([r['per_head'] for r in g]):.0f} | {mn([r['pooled'] for r in g]):.0f} | {mn([(r['h1_ent']+r['h2_ent'])/2 for r in u]):.3f} | "
              f"{mn([(r['h1_dead']+r['h2_dead'])/2 for r in u]):.3f} | {mn([r['a']['pr'] for r in G]):.1f} / {mn([r['a']['erank'] for r in G]):.0f} | "
              f"{mn([r['a_hat']['pr'] for r in G]):.1f} / {mn([r['a_hat']['erank'] for r in G]):.0f} | {mn([r['q']['pr'] for r in G]):.1f} / {mn([r['q']['erank'] for r in G]):.0f} | "
              f"{mn([r['q']['cos'] for r in G]):.3f} | {mn([r['q']['R'] for r in G]):.3f}")
    w0 = [r for r in R["usage"] if r["pass_"] == "w0"]
    if w0:
        g = [r for r in R["g2"] if r["pass_"] == "w0"]
        print("w0 (word-keyed read0): rows top1%% %.3f distinct/1k/head %.0f subkey dead %.3f" % (np.mean([r['r_top1'] for r in w0]), np.mean([r['per_head'] for r in g]), np.mean([(r['h1_dead']+r['h2_dead'])/2 for r in w0])))
