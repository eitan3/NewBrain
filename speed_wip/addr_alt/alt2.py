"""Second batch of address alternatives (plan_v2/18), same protocol as alt.py / addr_inv/sim.py.
usage: python -B alt2.py OUT.json [pass=2]
inputs : pca16 | pca32 (PCA-whitened top-r of the EMA-normalised a_hat, fit on seqs 0-15 of 6K, frozen)
address: flyhash   - per half a fixed sparse binary expansion (512 Kenyon cells, each sampling 6 input dims) of the
                     whitened input; winner-take-all = top-m cells (Dasgupta et al. 2017); keys never learned
         itq       - per half ITQ (Gong & Lazebnik 2011): fit a rotation of a 9-dim PCA/random block that minimises the
                     sign-quantisation error on the fit set, then 9 sign bits = 512 buckets, multi-probe scores
         opq_km    - OPQ-style: the whitened input is randomly rotated and split into two independent halves (the
                     halves are uncorrelated by construction), each half k-means'd (S=512) on the fit set
         rp_pk_sep - rp_pk with the two halves fed by DISJOINT PCA directions (odd / even components): independent halves
         simhash_b - simhash with 9 bits per half drawn from disjoint direction sets (odd / even PCA components)
"""
import os, sys, json, math, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("altbase", os.path.join(os.path.dirname(os.path.abspath(__file__)), "alt_common.py"))
cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)

t0 = time.time()
OUTF = sys.argv[1]
PASS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
S, m, km, L, H = cm.S, cm.m, cm.km, cm.L, cm.H
VERT = cm.VERT
gen = cm.gen


def itq(Z, iters=50):
    """ITQ rotation for the centred fit codes Z (n, b): argmin |sign(ZR) - ZR|."""
    b = Z.size(1)
    R = torch.linalg.qr(torch.randn(b, b, generator=gen))[0]
    for _ in range(iters):
        Bm = torch.sign(Z @ R)
        U, _, Vt = torch.linalg.svd(Z.T @ Bm)
        R = U @ Vt
    return R


RES = {}
for l in range(L):
    sets_raw = cm.level_sets(PASS, l)
    for inp in ("pca16", "pca32"):
        sets = cm.make_inputs(sets_raw, inp)
        din = sets["fit"].size(1)
        for h in range(H):
            g = torch.Generator().manual_seed(5000 + 1000 * l + 10 * h)
            perm = torch.randperm(din, generator=g)
            ev_idx, od_idx = perm[: din // 2], perm[din // 2:]        # disjoint direction sets for the two halves
            for ad in ("flyhash", "itq", "opq_km", "rp_pk_sep", "simhash_b"):
                if ad == "flyhash":
                    Ms = []
                    for _ in range(2):
                        Mx = torch.zeros(din, S)
                        for c in range(S):
                            Mx[torch.randperm(din, generator=g)[:6], c] = 1.0
                        Ms.append(Mx)
                    mu = [sets["fit"] @ Mx for Mx in Ms]
                    mm = [x.mean(0) for x in mu]; sd = [x.std(0) for x in mu]
                    score = lambda n_: tuple((sets[n_] @ Ms[i] - mm[i]) / sd[i] for i in range(2))
                elif ad in ("itq", "simhash_b"):
                    blocks = []
                    for idx in (ev_idx, od_idx):
                        Pb = torch.randn(idx.numel(), 9, generator=g)
                        Zf = sets["fit"][:, idx] @ Pb
                        med = Zf.median(0).values
                        R = itq(Zf - med) if ad == "itq" else torch.eye(9)
                        sd = ((Zf - med) @ R).std(0)
                        blocks.append((idx, Pb, med, R, sd))
                    def score(n_, blocks=blocks):
                        return tuple(((((sets[n_][:, idx] @ Pb) - med) @ R) / sd) @ VERT.T for idx, Pb, med, R, sd in blocks)
                elif ad == "rp_pk_sep":
                    Pa = torch.randn(ev_idx.numel(), 64, generator=g); Pb_ = torch.randn(od_idx.numel(), 64, generator=g)
                    K1 = F.normalize(torch.randn(S, 64, generator=g), dim=-1); K2 = F.normalize(torch.randn(S, 64, generator=g), dim=-1)
                    def score(n_, Pa=Pa, Pb_=Pb_, K1=K1, K2=K2):
                        return (F.normalize(sets[n_][:, ev_idx] @ Pa, dim=-1) @ K1.T, F.normalize(sets[n_][:, od_idx] @ Pb_, dim=-1) @ K2.T)
                else:  # opq_km
                    Rr = torch.linalg.qr(torch.randn(din, din, generator=g))[0]
                    def qh(n_, Rr=Rr):
                        z = sets[n_] @ Rr
                        return F.normalize(z[:, : din // 2], dim=-1), F.normalize(z[:, din // 2:], dim=-1)
                    qf = qh("fit")
                    K1 = cm.spherical_kmeans(qf[0], S); K2 = cm.spherical_kmeans(qf[1], S)
                    def score(n_, K1=K1, K2=K2, qh=qh):
                        a, b = qh(n_); return a @ K1.T, b @ K2.T
                RES.setdefault(f"{PASS}|{inp}|{ad}", []).append(dict(level=l, head=h, **cm.evaluate(score)))
        print(f"level {l} {inp} done {time.time() - t0:.0f}s", flush=True)
    json.dump(RES, open(OUTF, "w"))
cm.table(RES)
print("done %.0fs" % (time.time() - t0))
