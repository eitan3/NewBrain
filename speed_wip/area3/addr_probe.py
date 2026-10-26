"""Area 3: on the live run's step-1000 checkpoint (CPU, eval), measure
 - address overlap of the shared reads between consecutive passes / vs the first read (per level)
 - thresholded address reuse (cache rows across passes when the addressing input moved little)
 - weight concentration of w, head redundancy
 - eval-loss effect of: heads subset, reads on fewer passes, shared/private knock-outs, address caching.
Weak evidence (step 1000 of 12207, infancy, no retraining) - indicative only."""
import sys, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\speed_wip\area3")
import memory
from memory import Memory
from loop_brain import LoopBrain

torch.manual_seed(0)
torch.set_num_threads(16)
ck = torch.load(r"C:\Eitan\Python\NewBrain\speed_wip\area3\step1000.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"])
model = LoopBrain(**cfg)
missing = model.load_state_dict(ck["state"], strict=False)
print("load:", len(missing.missing_keys), "missing", len(missing.unexpected_keys), "unexpected", flush=True)
model.eval()
mem = model.mem
print("experts", model.blocks[0].mlp.E, "frozen", mem.is_frozen, flush=True)

val = np.memmap(r"X:\Datasets\wikipedia_en_20231101_wiki16k\val.bin", dtype=np.uint16, mode="r")
rng = np.random.RandomState(1)
B, T = 8, 512
st = rng.randint(0, len(val) - T - 1, B)
X = torch.from_numpy(np.stack([val[i:i + T].astype(np.int64) for i in st]))
Y = torch.from_numpy(np.stack([val[i + 1:i + T + 1].astype(np.int64) for i in st]))
K = 6

# ------------------------------------------------------------ hooks
REC = {"sel": [], "w": [], "a": [], "g": []}
orig_select, orig_online, orig_read = Memory.select, Memory.online, Memory.read_shared
CTRL = dict(cache=None, theta=None, reuse=[], read_mask=None, read_i=0, head_keep=None, priv_mask=None)


def select(self, a, a_hat, collect):
    out = orig_select(self, a, a_hat, collect)
    th = CTRL["theta"]
    if th is not None:
        c = CTRL["cache"]
        if c is not None:
            rel = (a - c["a"]).norm(dim=-1) / c["a"].norm(dim=-1).clamp(min=1e-12)      # (L, N)
            keep = rel < th
            CTRL["reuse"].append(keep.float().mean().item())
            m4 = keep[:, None, :, None]
            for k in ("i", "j", "I1", "I2", "rows"):
                out[k] = torch.where(m4, c[k], out[k])
            # cached entries keep their reference input (no creeping drift); refreshed ones update it
            a_ref = torch.where(keep[..., None], c["a"], a)
        else:
            a_ref = a
        CTRL["cache"] = dict(a=a_ref, **{k: out[k] for k in ("i", "j", "I1", "I2", "rows")})
    REC["sel"].append(out["rows"].clone())
    REC["a"].append(a.clone())
    return out


def online(self, a_hat, sel):
    out = orig_online(self, a_hat, sel)
    if "w" in out:
        REC["w"].append(out["w"].detach().clone())
    return out


def read_shared(self, w, sel, pbar):
    i = CTRL["read_i"]
    CTRL["read_i"] += 1
    hk = CTRL["head_keep"]
    if hk is not None:
        w = w.clone()
    M, g = orig_read(self, w, sel, pbar)
    if hk is not None:  # recompute M with only the first hk heads' gates (others zero)
        L, H, km, dv = self.L, self.H, self.km, self.dv
        N = w.size(2)
        rows = sel["rows"].permute(2, 0, 1, 3).reshape(N * L * H, km)
        ww = w.permute(2, 0, 1, 3).reshape(N * L * H, km)
        r = F.embedding_bag(rows, self.V, per_sample_weights=ww, mode="sum").view(N, L, H, dv)
        g2 = g.clone()
        g2[..., hk:] = 0
        u = (g2[..., None] * r).sum(2)
        M = torch.einsum("nlc,ldc->nd", u, self.w_o)
    REC["g"].append(g.detach().mean((0,)).clone())
    rm = CTRL["read_mask"]
    if rm is not None and not rm(i):
        M = torch.zeros_like(M)
    return M, g


Memory.select, Memory.online, Memory.read_shared = select, online, read_shared
orig_private = Memory.private


def private(self, q1, qb1, *a):
    pm = CTRL["priv_mask"]
    i = CTRL.setdefault("priv_i", 0)
    CTRL["priv_i"] = i + 1
    y = orig_private(self, q1, qb1, *a)
    if pm is not None and not pm(i):
        y = torch.zeros_like(y)
    return y


Memory.private = private


def reset(**kw):
    for k in REC:
        REC[k] = []
    CTRL.update(cache=None, theta=None, reuse=[], read_mask=None, read_i=0, head_keep=None, priv_mask=None, priv_i=0)
    CTRL.update(kw)
    mem.ko_shared = mem.ko_private = False


@torch.no_grad()
def loss(**kw):
    reset(**kw)
    model.K = K
    t0 = time.time()
    out = model(X, Y)
    l = out[1].item() if isinstance(out, tuple) else float("nan")
    return l, time.time() - t0


res = {}
l0, dt = loss()
res["base_loss"] = l0
print(f"base loss {l0:.4f} ({dt:.1f}s), reads {len(REC['sel'])}", flush=True)

# ---- address statistics from the base run
rows = REC["sel"]           # list of (L, H, N, km)
ws = REC["w"]
A = REC["a"]
L = mem.L


def overlap(r1, r2, w1):
    inter = (r1[..., :, None] == r2[..., None, :]).any(-1)                # (L, H, N, km)
    return inter.float().mean(-1).mean((1, 2)), (w1 * inter).sum(-1).mean((1, 2))


cons = []
for k in range(1, len(rows)):
    s, w = overlap(rows[k], rows[k - 1], ws[k])
    drift = ((A[k] - A[k - 1]).norm(dim=-1) / A[k - 1].norm(dim=-1)).mean(-1)
    cons.append(dict(read=k, set=s.tolist(), w=w.tolist(), input_drift=drift.tolist()))
res["consecutive_overlap"] = cons
vs_last = []
for k in range(len(rows) - 1):
    s, w = overlap(rows[k], rows[-1], ws[-1])
    vs_last.append(dict(read=k, set=s.tolist(), w=w.tolist()))
res["overlap_vs_last_read"] = vs_last
# weight concentration
wc = []
for w in ws:
    ent = -(w.clamp(min=1e-12) * w.clamp(min=1e-12).log()).sum(-1)
    wc.append(dict(max_w=w.max(-1).values.mean((1, 2)).tolist(), eff_rows=ent.exp().mean((1, 2)).tolist(),
                   top2_mass=w.topk(2, -1).values.sum(-1).mean((1, 2)).tolist()))
res["w_concentration"] = wc
# head redundancy: overlap of the row sets of head h vs head h' at the same (level, token)
hr = []
r = rows[-1]
for h1 in range(mem.H):
    for h2 in range(h1 + 1, mem.H):
        inter = (r[:, h1, :, :, None] == r[:, h2, :, None, :]).any(-1).float().mean(-1).mean(-1)
        hr.append(dict(h=(h1, h2), set_overlap_by_level=inter.tolist()))
res["head_overlap"] = hr
res["gates_by_read"] = [g.tolist() for g in REC["g"]]
print(json.dumps({k: res[k] for k in ("consecutive_overlap", "w_concentration")})[:3000], flush=True)

# ---- variants (eval loss, no retraining)
var = {}
var["shared_ko"] = loss(read_mask=lambda i: False)[0]
var["private_ko"] = loss(priv_mask=lambda i: False)[0]
var["both_ko"] = loss(read_mask=lambda i: False, priv_mask=lambda i: False)[0]
for hk in (1, 2):
    var[f"heads_first_{hk}"] = loss(head_keep=hk)[0]
n_reads = len(rows)
for name, f in (("reads_only_last", lambda i: i == n_reads - 1),
                ("reads_last3", lambda i: i >= n_reads - 3),
                ("reads_every_2nd", lambda i: (n_reads - 1 - i) % 2 == 0),
                ("reads_skip_read0", lambda i: i != 0)):
    var[name] = loss(read_mask=f)[0]
for name, f in (("private_last3", lambda i: i >= K - 3), ("private_every_2nd", lambda i: (K - 1 - i) % 2 == 0)):
    var[name] = loss(priv_mask=f)[0]
cache = {}
for th in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1e9):
    l, _ = loss(theta=th)
    cache[str(th)] = dict(loss=l, reuse_frac_by_read=list(CTRL["reuse"]),
                          mean_reuse=float(np.mean(CTRL["reuse"])) if CTRL["reuse"] else 0.0)
var["address_cache"] = cache
res["variants"] = var
print(json.dumps(var, indent=1), flush=True)
json.dump(res, open(r"C:\Eitan\Python\NewBrain\speed_wip\area3\addr_probe.json", "w"), indent=1)
