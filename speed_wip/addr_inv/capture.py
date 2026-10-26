"""Capture the memory addressing inputs on a fixed val batch (CPU, eval, no writes).
usage: capture.py CODE_DIR CKPT|fresh:REF_CKPT OUT.pt [n_seq]
Saves per fresh select call: k, a (L,N,448) fp16, rows/I1/I2 (real code); per read: k, gates (N,L,H) fp16;
plus the memory state needed to re-address offline, and the tokens."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
code, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
n_seq = int(sys.argv[4]) if len(sys.argv) > 4 else 32
sys.path.insert(0, code)
import numpy as np
import torch
torch.set_num_threads(4)
from memory import Memory
from loop_brain import LoopBrain
from spiking_lm import RoutedMLP

t0 = time.time()
fresh = src.startswith("fresh:")
ck = torch.load(src[6:] if fresh else src, map_location="cpu", weights_only=False, mmap=True)
cfg = dict(ck["cfg"])
step = 0 if fresh else ck.get("step")
torch.manual_seed(0)
model = LoopBrain(**cfg)
if not fresh:
    r = model.load_state_dict(ck["state"], strict=False)
    print("missing", len(r.missing_keys), r.missing_keys[:5], "unexpected", len(r.unexpected_keys), r.unexpected_keys[:5])
run_cfg = {}
cp = os.path.join(os.path.dirname(src[6:] if fresh else src), "config.json")
if os.path.exists(cp):
    run_cfg = json.load(open(cp))
for attr in ("mem_readdr_every", "mem_readdr_start", "mem_margin_passes"):
    if attr in run_cfg and hasattr(model, attr):
        setattr(model, attr, run_cfg[attr])
for mod in model.modules():
    if isinstance(mod, RoutedMLP):
        mod.noise, mod.eval_noise = 0.0, 0.0
del ck
mem = model.mem
mem.write_on = False
model.eval()
model.K = 6
print("step", step, "read0", mem.read0, "L,H,S,m,km", mem.L, mem.H, mem.S, mem.m, mem.km, "readdr", getattr(model, "mem_readdr_every", None),
      getattr(model, "mem_readdr_start", None), "experts", model.blocks[0].mlp.E, "frozen", mem.is_frozen, "load %.0fs" % (time.time() - t0), flush=True)

val = np.memmap(r"X:\Datasets\wikipedia_en_20231101_wiki16k\val.bin", dtype=np.uint16, mode="r")
rng = np.random.RandomState(1)
T = 512
st = np.sort(rng.randint(0, len(val) - T - 1, n_seq))
X = torch.from_numpy(np.stack([val[i:i + T].astype(np.int64) for i in st]))

REC = {"sel": [], "g": []}
CUR = {"k": None}
orig_hier, orig_select, orig_read = LoopBrain._hier_step, Memory.select, Memory.read_shared


def hier(self, s, read, address, k=0, **kw):
    CUR["k"], CUR["reuse"] = k, kw.get("sel") is not None
    return orig_hier(self, s, read, address, k=k, **kw)


def select(self, a, a_hat, collect):
    o = orig_select(self, a, a_hat, collect)
    REC["sel"].append(dict(k=CUR["k"], a=a.half(), rows=o["rows"].int(), I1=o["I1"].short(), I2=o["I2"].short()))
    return o


def read_shared(self, w, sel, pbar):
    M, g = orig_read(self, w, sel, pbar)
    REC["g"].append(dict(k=CUR["k"], reuse=CUR["reuse"], g=g.detach().half(), rows=sel["rows"].int()))
    return M, g


LoopBrain._hier_step, Memory.select, Memory.read_shared = hier, select, read_shared
B = 4
chunks = []
with torch.no_grad():
    for b in range(0, n_seq, B):
        REC["sel"].clear(); REC["g"].clear()
        model.run(X[b:b + B], K=6)
        chunks.append(dict(sel=list(REC["sel"]), g=list(REC["g"])))
        print("batch", b, "%.0fs" % (time.time() - t0), flush=True)


def cat(lst, key, dim):
    return torch.cat([c[key] for c in lst], dim)


ncall = len(chunks[0]["sel"])
sels = [dict(k=chunks[0]["sel"][i]["k"], a=cat([c["sel"][i] for c in chunks], "a", 1),
             rows=cat([c["sel"][i] for c in chunks], "rows", 2), I1=cat([c["sel"][i] for c in chunks], "I1", 2),
             I2=cat([c["sel"][i] for c in chunks], "I2", 2)) for i in range(ncall)]
nr = len(chunks[0]["g"])
gs = [dict(k=chunks[0]["g"][i]["k"], reuse=chunks[0]["g"][i]["reuse"], g=cat([c["g"][i] for c in chunks], "g", 0),
           rows=cat([c["g"][i] for c in chunks], "rows", 2)) for i in range(nr)]
state = {n: getattr(mem, n).detach().clone() for n in ("wq_ema", "k1_ema", "k2_ema", "norm_m", "norm_v", "U", "wcount")}
state["k1"], state["k2"], state["wq"] = mem.keys.k1.detach().clone(), mem.keys.k2.detach().clone(), mem.keys.wq.detach().clone()
state["gate_b"] = mem.gates.b.detach().clone()
torch.save(dict(step=step, X=X, starts=st, sels=sels, reads=gs, state=state, tau=mem.tau, m=mem.m, km=mem.km, S=mem.S,
                read0=mem.read0), out)
print("calls", [s["k"] for s in sels], "reads", [(g["k"], g["reuse"]) for g in gs], "done %.0fs" % (time.time() - t0))
