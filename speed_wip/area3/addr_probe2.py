"""Area 3: extra eval-only variants on the step-1000 checkpoint (re-uses addr_probe's hooks)."""
import sys, json
sys.argv = ["x"]
src = open(r"C:\Eitan\Python\NewBrain\speed_wip\area3\addr_probe.py").read()
src = src.split("res = {}")[0]          # model, data, hooks, loss()
exec(src)
out = {}
l0, _ = loss()
out["base"] = l0
ws = REC["w"]
out["top4_mass_by_read"] = [w.topk(4, -1).values.sum(-1).mean((1, 2)).tolist() for w in ws]
out["top3_mass_by_read"] = [w.topk(3, -1).values.sum(-1).mean((1, 2)).tolist() for w in ws]
n = len(REC["sel"])
for name, f in (("read0_only", lambda i: i == 0), ("read0_and_last", lambda i: i in (0, n - 1)),
                ("read0_last2", lambda i: i == 0 or i >= n - 2)):
    out[name] = loss(read_mask=f)[0]
# fewer rows: keep only the top-4 of the 6 weights (renormalised), all reads
orig_read2 = Memory.read_shared


def read_top4(self, w, sel, pbar):
    v, ix = w.topk(4, -1)
    w2 = torch.zeros_like(w).scatter(-1, ix, v)
    w2 = w2 / w2.sum(-1, keepdim=True)
    return orig_read2(self, w2, sel, pbar)


Memory.read_shared = read_top4
out["topk4_renorm"] = loss()[0]
Memory.read_shared = orig_read2
print(json.dumps(out, indent=1))
json.dump(out, open(r"C:\Eitan\Python\NewBrain\speed_wip\area3\addr_probe2.json", "w"), indent=1)
