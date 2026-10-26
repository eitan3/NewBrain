"""Replay a torch.cuda.memory snapshot's trace: the allocated-bytes timeline, its peak, and the live blocks at the
peak grouped by the first model-code frame (file:line function) - the ranked consumers.

    python analyze_snapshot.py snapshot_K10.pickle [base_alloc_GB]
"""
import collections
import json
import pickle
import sys
from pathlib import Path

path = Path(sys.argv[1])
base = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 0.0
snap = pickle.load(open(path, "rb"))
tr = [t for t in snap["device_traces"] if t]
ev = tr[0]
MODEL = ("loop_brain.py", "memory.py", "spiking_lm.py", "brain_opt.py", "vram_common.py", "variants.py")


def key(frames, depth=2):
    fr = [f for f in frames if any(f["filename"].endswith(m) for m in MODEL)]
    if not fr:
        fr = frames[:1]
    out = []
    for f in fr[:depth]:
        out.append(f"{Path(f['filename']).name}:{f['line']} {f['name']}")
    return " <- ".join(out) if out else "?"


live = {}
cur = base
peak, peak_i = cur, -1
timeline = []
for i, e in enumerate(ev):
    a = e["action"]
    if a == "alloc":
        live[e["addr"]] = (e["size"], i)
        cur += e["size"]
    elif a == "free_requested":
        cur -= live.pop(e["addr"])[0] if e["addr"] in live else e["size"]   # (untracked: allocated before recording)
    if cur > peak:
        peak, peak_i = cur, i
    if i % 200 == 0:
        timeline.append(cur)
# second replay: live set at the peak
live2 = {}
for i, e in enumerate(ev[:peak_i + 1]):
    if e["action"] == "alloc":
        live2[e["addr"]] = e
    elif e["action"] == "free_requested":
        live2.pop(e["addr"], None)
groups = collections.defaultdict(lambda: [0, 0])
groups1 = collections.defaultdict(lambda: [0, 0])
for e in live2.values():
    k = key(e["frames"])
    groups[k][0] += e["size"]
    groups[k][1] += 1
    k1 = key(e["frames"], 1)
    groups1[k1][0] += e["size"]
    groups1[k1][1] += 1
tot = sum(g[0] for g in groups.values())
print(f"events {len(ev)}, base {base / 1e9:.2f} GB, peak {peak / 1e9:.2f} GB at event {peak_i} "
      f"({ev[peak_i]['action']} {ev[peak_i]['size'] / 1e6:.1f} MB at {key(ev[peak_i]['frames'], 3)}); live traced at peak {tot / 1e9:.2f} GB")
print("\n-- top by first model frame")
rows1 = sorted(groups1.items(), key=lambda kv: -kv[1][0])
for k, (b, n) in rows1[:40]:
    print(f"{b / 1e9:7.3f} GB  {n:6d}  {k}")
print("\n-- top by two model frames")
rows = sorted(groups.items(), key=lambda kv: -kv[1][0])
for k, (b, n) in rows[:40]:
    print(f"{b / 1e9:7.3f} GB  {n:6d}  {k}")
# segments: reserved vs allocated (fragmentation at snapshot time)
segs = snap["segments"]
res = sum(s["total_size"] for s in segs)
alloc = sum(b["size"] for s in segs for b in s["blocks"] if b["state"] == "active_allocated")
print(f"\nsegments at snapshot end: reserved {res / 1e9:.2f} GB, allocated {alloc / 1e9:.2f} GB, n_segments {len(segs)}; "
      f"large(>1MB pool) {sum(s['total_size'] for s in segs if s['segment_type'] == 'large') / 1e9:.2f} GB")
out = dict(peak_gb=peak / 1e9, base_gb=base / 1e9, peak_event=peak_i, n_events=len(ev),
           top1=[(k, b / 1e9, n) for k, (b, n) in rows1[:60]], top2=[(k, b / 1e9, n) for k, (b, n) in rows[:60]],
           timeline_gb=[t / 1e9 for t in timeline])
path.with_suffix(".analysis.json").write_text(json.dumps(out, indent=1))
