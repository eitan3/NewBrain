"""eval_world.py on the CPU (2 threads, no CUDA) with one extra knockout for the blob memory (plan_v2/25d, LoopBrain.bm):
ko_blob = blob memory reads off (bm.read_mode = 'none', as train_lm's gm_val_ko_delta does). The existing knockouts of
the concept memory (model.mem: ko_shared / ko_private / ko_level*) are kept if the model has one.
Repo files are not modified; the patch lives only in this wrapper."""
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, r"C:\Eitan\Python\NewBrain\fruit-fly-sim")
import torch
torch.set_num_threads(2)
import eval_world as ew

_orig_ko = ew.knockouts


def knockouts(model, gate_modes=True):
    out = _orig_ko(model, gate_modes)
    if getattr(model, "bm", None) is not None:
        if out[0][0] == "model":
            out[0] = ("mem_on", None, None)
        out.insert(1, ("ko_blob", "bm.ko", True))   # (the gated 25d read ignores read_mode; bm.ko closes every pair)
        if os.environ.get("BLOBKO_ONLY") == "1":
            out = [o for o in out if o[0] == "ko_blob"]
    return out


class KO(ew.KO):
    def __init__(self, model, attr, value):
        if attr and attr.startswith("bm."):
            self.mem, self.attr, self.value = model.bm, attr[len("bm."):], value
            self.bm_rm = attr == "bm.ko"   # (also read_mode 'none', for the ungated read)
        else:
            super().__init__(model, attr, value)


    def __enter__(self):
        super().__enter__()
        if getattr(self, "bm_rm", False):
            self.old_rm, self.mem.read_mode = self.mem.read_mode, "none"

    def __exit__(self, *exc):
        if getattr(self, "bm_rm", False):
            self.mem.read_mode = self.old_rm
        super().__exit__(*exc)


ew.knockouts, ew.KO = knockouts, KO
_orig_load = ew.load_model


def load_model(a):
    out = _orig_load(a)
    m = out[0]
    bm = getattr(m, "bm", None)
    print(f"[wrapper] blob memory: {'none' if bm is None else f'read_mode {bm.read_mode}, fitted {bm._fitted}'}; "
          f"concept memory: {'none' if getattr(m, 'mem', None) is None else 'present'}; "
          f"threads {torch.get_num_threads()}, cuda available {torch.cuda.is_available()}", flush=True)
    return out


ew.load_model = load_model
ew.main()
