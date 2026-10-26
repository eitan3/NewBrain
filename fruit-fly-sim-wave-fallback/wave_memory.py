"""The wave memory behind the MemoryBase interface (memory_api.py, README_memory_api.md; plan_v2/26).

Two modes (--wave-mode, cfg wave_mode):

  stub  WaveMemory: plumbing only (no parameters, no buffers, reads zeros; the brain runs exactly as without memory)
  on    CodedWaveMemory:
          ADDRESS   wave_address.WaveAddress: an online SOM-like entry map + 3 residual levels x 32 slices of soft
                    top-2 codes, from the brain's own thought xf only, learned in-step on detached xf (local
                    competitive rule + homeostasis), no gradient; two speeds (fast during the warm-up, then slow)
          STORAGE   pluggable (wave_store.STORES, --wave-store): the global store, default "columns" =
                    wave_columns.WaveColumns (the WaveSheet v2 column store, SPEC_wavesheet_v2_cloud.md: 1,024 places x
                    3 layers x 16 columns, 4-bit key synapses, chain bits, 2 entries each read with its own residual
                    keys, a decoder A / B, the sweep to f*); "sheet" = wave_sheet.WaveSheet (the pre-v2 wave sheet of
                    plan 26 section 8.4, --wave-pack nibble | c2p3); the private stores (--wave-priv-levels >= 1), one
                    part per routed expert + one for the shared expert (HebbianSheet), off by default
          WRITES    pluggable selection + plan (wave_write.POLICIES, --wave-write; default "wave": count 1, or 2 on
                    surprise, and the second entry after a failed recall), once per TOKEN: each selected token at the
                    one in-window pass where its read-point gradient is largest; content = -dL/d(expert input) there,
                    normalised to the content scale (0.18 median |xf|); parts: global + the token's top-1 routed
                    expert + the shared expert
          CONFIRM   a read that was open and agrees with the brain's own correction (cos(read, -gradient) > --wave-
                    confirm) is a confirmed recall: store.confirm (the sheet: +1 along the recalled path, keyed by the
                    CURRENT codes: reconsolidation), for the written tokens and the best --wave-confirm-w others
          READ      gated, decided before the experts run (as the blob memory): g = sigmoid(gate(features) +
                    open_bias) from DETACHED address / store features, the read used where g > --wave-gate-floor;
                    open_bias is set during the first --wave-gate-warm steps from the gate's own score quantile so
                    about --wave-gate-open0 (25%, like 25d) of the (token, expert) pairs read, then fixed; plus a
                    per-expert gate row gate_e[e] (zero init; row Emax = the shared expert). The read
                    g * gain * store.read(codes) is added to every kept pair's expert input and to the shared
                    expert's input. gate, gains and the store's decoder are nn.Parameters trained only by the main
                    loss through the read; the read has no path to xf (the codes are detached)

Rules this module keeps (owner): separate from the brain's weights (its own state, memory.pt); every learning step is
inside the normal training step (after_backward / after_step), no offline phase; no memory-side loss (the address and
the store learn by local rules under no_grad); addresses come only from xf; no row / hash table (codes over neurons, a
sheet of synapses); the keep mask (--wave-drop, --wave-drop-mode seq: the spec's m *= keep, 10% of the SEQUENCES of
each training forward get m = 0, drawn per forward from the memory's own generator; the capture leaf still sees them;
forward = the earlier whole-forward dropout).

Train-only (--memory-train-only experts=...), two modes:
  private parts on (--wave-priv-levels >= 1): only the selected experts' private parts are written, only their read
    gains gain_p learn. The SHARED parts stay frozen: the address, the global store (no writes, no confirms, no
    forgetting), the gate and its open bias, gate_e, gain_g, the decoder, the write baseline and the content scale.
  global sheet only (the default, the column store; spec PART 4): owned = per selected expert e the --wave-owned (16)
    places with the highest owner_use[:, e] among places with owner_use > 0 (never an empty place; owner_use: integer
    counts of (entry place, routed expert) on 1,024 sampled tokens per training micro-batch, + the shared column).
    Writes and confirms only for tokens routed to a selected expert whose place is owned (masked before the quota and
    the top-k), the sweep over the owned places only; trainable: the gate rows gate_e[e] alone (a fresh AdamW).
    Frozen: the brain, every other gate row, the gate MLP and open bias, gain_g, the decoder (A, B, rho, scale), the
    address, the null statistics, theta_h, owner_use.
  The sheet / hebb global stores without private parts have no train-only mode (a clear error).
"""
import math

import torch
import torch.nn as nn
from torch.profiler import record_function as _rf      # (named regions for prof_wave.py; ~free when not profiling)

import wave_columns  # noqa: F401  (registers STORES["columns"])
import wave_sheet  # noqa: F401  (registers STORES["sheet"])
from memory_api import MemoryBase
from wave_address import Codes, WaveAddress
from wave_columns import WaveColumns
from wave_sheet import WaveSheet
from wave_gate import GatedReadout
from wave_store import STORES
from wave_write import POLICIES

WAVE_DEFAULTS = dict(
    wave_mode="on", wave_seed=1234,
    # the address (wave_address.py)
    wave_side=32, wave_levels=3, wave_slices=32, wave_codes=16, wave_topk=2, wave_entries=4, wave_warm=200,
    wave_anneal=2000, wave_upd_rows=4096, wave_fast=1000, wave_eta_slow=1.0 / 8192,
    # the storage (wave_sheet.py / wave_store.py)
    wave_store="columns", wave_lambda=4.0, wave_pack="nibble", wave_entry_keys="own", wave_owned=16,
    wave_map_beta=0.25, wave_alloc_gamma=0.0, wave_null_bins="load", wave_c_z0=2.0,
    wave_sheet_side=32, wave_sheet_units=64, wave_sheet_topk=1, wave_fill=0.12,
    wave_merge=0.5, wave_content_fast=2000, wave_read_chunk=2048,
    wave_store_dtype="fp16", wave_read_topk=1, wave_priv_levels=1, wave_priv_topk=1, wave_forget=0.0,
    wave_priv_write="read", wave_priv_cap=2, wave_null_every=16,
    # the pre-read gate + per-expert output layer (wave_gate.GatedReadout, v2.1): "pre" = decide before the read,
    # read only the open rows, o = exp(log_s) |x| (u + u D_e); "post" = the earlier post-read gate (_gate)
    wave_gate_mode="pre", wave_k_routed=6, wave_gate_cap_frac=0.5, wave_moments_bf16=0,
    # writes (wave_write.py)
    wave_write="spec", wave_write_w=16, wave_write_per_seq=1, wave_content=0.18, wave_confirm=0.5,
    wave_confirm_w=16, wave_z_conf=2.0, wave_target="grad+old", wave_strong_k=4, wave_null=256, wave_eot=-1,
    # the read (the schedule: zeros before --wave-read-pass, one recall there per forward, re-applied after)
    wave_read_pass=3, wave_cap=0.5, wave_lr=1e-3, wave_recon=0,
    wave_drop=0.1, wave_drop_mode="seq", wave_gate_open0=0.25, wave_gate_warm=200, wave_explore=0.05, wave_gate_floor=0.5,
    wave_max_experts=0)


def make_wave_memory(d, n_exp, **c):
    """LoopBrain's builder: cfg wave_* keys -> the module of that mode."""
    c = dict(WAVE_DEFAULTS, **c)
    if c["wave_mode"] == "stub":
        return WaveMemory(d, n_exp, max_experts=c["wave_max_experts"], wave_mode="stub")
    if c["wave_mode"] != "on":
        raise NotImplementedError(f"wave memory mode {c['wave_mode']!r}: 'stub' or 'on' (wave_memory.py)")
    return CodedWaveMemory(d, n_exp, **c)


class WaveMemory(MemoryBase):
    """The stub: no parameters, no state, reads zeros."""
    kind = "wave"

    def __init__(self, d, n_exp, max_experts=0, wave_mode="stub"):
        super().__init__()
        if wave_mode != "stub":
            raise NotImplementedError(f"WaveMemory is the stub; mode {wave_mode!r} is CodedWaveMemory")
        self.d, self.mode = int(d), wave_mode
        self.Emax = max(int(max_experts or 0), int(n_exp))
        self.n_live = int(n_exp)
        self.calls = 0                        # (host counter: reads since the last begin_forward; tests)

    def begin_forward(self):
        self.calls = 0

    def read(self, xf, e_k, t_k, p_k=None, slot_k=None, side=None, call=None):
        """Zeros: one (n_pairs, D) for the routed pairs, one (N, D) for the shared expert. No gradient."""
        self.calls += 1
        z = xf.detach()
        return z.new_zeros(e_k.size(0), z.size(-1)), z.new_zeros(z.size(0), z.size(-1))

    def on_grow(self, src, E0, split=True):
        self.n_live = E0 + len(src)
        self.Emax = max(self.Emax, self.n_live)

    def describe(self, args=None):
        return (f"wave memory (STUB: reads zeros, no parameters, no state; experts {self.n_live}"
                + (f"; train-only experts {self.train_only}" if self.train_only is not None else "") + ")")

    def log_record(self, step=None):
        return dict(wave_stub=1)

    def memory_cfg(self):
        return dict(wave_mode=self.mode)

    def write(self, *a, **k):
        raise NotImplementedError("WaveMemory.write: the stub (--wave-mode on is CodedWaveMemory)")

    def propagate(self, *a, **k):
        raise NotImplementedError("WaveMemory.propagate: not part of the settled design yet")

    def trainable_parameters(self, expert_ids):
        return []


def _pick(traces, tok, best):
    """The traces of `tok` at each token's chosen call: traces = one dict per call; best (n,) call index."""
    if traces[0] is None:
        return None
    ar = torch.arange(tok.numel(), device=tok.device)
    return {k: torch.stack([t[k][tok] for t in traces])[best, ar] for k in traces[0]}


class CodedWaveMemory(MemoryBase):
    """--wave-mode on (module doc)."""
    kind = "wave"

    def __init__(self, d, n_exp, **c):
        super().__init__()
        c = dict(WAVE_DEFAULTS, **c)
        unknown = set(c) - set(WAVE_DEFAULTS)
        assert not unknown, f"unknown wave-memory options {sorted(unknown)}"
        self._cfg = c
        self.d, self.mode, self.seed = int(d), c["wave_mode"], int(c["wave_seed"])
        self.Emax = max(int(c["wave_max_experts"] or 0), int(n_exp))
        self.n_live = int(n_exp)
        self.shared_part = self.Emax                  # the private part of the shared expert
        self.address = WaveAddress(d, side=c["wave_side"], levels=c["wave_levels"], slices=c["wave_slices"],
                                   codes=c["wave_codes"], topk=c["wave_topk"], entries=c["wave_entries"],
                                   seed=self.seed, warm=c["wave_warm"], anneal=c["wave_anneal"], fast=c["wave_fast"],
                                   eta_slow=c["wave_eta_slow"])
        St = STORES[c["wave_store"]]
        if issubclass(St, WaveColumns):
            assert c["wave_entry_keys"] in ("own", "shared"), f"--wave-entry-keys {c['wave_entry_keys']!r}: own | shared"
            self.store = St(d, 1, self.address, fill=c["wave_fill"], lam=c["wave_lambda"], seed=self.seed,
                            read_chunk=c["wave_read_chunk"])
            self.store.shared_keys = c["wave_entry_keys"] == "shared"
            self.store.n_read = int(c["wave_entries"])
            self.address.entry_keys = not self.store.shared_keys
        elif issubclass(St, WaveSheet):
            self.store = St(d, 1, self.address, side=c["wave_sheet_side"], units=c["wave_sheet_units"],
                            merge=c["wave_merge"], fill=c["wave_fill"], read_topk=c["wave_sheet_topk"],
                            levels=c["wave_levels"], fast=c["wave_content_fast"], seed=self.seed,
                            read_chunk=c["wave_read_chunk"], pack=c["wave_pack"])
        else:
            self.store = St(d, 1, self.address, use_loc=True, read_topk=c["wave_read_topk"],
                            dtype=c["wave_store_dtype"], decay=c["wave_forget"])
        Pt = STORES["hebb"]
        self.priv = (Pt(d, self.Emax + 1, self.address, use_loc=False, levels=c["wave_priv_levels"],
                        read_topk=c["wave_priv_topk"], dtype=c["wave_store_dtype"], decay=c["wave_forget"])
                     if c["wave_priv_levels"] else None)
        self.policy = POLICIES[c["wave_write"]](write_w=c["wave_write_w"], per_seq=c["wave_write_per_seq"],
                                                seed=self.seed)
        if hasattr(self.policy, "strong_k"):
            self.policy.strong_k = int(c["wave_strong_k"])
        self.n_feat = self.address.n_feats + 4
        with torch.random.fork_rng(devices=[]):       # (own stream: the global one is left where it was)
            torch.manual_seed(self.seed + 17)
            self.gate_router = nn.Sequential(nn.Linear(self.n_feat, 16), nn.GELU(), nn.Linear(16, 1))
        # (named "router": BrainOpt puts it on Adam, like the brain's routers)
        self.gain_g = nn.Parameter(torch.ones(()))
        self.gain_p = nn.Parameter(torch.ones(self.Emax + 1)) if self.priv is not None else None
        self.gate_e = nn.Parameter(torch.zeros(self.Emax + 1))   # (the per-expert gate rows; Emax = the shared expert)
        self.global_to = isinstance(self.store, WaveColumns)      # (train-only by owned places of the global sheet)
        if self.global_to:                            # (entry place x expert counts: the train-only owned places)
            self.register_buffer("owner_use", torch.zeros(self.store.P, self.Emax + 1, dtype=torch.int32))
        self.n_owned = int(c["wave_owned"])
        # private writes: "read" = only into the routed experts' parts that were READ at the recall pass (pair gate
        # open, or explored), the first --wave-priv-cap by routing slot, + the shared part if its gate read; "top1" =
        # the token's top-1 routed expert + the shared part (the earlier rule)
        assert c["wave_priv_write"] in ("read", "top1"), f"--wave-priv-write {c['wave_priv_write']!r}: read | top1"
        self.priv_write, self.priv_cap = c["wave_priv_write"], int(c["wave_priv_cap"])
        self.null_every = max(1, int(c["wave_null_every"]))     # (the null control every N optimizer steps)
        assert c["wave_gate_mode"] in ("pre", "post"), f"--wave-gate-mode {c['wave_gate_mode']!r}: pre | post"
        self.k_routed = int(c["wave_k_routed"])
        self.gr = (GatedReadout(d, self.Emax, k_routed=self.k_routed, cap_frac=c["wave_gate_cap_frac"],
                                explore=c["wave_explore"], cap=c["wave_cap"], moments_bf16=bool(c["wave_moments_bf16"]),
                                seed=self.seed + 31) if c["wave_gate_mode"] == "pre" else None)
        self.n_opt = 0
        # place balance arms: A = the map learns on the written thoughts with a write-count conscience (beta);
        # B = a write goes to the entry (of --wave-entries) maximising loc score - gamma sc log(load + 1)
        self.map_beta, self.alloc_gamma = float(c["wave_map_beta"]), float(c["wave_alloc_gamma"])
        if not self.global_to:                        # (the arms need the column store's places)
            self.map_beta = self.alloc_gamma = 0.0
        # the null control's bins: "fill" = the spec's 8 bins of the store's global fill over [0, 2 f*]; "load" = 8
        # bins of log2(1 + memories at the cue's winner place) (the column store: they see the per-place load)
        assert c["wave_null_bins"] in ("fill", "load"), f"--wave-null-bins {c['wave_null_bins']!r}: fill | load"
        self.null_bins = c["wave_null_bins"] if self.global_to else "fill"
        self.c_z0 = float(c["wave_c_z0"])             # (c_tok = clamp((z - c_z0) / 4, 0, 1); the spec's 2)
        self.register_buffer("kappa", torch.zeros(()))   # the content scale: EMA of the median |xf|
        self.register_buffer("open_bias", torch.zeros(()))
        # the null control (spec PART 2): familiarity of mismatched cues per fill bin; the confirmation threshold
        self.register_buffer("null_mu", torch.zeros(8))
        self.register_buffer("null_sd", torch.ones(8))
        self.register_buffer("null_n", torch.zeros(8, dtype=torch.long))
        self.register_buffer("theta_h", torch.tensor(0.05))
        self.register_buffer("gate_warm_left", torch.tensor(int(c["wave_gate_warm"]), dtype=torch.long))
        self.open0, self.gate_warm = float(c["wave_gate_open0"]), int(c["wave_gate_warm"])
        self.explore, self.floor = float(c["wave_explore"]), float(c["wave_gate_floor"])
        self.drop, self.upd_rows, self.content = float(c["wave_drop"]), int(c["wave_upd_rows"]), c["wave_content"]
        assert c["wave_drop_mode"] in ("seq", "forward"), f"--wave-drop-mode {c['wave_drop_mode']!r}: seq | forward"
        self.drop_mode = c["wave_drop_mode"]
        self.confirm_thr, self.confirm_w = float(c["wave_confirm"]), int(c["wave_confirm_w"])
        self.read_pass, self.cap, self.lr = int(c["wave_read_pass"]), float(c["wave_cap"]), float(c["wave_lr"])
        self.recon = bool(c["wave_recon"])
        self.z_conf, self.target_mode, self.n_null = float(c["wave_z_conf"]), c["wave_target"], int(c["wave_null"])
        if hasattr(self.store, "recon"):
            self.store.recon = self.recon
        self.eot = None if c["wave_eot"] < 0 else int(c["wave_eot"])
        self._opt, self._opt_pending = None, None     # (the memory's own AdamW: outside the brain optimizer / clip)
        self._recall, self._probe = None, None
        self._warm_left = int(c["wave_gate_warm"])    # (host mirror of gate_warm_left: no device sync)
        self.train_forward = False
        self.bench_open = False                      # (wave_bench.py: the gate bypassed, g = 1, gains 1)
        # host state (saved in the extra state)
        self.n_fwd = 0
        self.cnt = dict(fwd=0, dropped=0, seqs=0, writes_g=0, writes_p=0, updates=0, confirms=0, surprise=0,
                        recalls=0)
        # per-forward scratch
        self._live, self._dropped, self._recs, self._rows, self._sel = False, False, {}, [], None
        self._keep = None
        self._owned_share = None
        self._gp_last = None
        self._acc, self._gq = None, []
        self._train_parts, self._gain_snap, self._gain_hook, self._row_param = None, None, None, None

    # ------------------------------------------------------------------ state
    def get_extra_state(self):
        opt = self._opt.state_dict() if self._opt is not None else self._opt_pending
        return dict(address=self.address.host_state(), store=self.store.host_state(), n_fwd=self.n_fwd, n_opt=self.n_opt,
                    policy_calls=self.policy.n_calls, cnt=dict(self.cnt), warm_left=self._warm_left, opt=opt)

    def set_extra_state(self, st):
        self.address.load_host_state(st.get("address", {}))
        self.store.load_host_state(st.get("store", {}))
        self.n_fwd = int(st.get("n_fwd", 0))
        self.n_opt = int(st.get("n_opt", 0))
        self.policy.n_calls = int(st.get("policy_calls", 0))
        self.cnt = dict(self.cnt, **st.get("cnt", {}))
        self._warm_left = int(st.get("warm_left", self._warm_left))
        self._opt, self._opt_pending = None, st.get("opt")

    def memory_cfg(self):
        return dict(self._cfg)

    @property
    def ready(self):
        return self.address.ready

    # ------------------------------------------------------------------ the read
    def begin_forward(self):
        self._recs, self._rows, self._sel = {}, [], None
        self._recall, self._probe = None, None                   # (a new forward clears the recall cache)
        self._live = bool(self.training and self.train_forward and torch.is_grad_enabled())
        self._dropped = False
        self._keep = None
        if self._live:
            self.n_fwd += 1
            self.cnt["fwd"] += 1
            if self.drop > 0 and self.drop_mode == "forward":
                g = torch.Generator().manual_seed(self.seed * 7919 + self.n_fwd)
                self._dropped = bool(torch.rand((), generator=g) < self.drop)
                self.cnt["dropped"] += int(self._dropped)

    def _features(self, codes, Rg, mg, z, c):
        k = self.kappa.clamp(min=1e-6) * self.content
        return torch.cat([codes.feats, torch.log1p(mg)[:, None], (Rg.detach().norm(dim=-1) / k)[:, None],
                          (z / 10).clamp(-0.5, 2)[:, None], c[:, None]], -1).detach()

    def _gate(self, feats, n, call, e_k, t_k):
        """-> g (n,) per token (the shared expert's row gate_e[Emax]) and g (n_pairs,) per routed pair (row e)."""
        if self.bench_open:
            one = torch.ones(n, device=feats.device)
            return one, one[t_k]
        s = self.gate_router(feats).squeeze(-1)
        if self._live and self.train_only is None and self._warm_left > 0:
            self._gq.append(torch.quantile(s.detach().float(), 1.0 - self.open0))
        g = torch.sigmoid(s + self.open_bias + self.gate_e[self.shared_part])
        gp = torch.sigmoid(s[t_k] + self.open_bias + self.gate_e[e_k])
        if self._live and self.explore > 0:
            gen = torch.Generator(device=feats.device).manual_seed(self.seed * 1_000_003 + self.n_fwd * 101 + int(call or 0))
            u = torch.rand(n, generator=gen, device=feats.device)
            ex = torch.rand(n, generator=gen, device=feats.device) < self.explore
            g = torch.where(ex, u, g)
            gp = torch.where(ex[t_k], u[t_k], gp)
            floor = lambda x, e: torch.where((x > self.floor) | e, x, torch.zeros_like(x))   # (explored: read)
            return floor(g, ex), floor(gp, ex[t_k])
        floor = lambda x: torch.where(x > self.floor, x, torch.zeros_like(x))
        return floor(g), floor(gp)

    @torch.no_grad()
    def _recall_of(self, xf):
        """The recall (once per forward, at --wave-read-pass): codes, the global store's read + trace, features."""
        codes = self.address.encode(xf)
        Rg, tr = self.store.read_traced(codes)
        mg = tr["mass"] if tr is not None and "mass" in tr else self.store.mass(codes)
        mg = torch.zeros(len(codes), device=Rg.device) if mg is None else mg
        z, c = self._zc(mg, self._winner(tr))
        if self.bench_open:
            c = torch.ones_like(c)
        return dict(codes=codes, Rg=Rg, tr=tr, z=z, c_tok=c, feats=self._features(codes, Rg, mg, z, c),
                    xn=xf.detach().float().norm(dim=-1), xf=xf.detach() if self.map_beta else None)

    def _capped(self, r, xn):
        """The hint cap: |r| <= wave_cap |x| per row (the spec's 0.5 |x|)."""
        if not self.cap:
            return r
        f = (self.cap * xn / r.detach().norm(dim=-1).clamp(min=1e-6)).clamp(max=1.0)
        return r * f[:, None]

    def _apply_recall(self, rec, e_k, t_k, call=None, Rg=None, keep=None):
        """-> (r_pairs, r_shared, g) of one pass from the cached recall (the gate and the private reads per pass);
        keep (n,) or None: the per-sequence keep mask (m *= keep)."""
        codes = rec["codes"]
        if Rg is None:       # (the store's decode WITH grad: its decoder learns from the main loss; the trace is fixed)
            Rg = self.store.decode(rec["tr"]) if hasattr(self.store, "decode") and rec["tr"] is not None else rec["Rg"]
            Rg = Rg * rec["c_tok"][:, None]       # (confidence vs the null control: an empty store reads exactly 0)
        g, gp = self._gate(rec["feats"], len(codes), call, e_k, t_k)
        self._gp_last = gp.detach()                   # (the pair gates of this pass: "write only where read")
        gain_g = 1.0 if self.bench_open else self.gain_g
        r_shared = self._capped((g * gain_g)[:, None] * Rg, rec["xn"])
        r_pairs = self._capped((gp * gain_g)[:, None] * Rg[t_k], rec["xn"][t_k])
        if self.priv is not None:
            gain_p = torch.ones_like(self.gain_p) if self.bench_open else self.gain_p
            if e_k.numel():
                Rp = self.priv.read(codes.take(t_k), part=e_k)
                r_pairs = r_pairs + self._capped((g[t_k] * gain_p[e_k])[:, None] * Rp, rec["xn"][t_k])
            Rs = self.priv.read(codes, part=torch.full((len(codes),), self.shared_part, device=Rg.device))
            r_shared = r_shared + self._capped((g * gain_p[self.shared_part])[:, None] * Rs, rec["xn"])
        if keep is not None:
            r_pairs, r_shared = r_pairs * keep[t_k][:, None], r_shared * keep[:, None]
        return r_pairs, r_shared, g

    def _seq_keep(self, x):
        """The spec's keep mask (--wave-drop-mode seq): per training forward, each sequence kept w.p. 1 - --wave-drop
        (the memory's own generator, CPU); (N,) float per token, or None (eval / off)."""
        if not self._live or self.drop <= 0 or self.drop_mode != "seq":
            return None
        if self._keep is None:
            B = x.size(0) if x.dim() == 3 else 1
            T = x.numel() // (B * x.size(-1))
            g = torch.Generator().manual_seed(self.seed * 7919 + self.n_fwd)
            ks = torch.rand(B, generator=g) >= self.drop
            self.cnt["dropped"] += int(B - int(ks.sum()))
            self.cnt["seqs"] += B
            self._keep = ks.float().repeat_interleave(T).to(x.device)
        return self._keep

    def __call__(self, xf, buf, x, side, e_k, p_k, t_k, slot_k, counts, N, E, C, n_keep, call=None):
        """The schedule: passes < --wave-read-pass get nothing; at the read pass one recall per forward (cached by the
        forward, reused if the pass is re-entered); every later pass re-applies it (fresh gates, private reads with
        that pass's routing). One capture leaf per forward: its gradient is dL/dm summed over the applied passes."""
        with torch.autocast(xf.device.type, enabled=False):
            if self._live:
                self._collect_rows(xf, call)
            k = int(call or 0)
            if self._dropped or not self.ready or k < self.read_pass:
                return buf, x
            if self.gr is not None:
                return self._call_pre(xf, buf, x, e_k, p_k, t_k, slot_k, C, k)
            if self._recall is None:
                self._recall = self._recall_of(xf)
                self.cnt["recalls"] += int(self._live)       # (training recalls: one per forward)
                if self._live and self.train_only is None:
                    with torch.no_grad():
                        med = self._recall["xn"].median()
                        self.kappa.copy_(torch.where(self.kappa > 0, self.kappa.lerp(med, 0.01), med))
            rec = self._recall
            r_p, r_s, g = self._apply_recall(rec, e_k, t_k, call, keep=self._seq_keep(x))
            if self._live and torch.is_grad_enabled():
                if self._probe is None:
                    self._probe = torch.zeros(xf.size(0), xf.size(1), device=xf.device, requires_grad=True)
                    self._recs[0] = dict(codes=rec["codes"], probe=self._probe, e_k=e_k, t_k=t_k, slot_k=slot_k,
                                         g=g.detach(), Rg=rec["Rg"].detach(), trace=rec["tr"], z=rec["z"],
                                         c_tok=rec["c_tok"], xf=rec["xf"], gp=self._gp_last)
                r_p, r_s = r_p + self._probe[t_k], r_s + self._probe
                acc = torch.stack([g.detach().mean(), (g.detach() > 0).float().mean(), torch.ones((), device=g.device)])
                self._acc = acc if self._acc is None else self._acc + acc
        buf = buf.index_put((e_k, p_k), r_p.to(buf.dtype), accumulate=True)
        return buf, x + r_s.to(x.dtype).view(x.shape)

    # ------------------------------------------------------------------ the pre-read gate path (wave_gate.GatedReadout)
    def _call_pre(self, xf, buf, x, e_k, p_k, t_k, slot_k, C, k):
        """--wave-gate-mode pre (inside __call__'s autocast-off block): at the read pass ONE decision + recall per
        forward (cached): gr.decide picks the open (pair, sheet) rows before any lookup; the global sheet is read only
        on the decision's Kg tokens, the private sheets only on its Kp pair rows; u = c_tok kappa y_hat (unitless,
        |u| <= 1; private rows: y_hat), o = gr.readout (exp(log_s) |x| (u + u D_e), the detached cap), times the
        keep mask, injected; later passes re-inject the cached o of the pairs that read (gr.reuse). The capture leaf
        (the writes' dL/dm) is added at every applied pass, as in the post-gate path."""
        first = self._recall is None
        if first:
            self._recall = self._recall_pre(xf, e_k, t_k, slot_k, C, k)
            self.cnt["recalls"] += int(self._live)
            if self._live and self.train_only is None:
                with torch.no_grad():
                    med = self._recall["xn"].median()
                    self.kappa.copy_(torch.where(self.kappa > 0, self.kappa.lerp(med, 0.01), med))
            rec = self._recall
            with _rf("wave.readout"):
                o = self._readout_pre(rec)
            keep = self._seq_keep(x)
            if keep is not None:
                o = o * keep[rec["dec"].t][:, None]
            rec["o"] = o
            buf, x = self.gr.inject(rec["dec"], o, buf, x, e_k, p_k)
        else:
            rec = self._recall
            o_r, o_s = self.gr.reuse(rec["dec"], rec["o"], e_k, t_k)
            buf, x = self.gr.inject_reuse(o_r, o_s, buf, x, e_k, p_k)
        if self._live and torch.is_grad_enabled():
            if self._probe is None:
                self._probe = torch.zeros(xf.size(0), xf.size(1), device=xf.device, requires_grad=True)
                self._recs[0] = dict(codes=rec["codes"], probe=self._probe, e_k=rec["e_k"], t_k=rec["t_k"],
                                     slot_k=rec["slot_k"],      # (the READ pass's routing: gp_read's layout)
                                     g=rec["g_read"], Rg=rec["Rg"], trace=rec["trN"], z=rec["z"], c_tok=rec["c_tok"],
                                     xf=rec["xf"], gp=rec["gp_read"], read_g=rec["read_g_tok"])
            buf = buf.index_put((e_k, p_k), self._probe[t_k].to(buf.dtype), accumulate=True)
            x = x + self._probe.to(x.dtype).view(x.shape)
        return buf, x

    @torch.no_grad()
    def _recall_pre(self, xf, e_k, t_k, slot_k, C, k):
        """The decision and the no-grad part of the recall; per-token views (N rows) of the global recall for the
        writes / confirms / null control: tokens the gate did not open for the global sheet get z = -1e9, c_tok 0."""
        gr, st = self.gr, self.store
        with _rf("wave.address"):
            codes = self.address.encode(xf)
        N, P = len(codes), e_k.numel()
        xn = xf.detach().float().norm(dim=-1)
        f_tok = torch.stack([torch.log1p(xn), codes.feats[:, 0]], -1)
        f_pair = (-slot_k.float() / max(1, self.k_routed))[:, None]
        with _rf("wave.decide"):
            dec = gr.decide(xf, e_k, t_k, C, self.n_live, step=self.n_fwd, call=k, training=self._live, f_tok=f_tok,
                            f_pair=f_pair)
        cg = codes.take(dec.g_idx)
        with _rf("wave.recall_global"):
            tr = st.trace(cg) if hasattr(st, "trace") else st.read_traced(cg)[1]
        chi = tr["mass"] if tr is not None and "mass" in tr else st.mass(cg)
        z_g, c_g = self._zc(chi, self._winner(tr))
        if self.bench_open:
            c_g = torch.ones_like(c_g)
        kap = st.kappa_mean(tr) if hasattr(st, "kappa_mean") else torch.ones_like(c_g)
        dev = xf.device
        gv = dec.g_valid
        tok_g = torch.where(gv, dec.g_idx, torch.full_like(dec.g_idx, N))      # (unread rows -> a dummy row N)
        put = lambda base, v: base.index_put((tok_g,), v)[:N]
        z = put(torch.full((N + 1,), -1e9, device=dev), z_g)
        c_tok = put(torch.zeros(N + 1, device=dev), c_g * gv.float())
        trN = None
        if tr is not None:
            trN = {}
            for key_, v in tr.items():
                if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == len(cg):
                    trN[key_] = v.new_zeros((N + 1,) + tuple(v.shape[1:])).index_put((tok_g,), v)[:N]
        # "write only where read": the routed pairs / shared rows / global tokens the decision read
        rd = dec.read_p | dec.read_g
        rt = dec.p_idx < P
        gp_read = torch.zeros(P + 1, device=dev).index_put((torch.where(rt, dec.p_idx, torch.full_like(dec.p_idx, P)),),
                                                           (dec.read_p & rt).float())[:P]
        g_read = torch.zeros(N + 1, device=dev).index_put((torch.where(rt, torch.full_like(dec.t, N), dec.t),),
                                                          (dec.read_p & ~rt).float())[:N]
        read_tok = torch.zeros(N, dtype=torch.bool, device=dev).index_put((dec.t,), rd, accumulate=True)
        return dict(codes=codes, dec=dec, tr=tr, trN=trN, cg=cg, z_g=z_g, c_g=c_g, kap=kap, z=z, c_tok=c_tok,
                    xn=xn, xf=xf.detach() if self.map_beta else None, gp_read=gp_read, g_read=g_read,
                    read_g_tok=dec.gsel_used, read_tok=read_tok, Rg=torch.zeros(N, xf.size(1), device=dev),
                    e_k=e_k, t_k=t_k, slot_k=slot_k)

    def _readout_pre(self, rec):
        """u (unitless) for the decision's rows, WITH grad to the store's decoder, then gr.readout."""
        dec, gr = rec["dec"], self.gr
        tr = rec["tr"]
        D = self.d
        Rg = self.store.decode(tr) if hasattr(self.store, "decode") and tr is not None else \
            self.store.read(rec["cg"])
        with torch.no_grad():                         # (the global read per token: confirm / null control use it)
            N = len(rec["codes"])
            tok_g = torch.where(dec.g_valid, dec.g_idx, torch.full_like(dec.g_idx, N))
            rec["Rg"] = rec["Rg"].new_zeros(N + 1, D).index_put((tok_g,), Rg.detach() * rec["c_g"][:, None])[:N]
        u_g = (rec["c_g"] * rec["kap"])[:, None] * Rg / Rg.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        if self.priv is not None:
            with _rf("wave.recall_private"):
                Rp = self.priv.read(rec["codes"].take(dec.t), part=dec.e)
            u_p = Rp / Rp.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            u_p = u_g.new_zeros(dec.p_idx.numel(), D)
        return gr.readout(dec, u_p.float(), u_g.float(), rec["xn"])

    def read(self, xf, e_k, t_k, p_k=None, slot_k=None, side=None, call=None):
        """The MemoryBase read (no recording, no schedule; wbench_module.Harness uses it): (r_pairs, r_shared), None
        before ready."""
        if not self.ready:
            return None, None
        with torch.no_grad():
            r_p, r_s, _ = self._apply_recall(self._recall_of(xf), e_k, t_k, call)
        return r_p, r_s

    @torch.no_grad()
    def _collect_rows(self, xf, call):
        """A deterministic strided subsample of this pass's thoughts for the address update."""
        N = xf.size(0)
        per = max(1, self.upd_rows // 4)
        stride = max(1, math.ceil(N / per))
        self._rows.append(xf.detach()[(int(call or 0) % stride)::stride].float())

    # ------------------------------------------------------------------ writes (once per token) and learning
    def select_writes(self, logits, targets, idx=None, ce_h=None):
        if not self._live or self._dropped or not self.ready or not self._recs:
            self._sel = None
            return None
        B, T = targets.shape
        with torch.no_grad():
            if ce_h is not None:
                ce, H, V = ce_h[0].reshape(-1).float(), ce_h[1].reshape(-1).float(), int(ce_h[2])
            else:
                lp = torch.log_softmax(logits.detach().reshape(B * T, -1).float(), -1)
                ce = -lp.gather(1, targets.reshape(-1, 1)).squeeze(1)
                H = -(lp.exp() * lp).sum(-1)
                V = lp.size(1)
            allow = self._allow(self._recs[0]) if self._global_train_only else None
            if self.gr is not None and self._recall is not None and "read_tok" in self._recall:
                allow = self._recall["read_tok"] if allow is None else allow & self._recall["read_tok"]
            if allow is None:
                self._sel = self.policy.select(ce, H, V, B, T, idx=idx, eot=self.eot)
            else:
                self._sel = self.policy.select(ce, H, V, B, T, idx=idx, eot=self.eot, allow=allow)
        return self._sel

    @property
    def _global_train_only(self):
        return self.train_only is not None and self.priv is None and self.global_to

    def _routed_sel(self, rec):
        """(N,) bool: the token is routed to a selected train-only expert (the shared expert: every token)."""
        N = rec["probe"].size(0)
        parts = torch.tensor(self._train_parts, device=rec["e_k"].device)
        if self.shared_part in self._train_parts:
            return torch.ones(N, dtype=torch.bool, device=parts.device)
        hit = torch.isin(rec["e_k"], parts).float()
        return torch.zeros(N, device=hit.device).index_add_(0, rec["t_k"], hit) > 0

    def _allow(self, rec):
        """Train-only (global sheet): the write candidates = routed to a selected expert AND entry place 0 owned."""
        return self._routed_sel(rec) & self.store.owned[rec["codes"].loc[:, 0]]

    def _parts_ok(self, parts):
        if self._train_parts is None:
            return torch.ones_like(parts, dtype=torch.bool)
        return torch.isin(parts, torch.tensor(self._train_parts, device=parts.device))

    @torch.no_grad()
    def after_backward(self):
        if self.gr is not None and self.train_only is None:
            with torch.enable_grad():
                self.gr.after_backward()
        recs, sel, rows = self._recs, self._sel, self._rows
        self._recs, self._sel, self._rows = {}, None, []
        calls = [r for r in recs.values() if r["probe"].grad is not None]
        written = None
        if sel is not None and calls:
            with _rf("wave.write"):
                written = self._write(sel, calls)
        if calls and (self.train_only is None or self._global_train_only) and self.confirm_w > 0:
            self._confirm_sweep(calls, written)
        if calls and self.train_only is None and self.n_null and self.n_opt % self.null_every == 0:
            with _rf("wave.null"):
                self._null_update(calls[0])
        if calls and self.train_only is None and self.global_to:
            self._owner_count(calls[0])
        if rows and self.train_only is None:
            X = torch.cat(rows)
            if X.size(0) > self.upd_rows:
                X = X[::math.ceil(X.size(0) / self.upd_rows)]
            self.address.update(X)
            self.cnt["updates"] += 1
        return None

    @torch.no_grad()
    def _owner_count(self, rec, n=1024):
        """owner_use[E0, e] += 1 for the routed pairs (and the shared column) of n strided sample tokens (integer
        index_put_ accumulate: deterministic)."""
        E0 = rec["codes"].loc[:, 0]
        N = E0.numel()
        stride = max(1, N // n)
        w = ((rec["t_k"] % stride == 0) & (rec["t_k"] // stride < n)).int()
        self.owner_use.index_put_((E0[rec["t_k"]], rec["e_k"]), w, accumulate=True)
        t = torch.arange(N, device=E0.device)
        ws = ((t % stride == 0) & (t // stride < n)).int()
        self.owner_use.index_put_((E0, torch.full_like(E0, self.shared_part)), ws, accumulate=True)

    def _targets(self, recs, tok):
        """Per token: E = dL/dm (the capture leaf's gradient, summed over the applied passes) and the write target
        -e_hat at the content scale (--wave-target grad+old adds the old recall in _write). best: the record index."""
        G = torch.stack([r["probe"].grad[tok] for r in recs])            # (records, n, D): one record per forward
        gn = G.norm(dim=-1)
        best = gn.argmax(0)
        ar = torch.arange(tok.numel(), device=tok.device)
        gsel, gnorm = G[best, ar], gn[best, ar]
        target = -gsel / gnorm.clamp(min=1e-12)[:, None] * (self.kappa * self.content)
        return target, gnorm, best

    def _recall_rows(self, recs, tok, best, target):
        """The recall of rows `tok` and its confirmation (spec W3): h = cos(read, -e); confirmed = z >= --wave-z-conf
        and h >= theta_h (theta_h: the null pairs' 99.9th percentile, EMA)."""
        per = [r["codes"].take(tok) for r in recs]
        ar = torch.arange(tok.numel(), device=tok.device)
        codes = Codes.pick(per, best)
        pick = lambda key: torch.stack([r[key][tok] for r in recs])[best, ar]
        g_read, Rg, z, c_tok = pick("g"), pick("Rg"), pick("z"), pick("c_tok")
        h = torch.where(Rg.norm(dim=-1) > 1e-6, torch.nn.functional.cosine_similarity(Rg, target, dim=-1),
                        torch.zeros_like(z))
        ok = ((z >= self.z_conf) & (h >= self.theta_h)).float()
        tr = _pick([r["trace"] for r in recs], tok, best)
        return codes, tr, dict(open=g_read, cos=h, ok=ok, z=z, c_tok=c_tok, Rg=Rg, E=self.address.E)

    def _write(self, sel, recs):
        tok, strength = sel
        n = tok.numel()
        if n == 0:
            return tok
        target, gnorm, best = self._targets(recs, tok)
        strength = strength * (gnorm > 0).float()
        codes, tr, recall = self._recall_rows(recs, tok, best, target)
        if self.target_mode == "grad+old":     # (spec: normalize(-e_hat + max(h, 0) [c_tok >= .5] m_hat))
            Rh = recall["Rg"] / recall["Rg"].norm(dim=-1, keepdim=True).clamp(min=1e-12)
            w = recall["cos"].clamp(min=0) * (recall["c_tok"] >= 0.5).float()
            mix = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-12) + w[:, None] * Rh
            target = mix / mix.norm(dim=-1, keepdim=True).clamp(min=1e-12) * (self.kappa * self.content)
        strength, entries = self.policy.plan(tok, strength, recall)
        entries = self._alloc(codes, entries)
        if self.map_beta and self.train_only is None and recs[0]["xf"] is not None:
            self.address.learn_writes(recs[0]["xf"][tok[strength > 0]], self.map_beta)
        ok = recall["ok"]
        if self._global_train_only:          # (W3 / W4 candidates: routed to a selected expert; owned: the store masks)
            sel_ = self._routed_sel(recs[0])[tok].float()
            ok, strength = ok * sel_, strength * sel_ * self.store.owned[codes.loc[:, 0]].float()
        self.cnt["surprise"] += int((ok == 0).sum())
        if tr is not None:
            tr["ok"] = ok                                                  # (familiar rows reuse their recalled path)
        if self.train_only is None or self._global_train_only:
            s_g = strength
            if recs[0].get("read_g") is not None:                         # (pre gate: only where the global read)
                s_g = strength * recs[0]["read_g"][tok].float()
            self.store.confirm(codes, ok, trace=tr)                       # (confirm first, then the writes: spec)
            self.store.write(codes, target, s_g, entries=entries, trace=tr)
            self.cnt["writes_g"] += int((s_g > 0).sum())
            self.cnt["confirms"] += int(ok.sum())
        if self.priv is None:
            return tok
        if self.priv_write == "read":
            rows, parts = self._priv_read_targets(recs[0], tok, strength)
            keep = self._parts_ok(parts)
            rows, parts = rows[keep], parts[keep]
            if rows.numel():
                self.priv.write(codes.take(rows), target[rows], strength[rows], part=parts,
                                entries=None if entries is None else entries[rows])
                self.cnt["writes_p"] += int(rows.numel())
            return tok
        # the token's top-1 routed expert at the written pass (-1: dropped by capacity) and the shared expert
        top = torch.full((n,), -1, dtype=torch.long, device=tok.device)
        for i, r in enumerate(recs):
            tok2e = torch.full((r["probe"].size(0),), -1, dtype=torch.long, device=tok.device)
            m0 = r["slot_k"] == 0
            tok2e[r["t_k"][m0]] = r["e_k"][m0]
            top = torch.where(best == i, tok2e[tok], top)
        ar = torch.arange(n, device=tok.device)
        parts = torch.cat([top, torch.full((n,), self.shared_part, dtype=torch.long, device=tok.device)])
        rows = torch.cat([ar, ar])
        keep = (parts >= 0) & self._parts_ok(parts)
        rows, parts = rows[keep], parts[keep]
        if rows.numel():
            c2 = codes.take(rows)
            self.priv.write(c2, target[rows], strength[rows], part=parts,
                            entries=None if entries is None else entries[rows])
            self.cnt["writes_p"] += int(rows.numel())
        return tok

    @torch.no_grad()
    def _alloc(self, codes, entries):
        """Fix B (place balance, --wave-alloc-gamma > 0, the column store): each write goes to the ONE entry (of the
        --wave-entries read) maximising loc_score - gamma sc log(load + 1) (load: the store's memories per place);
        the read scans the same entries and keeps the clearest (chibar)."""
        if not self.alloc_gamma or not self.global_to or entries is None:
            return entries
        E = self.store._n_ent(codes)
        ld = self.store.load[codes.loc[:, :E]].float()
        sc = codes.loc_s[:, :E] - self.alloc_gamma * self.address.sc * torch.log(ld + 1)
        j = sc.argmax(1)                                               # (first index on ties)
        one = torch.nn.functional.one_hot(j, entries.size(1)).bool()
        return one & entries.any(1, keepdim=True)

    @torch.no_grad()
    def _priv_read_targets(self, rec, tok, strength):
        """(rows into tok, parts) of the private writes, --wave-priv-write read: per written token (strength > 0) its
        routed pairs that were read at the recall pass (pair gate > 0: open or explored), the first --wave-priv-cap by
        routing slot, + the shared part if the token's gate read."""
        e_k, t_k, slot_k, gp = rec["e_k"], rec["t_k"], rec["slot_k"], rec["gp"]
        N = rec["probe"].size(0)
        pos = torch.full((N,), -1, dtype=torch.long, device=tok.device)
        pos[tok] = torch.arange(tok.numel(), device=tok.device)
        on = (gp > 0) & (pos[t_k] >= 0) & (strength[pos[t_k].clamp(min=0)] > 0)
        keep = torch.zeros_like(on)
        used = torch.zeros(N, device=tok.device)
        for s_ in range(int(slot_k.max()) + 1 if slot_k.numel() else 0):        # (k slots: a short host loop)
            m_ = on & (slot_k == s_) & (used[t_k] < self.priv_cap)
            keep = keep | m_
            used.index_add_(0, t_k, m_.float())
        rows, parts = pos[t_k[keep]], e_k[keep]
        g_tok = rec["g"]
        sh = (g_tok[tok] > 0) & (strength > 0)
        ar = torch.arange(tok.numel(), device=tok.device)[sh]
        return (torch.cat([rows, ar]),
                torch.cat([parts, torch.full_like(ar, self.shared_part)]))

    def _confirm_sweep(self, recs, written):
        """Confirmed recalls among the tokens not written this micro-batch: at most --wave-confirm-w (16), the best by
        h x c_tok (spec W3)."""
        N = recs[0]["probe"].size(0)
        tok = torch.arange(N, device=recs[0]["probe"].device)
        target, gnorm, best = self._targets(recs, tok)
        _, tr_all, rc = self._recall_rows(recs, tok, best, target)
        ok = (rc["ok"] > 0) & (gnorm > 0)
        if self._global_train_only and tr_all is not None:   # (masked before the top-k: routed + owned winner place)
            ok = ok & self._routed_sel(recs[0]) & self.store.owned[
                tr_all["loc"].gather(1, tr_all["best"][:, None]).squeeze(1)]
        if written is not None and written.numel():
            ok[written] = False
        sc = torch.where(ok, rc["cos"] * rc["c_tok"], torch.full_like(rc["cos"], -2.0))
        top = sc.topk(min(self.confirm_w, N))
        pick = top.indices[top.values > -2.0]
        if pick.numel() == 0:
            return
        codes, tr, _ = self._recall_rows(recs, pick, best[pick], target[pick])
        self.store.confirm(codes, torch.ones(pick.numel(), device=pick.device), trace=tr)
        self.cnt["confirms"] += int(pick.numel())

    @torch.no_grad()
    def _null_update(self, rec):
        """The null control (spec PART 2): --wave-null tokens' cues with their own entry location but another token's
        keys (t + N/2); EMA mean / sd of their familiarity per fill bin -> z of real reads; theta_h from the null
        pairs' h' = cos(read_t, -e_(t + N/2)) on tokens with c_tok > 0 (99.9th percentile, floor .05)."""
        codes = rec["codes"]
        N = len(codes)
        t = torch.arange(0, N, max(1, N // self.n_null), device=codes.loc.device)[:self.n_null]
        o = (t + N // 2) % N
        null = Codes(codes.loc[t], codes.loc_s[t], codes.code[o], codes.code_w[o], codes.code_d[o], codes.feats[t],
                     None if codes.code_e is None else codes.code_e[o])
        if not self._null_stats(null):
            return
        g = rec["probe"].grad
        if g is not None:
            Rg, ct = rec["Rg"][t], rec["c_tok"][t]
            hp = torch.nn.functional.cosine_similarity(Rg, -g[o], dim=-1)
            hp = torch.where((ct > 0) & (Rg.norm(dim=-1) > 1e-6), hp, torch.full_like(hp, -2.0))
            srt = hp.sort(descending=True).values
            cnt = (hp > -2.0).sum()
            q = srt.gather(0, (cnt.float() * 0.001).floor().long().clamp(max=srt.numel() - 1).view(1)).squeeze(0)
            new = 0.99 * self.theta_h + 0.01 * torch.clamp(q, min=0.05)
            self.theta_h.copy_(torch.where(cnt > 0, new, self.theta_h))

    @torch.no_grad()
    def _null_stats(self, null):
        """EMA (rate .01) mean / sd of the null cues' familiarity per bin (one-hot sums: deterministic)."""
        chi = self.store.null_observe(null) if hasattr(self.store, "null_observe") else self.store.mass(null)
        if chi is None:
            return False
        b = self._bins(getattr(self.store, "last_null_place", None), chi.numel(), chi.device)
        oh = torch.nn.functional.one_hot(b, 8).float()                    # (n, 8)
        cnt = oh.sum(0)
        m_ = (oh.T @ chi) / cnt.clamp(min=1)
        v_ = ((oh.T @ (chi * chi)) / cnt.clamp(min=1) - m_ * m_).clamp(min=0)
        seen, first = cnt > 0, self.null_n == 0
        self.null_mu.copy_(torch.where(seen, torch.where(first, m_, self.null_mu.lerp(m_, 0.01)), self.null_mu))
        self.null_sd.copy_(torch.where(seen, torch.where(first, v_.sqrt(), (self.null_sd ** 2).lerp(v_, 0.01).sqrt()),
                                       self.null_sd))
        self.null_n.add_(seen.long())
        return True

    @torch.no_grad()
    def bench_null(self, xf, n=256):
        """(bench) one null-control observation on these thoughts, as _null_update does on a micro-batch: n cues with
        their own entry places and the keys of the token N/2 away."""
        codes = self.address.encode(xf)
        N = len(codes)
        t = torch.arange(0, N, max(1, N // n), device=codes.loc.device)[:n]
        o = (t + N // 2) % N
        self._null_stats(Codes(codes.loc[t], codes.loc_s[t], codes.code[o], codes.code_w[o], codes.code_d[o],
                               codes.feats[t], None if codes.code_e is None else codes.code_e[o]))

    def _fill_bin(self):
        """The fill bin (8 over [0, 2 f*]) of the store (a 0-dim tensor; no host sync)."""
        f = getattr(self.store, "fill_est", None)
        if f is None:
            return torch.zeros((), dtype=torch.long, device=self.null_mu.device)
        ft = float(getattr(self.store, "fill_target", 0.12)) or 0.12
        return (f / (2 * ft) * 8).long().clamp(0, 7)

    @staticmethod
    def _winner(tr):
        """The winner entry's place per token from a store trace, or None."""
        if tr is None or "loc" not in tr or "best" not in tr:
            return None
        return tr["loc"].gather(1, tr["best"][:, None]).squeeze(1)

    def _bins(self, places, n, device):
        """(n,) null-control bins: the global fill bin, or (--wave-null-bins load) log2(1 + the place's load)."""
        if self.null_bins == "load" and places is not None:
            ld = self.store.load[places].float()
            return torch.log2(ld + 1).floor().long().clamp(0, 7)
        return self._fill_bin().expand(n)

    def _zc(self, chi, places=None):
        """z and c_tok of real reads from their familiarity chi (spec: z = (chi - mu_b) / max(sd_b, .005),
        c_tok = clamp((z - 2) / 4, 0, 1); no null statistics yet -> c_tok 0: the read is 0)."""
        b = self._bins(places, chi.numel(), chi.device)
        z = (chi - self.null_mu[b]) / self.null_sd[b].clamp(min=0.005)
        c = ((z - self.c_z0) / 4).clamp(0, 1) * (self.null_n[b] > 0).float()
        return z, c

    def _step_params(self):
        """The memory's own AdamW (betas .9 / .99, wd 0, clip 1.0 on its own parameters): once per optimizer step,
        over the parameters that are trainable now (train-only: gain_p, or gate_e, alone); outside the brain's optimizer and clip
        (train_lm excludes bm.* there for --memory wave). Then the clamps."""
        own = {id(p) for p in self.gr.parameters()} if self.gr is not None else set()
        if self.gr is not None:                         # (the gate + output layer: its own groups / clip / AdamW)
            if self.train_only is None:
                self.gr.step()
            else:
                for p in self.gr.parameters():
                    p.grad = None
        params = [p for p in (self.parameters() if self.train_only is None else
                              self.trainable_parameters(self.train_only)) if p.requires_grad and id(p) not in own]
        for p in self.parameters():                     # (train-only: the frozen ones' gradients are dropped)
            if p.grad is not None and id(p) not in own and not any(p is q for q in params):
                p.grad = None
        if not params or all(p.grad is None for p in params):
            return
        if self._opt is None:
            sp = {id(p) for p in self.store.parameters()}         # (the store's decoder / gains: lr x 0.1, spec)
            groups = [dict(params=[p for p in params if id(p) not in sp], lr=self.lr),
                      dict(params=[p for p in params if id(p) in sp], lr=0.1 * self.lr)]
            self._opt = torch.optim.AdamW([g_ for g_ in groups if g_["params"]], betas=(0.9, 0.99), weight_decay=0.0)
            if self._opt_pending is not None:
                try:
                    self._opt.load_state_dict(self._opt_pending)
                except (ValueError, KeyError):
                    pass                                # (another parameter set, e.g. train-only: a fresh AdamW)
                self._opt_pending = None
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        self._opt.step()
        if hasattr(self.store, "after_param_step") and self.train_only is None:   # (train-only: the decoder frozen)
            self.store.after_param_step()
        self._opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.gain_g.clamp_(0.05, 4.0)
            self.gate_e.clamp_(-4.0, 4.0)
            if self.gain_p is not None:
                self.gain_p.clamp_(0.05, 4.0)
            if getattr(self.store, "alpha", None) is not None:
                self.store.alpha.clamp_(0.0, 4.0)

    def after_step(self, step, refit_every=0):
        self.n_opt += 1
        self._step_params()
        if self.train_only is None:
            self.store.forget(step)
            if self.priv is not None:
                self.priv.forget(step)
            if self._gq:                               # the gate's open bias during its warm-up (then fixed)
                with torch.no_grad():
                    q = torch.stack(self._gq).mean()
                    first = self._warm_left == self.gate_warm
                    self.open_bias.copy_(-q if first else self.open_bias.lerp(-q, 0.2))
                    self.gate_warm_left.sub_(1)
                self._warm_left -= 1
                self._gq = []
        else:
            self._gq = []
            if self.priv is not None:
                self.priv.forget(step, parts=self._train_parts)
            elif self._global_train_only:
                self.store.forget(step)                # (the store sweeps the owned places only)
            if self._gain_snap is not None:            # (the unselected rows stay exactly as they were)
                with torch.no_grad():
                    keep = torch.ones_like(self._gain_snap, dtype=torch.bool)
                    keep[self._train_parts] = False
                    row = getattr(self, self._row_param)
                    row.copy_(torch.where(keep, self._gain_snap, row))

    def on_grow(self, src, E0, split=True):
        src = [int(s) for s in (src.tolist() if torch.is_tensor(src) else src)]
        dst = list(range(E0, E0 + len(src)))
        assert not dst or dst[-1] < self.Emax, f"wave memory: {dst[-1] + 1} experts > --wave-max-experts {self.Emax}"
        self.n_live = E0 + len(src)
        if self.priv is not None and dst:
            self.priv.copy_parts(src, dst)
            with torch.no_grad():
                self.gain_p[dst] = self.gain_p[src]
        if dst and self.gr is not None:
            self.gr.on_grow(src, dst)
        if dst:                                        # (children = clones of the parent's gate row; owner counts
            with torch.no_grad():                      # split: each of the n children gets clone // n)
                self.gate_e[dst] = self.gate_e[src]
                if self.global_to:
                    n = {s_: src.count(s_) + (1 if split else 0) for s_ in set(src)}
                    div = torch.tensor([n[s_] for s_ in src], dtype=torch.int32, device=self.owner_use.device)
                    clone = self.owner_use[:, src] // div
                    self.owner_use[:, dst] = clone
                    if split:
                        for s_ in set(src):
                            self.owner_use[:, s_] //= n[s_]

    # ------------------------------------------------------------------ train-only
    def set_train_only(self, expert_ids):
        if expert_ids is not None and self.priv is None and not self.global_to:
            raise NotImplementedError(
                f"--memory-train-only with the global {type(self.store).__name__} store and no private parts: no "
                f"train-one-expert mode there. Use the column store (--wave-store columns, the default: owned places + "
                f"the gate row) or private parts (--wave-priv-levels 1: the experts' own stores + gains)")
        super().set_train_only(expert_ids)
        if self._gain_hook is not None:
            self._gain_hook.remove()
            self._gain_hook = None
        self._gain_snap = self._row_param = None
        if hasattr(self.store, "owned"):
            self.store.owned = None
        self._owned_share = None
        if expert_ids is None:
            self._train_parts = None
            self.address.frozen = self.policy.frozen = False
            return
        self._train_parts = sorted({self.shared_part if e < 0 else int(e) for e in self.train_only})
        self.address.frozen = self.policy.frozen = True
        self._opt, self._opt_pending = None, None     # (a fresh AdamW over the train-only parameters)
        self._row_param = "gain_p" if self.priv is not None else "gate_e"      # (by name: not a second registration)
        row = getattr(self, self._row_param)
        self._gain_snap = row.detach().clone()
        mask = torch.zeros_like(self._gain_snap)
        mask[self._train_parts] = 1
        self._gain_hook = row.register_hook(lambda g, _m=mask: g * _m)
        if self.priv is None:                          # (the global sheet: the owned places, spec PART 4)
            self.store.owned = self.owned_places(self._train_parts)

    @torch.no_grad()
    def owned_places(self, parts):
        """(P,) bool: per column e of `parts` the --wave-owned places with the highest owner_use[:, e] among those with
        owner_use[:, e] > 0 (first index on ties); the share of owner_use[:, e] they hold is kept for the log."""
        own = torch.zeros(self.owner_use.size(0), dtype=torch.bool, device=self.owner_use.device)
        share = []
        for e in parts:
            u = self.owner_use[:, e].long()
            k = min(self.n_owned, int((u > 0).sum()))
            if k:
                own[torch.sort(u, descending=True, stable=True).indices[:k]] = True
            share.append(float(u[own].sum()) / max(1, int(u.sum())))
        self._owned_share = share
        return own

    def trainable_parameters(self, expert_ids):
        if self.priv is not None:
            return [self.gain_p]
        return [self.gate_e] if self.global_to else []

    # ------------------------------------------------------------------ logging
    def describe(self, args=None):
        a, st = self.address, self.store
        s = (f"wave memory (on): address SOM {a.side}x{a.side} + {a.levels} levels x {a.S} slices x {a.M} codes, "
             f"top-{a.r}, {a.E} entries, warm {a.warm} / fast {a.fast} updates; store {type(st).__name__}"
             + (f" {st.side}x{st.side} hc x ({st.K} keys + {st.M} units), fill {st.fill_target}"
                if isinstance(st, WaveSheet) else "")
             + (f" {st.P} places x 3 layers x 16 columns, fill {st.fill_target}, chain lambda {st.lam_ch}, entry keys "
                f"{'shared' if st.shared_keys else 'own'}" if isinstance(st, WaveColumns) else "")
             + (" (PLACEHOLDER)" if st.placeholder else "")
             + (f" + private {self.Emax}+1 parts" if self.priv is not None else "")
             + f"; writes {type(self.policy).__name__} {self.policy.write_w}/micro-batch (<= {self.policy.per_seq}/seq);"
             f" gate open {self.open0} over {self.gate_warm} steps; drop {self.drop}; {self.nbytes() / 1e6:.1f} MB")
        if self.train_only is not None and self.priv is not None:
            s += f"; train-only experts {self.train_only} (address, global store, gate frozen)"
        elif self.train_only is not None:
            s += (f"; train-only experts {self.train_only}: {int(self.store.owned.sum())} owned places (share of the "
                  f"experts' owner counts {[round(x, 3) for x in self._owned_share]}), gate rows {self._train_parts}")
        return s

    def log_record(self, step=None):
        rec = {f"wmem_{k}": v for k, v in self.address.stats().items()}
        rec.update({f"wmem_{k}": v for k, v in self.cnt.items()})
        rec.update({f"wmem_{k}": v for k, v in self.policy.log().items()})
        rec.update({f"wmem_store_{k}": v for k, v in self.store.fill().items()})
        if self.priv is not None:
            rec.update({f"wmem_priv_{k}": v for k, v in self.priv.fill().items()})
        if self._acc is not None:
            a = self._acc.tolist()
            rec.update(wmem_gate_mean=a[0] / a[2], wmem_gate_open=a[1] / a[2])
            self._acc = None
        rec["wmem_gain_g"] = float(self.gain_g.detach())
        if self._owned_share is not None:
            rec["wmem_owned_share"] = self._owned_share
        if self.gr is not None:
            rec.update(self.gr.log_record())
        rec["wmem_kappa"] = float(self.kappa)
        rec["wmem_open_bias"] = float(self.open_bias)
        return rec

    def health(self):
        return {f"wmem_{k}": v for k, v in self.address.stats().items()}

    # ------------------------------------------------------------------ the W-bench adapter (wave_bench.py)
    @torch.no_grad()
    def bench_write(self, xf, target, e_k=None, t_k=None, call=None, learn=False):
        """One write per row through the real path: read (recall) first, then the policy's plan (count 1 / 2 on
        surprise, second entry), confirm, write. Content = target; private: the row's first routed expert in e_k
        (the capture's order) + the shared part. learn: the address also learns on these rows first."""
        if learn:
            self.address.update(xf)
        if self.map_beta:                          # (place balance A: the map learns on the write load, online)
            self.address.learn_writes(xf, self.map_beta)
        codes = self.address.encode(xf)
        n = xf.size(0)
        target = target.float()
        R, tr = self.store.read_traced(codes)
        cos = torch.nn.functional.cosine_similarity(R.detach(), target, dim=-1)
        ok = ((cos > self.confirm_thr) & (R.norm(dim=-1) > 0)).float()
        s, ent = self.policy.plan(torch.arange(n), torch.ones(n), dict(open=torch.ones(n), cos=cos, ok=ok,
                                                                        E=self.address.E))
        ent = self._alloc(codes, ent)
        if tr is not None:
            tr["ok"] = ok
        self.store.confirm(codes, ok, trace=tr)
        self.store.write(codes, target, s, entries=ent, trace=tr)
        j = ent.float().argmax(1) if ent is not None else torch.zeros(n, dtype=torch.long)
        self.last_write_place = torch.where(s > 0, codes.loc.gather(1, j[:, None]).squeeze(1),
                                            torch.full((n,), -1, dtype=torch.long))      # (bench diagnostics)
        if self.priv is not None:
            parts = [torch.full((n,), self.shared_part, dtype=torch.long)]
            rows = [torch.arange(n)]
            if e_k is not None and e_k.numel():
                first = torch.ones_like(t_k, dtype=torch.bool)
                first[1:] = t_k[1:] != t_k[:-1]
                rows.append(t_k[first])
                parts.append(e_k[first].clamp(max=self.Emax - 1))
            r, p = torch.cat(rows), torch.cat(parts)
            self.priv.write(codes.take(r), target[r], s[r], part=p)

    @torch.no_grad()
    def bench_confirm(self, xf, target):
        """Rehearsal (wave_bench / tests_wave_drift): read these thoughts, and confirm the recalls that match their
        content (reconsolidation under the current codes); no new write."""
        codes = self.address.encode(xf)
        R, tr = self.store.read_traced(codes)
        cos = torch.nn.functional.cosine_similarity(R, target.float(), dim=-1)
        ok = ((cos > self.confirm_thr) & (R.norm(dim=-1) > 0)).float()
        self.store.confirm(codes, ok, trace=tr)
        return float(ok.mean())
