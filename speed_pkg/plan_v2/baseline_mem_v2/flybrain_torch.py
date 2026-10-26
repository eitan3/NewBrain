"""PyTorch re-implementation of the Shiu et al. (2024) whole-brain fly LIF model.

Same equations and constants as model.py in philshiu/Drosophila_brain_model,
but runs many trials at once (batched) and on GPU if one is available.

Usage:
    python flybrain_torch.py --repo path/to/Drosophila_brain_model --version 783 \
        --trials 30 --seconds 1.0 --device cuda

The simulation is event-driven: each step only the outgoing synapses of neurons
that actually spiked are touched, so cost scales with activity, not with the
15M synapses.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# constants copied from model.py (default_params); units: mV and ms
P = dict(
    v_0=-52.0, v_rst=-52.0, v_th=-45.0,
    t_mbr=20.0,   # membrane time constant
    tau=5.0,      # synaptic time constant
    t_rfc=2.2,    # refractory period
    t_dly=1.8,    # synaptic delay
    w_syn=0.275,  # mV per synapse
    f_poi=250,    # Poisson input weight = w_syn * f_poi
    dt=0.1,       # Brian2 default time step
)

SUGAR_R = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265


def load_connectome(repo, version):
    repo = Path(repo)
    if version == "630":
        comp = repo / "2023_03_23_completeness_630_final.csv"
        con = repo / "2023_03_23_connectivity_630_final.parquet"
    else:
        comp, con = repo / "Completeness_783.csv", repo / "Connectivity_783.parquet"
    ids = pd.read_csv(comp, index_col=0).index.values
    df = pd.read_parquet(con, columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    return ids, df["Presynaptic_Index"].values, df["Postsynaptic_Index"].values, df["Excitatory x Connectivity"].values


class FlyBrain:
    """Batched LIF network. State tensors have shape (N, B): neurons x trials."""

    def __init__(self, n, pre, post, syn_count, device="cpu", params=P):
        self.p, self.n, self.dev = params, n, torch.device(device)
        # CSR sorted by presynaptic neuron, so a spike looks up its outgoing row
        order = np.argsort(pre, kind="stable")
        pre, post = pre[order], post[order]
        w = syn_count[order].astype(np.float32) * params["w_syn"]
        rowptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(rowptr, pre + 1, 1)
        self.rowptr = torch.from_numpy(np.cumsum(rowptr)).to(self.dev)
        self.col = torch.from_numpy(post.astype(np.int64)).to(self.dev)
        self.w = torch.from_numpy(w).to(self.dev)
        self.silenced = torch.zeros(n, dtype=torch.bool, device=self.dev)

        dt = params["dt"]
        self.a_v = math.exp(-dt / params["t_mbr"])
        self.a_g = math.exp(-dt / params["tau"])
        # exact solution of dv/dt = (v0 - v + g)/T, dg/dt = -g/tau over one step
        self.k_vg = params["tau"] / (params["tau"] - params["t_mbr"]) * (self.a_g - self.a_v)
        self.delay = int(round(params["t_dly"] / dt))
        # Brian2 integrates again on the step where t - lastspike >= t_rfc, i.e. 21 frozen
        # steps for 2.2 ms; our counter is decremented once before the first frozen step
        self.rfc_steps = int(round(params["t_rfc"] / dt)) - 1

    def silence(self, idx):
        self.silenced[idx] = True

    @torch.no_grad()
    def run(self, exc_idx, rate_hz=150.0, seconds=1.0, trials=30, seed=0, input_hits=None, record=None):
        """input_hits: optional bool tensor (steps, len(exc_idx), trials) replacing the random Poisson draws."""
        p, n, B, dev = self.p, self.n, trials, self.dev
        g_gen = torch.Generator(device=dev).manual_seed(seed)
        steps = int(round(seconds * 1000 / p["dt"]))
        v = torch.full((n, B), p["v_0"], device=dev)
        g = torch.zeros((n, B), device=dev)
        rfc_left = torch.zeros((n, B), dtype=torch.int32, device=dev)
        exc = torch.as_tensor(exc_idx, dtype=torch.long, device=dev)
        poi_p = rate_hz * p["dt"] / 1000.0
        poi_w = p["w_syn"] * p["f_poi"]
        # ring buffer of pending spikes: list of (pre_idx, trial_idx) per future step
        ring = [None] * (self.delay + 1)
        counts = torch.zeros((n, B), dtype=torch.int32, device=dev)
        spike_log = []  # (step, neuron, trial) kept on device, chunked
        self.trace = []  # v of `record` neurons (trial 0) after each step, for debugging

        for t in range(steps):
            # 1) integrate (frozen while refractory)
            active = rfc_left <= 0
            v_new = p["v_0"] + (v - p["v_0"]) * self.a_v + g * self.k_vg
            v = torch.where(active, v_new, v)
            g = torch.where(active, g * self.a_g, g)
            rfc_left -= 1
            # 2) threshold (reset happens after synaptic input, as in Brian2's schedule)
            spk = v > p["v_th"]
            any_spk = bool(spk.any())
            if any_spk:
                counts += spk.int()
                nz = spk.nonzero()
                spike_log.append(torch.cat([torch.full((len(nz), 1), t, device=dev), nz], 1))
                nz = nz[~self.silenced[nz[:, 0]]]
                ring[t % len(ring)] = nz
            else:
                ring[t % len(ring)] = None
            # 3) deliver spikes emitted `delay` steps ago
            due = ring[(t - self.delay) % len(ring)] if t >= self.delay else None
            if due is not None and len(due):
                pre, tri = due[:, 0], due[:, 1]
                start, cnt = self.rowptr[pre], self.rowptr[pre + 1] - self.rowptr[pre]
                tot = int(cnt.sum())
                if tot:
                    rep_start = torch.repeat_interleave(start, cnt)
                    offs = torch.arange(tot, device=dev) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
                    syn = rep_start + offs
                    tgt = self.col[syn] * B + torch.repeat_interleave(tri, cnt)
                    g.view(-1).index_add_(0, tgt, self.w[syn])
            # 4) Poisson drive onto stimulated neurons (adds straight to v, like PoissonInput)
            hits = input_hits[t].to(dev) if input_hits is not None else \
                torch.rand((len(exc), B), generator=g_gen, device=dev) < poi_p
            v[exc] += hits.float() * poi_w
            # Brian2 (linear method + "unless refractory") drops synaptic input that
            # arrives while a neuron is refractory: g stays 0 until it recovers
            g = torch.where(active, g, torch.zeros_like(g))
            # 5) reset neurons that spiked this step
            if any_spk:
                v = torch.where(spk, torch.full_like(v, p["v_rst"]), v)
                g = torch.where(spk, torch.zeros_like(g), g)
                rfc_left = torch.where(spk, torch.full_like(rfc_left, self.rfc_steps), rfc_left)
                rfc_left[exc] = 0  # Poisson targets have no refractory period
            if record is not None:
                self.trace.append(torch.stack([v[record, 0], g[record, 0]]).cpu())

        spikes = torch.cat(spike_log).cpu().numpy() if spike_log else np.zeros((0, 3), int)
        rates = counts.float().mean(1).cpu().numpy() / seconds
        return rates, spikes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Drosophila_brain_model")
    ap.add_argument("--version", default="783", choices=["630", "783"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="flybrain_rates.csv")
    a = ap.parse_args()

    t0 = time.time()
    ids, pre, post, w = load_connectome(a.repo, a.version)
    brain = FlyBrain(len(ids), pre, post, w, device=a.device)
    id2i = {f: i for i, f in enumerate(ids)}
    print(f"loaded {len(ids):,} neurons, {len(w):,} connections on {a.device} in {time.time()-t0:.1f}s")

    missing = [f for f in SUGAR_R if f not in id2i]
    if missing:
        print(f"warning: {len(missing)} sugar neuron id(s) not in v{a.version}, skipped: {missing}")
    t0 = time.time()
    rates, _ = brain.run([id2i[f] for f in SUGAR_R if f in id2i], a.rate, a.seconds, a.trials)
    print(f"simulated {a.trials} trials x {a.seconds}s in {time.time()-t0:.1f}s")
    print(f"active neurons: {(rates > 0).sum()}, MN9 rate: {rates[id2i[MN9]]:.1f} Hz")
    pd.DataFrame({"flywire_id": ids, "rate_hz": rates}).query("rate_hz > 0") \
        .sort_values("rate_hz", ascending=False).to_csv(a.out, index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
