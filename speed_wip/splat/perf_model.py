"""Analytic per-step cost of the splat read / write at the brain's scale (plan_v2/23 section 7).
B = 16, T = 512 -> N = 8,192 tokens per micro-batch, accum 2, reads per forward from the config, x1.33 for the
activation-checkpoint recompute of the forward. Op counts come from bench_cpu.json (TorchDispatchMode on the real
plain-PyTorch implementation in bench_gpu.py; size-independent). GPU: RTX 5090 Laptop, 896 GB/s (70% achievable
for gathers), ~35 us per op on Windows (plan 13: ~40 us/op observed incl. GPU work, ~25 us host)."""
import json
import os

from bench_gpu import CONFIGS

HERE = os.path.dirname(os.path.abspath(__file__))
ops = json.load(open(os.path.join(HERE, "bench_cpu.json")))
N, MB, RECOMP, BW, US_OP, NSPL = 8192, 2, 1.33, 896e9 * 0.7, 35e-6, 1 << 20
rows = []
for name, c in CONFIGS.items():
    V, d, k, dv, S, T = c["V"], c["d_m"], c["k"], c["dv"], c["S"], c["T"]
    nc = T * c["kc"] * c["cap"]
    Q = N * V
    cw = 2 * d + 1                                                     # packed coefficients per slot (fp32)
    hops = c["hops"]
    # forward bytes per read: cell scores (2T, Q, C) written + read by topk; pointer gather; candidate coefficient
    # gather (table read + temp write + bmm read); winners (x3); value gather
    b_cell = hops * 2 * T * Q * c["C"] * 4 * 2
    b_ptr = hops * Q * nc * 8 * 3
    b_cand = hops * Q * nc * cw * 4 * 3 * (2 if c["ray"] else 1)       # (ray: the second segment-end lookup)
    b_win = hops * Q * k * cw * 4 * 3
    b_val = Q * k * dv * 4
    b_fwd = b_cell + b_ptr + b_cand + b_win + b_val + N * 448 * 4 + Q * d * 4
    # backward: value index_add_ (read-modify-write) + embedding for d omega + winners' coef again + view GEMM grads
    b_bwd = Q * k * dv * 4 * 3 + b_win + N * 448 * 4 * 2
    fl = (N * 448 * V * d * 2 * 3 + hops * Q * T * d * c["C"] * 2 + hops * Q * nc * cw * 2 * S
          + hops * Q * k * cw * 2 * S * 3 + Q * k * dv * 2 * 3)
    t_read = (b_fwd * RECOMP + b_bwd) / BW                             # seconds per read (fwd + recompute + bwd)
    n_reads = c["reads"] * MB
    op_read = ops[name]["ops_read_fwd_bwd"]
    ops_step = op_read * n_reads * RECOMP + ops[name]["ops_write"]
    t_work = t_read * n_reads
    t_launch = ops_step * US_OP
    # transient VRAM: the largest temp of one read (cell scores or candidate coefficients); saved for backward
    vram_tmp = max(2 * T * Q * c["C"] * 4, Q * nc * cw * 4) / 1e9
    vram_saved = (Q * k * (cw + 2) * 4 + Q * k * 8 + Q * d * 4) / 1e9 * c["reads"]
    table = NSPL * (cw + 2) * 4 / 1e9 + 2 * NSPL * dv * 4 / 1e9 + T * c["C"] ** 2 * c["cap"] * 8 / 1e9
    rows.append(dict(cfg=name, Q=Q, cand=nc, GB_per_read=(b_fwd * RECOMP + b_bwd) / 1e9, GFLOP_per_read=fl / 1e9,
                     ops_read=op_read, ops_step=ops_step, ms_work_step=1e3 * t_work, ms_launch_step=1e3 * t_launch,
                     ms_step_est=1e3 * (max(t_work, t_launch) + 0.3 * min(t_work, t_launch)),
                     vram_tmp_GB=vram_tmp, vram_saved_GB=vram_saved, table_GB=table))
print("| config | queries / read | candidates | GB moved / read (fwd + recomp + bwd) | GFLOP / read | ops / read | "
      "ops / step | work ms / step | launch ms / step | est. ms / step | largest temp GB | saved GB | tables GB |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['cfg']} | {r['Q']:,} | {r['cand']} | {r['GB_per_read']:.2f} | {r['GFLOP_per_read']:.1f} | "
          f"{r['ops_read']} | {r['ops_step']:.0f} | {r['ms_work_step']:.0f} | {r['ms_launch_step']:.0f} | "
          f"{r['ms_step_est']:.0f} | {r['vram_tmp_GB']:.2f} | {r['vram_saved_GB']:.2f} | {r['table_GB']:.2f} |")
json.dump(rows, open(os.path.join(HERE, "perf_model.json"), "w"), indent=1)
