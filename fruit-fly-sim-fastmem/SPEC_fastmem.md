# FastMem spec (coordinator, from its MemT-1.0 review; 2026-10-25 21:12 UTC)
Write our own code from these equations. Do NOT copy MemT code (PolyForm NC license).

## What it is
- A fast-weight synapse matrix per expert. No slots, rows, KV list or places.
- Write: delta-rule (GDN-2 "stable") update.
- Read: q · M.
- MemT resets it per conversation. Ours persists across all of training (long-term).

## Shape
- Per brain expert e (all 193, including the shared one): M_e [H=4, d_k=512, d_v=512] fp32. That is 4 MB each, about 0.81 GB total.
- Global: M_g [32, 512, 512] (32 MB).
- Each M is its own tensor, saved in the memory file, and loadable / resettable per expert.

## Maths (from the expert input thought xf, 384-d)
- q = L2norm(xf Wq_{e,h}), k = L2norm(xf Wk_{e,h}), v = xf Wv_{e,h}
- beta = sigmoid(xf Wb), omega = sigmoid(xf Ww), per channel and dim
- lam = -exp(s_h) · softplus(xf Wl + bl), initialised so the decay is close to 1 (long-term, e.g. exp(lam) ≈ 1 − 1e-4)
- WRITE, per routed token in order:
  - U = sqrt(beta) · k
  - M <- (I − Uᵀ U) diag(exp(lam)) M + Uᵀ (omega · v)
- READ:
  - R = q M / sqrt(d_k)
  - out_h = RMSNorm(R_h) · 2·tanh(SiLU(xf Wg_h)/2)
  - concat over the 4 heads -> 2048
- Into the expert input: through the existing per-expert D_e [384, 2048] (init 0) and the existing pre-read gate.
- Global sheet: the same path, with its own gate.
- Memory off == no memory, bit-identical.

## In-step
- Each sequence scans its routed tokens causally, starting from the persistent M_t (chunked, chunk 64; WY or the simple chunked form).
- Token t reads M_t plus the updates from the tokens before t.
- After the step: M_{t+1} = M_t + mean_b(M_end_b − M_t), detached.
- Passes: read at every pass from the same M (no per-pass memory); write once, from the final pass.

## Gradient (default)
- Write side: xf.detach() for the inputs of k, v, beta, omega and lam.
- The main loss trains Wk, Wv, the gates and D_e through M within the step.
- The read side may pass gradient into xf (as now).
- No separate memory loss.

## Fallback
If per-expert scans over routed tokens are too slow or too complex: 16 memory experts with dense softmax routing from xf (MoS: a shared address matrix, separate read and write routers).

## Rules check
- Separate, swappable, fixed size: yes.
- No row/hash tables: yes (dense matrix).
- Addresses from thoughts: yes.
- No memory loss into thoughts: yes, with the detach.
- In-step: yes.
- Exploration: keep the 5% gate exploration constant.

## Run-critical CPU tests
- Write k->v, then reading k returns v.
- Overwrite.
- Causality.
- Off == none.
- Grads reach Wk, Wv and the gates with xf detached.
- Bytes stay fixed.
- Save/load of a single expert.
- Speed microbenchmark at 192 experts (estimated ms/step for read and write).

## Run (after Eitan's go)
- Small brain 48 -> 192 on world_change_mix with --data-mode stream (apply world_change\train_lm_stream.patch), 12,207 steps.
- Score: eval_change memory on vs off, plus the new-facts test.
