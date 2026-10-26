# WaveSheet v2.1 FINAL build brief (cloud, after 3 reviews; coordinator, 2026-10-25 10:05 UTC)

## Cover note
TIME BOX: Eitan wants the launch as soon as possible. Do NOT restart or rebuild what works. Use this brief as the spec and as the review checklist for the current build.

### MUST for this run (correctness or rules)
- the cap scale is detached (have);
- capacity-aware selection, 0 drops (have);
- gused, plus the isfinite(+inf) bug in PART 2;
- the shared expert reads global only (bp[:,6] = -inf);
- --wave-drop sequences don't write or confirm;
- wd 0 on D;
- z_conf 2 (have);
- unread = exactly 0, and off == none;
- splits copy everything (the 12K run grows 192->768 experts, so splits ARE P1);
- the D factored update from PART 3 (about 12 lines; saves 0.85 GiB and keeps the file at 1.94 GB). Keep fused AdamW only if the factored update fails its test.

### LAYOUT
Tell me now, in 3 lines, what the --wave-priv-levels private path stores per expert (places, bytes) and the global size. Eitan asked to shrink global and expand per expert. If it is not the PART 1 strip layout (global 256 places + 4 places per expert) and switching is quick, switch. If not quick, keep yours and say how it differs.

### DEFER (P2, tell me which)
budget controller (use --wave-budget fixed), train-only isolation, swap/load, CUDA graphs, arms, benches.

Report every difference between your code and this brief at the end.

## PART 1/5: priorities, changes, layout
Where it differs from my 09:36 message, THIS wins.

**P1** (needed for the first run; do now, in this order):
1. layout + memory rows;
2. private address;
3. gate + selection with --wave-budget fixed;
4. read path + Translate + detached cap;
5. gate loss;
6. write rule + confirmations + sweep;
7. D update;
8. must-pass tests (PART 5);
9. VRAM/time check.

**P2** (after the run starts): budget controller, train-only isolation, per-expert swap/load, CUDA graphs, arms, the T3p/T8/T4 benches. Splits are P1 only if the run will split soon.

### CHANGES vs 09:36
- **a) D storage and update.**
  - D [769,384,384] fp32 is a registered BUFFER, not a Parameter, plus a D_grad buffer, updated by our own factored update (PART 3).
  - No AdamW moments: -0.85 GiB VRAM, and the memory file is 1.94 GB instead of 2.84 GB.
  - Do NOT use torch.optim.Adafactor. Its step calls .item() 3x (host syncs), builds full-size temporaries, and scales steps by the RMS of the whole tensor (which couples experts).
  - If fused AdamW is already built and time is short, it fits the first run (~20.3 GiB est.), but switch before the long run (the file > 1.9 GB rule).
- **b) Shared expert.** The shared expert (id 768) has NO strip of its own by default: its pair takes only the global hint (bp[:,6] = -inf). Its 4 places stay allocated (arm --wave-shared-strip 1). C_sh = K_g.
- **c) Three budgets** (python ints):
  - K = hint pairs <= 7N/2 = 28,672;
  - K_pr = private reads <= 6N/2 = 24,576;
  - K_g = global-read tokens <= N/2 = 4,096.
  - Private reads are compacted out of the hint list, so a pair that only wants the global hint costs no private read.
- **d) Exploration.**
  - 5% of ROUTED pairs force a private read (K_xp = 2,458).
  - Independently, 5% of TOKENS force the global read, with the hint for all 7 pairs (K_xg = 410).
  - Constant, never annealed, off at eval.
- **e) Generators.** One generator per purpose (explore, keep, addr, sweep), reseeded per (step, micro-batch) on the host with manual_seed(python int). Resume then needs no generator state.
- **f) wd = 0 for D.** 0.01 at lr 1e-4 is a 1M-step time constant; it never pulled W_e to I.
- **g) Dropped sequences.** --wave-drop sequences still read (for gate targets) but write nothing and confirm nothing.
- **h) Confirmation.** Confirm at z >= z_conf = 2 (the measured precision is 0.93).
- **i) Top-k.** Every top-k uses tie-free int64 keys (PART 2).
- **j) Sweep.** Sweep 8 global places per step (was 4), plus 24 private.

### LAYOUT
- **Global:** 256 places (16x16 torus, ids 0..255).
- **Strips:** 769 strips x 4 places (id 256 + 4*m + j, where m = memory row).
- **DUMMY = 3,332:** all zero, never written or swept; a read there gives y = 0 exactly.
- **Arrays:** key_syn flat uint8 [3333*393,216 + 64], chain_bits [3333*49,152 + 64]; 1,474,412,672 B in all. int64 row ids.
- **Memory rows:**
  - row_of [769] int64 maps a brain expert id to a memory row (0..E_live-1, contiguous, assigned at on_grow; shared 768 -> 768).
  - Every per-expert tensor (strip places, P_e, D, R, Cv, gate rows, Wld_p) is indexed by memory row, so D[:E_live] is always a view even if brain ids are not contiguous.
- **Global address:** keep fix A (the map learns on write load, least-loaded of top-4, read top-4) at P_g = 256. Report its constants and re-run T3a-fix at 256 places.

## PART 2/5: private address, gate, selection
Everything here runs under no_grad, with autocast off and fp32 'ieee' matmul.

### PRIVATE ADDRESS (routed pairs, k<6)
- Pc = P_e[:E_live].reshape(4*E_live,384).
- Per 2,048-token chunk: s = xhat@Pc.T - 0.5*|Pc|^2.
- Take columns 4*row_of[eid]+j; mask j >= Pinit[row] with -inf; jstar = argmax.
- place_p = 256+4*row+jstar, or DUMMY if Pinit[row]==0 (then wp = False).
- Keep d_p and margin_p. Keys (Q_p, 3 levels) only for read and null pairs.
- Write place = read place.

### Learning
- **Warm-up** (step < write_start):
  - 1,024 sample tokens x 2 routed pairs;
  - first-sample init while Pinit<4;
  - rate max(1/32,1/(1+hits));
  - sample-hit conscience 0.05.
- **Slow phase:**
  - samples = this micro-batch's PRIVATE WRITES (<=32);
  - BMU = argmin |xhat-P|^2 + 0.05*log((Wld_p+1)/(mean of the expert's 4 + 1));
  - P += (xhat-P)/128;
  - one-hot GEMM.
- **Q_p:** the v2 Q rule on the 2,048 sample pairs.
- A sample-hit conscience is what collapsed T3a.

### GATE
- h = silu(G1_w@[W_gx@xhat ; f_tok] + G1_b), shape [N,16].
- bp = <Wp[row],h> + Up.f_pair + bp_b[row]; bg the same with Wg, Ug, bg_b; bp[:,6] = -inf.
- Init: Wp, Wg ~ N(0,0.01); bias init b0 for ~25% yes (calibrate, ~ -0.02). All gate params lr 1e-3.
- Save xhat, f_tok, f_pair, eid detached.

### SELECTION
The coordinator's CPU prototype: 148 dispatch ops, no syncs, all invariants hold.

```
tkey(x): n=x.numel(); b=x.contiguous().view(int32); k=where(b>=0,b,b^0x7FFFFFFF); return (k.long()<<32)+(n-1-arange(n))  # distinct; tie -> lower index
ar = arange(M), cache per length
LOW = iinfo(int64).min
xp6 = zeros(6N,bool).index_put_((topk(tkey(rand(6N)),K_xp).indices,),True); xp = cat([xp6.view(N,6),zeros(N,1,bool)],1)
xg = zeros(N,bool).index_put_((topk(tkey(rand(N)),K_xg).indices,),True)
SG = bg.amax(1); gs = where(xg,inf,where(SG>tau,SG,-inf)); GT0 = topk(tkey(gs),K_g).indices
gsel = zeros(N,bool).index_put_((GT0,), xg[GT0]|(SG[GT0]>tau))   # NOT isfinite(gs): +inf fails
wg = gsel[:,None]&(xg[:,None]|(bg>tau)); wp = xp|(bp>tau); want = wp|wg; forced = xp|(xg[:,None]&wg)
r = where(forced,inf,where(want,maximum(where(wp,bp,-inf),where(wg,bg,-inf)),-inf)).view(-1)
grp = where(eid==768,E_live,row_of[eid]).view(-1); key = tkey(r)
gr = empty(M).scatter_(0,argsort(key,descending=True),ar)
order = argsort(grp*65536+gr); gs2 = grp[order]
rk = empty(M).scatter_(0,order,ar-searchsorted(gs2,gs2))
cap = where(grp==E_live,C_sh,C); elig = (rk<cap)&want.view(-1)
S = topk(where(elig,key,LOW+ar),K).indices; valid = elig[S]; tok = S//7; gS = grp[S]
vp0 = valid&wp.view(-1)[S]; vg = valid&wg.view(-1)[S]
slot = where(valid, where(gS==E_live,E_live*C,gS*C)+rk[S], NSLOT)   # NSLOT = E_live*C+C_sh
PR = topk(tkey(where(vp0,where(xp.view(-1)[S],inf,bp.view(-1)[S]),-inf)),K_pr).indices
vp = zeros(K,bool).index_put_((PR,),vp0[PR])
gused = zeros(N,int32).index_put_((tok,),vg.int(),accumulate=True) > 0
GT = topk(tkey(where(gused,SG,-inf)),K_g).indices; gvalid = gused[GT]
```

**Rules:**
- C = 8*ceil(2*K*(6/7)/E_live/8): 256 at 192 experts, 64 at 768.
- No pair is dropped after selection.
- Count: want pairs excluded by cap or budget, demoted (vp0&~vp), DUMMY slack.
- Build vp_d, vg_d, pslot (row K = zero row) and gslot AFTER all masks.
- Eval: K_xp = K_xg = 0.
- Compute K etc. from the call's N.

## PART 3/5: read path, translation, D update

### READ LIST
- **Order:**
  1. [0,4K_g): the global entries of GT (gvalid ? E_g[GT[i],j] : DUMMY);
  2. then K_pr private (vp[PR[j]] ? place_p : DUMMY);
  3. then the null block (256 tokens x 4 global + 256 routed pairs), only on null steps. Null steps are every 16 steps, EMA rate 1-0.99^16; skip null rows whose place is DUMMY.
- **Chunks:** 8,192 reads each, last chunk partial, no padding; max 40,960 = 5 chunks.
- **Per read:** v2 R2-R4. The global winner is the clearest of 4 (R5). Confidence and readout follow v2 R6-R7, with per-sheet null stats.
- **Memory hygiene:** free gathered rows before the int32 cast, do '& 0xFFFF' in place, and do the chain bit-sum with dtype=uint8.

### HINT + TRANSLATION (grad on, autocast off)
```
y_g_full = zeros(N,384).index_put((GT,), y_g*gvalid[:,None]); y_p_full = zeros(K,384).index_put((PR,), y_p*vp[PR][:,None])
y_pair = keep_seq[tok][:,None]*(vg[:,None]*y_g_full[tok] + vp[:,None]*y_p_full)
y_pair = Capture.apply(y_pair, anchor)     # backward: E_pair[:K] += grad
X = zeros(NSLOT+1,384).index_put((slot,), y_pair)
outr,outs = Translate.apply(X[:E_live*C].view(E_live,C,384), X[E_live*C:NSLOT], D, E_live, D_grad)
o = cat([outr.reshape(-1,384), outs, zeros(1,384)])[slot]
o = (exp(log_s)*xnorm[tok])[:,None]*o
o = o*clamp(0.5*xnorm[tok]/o.norm(dim=-1).clamp_min(1e-6), max=1).detach()[:,None]   # cap scale DETACHED
```

**Translate:**
- **forward:** saves Xr, Xs; returns Xr+bmm(Xr,D[:E]) and Xs+Xs@D[768].
- **backward:**
  - if grad_buf is not None: grad_buf[:E].baddbmm_(Xr.transpose(1,2),gr); grad_buf[768].addmm_(Xs.t(),gs);
  - returns gr+bmm(gr,D[:E].transpose(1,2)), gs+gs@D[768].t(), None, None, None.
- grad_buf=None at eval and in train-only.
- Assert D_grad.data_ptr() is unchanged at forward_begin. Never set it to None; zero_() it after each memory step.

### PASS 3 OUT
```
r3 = zeros(7N+1,384,bf16).index_put((S,), o.bf16()); r_pairs = r3[:7N].view(N,7,384)[:,:6]; r_shared = [:,6]
```
Inject at the next pass = sum_k pi.detach()*r_pairs + w_sh*r_shared (v2 A5).

### PASSES 4-5
```
eq = e_p[:,:,None]==eid[:,None,:]; has = eq.any(-1); kidx = eq.to(int8).argmax(-1); idx = where(has, arange(N)[:,None]*7+kidx, 7N); r_p = r3[idx]
```
- Flat index + zero pad row: deterministic backward (no gather/scatter_add).
- Counters only on first entry, not on checkpoint recompute.
- Assert use_reentrant=False.

### UNREAD pairs
Unread pairs are those outside S, with want=0, excluded, or DUMMY slack. They get:
- no translation input;
- output exactly 0 at passes 3-5;
- zero D_grad rows;
- no write and no target.

### D UPDATE
In after_step, with our own code. Chunks of 256 memory rows over D[:E_live], plus row 768. b2=0.99, lr=1e-4, eps=1e-30, clip_d=1, wd=0.
```
G = D_grad[a:b]
R[a:b].mul_(b2).add_(vector_norm(G,dim=-1).square()/384, alpha=1-b2); Cv[a:b] the same with dim=-2
n[a:b] += 1; bc = 1 - b2**n[a:b].float()
rh = R[a:b]/bc[:,None]; ch = Cv[a:b]/bc[:,None]
V = rh[:,:,None]*ch[:,None,:]/rh.mean(-1).clamp_min(eps)[:,None,None] + eps
U = G*V.rsqrt_(); rms = vector_norm(U,dim=(-2,-1))/384
D[a:b].add_(U/(rms/clip_d).clamp_min(1)[:,None,None], alpha=-lr)
```
- A zero-grad row does not move; no cross-expert coupling; no syncs.
- Skip memory optimizer steps before write_start.
- Save/restore the trainer's matmul precision on the host and run the memory GEMMs at fp32 ieee.
- Assert no GradScaler.

## PART 4/5: gate loss, writes, confirmations, sweep, splits, reset

### ORDER
- **after_backward:**
  1. W1 signals;
  2. W2 calibration;
  3. W3 confirmations;
  4. W3b gate loss;
  5. W4 writes;
  6. W5 apply (confirm, then write; v2 W5);
  7. W6 counters + Wld;
  8. W7 address learning.
- **after_step:**
  1. memory AdamW: fused; gate lr 1e-3; decoder 1e-4; log_s 1e-4; betas .9/.99, wd 0, clip 1.0 per group;
  2. decoder constraint;
  3. D update;
  4. D_grad.zero_();
  5. sweep.
- All of this is skipped before write_start.

### GATE TARGET
- E_pair = dL/dy_pair from Capture.
- b_p = -<E_pair, vp*y_p_raw>, b_g = -<E_pair, vg*y_g_raw[tok]>.
  - raw = before keep_seq; for dropped sequences this is the benefit of adding the read.
- With the detached cap, b_p+b_g = -<dL/do,o> exactly, capped or not.
- sigma_b = where(sigma_b==0, rms, 0.99*sigma_b+0.01*rms); bt = clamp(b/max(sigma_b,1e-12),-5,5).

### LOSS
- Recompute bhat_p, bhat_g, chat_g = <Wc0,h>+<Uc0,f_tok>+bc0 and chat_p = <Wc1,h>+<Uc1,f_pair>+bc1 at the S rows, from the saved detached inputs.
- w = 4 on exploration parts (xp for private, xg tokens for global), else 1.
- L_b = sum w*(vp*Huber1(bhat_p-bt_p) + vg*Huber1(bhat_g-bt_g)) / max(sum w*(vp+vg),1).
- L_c = BCE vs conf labels on vp / vg rows.
- L_gate = L_b + 0.5*L_c; torch.autograd.backward(L_gate, inputs=gate_params).
- Log GATE_COLLAPSE if non-exploration reads < 1% of K for 200 steps while the exploration mean bt > 0.

### GLOBAL HINT GRAD
- e_g = sum_k vg_d[GT,k]*E_pair[pslot[GT,k]] (dense gather, deterministic).
- h_g = cos(y_g,-e_g); h_p = cos(y_p,-E_pair at PR); 0 where |y| < 1e-6.

### CONFIRM (only read sheets, kept sequences)
- conf_g = gused & keep & z_g>=2 & h_g>=theta_hg -> top 8 by h*c at the winner entry path.
- conf_p = vp & keep & z_p>=2 & h_p>=theta_hp -> top E_live//24.
- conf_tok = either.

### WRITE (Eitan 08:30)
- **Candidates:**
  - read_tok = (gused | any vp_d[:,:6]) & keep_seq;
  - cand = read_tok & ~conf_tok & (sigma >= q90 by kthvalue) & pos>=1;
  - per sequence, argmax sigma -> wt[16]; the top 4 by sigma get strength 2.
- **Sheets:**
  - global iff gused[wt], at the least-loaded of its 4 entries (Wld_g), with that entry's keys;
  - private: top-2 by pi (tie-free key) among k<6 with vp_d[wt,k], at that pair's read place with its cached keys.
- **Targets:**
  - global: normalize(-e_g_hat + max(h_g,0)*[c_g>=.5]*y_g_hat);
  - private: normalize(-E_hat + max(h_p,0)*[c_p>=.5]*y_p_hat);
  - send any target with raw norm < 1e-6 to scratch.
- Encode all <= 48 targets in one batch (v2 W4).
- W6: Wld_g, Wld_p = EMA (T_L 1,024) + births.

### SWEEP
- 8 global places per step + 24 over the live strips (device index list built at on_grow).
- The v2 W8 rule; g_sweep reseeded per step.

### SPLIT (on_grow, parent row p -> children)
1. Clone first.
2. Then index_copy_ into every child row:
   - the 4 places' key + chain bytes, the fill rows;
   - Wld_p/n, P_e, Ph_e/n, Pinit;
   - the D row, the R/Cv/n rows;
   - the gate rows Wp/Wg/bp/bg and their AdamW moments.
   - A child may reuse p.
3. Then update row_of (a host list + one device copy at on_grow; fine, not in the step), E_live, C, the conf cap and the sweep list.

### RESET
- **expert:e:**
  - zero its places, fill, Wld_p, D row and R/Cv/n rows;
  - gate rows back to init AND their AdamW moments zeroed;
  - Pinit = 0 (its reads go to DUMMY until re-init).
- **reset all:** the above, plus rebuild the optimizer state.

## PART 5/5: must-pass tests (P1), measurement, report

### P1 TESTS
CPU, tiny preset: P_g 16, 8 routed + shared, 4 places per strip, B2xT64.

- **U36:** off == none, bit-identical; before write_start == none (no gate ops, no gathers).
- **U23:**
  - autograd.grad(L_gate, xf) is None;
  - brain grads are identical with the gate loss on/off;
  - L_gate changes only gate .grad;
  - the main loss gives nonzero D_grad rows for read experts and grads to A, B, log_s;
  - d(sum out)/d(xf) = 0;
  - gate .grad is zero after the main backward.
- **U24:**
  - cap INACTIVE (s=1e-3) and ACTIVE (s=1, |y|~0.65): b_p+b_g = -<dL/do,o> within 1e-6 rel;
  - at small s, b within 10% of L(0)-L(y).
- **U21:**
  - |S| = K;
  - exploration counts exact, and all kept;
  - per-group count <= cap;
  - valid slots unique;
  - gused is a subset of gsel, and every vg token is in GT;
  - same seed -> same S;
  - planted ties -> the lower index wins;
  - a TorchDispatchMode guard finds no _local_scalar_dense, nonzero, masked_select or unique in the memory code.
- **U22:** unread pairs give exactly 0 at passes 3-5; the D_grad rows of unread experts are exactly 0.
- **U25:**
  - global-only read -> only global places change;
  - private-only (1 expert) -> only its 4 places;
  - both, with 3 private readers -> global + the top-2 by pi only;
  - no read -> nothing;
  - all of a token's global pairs excluded -> no global read and no global write;
  - keep=0 sequence -> no write, no confirmation;
  - a capped, confirmed read is confirmed;
  - a zero target is skipped.
- **U26:**
  - Translate vs naive per-pair (I+D_e): fwd <= 1e-5 rel, grad D <= 1e-5 rel, grad y <= 1e-6;
  - the grad_buf=None path works;
  - D never gets a .grad.
- **U39:**
  - zero-grad rows unchanged by the D update;
  - one expert's update ignores other rows' grads;
  - equals an fp64 reference within 1e-6;
  - no syncs.
- **U4c-v2.1:**
  - trainer checkpointing on vs off: E_pair, D_grad and the A/B/log_s grads bit-identical;
  - recall once, Translate.backward and Capture.backward once per micro-batch.
- **U28:** split copy (incl. a child reusing the parent row, and non-contiguous brain ids through row_of).
- **U34:** storage exactly 1,474,412,672 B; file within 1% + 64 KB of byte_budget().

### P2 TESTS
U27 reuse, U29 train-only bit-identity (snapshot ranking features), U30 swap/reset (D[e]==0 and init gate rows after one optimizer step with no reads), U31-U33, U35 (<= 1,600 launches per micro-batch), U38 budget controller, then T3p, T8, T4.

### MEASURE
Module smoke and vram_guard, announced to the coordinator first; the full run needs Eitan's typed go.

- **VRAM:**
  - reserved peak = torch.cuda.max_memory_reserved(), printed /2**30 AND /1e9, with --wave-budget fixed (worst case);
  - bar <= 21.0 in ONE stated unit;
  - expected ~19.5 GiB with the 15.8 brain (20.3 with AdamW on D);
  - if over: --wave-chunk 4096, then 3 places per strip, then brain savers.
- **Speed:**
  - ms/step and tok/s vs none on the same code; launches per step;
  - profile split: address, selection, recall, readout, Translate, D update, after_backward;
  - bar <= 1.10 x none;
  - if over: CUDA graphs first (one per budget, shared pool, chunk 4096 inside), then --wave-gread-frac 0.25.

### REPORT
- read shares (private, global) and the exploration share;
- the write mix (global / private / both per step);
- excluded / demoted / DUMMY-slack counts;
- the VRAM breakdown (sheets, D + D_grad, held, transients);
- any failing test;
- every place the built code differs from this brief.

## Coordinator follow-up (10:06 UTC)
YES, switch to the PART 1 strip layout now: global 256 places + 4 wave places per expert, in one flat key_syn/chain array, reusing the column-store code with a place offset. The Hebbian placeholder is not Eitan's per-expert wave memory, so it must not be the run's private path. Keep it only as an arm if it costs nothing. The time box holds: do the MUST list and defer the rest. Report when the strips pass U25/U34, plus the byte budget.
