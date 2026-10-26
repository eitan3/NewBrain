# WaveSheet v2: cloud spec (review input, 2026-10-25 06:05 UTC)

Source: the coordinator, combining 3 independent cloud reviews, all "build after fixes".
Use it as REVIEW INPUT for the in-progress build, not as a restart.

## Reconciliation

**MUST adopt**
- the no-sync / autocast-off / no_grad rules;
- the schedule:
  - pass 0 gives zeros;
  - one recall at pass 3, cached by (fwd_id, pass) and reused at passes 4-5;
  - recompute-safe;
  - plasticity in after_backward only;
- the null control giving z, c_tok and kappa;
- confirmation: z>=4 & h>=theta_h, at most 16 per micro-batch;
- writes:
  - 1 per sequence per micro-batch, at the most surprising non-confirmed token (sigma>=q90);
  - strength 2 for the top 4 of 16;
  - target = normalize(-e_hat + max(h,0)*old recall);
- Capture/E_buf with the anchor leaf;
- memory params in their own AdamW, excluded from the brain optimizer and the brain clip;
- the gate/scale clamps and the 0.5|x| cap;
- the determinism rules: unique index_put_, scratch pad, order-free dedupe;
- the chain fill controller, f*ch=0.0315;
- Stage A T0 (blob module at equal bytes on the deployed schedule), T2 and T6, before trusting the store;
- the U-tests you can reach.

**SHOULD adopt**
- 4-bit nibble storage (3 strengths x 5 phases + empty, 0.46 GB) as the default layout, because Eitan asked for no more space. Keep the 5-bit / byte layout only as a fallback flag, if the nibble path fails U20 or costs real time.
- One global sheet as the default: train-only = the 16 places expert e reads most. If private/shared parts are already built, keep them behind a flag for Eitan's decision.

**Keep yours as arms; cloud default otherwise**
- Reconsolidation becomes `--wave-recon`, default 0: the spec found its influx cut memory life to 400-700 steps.
- Your two-speed map is the spec's step<500 rule followed by a slow EMA with T=8192 (the same idea). Test T in {freeze, 32768, 8192, 2048, fast} in T3d.
- W4b's 2nd-entry write after a failed recall becomes an arm in T4 (write at E1), since the toy evidence supports it.
- Gate start: spec b0=0 with log_s=ln0.3 and the cap, unless the 25% init already passes the U-tests.

**When the evidence disagrees:** if a spec item contradicts measured evidence, keep the evidence and report the item and the number.

**GPU/RAM rule (Eitan uses the laptop)**
- Every GPU job calls torch.cuda.set_per_process_memory_fraction (cap ~16 GB) so it OOMs instead of spilling into shared RAM.
- torch threads 2.
- Every GPU job is logged to the coordinator before it starts.
- G1/G2 (training processes) need Eitan's typed go.

## PART 1/5: goal, pre-build gates, layout

**Goal.** Replace the placeholder store in fruit-fly-sim-wave with WaveSheet.
- Keep the address module (a), the MemoryBase wiring (b), tests_wave 27/27 and tests_memory_api 29/29.
- `--memory blob` stays bit-identical; `--memory none` is unchanged.
- CPU only. No training run without Eitan's typed go (G1, G2 only).

**Stage A first.** Offline on ckpt_6000: forward/backward of the frozen brain, no weight updates. STOP and report if a gate fails.

- **T0 Harness baselines on the deployed schedule** (recall from pass-3 xf; the same answer scored at passes 3, 4, 5):
  - the blob module at byte_budget 0.46 GB, ExactNN, and the (a) ideal codes;
  - both with no load and with crowd load (300K real wiki xf written first).
- **T1 (a) parity at 32x32:** >=0.50 cloze and >=0.66 held, mean over passes 1-5.
  - Report per level l the key agreement a_l at the right place (cue key == stored key) and P(stored key in cue top-2), for train-template returns, held-out templates, cloze cues, and same-person-other-relation.
  - Report place agreement, top1/top2.
  - Run the column model (PART 2) with these a_l.
- **T2 -dL/dm stability:**
  - median same-fact cos over >=3 train exposures >=0.3;
  - same-fact minus different-fact-same-template >=0.2;
  - train vs cloze >=0.25;
  - random-init decoder capture of real targets >=0.78.
- **T6 oracle:** inject m=alpha*|x|*(-e_hat of an earlier same-fact exposure) at passes 3-5, all experts, g=1, alpha in {0.1,0.2,0.3,0.5}.
  - Controls: another fact with the same template, and mean -e.
  - Pass: gain >=+20 pts cloze top-1 and >=+10 over the best control.
- **STOP if:**
  - T2 or T6 fail;
  - the column model predicts rare-return recall <0.4 at every fill arm;
  - T0 blob >= predicted wave T3a + 0.10 cloze.

**Constants**
- d384, N8192, P1024 (32x32 torus), L3, S32 slices x12 dims.
- Uk16, C16 cols x24 dims, Uc64, PHI5, F16, W3, lambda4, beta16, Emax769, R={3}.
- f*=0.12; f*ch=1-(1-f*)^(Uk/Uc)=0.0315; chunk2048.
- Fan-in of column c = slices (2c+q)%32, q<16.

**Synapse nibble s**
- 0 = empty; s = 1+5*(cnt-1)+ph, with cnt 1..3 and ph 0..4.
- cnt = (s-1)//5+1; ph = (s-1)%5.
- inc: s+5 if 1<=s<=10. dec: s-5 if s>5, else 0.
- efficacy = (cos,sin)(2*pi*ph/5) if s>0, else 0.

**Tensors** (own state file last.memory.pt, all counted by byte_budget)
- **key_syn:** flat uint8 [402,653,184 + 64 pad].
  - row(p,l,c,q,k) = ((((p*3+l)*16+c)*16+q)*16+k).
  - 32 B per row = 64 units; the low nibble is the even unit.
  - syn = row*64+v; byte = syn>>1; nib = syn&1.
- **chain_bits:** flat uint8 [50,331,648 + 64 pad].
  - row(p,l-1,c,i,vabove) = ((((p*2+l-1)*16+c)*3+i)*64+vabove).
  - 8 B per row; bit v = target unit v of layer l, col c.
- **pad:** scratch for dropped duplicate writes. Zeroed after every put, excluded from checksums, saved as zeros.
- **Address buffers (fp32):** mu[384], P[1024,384], P_hits[1024] (EMA), Q[3,32,16,12], Q_hits[3,32,16] (EMA).
- **Parameters** (own AdamW, betas .9/.99, wd 0, clip 1.0):
  - dec_A, dec_B [3,16,64,24], lr 1e-4;
  - W_gx[16,384], G1_w[16,24], G1_b[16], gate_w[769,16] N(0,.01), gate_b[769]=0, g_dec[384]=1, lr 1e-3;
  - log_s=ln0.3 clamped to [ln.05, ln1], and gate_b clamped to [-4,4], both lr 1e-4.
  - After each memory step: |A_v|=|B_v|=rho_l, and B_v orthogonal to A_v.
- **Other buffers:**
  - rho[3], init .137, .115, .096;
  - owner_use[1024,769] int32;
  - fill_key[1024,3], fill_ch[1024,2], owned[1024];
  - null stats med[3], q99[3], mu_b[8], sd_b[8];
  - theta_h; influx counters; generator state.
- **Total:** about 0.46 GB (459.8 MB). `--memory-cap-mb 600`.

## PART 2/5: address and recall (exact ops)

**Op rules**
- All address, recall and plastic ops run under torch.autocast(enabled=False), with .float() inputs and torch.no_grad (not inference_mode).
- No host syncs: no .item, nonzero, unique, bool-mask indexing or tensor-valued if.
- Quantiles by kthvalue/topk with a python int k.

**Address** (once per micro-batch)
- x = xf.detach().float(); xh = (x-mu)/max(|x-mu|,1e-6).
- s = xh@P.T - 0.5|P|^2; E0 = argmax; masked_fill -inf; E1 = argmax (first index on ties).
- d_w = |xh-P[E]|.
- Per entry j: r = xh-P[Ej]. For l=0..2:
  - Rv = r.view(N,32,12);
  - D = |Rv|^2 - 2einsum(nsd,skd->nsk,Rv,Q[l]) + |Q[l]|^2;
  - K[:,j,l] = argmin;
  - r -= Q[l][s,K].

**Address learning** (after_backward, on 1024 seeded sample tokens per micro-batch; frozen in train-only)
- step<500: keep (a)'s rule.
- step>=500:
  - BMU = argmin |xh-P|^2 + 0.01*log((hits+1)/(mean+1));
  - one-hot GEMM sums;
  - X += (1/T)(mean-X), only for nodes with samples, with T=8192 micro-batches (flag: 0 = freeze, fast = negative control);
  - applied to P (no neighbour pull), Q (residual slices assigned at the BMU), mu and hits.

**LUT16** [65536,8] fp32, rebuilt from the formula, never saved.
- For int16 value w and u=0..3, s=(w>>4u)&15 gives (re,im) of unit 4*idx+u.
- The arithmetic path must match it bit-exactly (U20).

**Recall** at pass 3, in chunks of 2048 tokens, both entries j=0,1
- **Layer 0:**
  - rows[n,2,16c,16q] = row(Ej,0,c,q,K[n,j,0,(2c+q)%32]);
  - G = key_syn.view(int64).view(-1,4)[rows] -> view int16 [n,2,16,16q,16w] -> permute to [...,16w,16q].contiguous() -> int32 & 0xFFFF -> F.embedding_bag(view(-1,16), LUT16, mode='sum') -> Z[n,2,16,64,2];
  - M0 = |Z|/16; v0 = argmax over units.
- **Layers 1, 2:**
  - Zk uses the level-l keys;
  - chain rows (Ej,l-1,c,i,v_{l-1}[n,j,(c+i-1)%16]) are gathered from chain_bits.view(int64);
  - bits = (x>>arange(64))&1; nch = sum over i;
  - M = (|Zk|+4*nch)/(16+12); v_l = argmax.
- **Entry choice:** chi[j,l] = mean_c max_v M; chibar_j = mean_l chi; w = argmax_j (entry 0 on ties). Arm route w0: choose w after layer 0.

**Null control** (256 tokens, t%32==0)
- cue = (E[t], K[(t+N/2)%N]); same layers and entry choice.
- EMA rate 0.01:
  - med_l and q99_l of the column maxima;
  - mu_b and sd_b of chibar_w per fill bin b (8 bins over [0,2f*], by the fill of E_w).

**Confidence**
- z = (chibar_w-mu_b)/max(sd_b,.005).
- c_tok = clamp((z-2)/4,0,1).
- kappa_lc = clamp((maxM_lc-med_l)/max(q99_l-med_l,.01),0,1).
- Empty storage gives y=0 exactly.

**Readout** (WITH grad, outside no_grad)
- Winner entry; per (l,c) the top-2 units by M; p = softmax(16*M_top2); theta = atan2(Im,Re).
- coefA = c_tok*kappa*p*cos(theta), coefB = c_tok*kappa*p*sin(theta), both detached.
- y = sum_l concat_c sum_j (coefA*A[l,c,idx_j] + coefB*B[l,c,idx_j]), via F.embedding on A.view(3072,24), and B likewise.

**Hint**
- m = exp(log_s)*|x|*(g_dec*y);
- m *= min(1, 0.5|x|/max(|m|,1e-6));
- m *= keep (10% of sequences, drawn at step_begin from the memory generator);
- m = Capture.apply(m, anchor).

**Features** q[n,8], detached
- clamp(z/10,-.5,2), c_tok, chi_w,l-med_l (3), the mean top1-top2 margin of M, d_w, fill(E_w).

**Cache per recall**
- E, K, w, v_l[n,3,16], ph_l = remainder(round(theta*5/(2pi)),5), chibar, z, c_tok, readout idx and coefs, q, m.

**Column model** (used by T1 and U10)
- One place, 16 cols x 3 layers, F16, Uk16, Uc64, PHI5, binary chain W3, lambda4.
- n in {26,66,131} memories, plus 8 same-person memories sharing keys with prob a_same.
- Cue keys match with prob a_l. Output = per-layer winner accuracy.

## PART 3/5: schedule, apply, teaching signal, writes, forgetting

**Schedule**
- The trainer calls forward_begin(fwd_id, training) at pass 0 of every forward, and read() gets pass_idx. Add both to MemoryBase if missing; they are active only for `--memory wave`.
- pass 0: zeros.
- pass in R: recall if (fwd_id,pass) is not cached.
- pass>=1: apply the latest cached recall from a pass <= current, else zeros.
- A new fwd_id clears the cache.
- Re-entry (checkpoint recompute) reuses the recall and rebuilds only readout, m and gates. Assert use_reentrant=False when checkpointing.
- Before write_start: skip the recall; m=0 still goes through Capture.

**Apply at pass p**
- xh_p detached (saved bf16).
- h = SiLU(G1_w@[W_gx@xh_p; q] + G1_b).
- a = sum(F.embedding(e_k,gate_w)*h) + gate_b[e_k]; the shared expert uses row 768.
- g = sigmoid(a), b0=0, no floor.
- r_pairs = g.bf16()[...,None]*m.bf16()[:,None,:], shape [n,6,384]; r_shared = g_sh*m.
- Next-pass inject = sum_k pi.detach()*r_pairs + r_shared*(shared mix weight). This trainer change is gated on `--memory wave`.

**Capture(m, anchor)**
- Identity forward. Backward: E_buf += grad (fp32), returns (grad, zeros).
- anchor = a 0-dim leaf with requires_grad=True, new per micro-batch, so E_buf is alive in train-only.
- E_buf = dL/dm summed over passes 3-5.
- Capture comes after the keep mask.

**after_backward** (before optimizer.step); skip if step<write_start (500), in eval, or when off.
- **W1:**
  - e = E_buf; e_hat = e/max(|e|,1e-12);
  - h = cos(m,-e) (0 if |m|<1e-6);
  - sigma = CE/(H+1) at the final pass.
- **W2:**
  - h' = cos(m_t,-e_((t+N/2)%N)) on tokens with c_tok>0;
  - theta_h = .99*theta_h + .01*max(.05,q99.9(h')), by descending sort + gather at floor(.001*count).
- **W3:**
  - conf = (z>=4) & (h>=theta_h); keep the top 16 by h*c_tok (invalid -> scratch).
  - Path at E_w: key (E_w,l,c,q,K[w,l,(2c+q)%32],v_l[c]). If present AND ph(s)==ph_l[c], then s+=5 if s<=10.
  - Chain bit (E_w,l-1,c,i,v_{l-1}[(c+i-1)%16]): set bit v_l[c].
  - Recon arm off by default.
- **W4:**
  - cand = not conf & sigma>=q90 (kthvalue) & pos>=1.
  - Per sequence, argmax sigma -> 16 slots (invalid if none).
  - st=2 for the top 4 of 16 by sigma, else 1.
  - Target y = normalize(-e_hat + max(h,0)*[c_tok>=.5]*m_hat).
  - Encode, for l=0..2:
    - Yc = res.view(16,24);
    - a = Yc.A[l]^T/rho_l, b = Yc.B[l]^T/rho_l, shape [16,64];
    - ph = remainder(round(atan2(b,a)*5/(2pi)),5);
    - sc = a*cos+b*sin; v* = argmax; coef = sc[v*];
    - res -= cos*A + sin*B at (v*,ph*);
    - rho_l = .99*rho_l + .01*mean(coef).
  - Write at E0 with keys K[:,0]:
    - key nibble = 1+5*(max(cnt_old,st)-1)+ph*;
    - chain bit (E0,l-1,c,i,v*_{l-1}[(c+i-1)%16]): set bit v*_l[c].
- **Dedupe:** apply W3 fully, then W4 reading the post-W3 bytes.
  1. List (syn,new); sort by syn*16+new; keep the last per syn.
  2. byte = syn>>1; delta = (new-old_nib)<<(4*(syn&1)).
  3. Segment-sum the deltas per byte (cumsum, run ends).
  4. Write old+delta at the last element of each byte run; the rest go to scratch.
  5. One index_put_; zero the scratch.
  6. Chain bits: the same with mask 1<<(bit&7).
- **Counters and logs:**
  - owner_use[E0,e_k[j]] += 1 (6 routed + column 768) on the 1024 samples (integer index_put_ accumulate).
  - Source labels go to a CPU log counter only. Influx counters per source.

**after_step**
- Memory AdamW once (grads summed over both micro-batches; memory params excluded from the brain optimizer and the brain clip), then renorm A/B.
- Sweep 16 places from sweep_ptr:
  - key block (p,l): fill = mean(s>0); n1 = mean(1<=s<=5); q = clamp((fill-f*)/max(n1,1e-6),0,1). Decrement each present synapse with prob q (uint8 randoms vs round(256q)).
  - chain block: if fill>f*ch, clear each set bit with prob (fill-f*ch)/fill.
- Train-only: owned places only.

## PART 4/5: modes, flags, implementation rules

**Modes**
- **Off:** zeros, bit-identical to none.
- **Reset store:** zero key_syn, chain_bits, owner_use, fill and the null stats; keep the address, decoder and params.
- **Reset all:** re-init; write_start = step+500.
- eps everywhere; atan2(0,0)=0.

**Swap**
- Load a state file.
- Refuse with a clear error if the layout version or P, L, S, Uk, C, Uc, PHI, F, W, packing, d or Emax differ.
- Same-run checkpoints only. Extra gate rows are ignored.

**Train-only e**
- owned = the 16 places with the highest owner_use[:,e], among places with owner_use[p,e]>0 (never empty places). Log the share.
- Trainable: leaf copies of gate_w[e] and gate_b[e] with a fresh AdamW, written back on save.
- Frozen: the brain, the other gate rows, W_gx, G1, log_s, g_dec, A, B, the address and the null stats.
- W3/W4 candidates need e among their 6 routed experts AND their place in owned (mask before topk and quota).
- Sweep owned places only.

**Splits (on_grow)**
- Clone the parent rows (and AdamW moments if present) first; the children get the clone.
- owner_use[:,child] = clone//n_children.
- Storage untouched.

**State file**
- Tensors and primitives only (torch.load weights_only=True).
- Temp file + os.replace with retry.
- Stamp the brain step and run id. The CUDA generator state falls back to CPU.
- byte_budget = sum of nbytes, equal to the file size within 1% + 64 KB.

**Determinism**
- index_put_ only with unique indices (scratch for the rest); integer index_add is fine.
- Float sums by one-hot GEMM; gate rows via F.embedding.
- torch.remainder, never fmod.
- Never index with a uint8 tensor (bool-mask trap).
- First index on ties.

**Flags v1** (other arms raise NotImplementedError)

| flag | default | arms / notes |
|---|---|---|
| `--memory` | wave | |
| `--wave-fill` | .12 | .06, .08, .2 |
| `--wave-writes-mb` | 16 | 8 |
| `--wave-strong-k` | 4 | 2, 16 |
| `--wave-chain` | 3 | 0, 1; `--wave-fanin 18` is the equal-bytes chain-0 control |
| `--wave-lambda` | 4 | 2, 6 |
| `--wave-route` | both | w0 |
| `--wave-readout` | top2 | wta |
| `--wave-evidence` | pres | count: x1, 1.5, 2 by cnt |
| `--wave-recon` | 0 | 1: the top 4 confirmed paths bind absent synapses at cnt 1, each using one write slot |
| `--wave-target` | grad+old | grad |
| `--wave-addr-t` | 8192 | 0, 32768, 2048, fast |
| `--wave-z-conf` | 4 | |
| `--wave-conf-max` | 16 | |
| `--wave-drop` | .1 | |
| `--wave-scale-init` | .3 | |
| `--wave-cap` | .5 | |
| `--wave-gate-b0` | 0 | |
| `--wave-chunk` | 2048 | 4096 |
| `--wave-unpack` | bag | arith |
| `--wave-null` | 256 | |
| `--wave-write-start` | 500 | |
| `--wave-intent` | 1.0 | 0.5 = recall only the top 50% of tokens by p=sigmoid(w_int.W_gx xh+b_int), b_int=2, 5% random always included, m*=p |

**v1.1** (specified, build later)
- `--wave-hop`: layers 1-2 step to the 4-neighbour whose prototype direction best matches the residual; the same rule at write.
- `--wave-phases 0`: weights-only 2-bit, Uc 128, chain W1, 470 MB.
- `--wave-read-passes`, `--wave-private-mb`, `--wave-infer-write`.

**Cost targets**
- 0.86 GB gathered per step: 50.7 KB per token-recall = 1536 key rows x 32 B + 192 chain rows x 8 B.
- Transient <=0.3 GB at chunk 2048; peak extra <=1.2 GB.
- About 1.3-1.5K launches per step.
- Step overhead <=10% of memory-off (estimate 30-60 ms).
- Fallbacks: chunk 4096, route w0, CUDA-graph capture of the recall (static shapes).

**Build order** (about 3 h)
1. Stage A scripts.
2. Storage + recall + encode/write + sweep + Harness adapter (unblocks T3).
3. Capture/E_buf + W1-W4 + slow address + gates (U1-U5, U14).
4. Train-only, splits, swap.

The data loader must supply a per-sequence source label for the write-mix log (not scoped yet).

## PART 5/5: tests and bars

**Unit tests** in tests_wave_store.py: the tiny preset (P=16 on 4x4, B2 x T64), plus one full-size shape/byte smoke. Keep 27/27 and 29/29 green.

- **U1** blob bit-identical (20 tiny steps).
- **U2** wave == none before write_start (loss and grads bit-identical; no gathers).
- **U3** reset store / reset all, then 50 steps: no NaN/Inf.
- **U4**
  - a: the grad of r wrt xf is zero (allow_unused);
  - b: brain grads identical with Capture on/off;
  - c: checkpoint on/off (real trainer setting): identical E_buf, one recall per micro-batch.
- **U5** two seeded CPU runs give identical checksums (storage minus pad, P, Q, mu, params) after 30 steps. A CUDA memory-only version runs under G1's go.
- **U6** train-only e:
  - the 83 brain tensors are unchanged;
  - only owned places, gate row e and counters change;
  - hints of tokens with all entries outside owned are bit-identical;
  - e=0 on a mostly empty sheet owns no empty place;
  - U6c: E_buf is nonzero and equal to the full-mode E_buf.
- **U7** save, fresh process, load: bit-identical recall on 1K xf; 192 and 768 experts load; a layout mismatch errors.
- **U7b** another run's memory: loss <= off + 0.5%.
- **U8** split children equal the parent.
- **U9** byte_budget <= cap, and constant.
- **U10** 2,000 entries at n<=26 per place:
  - clean unit accuracy >=0.99;
  - at a=0.6: >=0.85, and >= chain-off + 0.03;
  - at a=0.3 and n=131: within ±0.05 of the column model (about 0.64 at layer 2).
- **U11** at P=64, 2 writes/step, 1 place swept/step, plus synthetic confirmations:
  - key fill f* ±0.02; chain fill f*ch ±0.005;
  - strength-1 mean life within 20% of the model (2.4-3.5K steps);
  - strength-2 survival at 1,356 steps >=0.85;
  - chain life within 20% of key life.
- **U12** random-init decoder capture >=0.78 median (isotropic and anisotropic); units used >=90%; norm constraint within 1e-5.
- **U13** Harness smoke; ExactNN parity unchanged.
- **U14** permuted source labels leave the memory bit-identical.
- **U15** TorchDispatchMode count <=800 kernel ops per full micro-batch.
- **U16** scratch is zero and excluded.
- **U17** chain controller.
- **U18** recall identical with and without outer bf16 autocast.
- **U19** cache cleared across micro-batches and eval forwards.
- **U20** bag == arithmetic decode; two nibbles of one byte written in one batch both land.

**T3 W-bench full module** (deployed schedule, equal bytes)
- T3a no load: cloze >=0.80 x ideal pass-3 cloze (about 0.44); held >=0.85 x ideal; within ±0.05 of the T1 prediction.
- T3b-crowd (300K wiki written first, sweep on): cloze >=0.70 x T3a; held >=0.75 x T3a.
- Retention (interleaved): report vs blob.
- T3c arms at equal bytes: the chain must beat chain-0 (F18) by >=+0.03 cloze on T3b-crowd, else drop it.
- T3d slow address during writes: cloze >=0.95 x frozen; the fast rule reproduces about 0.02.

**T4 stream:** wiki xf background, W1 facts, 9 sightings at real gaps, real sigma, e with T2's cosine; 12K steps, scoring the last 6K.
- Bars:
  - rare returns >=0.5;
  - rare at gap >1.3K >=0.4;
  - late >=0.7;
  - rare cloze >= blob + 0.05 and >=0.25;
  - false confirms <=2 per micro-batch;
  - false hits <=0.02.
- Arms: fill x writes, strong-k, recon.

**T5 stream dump** (>=1K steps): >=50% of rare facts are written within their first 3 sightings.

**T7** ckpt_6000 vs ckpt_8000: code agreement >=0.8 x same-checkpoint, for the chosen T.

**G1** (Eitan's go): 200 steps, wave vs none.
- VRAM <= none + 1.2 GB and <=21.4 GiB.
- Time <=1.10x.
- Launches <= none + 2000.
- No NaN; no syncs (sync_debug_mode error, 20 steps).
- 32 writes per step.

**G2** (go): 1.5-2K steps of wave / none / blob (empty start).
- Loss <= none + 2 SE; off <= none + 0.5%.
- Rare cloze >= blob.
- Time <= 0.5x blob.
- Gate AUC vs the confirmed label >=0.65.
- Stop at step 1K unless s>=0.1 and g(confirmed) >= 1.5x g(other).

## Owner decisions (Eitan, 2026-10-25 08:06 UTC)
1. GO on WaveSheet, although its density per memory is about the same as the blobs'.
2. Rule 5 is OK for all three: gathering the synapse rows of the key neurons that fired, the best-match content write,
   and the fixed 16-state efficacy LUT.
3. Regular training trains everything (the brain and all memory parts). Train-one-expert is a later fine-tune mode
   only. The global sheet is the default; private parts stay behind a flag, OFF, until he answers "global only or
   private too".
4. No travelling wave now: the wave stays a coarse-to-fine sweep at a fixed place. The hop (v1.1) is the maximum next
   step.
5. Read at pass 3 for every token; each expert's gate decides how much to use it. --wave-intent stays 1.0 (off).
6. z_conf lowered 4 -> 2: z>=4 fired on 10 of 3,200 cues; precision at z>=2 is ~0.93. z_conf=3 is kept as an arm.
   (Coordinator decision, 2026-10-25 09:00 UTC.)
7. z_conf back to 4: the load-bin null spreads z, and at z>=2 confident-false was 0.107/0.127 (cloze/held); at z>=4
   cloze confident-false is 0.042. The readout ramp stays c = clamp((z-2)/4). (Coordinator decision, 2026-10-25 10:07 UTC.)

## Owner decisions (Eitan, 2026-10-25 10:10 UTC; recorded by the build agent)
1. The confidence scaling stays.
2. The memory file is 1.94 GB, with 4 places per expert (the v2.1 strip layout).
3. Split children copy the parent (redesigning splits is for a future version).
4. The shared expert has a gate like the routed experts: its pair (k = 6) reads the GLOBAL sheet under its own gate row
   (Wg[768], Ug, bg[768]), with the same exploration and the same target loss, and no private read (bp[:, 6] = -inf).
