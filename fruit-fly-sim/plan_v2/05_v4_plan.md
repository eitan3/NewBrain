# v4 plan: fair recipe + grown pass-routed membrane + grown wave expert pool (base: waves + workspace v2)

Planning agent report, 2026-10-22. Base flags: --state none --waves --slots 16 --ws-direct + base flags.
Memory probe: ~0.33 GB activations per active head; 384->768 experts adds ~3.6 GB (token x E fp32 routing
tensors per pass); base (6 heads, 384 experts) ~11 GB activations, real peak ~15 GB.

## Bugs / gaps to fix first
1. Newborn experts never selected in the base config: --route-aux sets balance "aux"; debt payback
   (debt/pay) only runs in auxfree mode -> newborn sits 2 below forever. Move debt payback (and
   debt[counts>=mean]=0) outside the auxfree branch in apply_balance and the non-deferred forward branch.
2. Checkpoint reload after loop growth: Neurogenesis.births_step sets cfg["heads"] to a list -> LoopBrain(**cfg)
   breaks. For LoopBrain store an int.
3. SSDMixer.ramping not saved -> checkpoint mid-ramp loses the ramp. After load set ramping = (ramp<1).any().
4. RoutedMLP.grow_experts must return src (parents) for the wave matrix.
5. Wave diagnostics: sheet positions from expert index shift as E grows -> use a fixed pos buffer.
6. Remove any --waves --neuro exit guard in train_lm.py.

## 1. Fair recipe vs v2
- --accum 2 (2 micro-batches of 16). Sample K once per optimizer step (same K both micro-batches) and the
  deep-supervision pass once per step (model.aux_k_fixed used by run()).
- Divide the whole micro loss (LM + router aux + z + 0.2 deep sup) by accum.
- Grad clip after the last micro-batch, opt.step, apply_balance (pending sums over both micro-batches = batch
  32; age += 1 once), neuro.step: once per optimizer step; neurogenesis/schedules count optimizer steps.
- tok_per_step = batch*accum*seq; maturity tau uses effective batch; train-loss EMA uses mean of micro losses;
  FLOPs counter matches.
- Data stream identical to v2 batch 32 for the same seed (two numpy draws of 16 == one draw of 32).
  --eval-batch 32 (default batch*accum). --anneal 0.05,0.2.
- Remaining (not linear) differences: aux balance loss (product of means) and capacity per micro-batch;
  homeostasis EMA twice per step; loop compute per token ~K x v2's. Diff v2 config.json for lr, warmup, decay,
  maturity, route flags, --neuro-every.

## 2. More word-to-word memory: 6 -> 36 heads, routed per pass (head groups)
- Pass-routed head groups: every head integrates every token (memory continuous). Buffer hgroup (H, long).
  The 6 initial heads = group 0, always on. 30 newborn heads round-robin into routed groups 1..5 (6 each).
  Pass k computes group 0 + group 1 + k mod 5: ~12 active heads per pass, 36 distinct heads total; choice
  depends only on the pass index (causal).
- Cost: ~4-5% step time and ~0.33 GB per ACTIVE head; idle heads cost only parameters (~123k each).
- Births exact: a pass's group membership never changes; newborn ramp 0 + zero wo columns.
- Workspace projections are sized by d and slots: head growth does not touch them; m (so the write m+o) is
  unchanged at birth.

## 3. Expert pool 192 -> 768 with growing wave connectivity
- Linear from 5% to 60% of training, --neuro-every 25, ~2 births per event, ramp birth mode.
- New Core.grow_experts(n): src = mlp.grow_experts(n); grow router_conn E -> E+n: old block unchanged;
  new row = parent's row (outgoing), new column = parent's column (incoming); twin<->parent entries and the
  diagonal 0; newborn<->newborn entries copied from their parents' entries. No noise.
- Exact births with waves: newborn score <= 1 + (parent bias - 2); in aux mode other biases are 0, so it is
  never selected at birth regardless of wave/fatigue terms; its row adds nothing (prev = 0); fatigue starts 0.
- router_s (64 -> E) already copies the parent's row.
- Fixed sheet: buffer pos (E, 2) on a fixed 16x12 sheet (cfg sheet_cols); newborn = parent's position +
  N(0, 0.3) jitter with wrap-around; rebuild adj2 from pos (same Gaussian, > 0.05); diagnostics use pos.
- carry_brain_state already pads 2-D state top-left (m 0, v median, n 0, born now) - lock in with a test.
- Memory fixes to stay under 20 GB: (a) wave term via F.embedding_bag(prev_idx, C, mode="sum") instead of
  dense prev @ C (exact, removes ~E^2 MACs per token per pass); (b) wrap the router (logits, side input, wave,
  sigmoid, z/aux losses) in torch.utils.checkpoint(use_reentrant=False).

## Schedule (200M tokens, 12,207 optimizer steps)
Births (heads and experts) steps 610-7,324; spike anneal 5-20%; last ramp ~8,324; WSD decay from 9,765;
warmup 2%. Growth ends well before the cool-down.

## Budget estimate (end state)
FLOPs ~ +25%; ~29-35k tok/s; peak ~16-18 GB with fixes (a)+(b) (~20.5 GB without). Must check with a GPU smoke
run born at final sizes: --run-name v4_smoke --start heads=36,experts=768 --tokens 3e6.

## Code changes
- spiking_lm.py: RoutedMLP.grow_experts returns src; debt payback in all balance modes; checkpointed route();
  forward accepts extra_idx and fatigue (wave built inside); SSDMixer: hgroup buffer,
  forward(x, write=None, group=None) slicing the active channels of wz/wx/wq/wk, wdt, convs, A_log, dt_bias,
  Dskip, norm_w, ramp and wo columns; grow_heads assigns groups round-robin; _load_from_state_dict restores
  ramping; Spike.forward / calibrate / homeostasis take a ch index; n_active excludes idle head groups.
- loop_brain.py: Core.grow_experts; pos buffer; adj2 from pos; embedding_bag wave; pass k sends
  group=1+k%G to mix; head_group and sheet_cols in cfg; aux_k_fixed.
- neurogenesis.py: call b.grow_experts when the block is a Core; keep cfg["heads"] an int for the loop.
- train_lm.py: --accum, --eval-batch, --head-group (0 = dense), --neuro-t 0.05,0.6; micro-batch loop per
  section 1; effective-batch tau, step counts, token and FLOPs counters.
- brain_opt.py: no change.

## Unit tests to add (double precision where exactness matters)
1. Accumulation equivalence (waves + slots, dropless, z-loss only, fixed K and aux_k): grads of one batch of 4
   == mean of two batches of 2; pending/pending_elig match; bias after apply_balance matches.
2. Exact head births in the loop (dense and grouped): output unchanged after each birth at several K; groups
   filled round-robin; a head gets zero grad on passes not using its group; output changes once ramp > 0.
3. Grouped mixer with all heads active == dense mixer.
4. Exact expert births with waves (aux and auxfree): output unchanged; router_conn (E+n)^2 with old block
   intact and parent row/col copied (zero diagonal, zero twin entries); pos, adj2, router_s extended; in aux
   mode after ~1,100 apply_balance calls the debt is 0 and newborns are selected.
5. carry_brain_state on router_conn: old block equal; new rows/cols m=0, v=median, n=0, born=t; next step finite.
6. Checkpoint reload after growth, mid-ramp: LoopBrain(**cfg) + strict load gives identical output.
7. Neurogenesis on the loop: heads and experts reach targets, optimizer rebuilt at each change.
8. embedding_bag wave == prev @ C in values and gradients.
9. Checkpointed routing == plain routing in values and gradients.
All 70 existing tests must still pass.

## Risks
Newborns have no fatigue (abrupt adoption mid-ramp); copied outgoing rows double excitation toward the parent's
targets; head groups get uneven gradient (group 5 only trains when K >= 5); memory uncertain until smoke run;
throughput drops gradually as heads/experts grow; aux loss scales with E (should hold at balance).

## Command
python -u train_lm.py --run-name v4_grow_fair --arch loop --spike graded --homeo 0.01 --homeo-mode floor
 --opt brain --batch 16 --accum 2 --eval-batch 32 --route-aux 0.02 --capacity 1.5 --moe 768,6,192,384,auxfree
 --heads 36 --start heads=6,experts=192 --neuro --neuro-every 25 --neuro-t 0.05,0.6 --head-group 6
 --state none --waves --slots 16 --ws-direct --anneal 0.05,0.2 --tokens 200e6
