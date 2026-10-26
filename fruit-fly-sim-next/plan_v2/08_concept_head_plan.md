# Thinking-concept head: uncertainty of the brain's internal thought, fed back into the loop - 2026-10-23

Eitan's idea: next to the regular output ("what the brain wants to say"), a head on the INTERNAL thinking that
predicts several tokens at once - not exactly, but the direction of the thought. If the thought is unclear, that
prediction becomes vague / noisy: that is the brain's uncertainty, and the loop should use it (keep thinking,
change what it does). Useful mostly for long tasks (coding, research); on TinyStories expect small effects - the
first goal is to see whether the signal is real.

## v1 (implemented behind flags, default off)
Where: on the WORKSPACE s after every pass (16 slots x 64 = the brain's internal state, never decoded by the LM
head), not on x.
- Head: pooled workspace  n_s(s).flatten (1024) -> Linear(1024, 128) -> GELU -> Linear(128, V)  (low-rank to the
  vocabulary; 0.66M params).
- Target (`--concept-w W`, `--concept-n 8`): the BAG of the next N tokens, order-free: loss = mean over the N
  future tokens of -log softmax(c)[token] (softmax-over-bag cross-entropy; tokens past the sequence end masked).
  Averaged over the backpropagated passes; weight W.
- Uncertainty per word and pass: u = [normalised entropy of the bag distribution, KL(c_k || c_{k-1})] - how vague
  the thought is, and whether it is still moving.
- Feedback:
  (a) `--concept-feedback 1`: a learned vector W_u u (zero-initialised: no change at start) is added to every
      workspace slot after each pass, so the brain can act on "I'm unsure" (u itself is detached; the brain learns
      how to use it).
  (b) with `--halt 1`: u is added to the stop head's features (d + 3 + 2).
- Diagnostics (loop report, when the concept head is on): concept loss at K=6; mean concept entropy per pass; and
  the key test - AUC of "concept uncertainty at pass k" for "more passes will help by > 0.05 nats", next to the
  AUC of the LM-prediction-change (KL) signal from the uncertainty probe (0.74-0.78).
- Cost: one 128 x V projection per pass (~0.5M MACs per word, ~10% of a pass's body), + the softmax.

## Interactions
- deep-sup: independent (deep-sup trains every pass's OUTPUT, the concept head trains the WORKSPACE to hold the
  direction of the next words).
- halt: the stop head gets the concept uncertainty as 2 extra features; it still never shapes the brain.
- noise: routing temperature is independent.

## v2 (plan only)
Latent concept target instead of a bag of tokens: an EMA (slow-moving copy) encoder summarises the next N tokens
into a vector z; the head predicts a Gaussian (mean, log-variance) of z; the variance is the uncertainty (JEPA-like).
Risks: collapse (z constant -> trivially predictable): needs the EMA target + variance / covariance regularisation
(VICReg-style) or a frozen pretrained encoder (e.g. the earlier NCM concept tokenizer); and the variance head can
cheat by inflating variance (use proper Gaussian NLL, clamp log-var). v1 first: if the bag-of-words uncertainty
does not predict "needs more thinking" better than the KL signal, the latent version is unlikely to either.

## Tests (CPU)
off -> no head, nothing changes; loss equals the hand-computed bag CE; future positions past the end are masked;
the feedback vector starts at zero (exact) and gets gradient; the halt head takes the 2 extra features; cfg rebuild
+ strict reload.

## Runs (proposal, not launched)
Paired 36M-token tests vs split: (1) + noise 0.3 constant (eval clean and at 0.3), (2) + deep-sup 0.3 + halt,
(3) + concept-w 0.1 + concept-feedback (+ halt).
