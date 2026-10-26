# Latent thinking concept (next-chunk latent) + usefulness - 2026-10-23

Eitan: no thinking dataset, no outside model. (1) the thought predicts the LATENT of the upcoming text, (2) the
thought is judged by its usefulness (Quiet-STaR style). Both behind flags, default off; extends plan 08.

## 1. Next-chunk latent target (self-supervised, JEPA / LCM-like)
Chunk: the next N = 16 tokens (t+1 .. t+16), not "the next sentence". Why: BPE merges punctuation with quotes
and spaces (dialogue is everywhere in TinyStories), so sentence ends are messy token-wise; a TinyStories sentence
is ~10-15 tokens, so 16 tokens covers about one sentence; and a fixed window is vectorisable (cumsum) and has the
same horizon for every word. N is a flag (--think-n).

Target z*_t = mean over the chunk of the brain's OWN final representation normf(x_K) at positions t+1..t+16,
with stop-gradient. Why not an EMA encoder:
- an EMA copy of the whole brain doubles the forward cost; an EMA "light pooled encoder over embeddings" gives a
  bag-of-embeddings target, i.e. almost the bag-of-words target we already have;
- normf(x_K) is free (the same forward computes every position in parallel) and is anchored by the next-token LM
  loss: it must stay informative to predict the next word, so it cannot collapse to a constant (the usual JEPA
  risk). Stop-gradient keeps the target from chasing the prediction.
- Safety anyway: log the target's std per dimension (collapse detector) and the prediction's std.
Causality: z*_t uses tokens after t only as a LABEL; the head at t reads only the workspace at t (causal).

Head (--think-target latent): pooled workspace n_s(s).flatten (1024) -> Linear 256 -> GELU -> Linear 2d (mean mu,
log-variance lv), every pass. Loss: Gaussian NLL 0.5 * (lv + (z* - mu)^2 / exp(lv)), lv clamped to [-6, 4], mean
over dims and positions with a full chunk... (positions near the end use the tokens that exist; the last position
has no future and is masked); averaged over the backpropagated passes; weight --think-w.
Uncertainty u = [predicted variance (mean lv, scaled), how much mu still moves: |mu_k - mu_{k-1}| / |mu_k|],
detached -> the stop head (2 features, as in plan 08) and (--think-fb) the workspace.

## 2. Usefulness (Quiet-STaR style, by gradient instead of RL)
--think-use: the predicted concept mu itself is written back into the workspace each pass through a zero-initialised
Linear(d -> 64) added to every slot (no change at birth). It gets no label beyond (1): its usefulness signal is the
next-word loss. Gradient path, without a new readout: on the NEXT pass the workspace read enters the core input u,
which feeds the membrane scan (causal time mixing) - so the loss of LATER words t' > t flows back through the
membrane into the workspace of word t, into mu_t and the concept head. This holds for passes inside the bptt window
(6), i.e. every training step. A separate multi-horizon readout is therefore not needed.
Diagnostic (loop report): val at K=6 with the concept write-back as trained vs with it zeroed: usefulness = the
loss it saves (positive = the thought helps the words that follow).

## Cost / memory
Head 1024x256 + 256x768 = 0.46M MACs per word per pass (~5% of a pass); write-back 384x64; target = a cumsum.
Memory: mu/lv per pass, B x T x 768 floats (25 MB at B=16) for the bptt passes.

## Tests (CPU)
off -> unchanged; target = hand-computed window mean of the stop-grad final representation (no gradient into it);
NLL value; masked last position; write-back zero-init exact, gets gradient from the LM loss of LATER positions
(gradient on mu_t from a loss on positions > t only); u feeds the stop head; cfg rebuild + strict reload.

## Run (proposal, queue only after approval)
Test 4 = pair3 recipe + --think-target latent --think-n 16 --think-use 1 (single change over pair3: latent target
and usefulness write-back instead of the bag target).
