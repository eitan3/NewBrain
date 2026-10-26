# Design v3: the looped brain (Eitan's design, 2026-10-22) - draft for review

Replaces the 6-layer stack. One set of weights is reused for K passes ("ticks"), like the same cortex
processing the input again and again. NCM is removed.

## Eitan's design (his words, condensed)
1. embedding
2. enter a loop
3. a routing layer selects N experts
4. each expert is a spiking module with M_i neurons (experts can have different sizes)
5. back to the start of the loop: new routing, but the SAME brain (same expert weights), not new layers
6. after the loop, a decoder turns the latent into text
7. an internal state (like an RNN): starts as zeros/random, goes into the loop; the loop outputs a new
   prediction x (where the embedding went) and a new internal state s (where the zeros went).
   The decoder never reads s. Goal of s: more connectivity and memory between experts.
8. an expert cannot be routed to itself again on the next pass; the signal must go through other experts
   first (refractory period).

## Concrete formulation (my draft)
Per token t (sequence of T tokens, batch B), d = width of x, d_s = width of s:

    e_t = Embed(token_t)
    x_0 = e,  s_0 = 0
    for k = 1..K:                                         (same weights every pass)
        u   = RMSNorm( W_in [x_{k-1} ; s_{k-1} ; e] )     input injection: e every pass
        m   = Temporal(u)          shared leaky-membrane population over the TOKEN sequence
                                   (Mamba-2-style gated linear recurrence, chunked parallel scan,
                                    graded spikes on its output) -> gives each token its context;
                                   each pass runs its own scan over all T tokens
        h   = u + m
        router: logits = W_r h + bias_e (balancer, selection only);
                experts used by this token on pass k-1 are masked out (refractory);
                choose top-N of E experts; gates = renormalised sigmoid scores
        o   = sum_{i in chosen} g_i * Expert_i(h)        Expert_i: up (d -> M_i), graded spikes (0..4),
                                                          down (M_i -> d); M_i differs per expert
              (+ an always-on shared expert)
        x_k = x_{k-1} + W_x [h ; o]                       residual update of the "thought"
        z   = sigmoid( W_z [o ; s_{k-1}] ),  c = tanh( W_c [o ; s_{k-1}] )
        s_k = (1 - z) * s_{k-1} + z * c                    GRU-style gated working memory
    logits_t = Decoder( RMSNorm(x_K) )                    s_K is not decoded

- Word-to-word context lives in the Temporal membranes (inside the loop). Open question B: should s also
  be carried from token t to t+1 (classic RNN across time)? Default: no (keeps training parallel).
- Training: K sampled per step (e.g. log-normal-Poisson, mean 6, range 2..12); truncated backprop through
  the last 4 passes (earlier passes under no_grad); loss = next-token CE (+ router z-loss).
- Spikes: graded 0..4 with learned thresholds, "floor" homeostasis. Optimizer: BrainOpt (Muon + AdamW).
- Growth: experts are born (copy of a busy expert, zero output, ramp-in) and experts grow neurons (M_i).
- Shared weights => far fewer parameters than a 6-layer stack at the same compute; capacity goes into many
  experts (e.g. 64 of mixed sizes 96/192/384, N = 6 per pass).

## Context: what we know from v1/v2 (TinyStories, ~13M active params, 200M tokens)
- 6-layer stack with routing + spikes + growth + Muon (organism v2): val loss 1.285 (full-size control 1.262);
  transformer 1.317. Seeds vary +-0.003.
- Earlier reviews for a looped design (DESIGN_graph_brain.md section 3): loop over depth for the whole
  sequence (not a per-token inner loop), input injection, sandwich norm, random h0, random depth training,
  truncated BPTT, exit when the answer stops changing, no joint halting training at first.
