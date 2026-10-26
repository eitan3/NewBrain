# Plan 25d outline: global + per-expert blob memory, the expert decides

Status: plan only, nothing is built. Replaces 25c (per-pass banks, rejected; kept as a record). The full plan with the
math is written after you've checked this outline. Any default you haven't confirmed yet is marked [default].

## 1. What the memory is
- **Only blobs (splats).** No hash/row table anywhere.
- **One global blob space** that every expert can read.
- **One private blob space per expert**: 768 routed experts plus the always-on shared expert. When an expert splits,
  the child gets a copy of the parent's private blobs.
- **Nothing is tied to the pass number.** Every pass uses the same spaces, so it works the same with 6 passes or 40.
  How often any pass reads is up to the brain.
- A blob = a centre (key, 64-d), a content vector, plus strength, age and use count. The width is one number per space.

## 2. The address
- The key is **the input the expert receives for that token, at that pass**, taken before any memory is added.
- In the current code, all experts a token is routed to receive the **same** input (the token's state at that layer).
  So the global search runs **once per token per pass**, and every routed expert sees its result.
- It's mapped to 64-d by a **fixed** centring/scaling and a fixed random projection, measured once early in
  training and **never learned**. No token keys. The address still comes only from the thought.

## 2b. What the gate and the address see (checked in loop_brain.py)
The order inside one pass k (`core.forward`, then `run()`):
1. `r = read(s, x)`: the token reads the workspace s. That s already holds every earlier pass's writes, the concept
   feedback (`hier.fb`) and the previous memory output (`_slot_write` into slot 15).
2. `u = norm(x + embed(token) + W_ro r + inject)`:
   - x = the running thought from all previous passes;
   - inject = the previous pass's concept write-back (Σ_l B_l A_l μ_l, all levels) + the previous pass's memory read M.
3. `h = u + mixer(u)`: the SSD membranes add the context of earlier words.
4. **Expert input `xf = norm(h)`**. The router also gets `side(s)`, a workspace summary.
5. MoE (`o`), then the workspace write `s = write(s, m + o)` and the x update.
6. After the core pass: the concept heads on the new s give μ_l for every level. The memory read M (today's rows) is
   computed, and both go into the next pass's inject.

**Already in xf:**
- the token;
- all earlier passes' thought (x);
- the workspace as read at this position (which includes the earlier write-backs, concept feedback and memory slot);
- every concept level from the previous pass (through the rank-64 write-back);
- the previous memory read;
- the context.

**Not in xf:**
- this pass's concept vectors and workspace update. These are computed after the MoE, so they can't exist yet; using
  them would need an extra pass;
- the full-detail μ_l (only its rank-64 write-back is in);
- the per-level uncertainty;
- the previous read's match quality;
- the router's workspace summary.

**Design rule:**
- **Address** = xf, which already combines token, thoughts, concepts, previous memory and context. It's mapped to 64-d
  by a fixed centring/scaling + a fixed random projection, never learned, the same map for global and private.
- **Gate input** = xf + side(s) + the previous pass's μ_l of each level (as produced, causal) + the previous read's
  match quality + this read's match quality (distance, margin; global and private). All detached: the gate never
  pushes the thought.

## 3. What happens inside one expert layer (every pass)
1. The router picks the token's experts, as today.
2. **Search (no gradient):**
   - global: the nearest blobs to the token's key, once per token;
   - private: for each (token, chosen expert), the nearest blobs in that expert's own space, all experts batched
     in one call.
3. **Each chosen expert's gate decides**, before the expert runs: "do I need memory for this token?" It looks at
   the input it received plus the match quality (distance to the best blob, and how clearly it wins), for both
   global and private.
   - Read where the predicted gain > 0. Safety: at most ~50% of an expert's tokens read (speed), and ~5% random
     reads, so the gain stays measurable. No imposed budget.
4. **Read:** blend the nearest blobs by distance (a Gaussian kernel), global + private.
5. **Inject:** the read is added to the expert's input, together with everything else it receives. Then the expert
   runs on it. Contents are stored at model width (or go through one shared projection), so no 768 separate output
   matrices.

## 4. Writing (inside the training step, no extra loop, no dataset pointers)
- After backward, surprising positions (chosen from the brain's own signals only) write:
  - a blob into the **private** space of each expert that processed the token (key = the expert's input, content =
    what the gradient says would have helped);
  - a blob into the **global** space.
- A near-duplicate is **merged** instead of copied, with a threshold set so unrelated facts are not merged.
- When a space is full, the **weakest** blob is evicted, never one used in this step.

## 5. Keeping blobs findable while thoughts drift
- When a read clearly helped (confirmed recall), the blob's centre moves toward the current key.
- Strict renewal on a very close match, with an anti-capture guard so a blob can't be pulled away by unrelated
  thoughts:
  - clear-winner margin;
  - capped move size;
  - per-blob and per-space move counts;
  - logged travel distance.
- Slow fading, used only for eviction. The kernel width is set from real same-fact distances, with a floor.
- **Fill rate is logged per space over time.** Slow filling is the main risk without a backup memory.

## 6. What learns
- By gradient: blob contents, the injection, and the experts.
- The main loss flows back through the read into the input: both the content and which blobs matched.
- **The gates never push the thought** (stop-gradient). Each gate is trained after backward from the measured gain of
  its reads.
- Keys are never trained by gradient.

## 7. Sizes [default]
- Global: about 256-512K blobs.
- Private: about 256-512 blobs per expert × 769 experts, about 200-400K blobs.
- Roughly 0.6-1.5 GB in all, with exact numbers and the ≤ 20 GB check in the full plan.

## 8. Speed
- One global search per token per pass. The private searches are tiny (a few hundred blobs per expert) and batched.
- The gates and the 50% cap bound the reads.
- Target: memory adds ≤ 3% to the step. Going faster overall is the separate speed track.

## 9. Checks before building, all with the NEW address (expert input)
- **Paraphrase:** the same fact in other wording finds the same blob.
- **Rare facts reach the writes.**
- **Gate separability:** can a gate tell where reading helps?
- **Drift** of the expert-input address from step 6K to 7K.
- **Cross-pass:** a blob written at pass i is found from pass j (all passes share one space).

## 10. GO criteria and early stop (same as 25c), vs the old run and the control at the same steps
- Late facts ≥ 0.27 at 7K (old 0.280) and ≥ 0.28 at the end (control 0.260).
- Rare facts ≥ 0.035 at 7K (old 0.016) and ≥ 0.045 at the end.
- Harm on other world text ≤ +0.04 nats (old +0.13).
- ≤ 3% slower; ≤ 20 GB.
- Early stop at 7K if late facts < 0.24 or rare facts ≤ 0.016.

## 11. Risks
- No backup memory: blobs must be written before they help.
- The cross-pass check: if a token's expert input changes a lot from pass to pass, a blob written at one pass may not
  be found at another.
- Per-expert spaces only see the tokens routed to that expert. Rare facts may be spread thin across experts, which is
  what the global space is for.
