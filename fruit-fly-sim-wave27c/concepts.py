"""Next-Concept Modeling (NCM, Kumbong et al. 2026, arXiv 2610.14162) adapted to our streaming brain.

1. ConceptTokenizer: a multiscale residual VQ-VAE over chunks of C tokens. Level k quantises the
   residual left by levels < k at l_k positions (l = 1, 2, 4, ..., C), so level 1 is one "gist" code per
   chunk and finer levels add detail. Nested dropout (train on a random number of levels) makes the
   coarse levels carry the important meaning. Trained first (train_concepts.py), then frozen.

2. ConceptHead + top-down plan (inside SpikingLM, see concept= option):
   - the brain's LOWER blocks run over the whole sequence;
   - at the last token of each chunk j, the head reads the lower-block state and predicts the codes of
     chunk j+1 coarse-to-fine (levels 1..K' in parallel within a level, NCM-style), trained with
     cross-entropy against the frozen tokenizer's true codes;
   - the head's OWN prediction (its soft belief: softmax-weighted codebook vectors) is reconstructed
     to C positions and added to the input of the UPPER blocks for every token of chunk j+1.
   The plan comes only from the past, so training is causal and parallel and there is no
   train/test mismatch (true codes are used only as loss targets).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_INTERP = {}


def _interp_matrix(n_in, n_out, device, dtype):
    """(n_out, n_in) matrix equal to linear interpolation (align_corners=False), built once per size."""
    key = (n_in, n_out, device, dtype)
    if key not in _INTERP:
        eye = torch.eye(n_in)[None]                                   # (1, n_in, n_in) as channels x length
        M = F.interpolate(eye, size=n_out, mode="linear", align_corners=False)[0].T  # (n_out, n_in)
        _INTERP[key] = M.to(device=device, dtype=dtype)
    return _INTERP[key]


def resize(z, n):
    """(N, l, d) -> (N, n, d) by linear interpolation along the sequence (the R_k / Down_k operators).
    Done as a tiny matmul: torch's GPU upsample_linear1d kernel took ~30 ms per call for this layout."""
    if z.size(1) == n:
        return z
    return torch.einsum("oi,bid->bod", _interp_matrix(z.size(1), n, z.device, z.dtype), z)


def nearest(u, cb):
    """Index of the nearest codebook row for each row of u (one matmul; much faster than cdist here)."""
    u, cb = u.float(), cb.float()
    return (cb.pow(2).sum(1)[None] - 2 * u @ cb.T).argmin(-1)


class ConceptTokenizer(nn.Module):
    def __init__(self, vocab, d=256, chunk=32, codebook=1024, heads=4):
        super().__init__()
        self.chunk, self.d, self.V = chunk, d, codebook
        self.levels = [2 ** i for i in range(int(math.log2(chunk)) + 1)]  # 1, 2, 4, ..., chunk
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(chunk, d) * 0.02)
        self.enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.codebooks = nn.Parameter(torch.randn(len(self.levels), codebook, d) * 0.02)
        self.dec = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, vocab)

    def encode_latent(self, x):  # x: (N, C) token ids
        # unit-scale latents, so codebook distances and VQ losses stay well conditioned
        return F.layer_norm(self.enc(self.emb(x) + self.pos), (self.d,))

    def quantize(self, z, n_levels=None):
        """Residual multiscale VQ. Returns codes per level, the cumulative reconstruction (N, C, d) and
        the VQ losses (codebook + commitment)."""
        n_levels = n_levels or len(self.levels)
        resid, recon = z, torch.zeros_like(z)
        codes, vq = [], 0.0
        for k in range(n_levels):
            u = resize(resid, self.levels[k])                              # Down_k
            cb = self.codebooks[k]
            idx = nearest(u.reshape(-1, self.d), cb).view(u.shape[:2])
            e = cb[idx]
            vq = vq + F.mse_loss(e, u.detach()) + 0.25 * F.mse_loss(u, e.detach())
            e = u + (e - u).detach()                                        # straight-through
            recon = recon + resize(e, self.chunk)                           # R_k
            resid = z - recon
            codes.append(idx)
        return codes, recon, vq

    def decode(self, recon):
        return self.out(self.dec(recon + self.pos))

    def recon_from_codes(self, codes):
        """Cumulative reconstruction from (possibly fewer than K) levels of codes."""
        recon = 0
        for k, idx in enumerate(codes):
            recon = recon + resize(self.codebooks[k][idx], self.chunk)
        return recon

    def forward(self, x, nested_p=0.5):
        z = self.encode_latent(x)
        K = len(self.levels)
        # nested dropout: with prob nested_p train on a random prefix of the levels
        n = K if (not self.training or torch.rand(()) > nested_p) else int(torch.randint(1, K, ()))
        codes, recon, vq = self.quantize(z, n)
        logits = self.decode(recon)
        ce = F.cross_entropy(logits.flatten(0, 1), x.flatten())
        return ce, vq, codes, logits

    @torch.no_grad()
    def codes(self, x, n_levels):
        return self.quantize(self.encode_latent(x), n_levels)[0]


class ConceptHead(nn.Module):
    """Predicts the next chunk's coarse concept codes from one brain state, coarse to fine.
    Positions: [prefix] + l_1 + l_2 + ... + l_K' slots; block-causal over levels (a level sees the
    prefix and the cumulative reconstruction of the coarser levels, never finer ones)."""

    def __init__(self, d_model, tok: ConceptTokenizer, n_levels=3, d=256, layers=2, heads=4):
        super().__init__()
        self.tok, self.K = tok, n_levels
        self.lv = tok.levels[:n_levels]
        self.d = d
        self.inp = nn.Linear(d_model, d)
        self.rec_in = nn.Linear(tok.d, d)
        self.lvl_emb = nn.Parameter(torch.randn(n_levels + 1, d) * 0.02)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                                               norm_first=True) for _ in range(layers))
        self.out = nn.ModuleList(nn.Linear(d, tok.V) for _ in range(n_levels))
        sizes = [1] + self.lv
        lvl_of = torch.cat([torch.full((s,), i) for i, s in enumerate(sizes)])
        # position p may attend to q if q's level < p's level, or q is in the prefix / its own level
        mask = (lvl_of[None, :] < lvl_of[:, None]) | (lvl_of[None, :] == 0) | (lvl_of[None, :] == lvl_of[:, None])
        self.register_buffer("attn_mask", ~mask, persistent=False)
        self.register_buffer("lvl_of", lvl_of, persistent=False)

    def _run(self, h, recons):
        """h: (N, d_model) brain states; recons[k]: cumulative reconstruction of levels < k (N, C, d_tok)."""
        parts = [self.inp(h)[:, None] + self.lvl_emb[0]]
        for k in range(self.K):
            parts.append(self.rec_in(resize(recons[k], self.lv[k])) + self.lvl_emb[k + 1])
        z = torch.cat(parts, 1)
        for blk in self.blocks:
            z = blk(z, src_mask=self.attn_mask)
        out, start = [], 1
        for k in range(self.K):
            out.append(self.out[k](z[:, start:start + self.lv[k]]))
            start += self.lv[k]
        return out  # list of (N, l_k, V) logits

    def teacher_logits(self, h, codes, corrupt=0.1):
        """Training: condition each level on the TRUE coarser codes (some randomly replaced)."""
        recons, recon = [], torch.zeros(h.size(0), self.tok.chunk, self.tok.d, device=h.device, dtype=h.dtype)
        for k in range(self.K):
            recons.append(recon)
            c = codes[k]
            if self.training and corrupt:
                c = torch.where(torch.rand(c.shape, device=c.device) < corrupt, torch.randint_like(c, self.tok.V), c)
            recon = recon + resize(self.tok.codebooks[k][c].to(h.dtype), self.tok.chunk)
        return self._run(h, recons)

    def plan(self, h):
        """The brain's own coarse-to-fine belief about the next chunk (soft, differentiable):
        returns the cumulative reconstruction (N, C, d_tok) after K' levels."""
        recon = torch.zeros(h.size(0), self.tok.chunk, self.tok.d, device=h.device, dtype=h.dtype)
        recons = [recon] * self.K
        for k in range(self.K):  # K' small passes: level k uses the soft reconstruction of levels < k
            logits = self._run(h, recons)[k]
            e = logits.float().softmax(-1) @ self.tok.codebooks[k].float()
            recon = recon + resize(e.to(h.dtype), self.tok.chunk)
            recons = recons[:k + 1] + [recon] * (self.K - k - 1)
        return recon
