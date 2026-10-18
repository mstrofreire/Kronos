"""Fast multi-path Kronos sampling with a key/value cache.

The stock `auto_regressive_inference` re-runs the whole context through the
Transformer at every step and for every sampled path, although the history is
identical for all paths. Here the history (prefix) is encoded once per series and
its attention keys/values are cached; each new step only processes the newly
sampled token of every path. Results match the reference implementation in eval
mode as long as lookback + horizon <= max_context (no sliding window).

Models are run in eval mode: the upstream KronosPredictor never calls .eval(),
which leaves dropout active during inference.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.kronos import sample_from_logits, calc_time_stamps

FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def _split_heads(x, n_heads):
    b, t, d = x.shape
    return x.view(b, t, n_heads, d // n_heads).transpose(1, 2)


def _rope_at(rotary, x, pos):
    """Apply the rotary embedding of absolute position `pos` to x of shape [B, H, 1, hd]."""
    freqs = pos * rotary.inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return x * emb.cos() + rotary._rotate_half(x) * emb.sin()


def _two_part_attention(q, k_pre, v_pre, k_suf, v_suf):
    """Attention of q [P,H,1,hd] over a shared prefix [1,H,L,hd] plus per-path suffix [P,H,t,hd]."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = [torch.matmul(q, k_pre.transpose(-1, -2)) * scale]
    if k_suf is not None:
        scores.append(torch.matmul(q, k_suf.transpose(-1, -2)) * scale)
    w = torch.softmax(torch.cat(scores, dim=-1), dim=-1)
    L = k_pre.size(-2)
    out = torch.matmul(w[..., :L], v_pre)
    if k_suf is not None:
        out = out + torch.matmul(w[..., L:], v_suf)
    return out


def _merge_heads(x):
    p, h, t, hd = x.shape
    return x.transpose(1, 2).reshape(p, t, h * hd)


@torch.no_grad()
def sample_paths(tokenizer, model, x, x_stamp, y_stamp, paths, T=1.0, top_k=0, top_p=0.9, clip=5):
    """Sample `paths` forecasts for one normalized series.

    x: [L, 6] normalized features, x_stamp: [L, 5], y_stamp: [H, 5] time features.
    Returns normalized predictions of shape [paths, H, 6].
    """
    device = next(model.parameters()).device
    L, H = x.shape[0], y_stamp.shape[0]
    x = torch.clip(torch.as_tensor(x, dtype=torch.float32, device=device), -clip, clip)[None]
    stamp = torch.as_tensor(np.concatenate([x_stamp, y_stamp]), dtype=torch.float32, device=device)[None]

    s1, s2 = tokenizer.encode(x, half=True)  # each [1, L]
    blocks = model.transformer
    n_heads = blocks[0].self_attn.n_heads

    # Prefix pass: identical for all paths, run once and cache keys/values.
    h = model.embedding([s1, s2]) + model.time_emb(stamp[:, :L])
    prefix_kv = []
    for blk in blocks:
        attn = blk.self_attn
        hn = blk.norm1(h)
        q = _split_heads(attn.q_proj(hn), n_heads)
        k = _split_heads(attn.k_proj(hn), n_heads)
        v = _split_heads(attn.v_proj(hn), n_heads)
        q, k = attn.rotary(q, k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + attn.out_proj(_merge_heads(a))
        h = h + blk.ffn(blk.norm2(h))
        prefix_kv.append((k, v))
    ctx = model.norm(h)  # [1, L, d]

    dep = model.dep_layer.cross_attn
    dep_heads = dep.n_heads  # the dependency layer has its own head count
    dk_pre = _split_heads(dep.k_proj(ctx), dep_heads)
    dv_pre = _split_heads(dep.v_proj(ctx), dep_heads)

    last_ctx = ctx[:, -1:].expand(paths, 1, ctx.size(-1))
    suffix_kv = [(None, None) for _ in blocks]
    dk_suf = dv_suf = None
    gen_pre, gen_post = [], []

    for i in range(H):
        s1_logits = model.head(last_ctx)[:, -1]
        pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p)  # [P, 1]
        # Rotary on a length-1 query uses position 0 (identity), so no RoPE in the cross-attention.
        q = _split_heads(dep.q_proj(model.embedding.emb_s1(pre)), dep_heads)
        a = _two_part_attention(q, dk_pre, dv_pre, dk_suf, dv_suf)
        x2 = model.dep_layer.norm(last_ctx + dep.out_proj(_merge_heads(a)))
        post = sample_from_logits(model.head.cond_forward(x2)[:, -1], temperature=T, top_k=top_k, top_p=top_p)
        gen_pre.append(pre)
        gen_post.append(post)
        if i == H - 1:
            break

        pos = L + i
        h = model.embedding([pre, post]) + model.time_emb(stamp[:, pos:pos + 1]).expand(paths, 1, -1)
        for j, blk in enumerate(blocks):
            attn = blk.self_attn
            hn = blk.norm1(h)
            q = _rope_at(attn.rotary, _split_heads(attn.q_proj(hn), n_heads), pos)
            k = _rope_at(attn.rotary, _split_heads(attn.k_proj(hn), n_heads), pos)
            v = _split_heads(attn.v_proj(hn), n_heads)
            ks, vs = suffix_kv[j]
            ks = k if ks is None else torch.cat([ks, k], dim=2)
            vs = v if vs is None else torch.cat([vs, v], dim=2)
            suffix_kv[j] = (ks, vs)
            k_pre, v_pre = prefix_kv[j]
            a = _two_part_attention(q, k_pre, v_pre, ks, vs)
            h = h + attn.out_proj(_merge_heads(a))
            h = h + blk.ffn(blk.norm2(h))
        last_ctx = model.norm(h)  # [P, 1, d]
        dk_new = _split_heads(dep.k_proj(last_ctx), dep_heads)
        dv_new = _split_heads(dep.v_proj(last_ctx), dep_heads)
        dk_suf = dk_new if dk_suf is None else torch.cat([dk_suf, dk_new], dim=2)
        dv_suf = dv_new if dv_suf is None else torch.cat([dv_suf, dv_new], dim=2)

    full_pre = torch.cat([s1.expand(paths, -1), torch.cat(gen_pre, dim=1)], dim=1)
    full_post = torch.cat([s2.expand(paths, -1), torch.cat(gen_post, dim=1)], dim=1)
    z = tokenizer.decode([full_pre, full_post], half=True)
    return z[:, -H:, :].cpu().numpy()


def forecast_paths(tokenizer, model, df, x_timestamp, y_timestamp, paths, T=1.0, top_k=0, top_p=0.9, clip=5):
    """Like KronosPredictor.predict, but returns every sampled path: array [paths, H, 6] in price units."""
    x = df[FEATURES].values.astype(np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0)
    xn = np.clip((x - mean) / (std + 1e-5), -clip, clip)
    x_stamp = calc_time_stamps(pd.Series(x_timestamp).reset_index(drop=True)).values.astype(np.float32)
    y_stamp = calc_time_stamps(pd.Series(y_timestamp).reset_index(drop=True)).values.astype(np.float32)
    preds = sample_paths(tokenizer, model, xn, x_stamp, y_stamp, paths, T, top_k, top_p, clip)
    return preds * (std + 1e-5) + mean
