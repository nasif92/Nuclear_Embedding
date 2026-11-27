"""Notebook-style Set Transformer for nucleus classification (a port of NucleusSetTransformer from the CoNIC notebook).

How it differs from train_set.py:
  train_set.py        : one set PER NUCLEUS (the nucleus + its K nearest), prediction read from the centre slot only.
  train_partition.py  : the slide's nuclei are split into spatially compact, NON-overlapping groups of <= --set_size
                        nuclei (like the notebook's spatial_sets); the transformer sees a whole group at once and
                        classifies EVERY nucleus in it jointly (permutation-equivariant, no positional encoding).

Same model as the notebook: Linear+GELU embedding, 2 pre-norm self-attention blocks (hidden 64, 4 heads, ff 128),
LayerNorm + Linear head. Groups are formed from centroids only (labels are never used). Because the partition is
fixed in the notebook, here it is re-drawn every epoch with a random rotation (so group edges move) and, at test
time, predictions can be averaged over --tta rotated partitions.

Controls (same script, same sampling/seeds/final-epoch rule as train.py):
    --set_size 48                      the model
    --set_size 1                       same network, every nucleus alone (architecture/optimizer control)
    --set_size 48 --shuffle_sets       random same-slide groups instead of spatial ones

    python train_partition.py --input qupath --subtypes Ki67 --seed 42 --set_size 48
    python train_partition.py --input qupath --subtypes Ki67 --seed 42 --set_size 1
    python train_partition.py --input qupath --subtypes Ki67 --seed 42 --set_size 48 --shuffle_sets

CAUTION (read the "mixed sets" lines in the report): labels here come from annotation REGIONS, so most groups are
single-class. A set model can then do well by classifying the whole group, which says little about groups that
straddle a boundary. The report therefore also gives accuracy on groups that contain both classes.
Groups here are built from annotated nuclei only (as in train_set.py); there is no positional input to the model.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


# --------------------------------------------------------------------------------------------- groups
def kd_groups(xy, max_size):
    """Split points into compact groups of <= max_size by recursive median cuts along the wider axis (label-free)."""
    out, stack = [], [np.arange(len(xy))]
    while stack:
        ix = stack.pop()
        if len(ix) <= max_size:
            if len(ix):
                out.append(ix)
            continue
        p = xy[ix]
        ax = int(np.ptp(p[:, 0]) < np.ptp(p[:, 1]))
        order = np.argsort(p[:, ax], kind="stable")
        h = len(ix) // 2
        stack += [ix[order[:h]], ix[order[h:]]]
    return out


def make_groups(xy, slide, max_size, theta=0.0, rng=None, shuffle=False):
    """List of index arrays into xy; every nucleus appears in exactly one group. Groups never cross slides."""
    c, s = math.cos(theta), math.sin(theta)
    xyr = xy @ np.array([[c, -s], [s, c]])
    groups = []
    for sl in np.unique(slide):
        ix = np.flatnonzero(slide == sl)
        if shuffle:
            perm = ix[rng.permutation(len(ix))]
            groups += [perm[i:i + max_size] for i in range(0, len(perm), max_size)]
        elif max_size == 1:
            groups += [ix[i:i + 1] for i in range(len(ix))]
        else:
            groups += [ix[g] for g in kd_groups(xyr[ix], max_size)]
    return groups


def pad_batch(groups, size):
    """(B, size) int64 index matrix, -1 = padding."""
    m = np.full((len(groups), size), -1, np.int64)
    for i, g in enumerate(groups):
        m[i, :len(g)] = g
    return m


# --------------------------------------------------------------------------------------------- model (as in the notebook)
class SetBlock(nn.Module):
    def __init__(self, dim=64, heads=4, ff=128, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, dim), nn.Dropout(dropout))

    def forward(self, x, valid):
        q = self.norm1(x)
        a, _ = self.attn(q, q, q, key_padding_mask=~valid, need_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x))


class NucleusSetTransformer(nn.Module):
    def __init__(self, d, n_out=2, hidden=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.blocks = nn.ModuleList([SetBlock(hidden, heads, ff=2 * hidden, dropout=dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_out))

    def forward(self, x, valid):
        x = self.embed(x)
        for b in self.blocks:
            x = b(x, valid)
        return self.head(x)


# --------------------------------------------------------------------------------------------- prediction
@torch.no_grad()
def predict_probs(model, XT, xy, slide, size, n_tta, seed, shuffle, dev, bs=256):
    """Softmax probabilities per nucleus, averaged over n_tta partitions (first one unrotated)."""
    model.eval()
    n = len(xy)
    probs = None
    rng = np.random.RandomState(seed + 12345)
    for t in range(n_tta):
        theta = 0.0 if t == 0 else rng.uniform(0, 2 * math.pi)
        groups = make_groups(xy, slide, size, theta, rng, shuffle)
        for i in range(0, len(groups), bs):
            idx = torch.as_tensor(pad_batch(groups[i:i + bs], size), device=dev)
            valid = idx >= 0
            logit = model(XT[idx.clamp(min=0)] * valid[..., None], valid)
            p = torch.softmax(logit, -1)
            if probs is None:
                probs = torch.zeros(n, p.shape[-1], device=dev)
            probs.index_add_(0, idx[valid], p[valid])
    return (probs / n_tta) if probs is not None else torch.zeros(0, 2, device=dev)


def mixed_set_stats(y, pred, groups):
    """Accuracy on nuclei in groups that contain both classes vs single-class groups."""
    mixed = np.zeros(len(y), bool)
    for g in groups:
        if len(g) > 1 and len(np.unique(y[g])) > 1:
            mixed[g] = True
    out = {"frac_nuclei_in_mixed_sets": float(mixed.mean()) if len(y) else 0.0}
    for name, m in (("acc_mixed_sets", mixed), ("acc_pure_sets", ~mixed)):
        out[name] = float((pred[m] == y[m]).mean()) if m.any() else None
    return out


def near_other_class_stats(xy, y, slide, pred, mpp, near_um):
    """Accuracy for annotated nuclei within near_um of an annotated nucleus of ANOTHER class (e.g. a tumour nucleus
    next to a stroma annotation) vs the rest. Distances in µm via a single --mpp."""
    from scipy.spatial import cKDTree
    d = np.full(len(y), np.inf)
    for s in np.unique(slide):
        on = np.flatnonzero(slide == s)
        for c in np.unique(y[on]):
            q, o = on[y[on] == c], on[y[on] != c]
            if len(o):
                d[q] = cKDTree(xy[o]).query(xy[q], k=1)[0] * mpp
    near = d < near_um
    ok = pred == y
    return {"near_um": near_um, "frac_near_other_class": float(near.mean()) if len(y) else 0.0,
            "acc_near_other_class": float(ok[near].mean()) if near.any() else None,
            "acc_far_from_other_class": float(ok[~near].mean()) if (~near).any() else None}


# --------------------------------------------------------------------------------------------- main
def main():
    import lemon_data as D
    import lemon_paths as P
    from train import report

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["qupath", "fusion", "lemon"], default="qupath",
                    help="lemon = embeddings only (--lemon file: LEMON or Nuc2Vec)")
    ap.add_argument("--subtypes", nargs="+", default=["Ki67"])
    ap.add_argument("--classes", nargs="+", default=["Tumor", "Non Tumor"])
    ap.add_argument("--feature_set", default="all_derived", choices=sorted(D.FEATURE_SETS))
    ap.add_argument("--lemon", default=P.LEMON_NPZ)
    ap.add_argument("--lemon_pca", type=int, default=0)
    ap.add_argument("--set_size", type=int, default=48, help="max nuclei per group (1 = every nucleus alone)")
    ap.add_argument("--shuffle_sets", action="store_true", help="control: random same-slide groups")
    ap.add_argument("--tta", type=int, default=1, help="average predictions over this many rotated partitions at test")
    ap.add_argument("--strata_size", type=int, default=48,
                    help="mixed-label vs single-class accuracy is always computed on this spatial partition, "
                         "so every model (including --set_size 1 and --shuffle_sets) is stratified the same way")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_sets", type=int, default=64, help="groups per optimisation step")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--reweight_power", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.1, help="dropout inside the attention blocks")
    ap.add_argument("--slide_norm", choices=["none", "center", "z"], default="none",
                    help="per-slide centring (center) or centring+scaling (z) of the globally z-scored QuPath features")
    ap.add_argument("--slide_norm_stats", choices=["scored", "all"], default="all",
                    help="per-slide statistics from every detection in the slide's QuPath export (all) or the scored nuclei")
    ap.add_argument("--cv_folds", type=int, default=0,
                    help="tuning: split TRAINING slides into this many folds (fixed assignment, same for every config)")
    ap.add_argument("--fold", type=int, default=0, help="which training-slide fold to hold out (0 .. cv_folds-1)")
    ap.add_argument("--skip_test", action="store_true", help="tuning: do not score the reference / labelled-binary slides")
    ap.add_argument("--monitor_every", type=int, default=10, help="print holdout accuracy every N epochs (with --cv_folds)")
    ap.add_argument("--mpp", type=float, default=0.25, help="µm per level-0 pixel, for the near-other-class metric")
    ap.add_argument("--near_um", type=float, default=20.0, help="'next to another class' threshold for that metric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    ap.add_argument("--cache_dir", default=P.CACHE_DIR)
    ap.add_argument("--no_labelled", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = list(a.classes); c2i = {c: i for i, c in enumerate(classes)}
    print(f"input={a.input}  subtypes={a.subtypes}  set_size={a.set_size}  shuffle_sets={a.shuffle_sets}  "
          f"tta={a.tta}  slide_norm={a.slide_norm}/{a.slide_norm_stats}  seed={a.seed}  device={dev}")

    tr = D.label_and_filter(D.load_training(a.subtypes, a.cache_dir), classes, "train")
    te = D.label_and_filter(D.load_reference(a.subtypes, a.cache_dir), classes, "reference")
    tl = None
    if not a.no_labelled:
        lb = D.load_reference_labelled(a.subtypes, a.cache_dir)
        if lb is not None:
            tl = D.label_and_filter(lb, classes, "labelled test")
    if tl is None or tl.empty:
        tl = te.iloc[0:0].copy()
    overlap = set(tr["wsi_name"]) & (set(te["wsi_name"]) | set(tl["wsi_name"]))
    if overlap:
        raise SystemExit(f"reference slides also in training: {sorted(overlap)[:5]}")
    va = tr.iloc[0:0].copy()
    if a.cv_folds:
        slides = sorted(tr["slide_uid"].unique())
        np.random.RandomState(0).shuffle(slides)                 # seed 0: same folds for every config and run seed
        held = [s for i, s in enumerate(slides) if i % a.cv_folds == a.fold]
        m = tr["slide_uid"].isin(held).to_numpy()
        va, tr = tr[m].reset_index(drop=True), tr[~m].reset_index(drop=True)
        print(f"  CV fold {a.fold}/{a.cv_folds}: holding out {len(held)} training slides: {held}")
    print(f"training: {len(tr)} nuclei | {tr['slide_uid'].nunique()} slides   holdout: {len(va)}   "
          f"reference: {len(te)}   labelled-binary: {len(tl)}")

    # ------------------------------------------------------------- inputs (statistics from training rows only)
    cols = [c for c in D.FEATURE_SETS[a.feature_set]
            if c in tr.columns and pd.api.types.is_numeric_dtype(tr[c]) and tr[c].notna().any()]
    med = tr[cols].median()
    raw = {k: d.reindex(columns=cols).astype(np.float64).fillna(med).to_numpy() for k, d in (("tr", tr), ("va", va), ("te", te), ("tl", tl))}
    mu, sd = raw["tr"].mean(0), raw["tr"].std(0); sd[sd == 0] = 1
    X = {k: np.clip((v - mu) / sd, -20, 20) for k, v in raw.items()}
    if a.slide_norm != "none" and a.input != "lemon":
        import slide_stats as SS
        fr_ = {"tr": tr, "va": va, "te": te, "tl": tl}
        ws = (SS.zscored_whole_slide_stats(list(fr_.values()), cols, a.subtypes, a.cache_dir, mu, sd, med.to_numpy())
              if a.slide_norm_stats == "all" else None)
        X = {k: np.clip(SS.apply_slide_norm(v, fr_[k], a.slide_norm, ws), -20, 20) for k, v in X.items()}
        print(f"  slide_norm={a.slide_norm} ({a.slide_norm_stats}) on the QuPath features")
    if a.input in ("fusion", "lemon"):
        idx = D.EmbeddingIndex(a.lemon)
        E = {k: idx.lookup(d, k).astype(np.float64) if len(d) else np.zeros((0, idx.dim)) for k, d in (("tr", tr), ("va", va), ("te", te), ("tl", tl))}
        if a.lemon_pca:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=a.lemon_pca, random_state=a.seed).fit(E["tr"])
            E = {k: pca.transform(v) if len(v) else np.zeros((0, a.lemon_pca)) for k, v in E.items()}
        emu, esd = E["tr"].mean(0), E["tr"].std(0); esd[esd == 0] = 1
        Eb = {k: np.clip((E[k] - emu) / esd, -20, 20) for k in E}
        X = Eb if a.input == "lemon" else {k: np.concatenate([X[k], Eb[k]], 1) for k in X}
    print(f"  input columns: {X['tr'].shape[1]}")

    frames = {"tr": tr, "va": va, "te": te, "tl": tl}
    XY = {k: (d[["cx_wsi", "cy_wsi"]].to_numpy(float) if len(d) else np.zeros((0, 2))) for k, d in frames.items()}
    SL = {k: d["slide_uid"].to_numpy() for k, d in frames.items()}
    Y = {k: d["_label"].map(c2i).to_numpy() for k, d in frames.items()}
    XT = {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in X.items()}
    YT = torch.as_tensor(Y["tr"], dtype=torch.long, device=dev)

    # ------------------------------------------------------------- train
    model = NucleusSetTransformer(X["tr"].shape[1], len(classes), a.hidden, a.heads, a.layers, a.dropout).to(dev)
    print(f"NucleusSetTransformer hidden={a.hidden} layers={a.layers} heads={a.heads} dropout={a.dropout}: "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    cnt = Counter(Y["tr"].tolist())
    cw = torch.as_tensor([1.0 / cnt[i] ** a.reweight_power for i in range(len(classes))], dtype=torch.float32, device=dev)
    cw = cw / cw.mean()
    crit = nn.CrossEntropyLoss(weight=cw, ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1.0, (e + 1) / max(1, a.warmup_epochs)) * 0.5 * (1 + math.cos(math.pi * e / a.epochs)))
    rng = np.random.RandomState(a.seed)

    for ep in range(1, a.epochs + 1):
        model.train()
        groups = make_groups(XY["tr"], SL["tr"], a.set_size, rng.uniform(0, 2 * math.pi), rng, a.shuffle_sets)
        order = rng.permutation(len(groups))
        tot, nb = 0.0, 0
        for i in range(0, len(order), a.batch_sets):
            idx = torch.as_tensor(pad_batch([groups[j] for j in order[i:i + a.batch_sets]], a.set_size), device=dev)
            valid = idx >= 0
            y = YT[idx.clamp(min=0)].masked_fill(~valid, -100)
            opt.zero_grad()
            logit = model(XT["tr"][idx.clamp(min=0)] * valid[..., None], valid)
            loss = crit(logit.reshape(-1, logit.shape[-1]), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        msg = f"  epoch {ep:03d}/{a.epochs}  train_loss={tot / max(nb, 1):.4f}  sets/epoch={len(groups)}  lr={opt.param_groups[0]['lr']:.2e}"
        if len(va) and (ep == 1 or ep % a.monitor_every == 0 or ep == a.epochs):
            pv = predict_probs(model, XT["va"], XY["va"], SL["va"], a.set_size, 1, a.seed, a.shuffle_sets, dev).argmax(1).cpu().numpy()
            ok = pd.Series(pv == Y["va"])
            msg += f"  holdout_acc={ok.mean():.4f}  holdout_slide_mean={ok.groupby(SL['va']).mean().mean():.4f}"
        print(msg, flush=True)

    # ------------------------------------------------------------- evaluate (final epoch, never selected on test)
    def evaluate(split, title):
        if not len(frames[split]):
            return {}, {}, {}
        p = predict_probs(model, XT[split], XY[split], SL[split], a.set_size, a.tta, a.seed, a.shuffle_sets, dev)
        pred = p.argmax(1).cpu().numpy()
        res, per = report(title, Y[split], pred, classes, frames[split]["subtype"].to_numpy(), SL[split])
        sgroups = make_groups(XY[split], SL[split], a.strata_size, 0.0, np.random.RandomState(0), False)
        ms = mixed_set_stats(Y[split], pred, sgroups)
        print(f"  spatial {a.strata_size}-groups (same for every model): {ms['frac_nuclei_in_mixed_sets']:.1%} of nuclei in "
              f"mixed-label groups; acc mixed={ms['acc_mixed_sets']}  acc pure={ms['acc_pure_sets']}")
        nb_ = near_other_class_stats(XY[split], Y[split], SL[split], pred, a.mpp, a.near_um)
        print(f"  next to another annotated class (<{a.near_um:g} µm): {nb_['frac_near_other_class']:.1%} of nuclei; "
              f"acc near={nb_['acc_near_other_class']}  acc far={nb_['acc_far_from_other_class']}")
        ms = {**ms, **nb_}
        return res, per, ms

    res_v, per_v, ms_v = evaluate("va", f"CV HOLDOUT (fold {a.fold}/{a.cv_folds})") if len(va) else ({}, {}, {})
    res, per, ms, res_l, per_l, ms_l = {}, {}, {}, {}, {}, {}
    if not a.skip_test:
        res, per, ms = evaluate("te", "EXTERNAL TEST")
        res_l, per_l, ms_l = evaluate("tl", "EXTERNAL TEST (labelled-binary)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        torch.save({"model_state": model.state_dict(), "args": vars(a), "classes": classes, "qupath_cols": cols,
                    "qupath_medians": med.to_numpy(), "qupath_mu": mu, "qupath_sd": sd}, a.out)
    print("RESULT " + json.dumps({"model": "partition_set", "input": a.input, "set_size": a.set_size,
                                  "shuffle_sets": a.shuffle_sets, "tta": a.tta, "seed": a.seed,
                                  "slide_norm": a.slide_norm, "slide_norm_stats": a.slide_norm_stats,
                                  "subtypes": a.subtypes, "lemon_pca": a.lemon_pca,
                                  "hidden": a.hidden, "layers": a.layers, "dropout": a.dropout,
                                  "weight_decay": a.weight_decay, "lr": a.lr, "epochs": a.epochs,
                                  "cv_folds": a.cv_folds, "fold": a.fold,
                                  "holdout_slide_mean": res_v, "holdout_per_slide": per_v, "holdout_mixed": ms_v,
                                  "n_params": int(sum(p.numel() for p in model.parameters())),
                                  "slide_mean": res, "per_slide": per, "mixed": ms,
                                  "slide_mean_labelled": res_l, "per_slide_labelled": per_l, "mixed_labelled": ms_l,
                                  "out": a.out}))


if __name__ == "__main__":
    main()
