"""Paper-style analysis (Nuc2Vec Figs 3-4): cluster the learned nucleus embeddings and look at the clusters.

    python nuc2vec/cluster.py --encoder nuc2vec/runs/pooled/encoder.pt --stains Ki67 ER PR --out nuc2vec/figs/pooled

What it does (paper section 2.4, scaled to a sample so Ward linkage fits in memory):
  1. embed --n_cluster random nuclei of the pretraining sample (central 128 px of the cached patches, no augmentation)
  2. Ward hierarchical clustering of the embeddings (dendrogram cut into --branches branches, like the paper's 11)
  3. flat clusters with HDBSCAN's excess-of-mass selection (sklearn.cluster.HDBSCAN, --min_cluster_size)
  4. for the largest clusters, a montage of random member nuclei; for the clusters with the LOWEST slide entropy, the
     slide composition next to a montage (paper Fig 4 used cancer type; here a low-entropy cluster means the model groups
     nuclei by slide - a stain/scanner shortcut)
  5. clusters.csv with size, slide entropy, stain shares and, for the annotated nuclei in the sample, the Tumor fraction.
     Labels are used ONLY here, for evaluation; the encoder never sees them.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

import common as C


def load_sample(stains, data_dir, n, seed):
    frames = []
    for st in stains:
        m = pd.read_csv(os.path.join(data_dir, st, "manifest.csv"))
        m["root"] = os.path.join(data_dir, st)
        frames.append(m)
    m = pd.concat(frames, ignore_index=True)
    if len(m) > n:
        m = m.sample(n, random_state=seed).reset_index(drop=True)
    return m


def load_patches(m):
    """uint8 (N,176,176,3) for the manifest rows, from the cached shards."""
    out = np.empty((len(m), C.PATCH_PX, C.PATCH_PX, 3), np.uint8)
    for (root, shard), g in m.groupby(["root", "shard"]):
        a = np.load(os.path.join(root, shard), mmap_mode="r")
        out[g.index.to_numpy()] = a[g["row"].to_numpy()]
    return out


def embed(model, patches, dev, bs=512):
    z = np.empty((len(patches), model.dim), np.float32)
    for i in range(0, len(patches), bs):
        x = torch.from_numpy(np.ascontiguousarray(C.center_crop_u8(patches[i:i + bs]))).to(dev).permute(0, 3, 1, 2).float().div(255)
        with torch.inference_mode():
            z[i:i + bs] = model(C.normalise(x)).cpu().numpy()
    return z


def entropy_bits(counts):
    p = np.asarray(counts, float); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def draw_panel(patches, idx, title, cols=5, rows=4, tile=80, rng=None):
    W, H = cols * tile, rows * tile + 16
    im = Image.new("RGB", (W, H), "white")
    ImageDraw.Draw(im).text((2, 2), title, fill="black")
    pick = rng.choice(idx, size=min(len(idx), cols * rows), replace=False)
    for k, i in enumerate(pick):
        t = Image.fromarray(C.center_crop_u8(patches[i:i + 1])[0]).resize((tile, tile), Image.Resampling.BILINEAR)
        im.paste(t, ((k % cols) * tile, 16 + (k // cols) * tile))
    return im


def page(panels, per_row=3):
    pw, ph = panels[0].size
    rows = (len(panels) + per_row - 1) // per_row
    pg = Image.new("RGB", (per_row * (pw + 8), rows * (ph + 8)), "white")
    for k, p in enumerate(panels):
        pg.paste(p, ((k % per_row) * (pw + 8), (k // per_row) * (ph + 8)))
    return pg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--stains", nargs="+", required=True)
    ap.add_argument("--data_dir", default=os.path.join(C.HERE, "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cluster", type=int, default=20000, help="nuclei to cluster (Ward needs O(n^2) memory: 20k = 1.6 GB)")
    ap.add_argument("--min_cluster_size", type=int, default=0, help="HDBSCAN; default 0.4%% of --n_cluster (paper: 400 of 100k)")
    ap.add_argument("--branches", type=int, default=11)
    ap.add_argument("--top", type=int, default=12, help="largest clusters to show")
    ap.add_argument("--low_entropy", type=int, default=6, help="slide-specific clusters to show")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(a.encoder, map_location="cpu")
    model = C.Nuc2Vec(ck["dim"], "none"); model.load_state_dict(ck["model"]); model = model.to(dev).eval()

    m = load_sample(a.stains, a.data_dir, a.n_cluster, a.seed)
    print(f"{len(m):,} nuclei, {m['wsi_name'].nunique()} slides, stains {a.stains}")
    patches = load_patches(m)
    Z = embed(model, patches, dev)

    from scipy.cluster import hierarchy as H
    try:
        import fastcluster
        L = fastcluster.linkage(Z, method="ward")
    except ImportError:
        print("  (fastcluster not installed: using scipy; pip install fastcluster is faster)")
        L = H.linkage(Z, method="ward")
    branch = H.fcluster(L, t=a.branches, criterion="maxclust")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    H.dendrogram(L, p=60, truncate_mode="lastp", no_labels=True, color_threshold=L[-(a.branches - 1), 2], ax=ax)
    ax.set_title(f"Ward dendrogram of Nuc2Vec embeddings ({len(m):,} nuclei, {a.branches} branches)")
    fig.savefig(os.path.join(a.out, "dendrogram.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

    from sklearn.cluster import HDBSCAN
    mcs = a.min_cluster_size or max(30, int(0.004 * len(m)))
    lab = HDBSCAN(min_cluster_size=mcs, cluster_selection_method="eom").fit_predict(Z)
    m["cluster"], m["branch"] = lab, branch
    ncl = int((np.unique(lab) >= 0).sum())
    print(f"HDBSCAN (min_cluster_size {mcs}): {ncl} clusters, {(lab < 0).mean():.1%} of nuclei unclustered")

    # --- Ward branches: every nucleus belongs to one, so these are not limited to HDBSCAN's dense cores ---
    if "label" in m:
        ann_all = m["label"].isin(["Tumor", "Non Tumor"])
        if ann_all.sum() > 20:
            from sklearn.metrics import normalized_mutual_info_score as _nmi
            la = m.loc[ann_all, "label"]
            base = la.value_counts(normalize=True).iloc[0]
            pur = m[ann_all].groupby("branch")["label"].agg(lambda x: x.value_counts().iloc[0]).sum() / ann_all.sum()
            print(f"Ward branches ({a.branches}): {int(ann_all.sum()):,} annotated nuclei; label purity {pur:.3f} "
                  f"(majority-class baseline {base:.3f}); NMI(branch, label) {_nmi(la, m.loc[ann_all, 'branch']):.3f}")
    bt = []
    for b in sorted(m["branch"].unique()):
        g = m[m["branch"] == b]
        cnt = g["wsi_name"].value_counts()
        an = g[g["label"].isin(["Tumor", "Non Tumor"])] if "label" in g else g.iloc[0:0]
        bt.append({"branch": int(b), "size": len(g), "n_slides": len(cnt), "top_slide_share": cnt.iloc[0] / len(g),
                   "slide_entropy_norm": entropy_bits(cnt.values) / max(np.log2(m["wsi_name"].nunique()), 1e-9),
                   **{f"share_{x}": float((g["subtype"] == x).mean()) for x in a.stains},
                   "n_annotated": len(an), "frac_tumor": float((an["label"] == "Tumor").mean()) if len(an) else np.nan})
    bt = pd.DataFrame(bt)
    bt.to_csv(os.path.join(a.out, "branches.csv"), index=False)
    print("\nWard branches (all nuclei):")
    print(bt.round(3).to_string(index=False))
    pan = [draw_panel(patches, np.flatnonzero(branch == r.branch),
                      f"branch {r.branch} n={r.size} Tumor {r.frac_tumor:.0%} (annot {r.n_annotated})", rng=rng) for r in bt.itertuples()]
    page(pan, per_row=4).save(os.path.join(a.out, "branches.png"))

    slides = sorted(m["wsi_name"].unique())
    rows = []
    for c in sorted(set(lab) - {-1}):
        g = m[m["cluster"] == c]
        cnt = g["wsi_name"].value_counts()
        ann = g["label"].isin(["Tumor", "Non Tumor"]) if "label" in g else pd.Series(False, index=g.index)
        rows.append({"cluster": c, "size": len(g), "branch": int(g["branch"].mode().iat[0]), "n_slides": len(cnt),
                     "top_slide": cnt.index[0], "top_slide_share": cnt.iloc[0] / len(g),
                     "slide_entropy_norm": entropy_bits(cnt.values) / max(np.log2(len(slides)), 1e-9),
                     **{f"share_{s}": float((g["subtype"] == s).mean()) for s in a.stains},
                     "n_annotated": int(ann.sum()), "frac_tumor": float((g.loc[ann, "label"] == "Tumor").mean()) if ann.sum() else np.nan})
    cl = pd.DataFrame(rows)
    cl.to_csv(os.path.join(a.out, "clusters.csv"), index=False)
    if len(cl):
        lbl = m[m["cluster"] >= 0]
        lbl = lbl[lbl["label"].isin(["Tumor", "Non Tumor"])] if "label" in lbl else lbl.iloc[0:0]
        if len(lbl) > 20:
            from sklearn.metrics import normalized_mutual_info_score
            pur = lbl.groupby("cluster")["label"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(lbl)
            print(f"annotated nuclei in clusters: {len(lbl):,}; label purity {pur:.3f} (majority-class baseline {lbl['label'].value_counts(normalize=True).iloc[0]:.3f}); NMI(cluster, label) "
                  f"{normalized_mutual_info_score(lbl['label'], lbl['cluster']):.3f}")
        print(f"slide entropy (1 = spread over all slides, 0 = one slide): median {cl['slide_entropy_norm'].median():.2f}, "
              f"clusters with > 50% from one slide: {(cl['top_slide_share'] > 0.5).sum()} of {len(cl)}")

        big = cl.sort_values("size", ascending=False).head(a.top)
        panels = [draw_panel(patches, np.flatnonzero(lab == r.cluster),
                             f"cluster {r.cluster}  n={r.size}  branch {r.branch}  slides={r.n_slides}", rng=rng) for r in big.itertuples()]
        page(panels).save(os.path.join(a.out, "clusters_largest.png"))
        low = cl[cl["size"] >= mcs].sort_values("slide_entropy_norm").head(a.low_entropy)
        panels = []
        for r in low.itertuples():
            g = m[m["cluster"] == r.cluster]["wsi_name"].value_counts().head(5)
            t = f"cluster {r.cluster} n={r.size}  slide entropy {r.slide_entropy_norm:.2f}\n" + "  ".join(f"{k[:10]} {v / r.size:.0%}" for k, v in g.items())
            p = draw_panel(patches, np.flatnonzero(lab == r.cluster), t.split("\n")[0], rng=rng)
            ImageDraw.Draw(p).text((2, p.size[1] - 12), t.split("\n")[1][:70], fill="black")
            panels.append(p)
        if panels:
            page(panels).save(os.path.join(a.out, "clusters_lowest_slide_entropy.png"))
    m[["key", "subtype", "wsi_name", "cluster", "branch"]].to_csv(os.path.join(a.out, "assignments.csv"), index=False)
    print(f"wrote {a.out}/: branches.png, branches.csv, dendrogram.png, clusters_largest.png, clusters_lowest_slide_entropy.png, clusters.csv, assignments.csv")


if __name__ == "__main__":
    main()