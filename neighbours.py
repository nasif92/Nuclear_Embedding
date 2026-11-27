"""Paper-style analysis (Nuc2Vec Fig 2): nearest neighbours of a nucleus under different representations, plus numbers.

    python nuc2vec/neighbours.py --subtype Ki67 --emb lemon=embeddings/lemon_vits8.npz n2v=embeddings/nuc2vec_pooled.npz \
        --out nuc2vec/figs/neighbours_Ki67.png

Pool: annotated TRAINING nuclei of one stain (data/nuclei_manifest.csv, split=train), --pool random ones. Rows of the
figure: QuPath handcrafted features (z-scored, Euclidean) and each embedding. Each tile shows the neighbour's label
(T = Tumor, N = Non Tumor). The paper compares by eye; this also prints, over --n_eval random queries:
    same-label@k  fraction of the k neighbours with the query's label (pathologist region labels)
    same-slide@k  fraction of the k neighbours from the query's own slide (high = the representation clusters by slide)
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

import common as C
import lemon_data as D
import lemon_paths as P


def qupath_matrix(m, subtype, cache_dir, feature_set="all_derived"):
    tr = D.label_and_filter(D.load_training([subtype], cache_dir), ["Tumor", "Non Tumor"], "train")
    tr["key"] = D.nucleus_keys(tr).to_numpy()
    cols = [c for c in D.FEATURE_SETS[feature_set] if c in tr.columns and tr[c].notna().any()]
    t = tr.set_index("key").reindex(m["key"])
    X = t[cols].astype(float)
    X = X.fillna(X.median())
    return ((X - X.mean()) / X.std().replace(0, 1)).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=P.MANIFEST_CSV)
    ap.add_argument("--subtype", required=True)
    ap.add_argument("--emb", nargs="+", required=True, help="name=path.npz ...")
    ap.add_argument("--pool", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default=P.CACHE_DIR)
    ap.add_argument("--wsi_mpp", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, dtype={"key": str})
    m = m[(m["split"] == "train") & (m["subtype"] == a.subtype)]
    if len(m) > a.pool:
        m = m.sample(a.pool, random_state=a.seed)
    m = m.reset_index(drop=True)
    reps = {"QuPath": qupath_matrix(m, a.subtype, a.cache_dir)}
    for item in a.emb:
        name, path = item.split("=", 1)
        reps[name] = D.EmbeddingIndex(path).lookup(m, name).astype(np.float64)
    lab = m["label"].to_numpy(); slide = m["wsi_name"].to_numpy()
    rng = np.random.RandomState(a.seed)
    qi = rng.choice(len(m), min(a.n_eval, len(m)), replace=False)
    nn_idx = {}
    print(f"{a.subtype}: pool {len(m):,} annotated training nuclei, {m['wsi_name'].nunique()} slides; k={a.k}")
    print(f"{'representation':<12}{'dim':>5}{'same-label@k':>14}{'same-slide@k':>14}")
    for name, X in reps.items():
        nn = NearestNeighbors(n_neighbors=a.k + 1).fit(X)
        idx = nn.kneighbors(X[qi], return_distance=False)[:, 1:]
        nn_idx[name] = (qi, idx)
        print(f"{name:<12}{X.shape[1]:>5}{(lab[idx] == lab[qi][:, None]).mean():>14.3f}{(slide[idx] == slide[qi][:, None]).mean():>14.3f}")
    # figure: the first --queries queries; one row block per representation
    tile, qn = 72, a.queries
    cache = {}

    def patch(i):
        if i not in cache:
            r = m.iloc[i]
            p = C.read_patches(r["wsi_path"], np.array([[r["x"], r["y"]]]), mpp=a.wsi_mpp, workers=1)
            cache[i] = Image.fromarray(C.center_crop_u8(p)[0]).resize((tile, tile))
        return cache[i]

    names = list(reps)
    cols = a.k + 1
    W, H = qn * (cols * tile + 10), len(names) * (tile + 4) * 1 + 14
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)
    for j in range(qn):
        x0 = j * (cols * tile + 10)
        d.text((x0 + 2, 1), f"query {j + 1}", fill="black")
        for r, name in enumerate(names):
            qidx, idx = nn_idx[name]
            y0 = 14 + r * (tile + 4)
            for c, i in enumerate([qidx[j]] + list(idx[j])):
                fig.paste(patch(i), (x0 + c * tile, y0))
                d.text((x0 + c * tile + 2, y0 + 1), ("[" if c == 0 else "") + ("T" if lab[i] == "Tumor" else "N") + ("]" if c == 0 else ""), fill="red")
            if j == 0:
                d.text((x0 + 2, y0 + tile - 11), name, fill="blue")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.save(a.out)
    print(f"saved {a.out}  (rows: {', '.join(names)}; first tile of each block = the query)")


if __name__ == "__main__":
    main()
