"""Does an embedding encode WHICH SLIDE a nucleus came from? (a stain / scanner shortcut check)

    python nuc2vec/probe.py --subtype Ki67 --emb lemon=embeddings/lemon_vits8.npz n2v=embeddings/nuc2vec_pooled.npz

Uses training-slide nuclei of the manifest only, --per_slide random nuclei per slide, a 50/50 split WITHIN each slide,
and a multinomial logistic regression on z-scored embeddings. Accuracy is compared with chance (1 / number of slides).
High accuracy = the embedding carries slide identity (colour / scanner), which a classifier can exploit on training
slides but which does not transfer to new slides. Also run on the QuPath features with --qupath.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import common as C  # noqa: F401  (puts LEMON/ on sys.path)
import lemon_data as D
import lemon_paths as P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=P.MANIFEST_CSV)
    ap.add_argument("--subtype", required=True)
    ap.add_argument("--emb", nargs="+", required=True, help="name=path.npz ...")
    ap.add_argument("--per_slide", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, dtype={"key": str})
    m = m[(m["split"] == "train") & (m["subtype"] == a.subtype)]
    rng = np.random.RandomState(a.seed)
    m = m.sample(frac=1.0, random_state=a.seed).groupby("wsi_name", sort=True).head(a.per_slide).reset_index(drop=True)
    y = pd.factorize(m["wsi_name"])[0]
    test = np.zeros(len(m), bool)
    for s in np.unique(y):
        ix = np.flatnonzero(y == s); rng.shuffle(ix); test[ix[: len(ix) // 2]] = True
    print(f"{a.subtype}: {len(m):,} training nuclei, {len(np.unique(y))} slides, chance = {1 / len(np.unique(y)):.3f}")
    for item in a.emb:
        name, path = item.split("=", 1)
        X = D.EmbeddingIndex(path).lookup(m, name).astype(np.float64)
        X = (X - X[~test].mean(0)) / (X[~test].std(0) + 1e-8)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(X[~test], y[~test])
        print(f"  {name:<10} dim={X.shape[1]:<4} slide-ID accuracy = {clf.score(X[test], y[test]):.3f}")


if __name__ == "__main__":
    main()
