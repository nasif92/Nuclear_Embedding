"""Step 3: embed every nucleus listed in the LEMON manifest with a frozen Nuc2Vec encoder (inference only, no
augmentation: the central 128 px of the same 0.25 um/px patch used in pretraining). Output has the same layout as the
LEMON file (features, key), so train.py / train_partition.py read it with --lemon.

    python nuc2vec/embed.py --encoder nuc2vec/runs/pooled/encoder.pt --out embeddings/nuc2vec_pooled.npz
    python nuc2vec/embed.py --encoder nuc2vec/runs/Ki67/encoder.pt --subtypes Ki67 --out embeddings/nuc2vec_Ki67.npz

The manifest (data/nuclei_manifest.csv from prepare_manifest.py) must include the labelled-binary nuclei if you want
them scored; rebuild it WITHOUT --no_labelled first. Resumable per slide.
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd
import torch

import common as C
import lemon_paths as P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--manifest", default=P.MANIFEST_CSV)
    ap.add_argument("--out", required=True)
    ap.add_argument("--subtypes", nargs="+", default=None)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--wsi_mpp", type=float, default=None)
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(a.encoder, map_location="cpu")
    model = C.Nuc2Vec(ck["dim"], "none"); model.load_state_dict(ck["model"]); model = model.to(dev).eval()
    df = pd.read_csv(a.manifest, dtype={"key": str})
    if a.subtypes:
        df = df[df["subtype"].isin(a.subtypes)]
    print(f"{len(df):,} nuclei, {df['wsi_path'].nunique()} slides; splits: {df['split'].value_counts().to_dict()}")
    tag = hashlib.md5(open(a.encoder, "rb").read(1 << 20)).hexdigest()[:8]
    shard_dir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "n2v_shards_" + tag)
    os.makedirs(shard_dir, exist_ok=True)
    feats, keys = [], []
    groups = list(df.groupby("wsi_path", sort=False))
    for n, (wsi, g) in enumerate(groups, 1):
        sp = os.path.join(shard_dir, os.path.splitext(os.path.basename(wsi))[0] + "_" + hashlib.md5("|".join(g["key"]).encode()).hexdigest()[:10] + ".npz")
        if not os.path.exists(sp):
            f = np.empty((len(g), ck["dim"]), np.float32)
            xy = g[["x", "y"]].to_numpy()
            for i in range(0, len(g), a.batch_size):
                p = C.read_patches(wsi, xy[i:i + a.batch_size], mpp=a.wsi_mpp, workers=a.workers)
                x = torch.from_numpy(np.ascontiguousarray(C.center_crop_u8(p))).to(dev).permute(0, 3, 1, 2).float().div(255)
                with torch.inference_mode():
                    f[i:i + len(p)] = model(C.normalise(x)).cpu().numpy()
            np.savez(sp + ".tmp.npz", features=f, key=g["key"].to_numpy().astype(str)); os.replace(sp + ".tmp.npz", sp)
            print(f"[{n}/{len(groups)}] {os.path.basename(wsi)}: {len(g):,} nuclei")
        else:
            print(f"[{n}/{len(groups)}] {os.path.basename(wsi)}: done, skipping")
        with np.load(sp, allow_pickle=False) as z:
            feats.append(z["features"]); keys.append(z["key"])
    feats, keys = np.concatenate(feats), np.concatenate(keys)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, features=feats, key=keys, arch="nuc2vec-resnet34", crop_um=C.CROP_PX * C.MPP_TARGET, output_px=C.CROP_PX)
    print(f"Saved {feats.shape[0]:,} x {feats.shape[1]} embeddings -> {a.out}")


if __name__ == "__main__":
    main()
