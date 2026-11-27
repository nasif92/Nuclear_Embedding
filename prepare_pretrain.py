"""Step 1 of Nuc2Vec pretraining: sample unlabelled nuclei near the annotated regions of the TRAINING slides and cache
their patches. Labels are never used. Reference slides are never read.

    python nuc2vec/prepare_pretrain.py --subtypes Ki67 ER PR --per_slide 3500
    -> nuc2vec/data/<stain>/manifest.csv + shards/<slide>.npy      (resumable: finished slides are skipped)

Nuclei come from the same context the set models see: every detection within --context_um of an annotated nucleus
(set_transformers/train_context.py loader, cached in set_transformers/cache_ctx). Sampling is uniform over those
detections, --per_slide per slide, so one large slide cannot dominate.
"""
import argparse
import importlib.util
import os
import types

import numpy as np
import pandas as pd

import common as C
import lemon_data as D
import lemon_paths as P


def load_train_context():
    spec = importlib.util.spec_from_file_location("train_context", os.path.join(C.PARENT, "set_transformers", "train_context.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wsi_lookup(names, manifest, extra_map, roots):
    """wsi_name -> path: from the LEMON manifest when it knows the slide, else by searching the WSI roots."""
    out = {}
    if manifest and os.path.exists(manifest):
        m = pd.read_csv(manifest, usecols=["wsi_name", "wsi_path"]).drop_duplicates("wsi_name")
        out.update(dict(zip(m["wsi_name"].astype(str), m["wsi_path"].astype(str))))
    if extra_map:
        m = pd.read_csv(extra_map)
        out.update(dict(zip(m["wsi_name"].astype(str), m["wsi_path"].astype(str))))
    missing = [n for n in names if n not in out]
    if missing:
        import prepare_manifest as PM
        found = PM.index_wsis(roots)
        for n in missing:
            c = found.get(n.lower(), [])
            if len(c) >= 1:
                out[n] = c[0]
    bad = [n for n in names if n not in out]
    if bad:
        raise SystemExit(f"no WSI file for {len(bad)} slide(s), e.g. {bad[:5]}; pass --wsi_map (wsi_name,wsi_path)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtypes", nargs="+", default=["Ki67", "ER", "PR"])
    ap.add_argument("--per_slide", type=int, default=3500)
    ap.add_argument("--context_um", type=float, default=100.0)
    ap.add_argument("--mpp", type=float, default=0.25, help="fallback um/px for the context radius (as train_context.py)")
    ap.add_argument("--patch_px", type=int, default=C.PATCH_PX)
    ap.add_argument("--wsi_mpp", type=float, default=None, help="override um/px when a WSI has no MPP metadata")
    ap.add_argument("--manifest", default=P.MANIFEST_CSV, help="LEMON manifest, used only to find each slide's WSI file")
    ap.add_argument("--wsi_map", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(C.HERE, "data"))
    ap.add_argument("--ctx_cache_dir", default=os.path.join(C.PARENT, "set_transformers", "cache_ctx"))
    ap.add_argument("--cache_dir", default=P.CACHE_DIR)
    a = ap.parse_args()

    TC = load_train_context()
    classes = ["Tumor", "Non Tumor"]
    ctx = types.SimpleNamespace(context_um=a.context_um, mpp=a.mpp, ctx_cache_dir=a.ctx_cache_dir)
    ref_names = set(D.load_reference(a.subtypes, a.cache_dir)["wsi_name"].astype(str))
    for st in a.subtypes:
        df = TC.load_root_all(P.TRAIN_FEAT_DIR, P.TRAIN_ANN_DIRS, [st], P.TRAIN_FEAT_PATTERN, classes, ctx, f"nuc2vec train: {st}")
        if df is None or df.empty:
            print(f"{st}: no training detections"); continue
        df = df[df["subtype"] == st].reset_index(drop=True)
        if set(df["wsi_name"].astype(str)) & ref_names:
            raise SystemExit("a reference slide is among the training slides - refusing to pretrain on it")
        if "Max diameter µm" in df.columns:
            diam_um = df["Max diameter µm"].astype(float)
        elif "Area µm^2" in df.columns:
            diam_um = 2 * np.sqrt(df["Area µm^2"].astype(float) / np.pi)
        else:
            diam_um = pd.Series(10.0, index=df.index)
        df["diam_px"] = (diam_um.fillna(diam_um.median()).clip(2, 40) / C.MPP_TARGET).to_numpy()
        df["key"] = D.nucleus_keys(df).to_numpy()
        paths = wsi_lookup(sorted(df["wsi_name"].astype(str).unique()), a.manifest, a.wsi_map, P.WSI_ROOTS)
        root = os.path.join(a.out_dir, st)
        os.makedirs(os.path.join(root, "shards"), exist_ok=True)
        rng = np.random.RandomState(a.seed)
        parts = []
        slides = sorted(df["wsi_name"].astype(str).unique())
        print(f"\n{st}: {len(df):,} detections near annotations in {len(slides)} training slides -> up to {a.per_slide:,} each")
        for n, name in enumerate(slides, 1):
            g = df[df["wsi_name"].astype(str) == name]
            pick = rng.choice(len(g), size=min(a.per_slide, len(g)), replace=False)
            g = g.iloc[np.sort(pick)].sort_values(["cy_wsi", "cx_wsi"]).reset_index(drop=True)
            sp = os.path.join(root, "shards", f"{name}.npy")
            mp = os.path.join(root, "shards", f"{name}.csv")
            meta = pd.DataFrame({"key": g["key"], "subtype": st, "wsi_name": name, "det_index": g["det_index"].astype(int),
                                 "x": g["cx_wsi"], "y": g["cy_wsi"], "diam_px": g["diam_px"], "label": g["_label"].to_numpy(), "shard": f"shards/{name}.npy",
                                 "row": np.arange(len(g))})
            if os.path.exists(sp) and os.path.exists(mp):
                print(f"[{n}/{len(slides)}] {name}: done, skipping")
                parts.append(pd.read_csv(mp)); continue
            patches = C.read_patches(paths[name], g[["cx_wsi", "cy_wsi"]].to_numpy(), a.patch_px, mpp=a.wsi_mpp, workers=a.workers)
            np.save(sp + ".tmp.npy", patches); os.replace(sp + ".tmp.npy", sp)
            meta.to_csv(mp, index=False)
            parts.append(meta)
            print(f"[{n}/{len(slides)}] {name}: {len(g):,} patches ({patches.nbytes / 1e6:.0f} MB)")
        man = pd.concat(parts, ignore_index=True)
        man.to_csv(os.path.join(root, "manifest.csv"), index=False)
        print(f"{st}: wrote {len(man):,} patches from {man['wsi_name'].nunique()} slides -> {root}/manifest.csv")
    print("\nNext: python nuc2vec/pretrain.py --stains " + " ".join(a.subtypes) + " --name pooled   (check aug_check.png first)")


if __name__ == "__main__":
    main()
