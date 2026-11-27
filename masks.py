"""Optional step 1b: per-nucleus MASKS for 'background replacement' (instead of the square box).

The detection GeoJSONs hold every nucleus outline. For each cached patch this rasterises that nucleus's polygon into the
patch's pixel grid (same centre, same 0.25 um/px scale as prepare_pretrain.py) and stores it bit-packed next to the shard:

    python nuc2vec/masks.py --subtypes Ki67 ER PR
    -> nuc2vec/data/<stain>/shards/<slide>.mask.npy   (N, 3872) uint8 = 176*176 bits per nucleus; resumable

Pretraining then uses it with  pretrain.py ... --bg_mode mask  (see common.Augment). No WSI pixels are read here, only the
detection GeoJSON of each slide, and nothing from the reference slides.

Safety check: the polygon at det_index must sit on the stored centroid; a slide whose polygons do not line up is skipped
(wrong file or index convention) instead of writing masks for the wrong nuclei.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import common as C
import lemon_paths as P

S = C.PATCH_PX


def _rings(geom):
    if not geom:
        return []
    t, c = geom.get("type"), geom.get("coordinates")
    if t == "Polygon":
        return [c[0]]
    if t == "MultiPolygon":
        return [p[0] for p in c]
    return []


def feature_rings(f):
    g = f.get("geometry")
    r = _rings(g)
    if not r and f.get("nucleusGeometry"):
        r = _rings(f["nucleusGeometry"])
    return r


def rasterise(feats, det_index, xy, w, h, diam_px, size=S):
    """feats: list of GeoJSON features; det_index/xy/diam_px: per-patch arrays. w, h: patch size in level-0 pixels.
    Returns (bool masks (N,size,size), n_fallback, median centre error in patch px)."""
    n = len(det_index)
    out = np.zeros((n, size, size), bool)
    fallback, err = 0, []
    for k in range(n):
        x0, y0 = xy[k][0] - w / 2.0, xy[k][1] - h / 2.0
        rings = feature_rings(feats[int(det_index[k])])
        im = Image.new("1", (size, size), 0)
        dr = ImageDraw.Draw(im)
        ok = False
        allp = []
        for ring in rings:
            pts = [((px - x0) * size / w, (py - y0) * size / h) for px, py in ring]
            if len(pts) >= 3:
                dr.polygon(pts, outline=1, fill=1)
                ok = True
                allp += pts
        if ok:
            a = np.asarray(allp)
            err.append(float(np.hypot(*(0.5 * (a.min(0) + a.max(0)) - size / 2.0))))
            m = np.asarray(im, bool)
        else:
            m = None
        if m is None or m.sum() < 4:
            yy, xx = np.mgrid[:size, :size]
            r = max(diam_px[k] / 2.0, 3.0)
            m = (yy - (size - 1) / 2.0) ** 2 + (xx - (size - 1) / 2.0) ** 2 <= r * r
            fallback += 1
        out[k] = m
    return out, fallback, float(np.median(err)) if err else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subtypes", nargs="+", default=["Ki67", "ER", "PR"])
    ap.add_argument("--data_dir", default=os.path.join(C.HERE, "data"))
    ap.add_argument("--wsi_mpp", type=float, default=None, help="override um/px when a WSI has no MPP metadata")
    ap.add_argument("--manifest", default=P.MANIFEST_CSV)
    ap.add_argument("--wsi_map", default=None)
    ap.add_argument("--max_centre_err", type=float, default=0.5, help="median polygon-centre error allowed, in nucleus diameters")
    a = ap.parse_args()

    import nuclei_io as NI
    import prepare_pretrain as PP
    for st in a.subtypes:
        root = os.path.join(a.data_dir, st)
        man = pd.read_csv(os.path.join(root, "manifest.csv"))
        files = {rec["slide_name"]: rec["feat_path"]
                 for rec in NI.iter_slides(P.TRAIN_FEAT_DIR, None, [st], P.TRAIN_FEAT_PATTERN, use_classification=True)}
        slides = sorted(man["wsi_name"].astype(str).unique())
        paths = PP.wsi_lookup(slides, a.manifest, a.wsi_map, P.WSI_ROOTS)
        print(f"\n{st}: masks for {len(slides)} slides")
        for n, name in enumerate(slides, 1):
            outp = os.path.join(root, "shards", f"{name}.mask.npy")
            if os.path.exists(outp):
                print(f"[{n}/{len(slides)}] {name}: done, skipping"); continue
            g = man[man["wsi_name"].astype(str) == name].sort_values("row")
            fp = files.get(name) or next((v for k, v in files.items() if k.startswith(name) or name.startswith(k)), None)
            if fp is None:
                print(f"[{n}/{len(slides)}] {name}: no detection file found - skipped"); continue
            mx, my = C.slide_mpp(C.open_slide(paths[name]), a.wsi_mpp)
            w, h = max(1, round(S * C.MPP_TARGET / mx)), max(1, round(S * C.MPP_TARGET / my))
            feats = NI.load_geojson_detections(fp)
            masks, nfb, err = rasterise(feats, g["det_index"].to_numpy(), g[["x", "y"]].to_numpy(), w, h, g["diam_px"].to_numpy())
            del feats
            med_d = float(g["diam_px"].median())
            if not np.isfinite(err) or err > a.max_centre_err * med_d:
                print(f"[{n}/{len(slides)}] {name}: polygons do not line up with the stored centroids "
                      f"(median error {err:.1f} px vs nucleus {med_d:.0f} px) - skipped, check det_index / file"); continue
            packed = np.packbits(masks.reshape(len(masks), -1), axis=1)
            np.save(outp + ".tmp.npy", packed); os.replace(outp + ".tmp.npy", outp)
            print(f"[{n}/{len(slides)}] {name}: {len(g):,} masks, {nfb} disc fallbacks, centre error {err:.2f} px, "
                  f"mean mask area {masks.sum((1, 2)).mean():.0f} px")
    print("\nNext: python nuc2vec/pretrain.py --stains Ki67 ER PR --name pooled_mask --bg_mode mask --check_only")


if __name__ == "__main__":
    main()