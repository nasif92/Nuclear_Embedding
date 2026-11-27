"""Step 2: contrastive pretraining (Nuc2Vec-style). Labels are not used anywhere.

    python nuc2vec/pretrain.py --stains Ki67 ER PR --name pooled        # one encoder for all three stains
    python nuc2vec/pretrain.py --stains Ki67 --name Ki67                # a stain-specific encoder
    python nuc2vec/pretrain.py --stains Ki67 --name test --epochs 1 --max_items 2000 --check_only

Each item: the nucleus patch plus two RANDOM patches from the same slide as backgrounds. Two independently augmented
views (each with its own pasted background) are pulled together (alignment) and all embeddings are spread over the
sphere (uniformity). A grid of augmented views is written to <out>/aug_check.png before training - look at it.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import common as C


class PatchSet(Dataset):
    def __init__(self, roots, max_items=0, seed=0, use_masks=False):
        frames = []
        for r in roots:
            m = pd.read_csv(os.path.join(r, "manifest.csv"))
            m["root"] = r
            frames.append(m)
        self.m = pd.concat(frames, ignore_index=True)
        if max_items and len(self.m) > max_items:
            self.m = self.m.sample(max_items, random_state=seed).reset_index(drop=True)
        self.use_masks = use_masks
        self.mask_arrays = {}
        self.slide_rows = {}            # (root, shard) -> row ids of the full shard
        self.arrays = {}
        self.key = (self.m["root"] + "|" + self.m["shard"]).to_numpy()
        self.row = self.m["row"].to_numpy()
        self.diam = self.m["diam_px"].to_numpy(np.float32)
        self.root = self.m["root"].to_numpy()
        self.shard = self.m["shard"].to_numpy()
        self.n_in_shard = self.m.groupby(["root", "shard"])["row"].transform("max").to_numpy() + 1

    def __len__(self):
        return len(self.m)

    def _arr(self, i):
        k = self.key[i]
        a = self.arrays.get(k)
        if a is None:
            a = self.arrays[k] = np.load(os.path.join(self.root[i], self.shard[i]), mmap_mode="r")
        return a

    def _mask(self, i):
        if not self.use_masks:
            return torch.zeros(1, 1, dtype=torch.uint8)
        k = self.key[i]
        a = self.mask_arrays.get(k)
        if a is None:
            p = os.path.join(self.root[i], self.shard[i].replace(".npy", ".mask.npy"))
            if not os.path.exists(p):
                raise SystemExit(f"--bg_mode mask but {p} is missing; run nuc2vec/masks.py first")
            a = self.mask_arrays[k] = np.load(p, mmap_mode="r")
        bits = np.unpackbits(np.asarray(a[self.row[i]]))[:C.PATCH_PX * C.PATCH_PX]
        return torch.from_numpy(bits.reshape(C.PATCH_PX, C.PATCH_PX).copy())

    def __getitem__(self, i):
        a = self._arr(i)
        rng = np.random.default_rng()
        n = len(a)
        b1, b2 = rng.integers(0, n, 2)
        return (torch.from_numpy(np.ascontiguousarray(a[self.row[i]])), torch.from_numpy(np.ascontiguousarray(a[b1])),
                torch.from_numpy(np.ascontiguousarray(a[b2])), torch.tensor(self.diam[i]), self._mask(i))


def save_grid(aug, ds, path, n=8, device="cpu"):
    from PIL import Image
    idx = np.random.RandomState(0).choice(len(ds), n, replace=False)
    x, b1, b2, d, m = [torch.stack(t) for t in zip(*[ds[int(i)] for i in idx])]
    x, b1, b2, d, m = x.to(device), b1.to(device), b2.to(device), d.to(device), m.to(device)
    v1, v2 = aug(x, b1, d, m), aug(x, b2, d, m)
    plain = C.center_crop_u8(x.cpu().numpy())
    den = lambda t: (t * torch.tensor(C.IMAGENET_STD, device=t.device).view(1, 3, 1, 1)
                     + torch.tensor(C.IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)).clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    rows = [np.concatenate(list(plain), 1), np.concatenate(list(den(v1)), 1), np.concatenate(list(den(v2)), 1)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)
    print(f"  saved {path}: row 1 = raw nuclei, rows 2-3 = two augmented views of each")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stains", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data_dir", default=os.path.join(C.HERE, "data"))
    ap.add_argument("--out_dir", default=os.path.join(C.HERE, "runs"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--uniform_weight", type=float, default=1.0)
    ap.add_argument("--no_bg_replace", action="store_true", help="ablation: no background replacement")
    ap.add_argument("--bg_mode", choices=["box", "mask"], default="box",
                    help="what is kept when the background is replaced: the square box (paper-like, default) or the nucleus polygon (needs masks.py)")
    ap.add_argument("--mask_dilate", type=int, default=2, help="--bg_mode mask: grow the polygon by this many px (0 = exact outline)")
    ap.add_argument("--p_color", type=float, default=0.5, help="probability of colour jitter (paper: 0.5)")
    ap.add_argument("--p_geo", type=float, default=0.5, help="probability of each of: rotation, flip x, flip y, random zoom (paper: 0.5)")
    ap.add_argument("--p_blur", type=float, default=0.5, help="probability of Gaussian blur (paper: 0.5)")
    ap.add_argument("--jitter", nargs=4, type=float, default=[0.4, 0.4, 0.4, 0.1], help="brightness contrast saturation hue")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max_items", type=int, default=0, help="debug: use a random subset")
    ap.add_argument("--check_only", action="store_true", help="write aug_check.png and stop")
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    ds = PatchSet([os.path.join(a.data_dir, s) for s in a.stains], a.max_items, a.seed, use_masks=a.bg_mode == "mask")
    print(f"{len(ds):,} patches | {ds.m['wsi_name'].nunique()} slides | stains {a.stains} | device {dev}")
    aug = C.Augment(bg_replace=not a.no_bg_replace, p_color=a.p_color, jitter=tuple(a.jitter), p_rot=a.p_geo, p_flip=a.p_geo,
                    p_scale=a.p_geo, p_blur=a.p_blur, bg_mode=a.bg_mode, mask_dilate=a.mask_dilate).to(dev)
    save_grid(aug, ds, os.path.join(out, "aug_check.png"), device=dev)
    if a.check_only:
        return
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                    persistent_workers=a.workers > 0, pin_memory=dev.type == "cuda")
    model = C.Nuc2Vec(a.dim, a.weights).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * len(dl)
    warm = a.warmup_epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + np.cos(np.pi * s / total)))
    amp = dev.type == "cuda"
    start = 0
    ck = os.path.join(out, "last.pt")
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); start = st["epoch"]
        print(f"  resumed from epoch {start}")
    json.dump(vars(a), open(os.path.join(out, "args.json"), "w"), indent=1)
    for ep in range(start + 1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = al = un = r1 = 0.0
        for step, (x, b1, b2, d, mk) in enumerate(dl, 1):
            x, b1, b2, d, mk = x.to(dev, non_blocking=True), b1.to(dev, non_blocking=True), b2.to(dev, non_blocking=True), d.to(dev), mk.to(dev)
            with torch.no_grad():
                v1, v2 = aug(x, b1, d, mk), aug(x, b2, d, mk)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                z = model(torch.cat([v1, v2]))
            z1, z2 = z.chunk(2)
            la = C.align_loss(z1, z2)
            lu = (C.uniform_loss(z1) + C.uniform_loss(z2)) / 2
            loss = la + a.uniform_weight * lu
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            with torch.no_grad():                       # does view 1 of nucleus i find view 2 of nucleus i in the batch?
                r1 += (z1.float() @ z2.float().T).argmax(1).eq(torch.arange(len(z1), device=dev)).float().mean().item()
            tot += loss.item(); al += la.item(); un += lu.item()
        n = len(dl)
        print(f"epoch {ep:02d}/{a.epochs}  loss={tot / n:.4f}  align={al / n:.4f}  uniform={un / n:.4f}  top1={r1 / n:.3f} (chance {1 / a.batch_size:.3f})  "
              f"lr={opt.param_groups[0]['lr']:.2e}  {len(ds) / (time.time() - t0):.0f} nuclei/s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, ck)
    torch.save({"model": model.state_dict(), "dim": a.dim, "stains": a.stains, "args": vars(a)}, os.path.join(out, "encoder.pt"))
    print(f"saved encoder -> {out}/encoder.pt\nNext: python nuc2vec/embed.py --encoder {out}/encoder.pt --out embeddings/nuc2vec_{a.name}.npz")


if __name__ == "__main__":
    main()