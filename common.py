"""Shared pieces for the Nuc2Vec-style pretraining (Feng, Vanderbilt & Fuchs, MIDL 2021, adapted to our IHC slides).

Paper recipe kept: 128x128 px nucleus patches at 40x (0.25 um/px), ResNet34 -> Linear 512 + ReLU -> 128-d embedding,
alignment + uniformity loss (Wang & Isola 2020), augmentations (random scale 0.5-1, colour jitter, Gaussian blur,
rotation, flips, each with 50 % probability) plus 'background replacement' (keep the nucleus box, paste a random
region of the same slide around it).
Deviations (stated, not hidden): the uniformity term uses the other items of the same mini-batch (no MoMo queue);
augmentations run on the GPU; the nucleus box is a square from the QuPath max diameter, not the segmentation mask (--bg_mode mask uses the
GeoJSON nucleus polygon instead; see masks.py).
"""
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)            # LEMON/
for _p in (PARENT, os.path.join(PARENT, "set_transformers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MPP_TARGET = 0.25        # um per pixel of the stored/embedded patches (40x)
PATCH_PX = 176           # stored patch; the model sees a 128 px window of it (room for rotation without empty corners)
CROP_PX = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ------------------------------------------------------------------------------------------- reading the slides
def open_slide(path):
    import openslide
    return openslide.OpenSlide(path)


def slide_mpp(slide, override=None):
    if override:
        return float(override), float(override)
    mx, my = (slide.properties.get(k) for k in ("openslide.mpp-x", "openslide.mpp-y"))
    try:
        mx, my = float(mx), float(my)
    except (TypeError, ValueError):
        mx = my = float("nan")
    if not (np.isfinite(mx) and np.isfinite(my) and mx > 0 and my > 0):
        raise ValueError("slide has no openslide.mpp-x/y; pass --mpp (um per level-0 pixel)")
    return mx, my


def read_patches(wsi_path, xy, patch_px=PATCH_PX, mpp_target=MPP_TARGET, mpp=None, workers=8):
    """uint8 (N, patch_px, patch_px, 3): a patch_px*mpp_target um square around each level-0 centroid, resized to
    patch_px (same crop logic as extract_lemon.py). Rows are read in the order given."""
    from PIL import Image
    import threading
    slide0 = open_slide(wsi_path)
    mx, my = slide_mpp(slide0, mpp)
    um = patch_px * mpp_target
    w, h = max(1, round(um / mx)), max(1, round(um / my))
    local = threading.local()
    out = np.empty((len(xy), patch_px, patch_px, 3), np.uint8)

    def one(i):
        s = getattr(local, "s", None)
        if s is None:
            s = local.s = slide0 if threading.current_thread() is threading.main_thread() else open_slide(wsi_path)
        x, y = xy[i]
        im = s.read_region((int(round(x - w / 2)), int(round(y - h / 2))), 0, (w, h)).convert("RGB")
        if im.size != (patch_px, patch_px):
            im = im.resize((patch_px, patch_px), Image.Resampling.BICUBIC)
        out[i] = np.asarray(im)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        list(ex.map(one, range(len(xy))))
    return out


# ------------------------------------------------------------------------------------------- model
class Nuc2Vec(nn.Module):
    """ResNet34 -> Linear(512,512) -> ReLU -> Linear(512,128); output is L2-normalised (paper: 128-d on the sphere)."""

    def __init__(self, dim=128, weights="imagenet"):
        super().__init__()
        import torchvision
        w = None
        if weights == "imagenet":
            try:
                w = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
                body = torchvision.models.resnet34(weights=w)
            except Exception as e:                                   # no network on the machine: say so, do not hide it
                print(f"  WARNING: could not load ImageNet weights ({type(e).__name__}: {e}); starting from random init")
                body = torchvision.models.resnet34(weights=None)
        else:
            body = torchvision.models.resnet34(weights=None)
        body.fc = nn.Identity()
        self.body = body
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, dim))
        self.dim = dim

    def forward(self, x):
        return F.normalize(self.head(self.body(x)).float(), dim=1)


def normalise(x01):
    m = torch.tensor(IMAGENET_MEAN, device=x01.device).view(1, 3, 1, 1)
    s = torch.tensor(IMAGENET_STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s


def center_crop_u8(patches, crop_px=CROP_PX):
    o = (patches.shape[1] - crop_px) // 2
    return patches[:, o:o + crop_px, o:o + crop_px]


# ------------------------------------------------------------------------------------------- loss
def align_loss(z1, z2, alpha=2):
    return (z1 - z2).norm(dim=1).pow(alpha).mean()


def uniform_loss(z, t=2):
    return torch.pdist(z.float(), p=2).pow(2).mul(-t).exp().mean().log()


# ------------------------------------------------------------------------------------------- augmentation (GPU, batched)
class Augment(nn.Module):
    """uint8 patches (B,S,S,3) -> normalised float (B,3,128,128). Every operation is drawn independently per sample."""

    def __init__(self, out_px=CROP_PX, p_scale=0.5, scale=(0.5, 1.0), p_rot=0.5, p_flip=0.5, p_color=0.5,
                 jitter=(0.4, 0.4, 0.4, 0.1), p_blur=0.5, blur_sigma=(0.1, 2.0), bg_replace=True, translate_px=4.0,
                 box_margin=1.15, bg_mode="box", mask_dilate=2):
        super().__init__()
        self.__dict__.update(out_px=out_px, p_scale=p_scale, scale=scale, p_rot=p_rot, p_flip=p_flip, p_color=p_color,
                             jitter=jitter, p_blur=p_blur, blur_sigma=blur_sigma, bg_replace=bg_replace,
                             translate_px=translate_px, box_margin=box_margin, bg_mode=bg_mode, mask_dilate=mask_dilate)

    def forward(self, x, bg, diam_px, mask=None):
        dev = x.device
        x = x.permute(0, 3, 1, 2).float().div_(255)
        B, _, S, _ = x.shape
        U = lambda lo=0.0, hi=1.0: torch.rand(B, device=dev) * (hi - lo) + lo
        if self.bg_replace:
            bg = bg.permute(0, 3, 1, 2).float().div_(255)
            half = (diam_px.to(dev) * self.box_margin / 2).clamp(min=6.0).view(B, 1, 1)
            c = (S - 1) / 2
            ax = (torch.arange(S, device=dev).float() - c).abs().view(1, 1, S)
            inside = (ax <= half) & (ax.view(1, S, 1) <= half)
            if self.bg_mode == "mask" and mask is not None:
                # keep the nucleus polygon (dilated by mask_dilate px, 1 px soft edge), paste the other patch elsewhere
                m = mask.to(dev).float()[:, None]
                d = int(self.mask_dilate)
                if d > 0:
                    m = F.max_pool2d(m, 2 * d + 1, stride=1, padding=d)
                alpha = F.avg_pool2d(m, 3, stride=1, padding=1)
                x = x * alpha + bg * (1 - alpha)
            else:
                x = torch.where(inside[:, None], x, bg)
        # geometry: rotation / flips / zoom as one affine sampling of the central window
        rot = torch.where(U() < self.p_rot, U(0, 2 * math.pi), torch.zeros(B, device=dev))
        sx = torch.where(U() < self.p_flip, -torch.ones(B, device=dev), torch.ones(B, device=dev))
        sy = torch.where(U() < self.p_flip, -torch.ones(B, device=dev), torch.ones(B, device=dev))
        z = torch.where(U() < self.p_scale, U(*self.scale), torch.ones(B, device=dev))
        w = (self.out_px / S) * z
        cos, sin = torch.cos(rot), torch.sin(rot)
        t = (torch.rand(B, 2, device=dev) * 2 - 1) * (self.translate_px / (S / 2))
        theta = torch.stack([torch.stack([cos * sx * w, -sin * sy * w, t[:, 0]], 1),
                             torch.stack([sin * sx * w, cos * sy * w, t[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, (B, 3, self.out_px, self.out_px), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        # colour jitter (brightness, contrast, saturation, hue)
        jb, jc, js, jh = self.jitter
        on = (U() < self.p_color).view(B, 1, 1, 1)
        gray_w = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
        y = x * U(1 - jb, 1 + jb).view(B, 1, 1, 1)
        m = (y * gray_w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
        y = (y - m) * U(1 - jc, 1 + jc).view(B, 1, 1, 1) + m
        g = (y * gray_w).sum(1, keepdim=True)
        y = (y - g) * U(1 - js, 1 + js).view(B, 1, 1, 1) + g
        ang = U(-jh, jh) * 2 * math.pi                                  # hue: rotate chroma in YIQ space
        T = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
        Ti = torch.linalg.inv(T)
        ca, sa = torch.cos(ang), torch.sin(ang)
        R = torch.zeros(B, 3, 3, device=dev)
        R[:, 0, 0] = 1; R[:, 1, 1] = ca; R[:, 1, 2] = -sa; R[:, 2, 1] = sa; R[:, 2, 2] = ca
        M = Ti.unsqueeze(0) @ R @ T.unsqueeze(0)
        y = torch.einsum("bij,bjhw->bihw", M, y)
        x = torch.where(on, y.clamp(0, 1), x)
        # Gaussian blur (sigma ~ 0 = identity for the samples that are not blurred)
        sig = torch.where(U() < self.p_blur, U(*self.blur_sigma), torch.full((B,), 1e-3, device=dev))
        k = torch.arange(-4, 5, device=dev).float().view(1, 9)
        ker = torch.exp(-(k ** 2) / (2 * sig.view(B, 1) ** 2))
        ker = ker / ker.sum(1, keepdim=True)
        kk = ker.repeat_interleave(3, 0)                                  # (B*3, 9)
        xb = x.reshape(1, B * 3, self.out_px, self.out_px)
        xb = F.conv2d(F.pad(xb, (4, 4, 0, 0), mode="reflect"), kk.view(B * 3, 1, 1, 9), groups=B * 3)
        xb = F.conv2d(F.pad(xb, (0, 0, 4, 4), mode="reflect"), kk.view(B * 3, 1, 9, 1), groups=B * 3)
        return normalise(xb.reshape(B, 3, self.out_px, self.out_px))