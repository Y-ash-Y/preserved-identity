# src/portrait.py
"""
Region-routed portrait: rebuild the target's FACE from the source's foreground OBJECT
pixels, and the target's background from the source's background pixels.

Pipeline:
  1. Auto-orient the target so the face is upright; crop to a portrait around it.
  2. Segment the crop into face (foreground) vs background.
  3. Segment the source into object (foreground) vs background.
  4. Match like-to-like: source object -> target face, source background -> target bg.
     Each region is filled by optimal-transport rearrangement of its source pixels.
  5. Adopt the target's lightness/contrast (luma recolor) for resemblance.

The result is an upright face portrait built from the source's own pixels, routed by
semantics so the rich object pixels build the face instead of being wasted as flat fill.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

from .pixel_rearrange import to_feature, solve_assignment_recursive, apply_recolor
from .segment import upright_face_crop, foreground_mask


def _resample(pixels: np.ndarray, n: int, rng) -> np.ndarray:
    """Return exactly `n` pixels drawn from `pixels` (without replacement when possible)."""
    if len(pixels) == 0:
        return np.zeros((n, 3))
    replace = n > len(pixels)
    idx = rng.choice(len(pixels), size=n, replace=replace)
    return pixels[idx]


def _fill_region(src_pixels, tgt_pixels, space, leaf, rng, k_clusters=6):
    """
    Rearrange a bag of `src_pixels` to fill `tgt_pixels`, with color-cluster
    sub-matching: sort both sides by lightness, split into `k_clusters` equal tonal
    bins, and rearrange within each bin. This enforces tonal correspondence
    (dark source pixels -> hair/shadow, mid -> skin) instead of leaving it to chance.
    """
    n = len(tgt_pixels)
    bag = _resample(src_pixels, n, rng)            # high-res pool -> subsample, no dup
    tgt_feat = to_feature(tgt_pixels, space)
    bag_feat = to_feature(bag, space)

    out = np.empty_like(tgt_pixels)
    t_order = np.argsort(tgt_feat[:, 0], kind="stable")   # by lightness
    s_order = np.argsort(bag_feat[:, 0], kind="stable")
    bounds = np.linspace(0, n, k_clusters + 1).astype(int)
    for i in range(k_clusters):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        ti, si = t_order[a:b], s_order[a:b]               # matched tonal bins
        perm = solve_assignment_recursive(bag_feat[si], tgt_feat[ti], leaf=leaf)
        out[ti] = bag[si][perm]
    return out


def _region_pixels(path: str, seg: str, long_edge: int = 1600):
    """Load a source at high resolution and split into (foreground, background) pixel
    pools. High res means each region has enough pixels to subsample without
    duplicating, preserving its color distribution and adding detail."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > long_edge:
        scale = long_edge / max(img.size)
        img = img.resize((max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))))
    mask = foreground_mask(img, method=seg).reshape(-1)
    rgb = np.asarray(img, dtype=np.float64).reshape(-1, 3)
    return rgb[mask], rgb[~mask]


def make_portrait(
    source_path: str,
    target_path: str,
    out_path: str | None = None,
    size: int = 512,
    margin: float = 0.6,
    space: str = "lab",
    recolor: float = 0.85,
    recolor_mode: str = "luma",
    leaf: int = 64,
    seg: str = "deeplab",
    seed: int = 0,
) -> Image.Image:
    rng = np.random.default_rng(seed)

    crop, face_rect = upright_face_crop(target_path, size=size, margin=margin)
    w, h = crop.size
    tgt_rgb = np.asarray(crop, dtype=np.float64).reshape(-1, 3)

    # target: face (foreground) vs background within the crop
    tmask = foreground_mask(crop, method=seg, rect=face_rect).reshape(-1)

    # source: object (foreground) vs background pixel pools at high resolution
    src_fg, src_bg = _region_pixels(source_path, seg)

    out = np.empty_like(tgt_rgb)
    fg_pos, bg_pos = np.where(tmask)[0], np.where(~tmask)[0]
    if len(fg_pos):
        out[fg_pos] = _fill_region(src_fg, tgt_rgb[fg_pos], space, leaf, rng)
    if len(bg_pos):
        out[bg_pos] = _fill_region(src_bg, tgt_rgb[bg_pos], space, leaf, rng)

    out = apply_recolor(out, tgt_rgb, recolor, recolor_mode, w, h)
    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3))
    if out_path:
        out_img.save(out_path)
    return out_img


def main():
    import argparse
    p = argparse.ArgumentParser(description="Region-routed upright face portrait from a source image.")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="results/portrait.png")
    p.add_argument("--size", type=int, default=512, help="portrait longer-edge length")
    p.add_argument("--margin", type=float, default=0.6, help="crop margin around the face box")
    p.add_argument("--recolor", type=float, default=0.85)
    p.add_argument("--recolor-mode", choices=["luma", "blend"], default="luma")
    p.add_argument("--seg", choices=["deeplab", "grabcut"], default="deeplab",
                   help="subject segmentation: deeplab (clean, downloads weights) or grabcut (offline)")
    args = p.parse_args()
    make_portrait(args.source, args.target, args.out, size=args.size, margin=args.margin,
                  recolor=args.recolor, recolor_mode=args.recolor_mode, seg=args.seg)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
