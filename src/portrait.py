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


def _fill_region(src_pixels, tgt_pixels, space, leaf, rng):
    """Rearrange a resampled bag of `src_pixels` to best match `tgt_pixels` (same count)."""
    n = len(tgt_pixels)
    bag = _resample(src_pixels, n, rng)
    perm = solve_assignment_recursive(to_feature(bag, space), to_feature(tgt_pixels, space), leaf=leaf)
    return bag[perm]


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

    # source: object (foreground) vs background, sampled on a grid comparable to the crop
    src_img = Image.open(source_path).convert("RGB").resize((w, h))
    smask = foreground_mask(src_img, method=seg).reshape(-1)
    src_rgb = np.asarray(src_img, dtype=np.float64).reshape(-1, 3)
    src_fg, src_bg = src_rgb[smask], src_rgb[~smask]

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
