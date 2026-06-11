# src/portrait.py
"""
Source-anchored portrait: rearrange a source image's OWN pixels so its subject
resembles the target's face, while keeping the source's size, color histogram,
contrast and brightness exactly, and leaving the source background untouched.

Pipeline:
  1. The output is the source itself (optionally capped in size). Every source pixel
     is used exactly once -> identical histogram / contrast / brightness.
  2. Segment the source into subject (foreground) vs background. Segmentation is
     SEMANTIC (Mask R-CNN instance masks): a blue book held against a blue wall stays
     foreground; the wall stays background -- not separable by color alone.
  3. The target's face, auto-oriented upright, is fitted into the source subject's
     bounding box to act as a per-pixel guide.
  4. The subject's pixels are rearranged (optimal transport, with tonal cluster
     sub-matching) to match that guide -> the subject takes the target's face.
  5. Background pixels are left exactly as they were.

The result is genuinely the source's pixels, rearranged -- same size, same colors,
same contrast -- with the subject morphed toward the target's face.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

from .pixel_rearrange import to_feature, solve_assignment_recursive
from .segment import upright_face_crop, subject_mask


def _fill_region(src_pixels, tgt_pixels, space, leaf, k_clusters=6):
    """
    Rearrange `src_pixels` (a permutation; len == len(tgt_pixels)) to match
    `tgt_pixels`, with color-cluster sub-matching: sort both by lightness, split into
    `k_clusters` equal tonal bins, rearrange within each bin. Enforces tonal
    correspondence (dark -> hair/shadow, mid -> skin) and stays a strict permutation.
    """
    n = len(tgt_pixels)
    tgt_feat = to_feature(tgt_pixels, space)
    src_feat = to_feature(src_pixels, space)
    out = np.empty_like(tgt_pixels)
    t_order = np.argsort(tgt_feat[:, 0], kind="stable")
    s_order = np.argsort(src_feat[:, 0], kind="stable")
    bounds = np.linspace(0, n, k_clusters + 1).astype(int)
    for i in range(k_clusters):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        ti, si = t_order[a:b], s_order[a:b]
        perm = solve_assignment_recursive(src_feat[si], tgt_feat[ti], leaf=leaf)
        out[ti] = src_pixels[si][perm]
    return out


def _bbox(mask: np.ndarray):
    """Bounding box (x0, y0, x1, y1) of a boolean (H, W) mask."""
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_guide(target_path: str, out_size, subject_box) -> np.ndarray:
    """
    Build a per-pixel color guide the size of the output (H*W, 3): the target's
    upright face fitted into the source subject's bounding box using a "cover" fit
    (preserve aspect, fill the box, center-crop the overflow) so the face is not
    distorted; elsewhere neutral grey.
    """
    w, h = out_size
    x0, y0, x1, y1 = subject_box
    bw, bh = x1 - x0, y1 - y0
    face_crop, _ = upright_face_crop(target_path, size=max(bw, bh))
    fw, fh = face_crop.size
    scale = max(bw / fw, bh / fh)                         # cover the box
    face_crop = face_crop.resize((max(bw, round(fw * scale)), max(bh, round(fh * scale))))
    cw, ch = face_crop.size
    left, top = (cw - bw) // 2, (ch - bh) // 2            # center-crop to the box
    face_crop = face_crop.crop((left, top, left + bw, top + bh))
    guide = np.full((h, w, 3), 128.0)
    guide[y0:y1, x0:x1] = np.asarray(face_crop, dtype=np.float64)
    return guide.reshape(-1, 3)


def make_portrait(
    source_path: str,
    target_path: str,
    out_path: str | None = None,
    max_dim: int = 1600,
    space: str = "lab",
    leaf: int = 64,
    seg: str = "maskrcnn",
) -> Image.Image:
    # 1. output = the source (capped in size, aspect preserved -> proportions kept)
    src = Image.open(source_path).convert("RGB")
    if max(src.size) > max_dim:
        s = max_dim / max(src.size)
        src = src.resize((max(1, round(src.size[0] * s)), max(1, round(src.size[1] * s))))
    w, h = src.size
    src_rgb = np.asarray(src, dtype=np.float64).reshape(-1, 3)

    # 2. semantic subject mask (objects incl. held items)
    fg = subject_mask(src, method=seg).reshape(-1)
    fg_pos = np.where(fg)[0]
    if len(fg_pos) == 0:                       # nothing detected -> return source unchanged
        if out_path:
            src.save(out_path)
        return src

    # 3. target face guide, fitted into the subject's bounding box
    guide = build_guide(target_path, (w, h), _bbox(fg.reshape(h, w)))

    # 4. rearrange subject pixels to the guide (strict permutation); 5. keep background
    out = src_rgb.copy()
    out[fg_pos] = _fill_region(src_rgb[fg_pos], guide[fg_pos], space, leaf)

    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3))
    if out_path:
        out_img.save(out_path)
    return out_img


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Rearrange a source image's own pixels so its subject resembles the target's face.")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="results/portrait.png")
    p.add_argument("--max-dim", type=int, default=1600, help="cap on output's longer edge")
    p.add_argument("--seg", choices=["maskrcnn", "deeplab", "grabcut"], default="maskrcnn",
                   help="subject segmentation (maskrcnn = objects incl. held items)")
    args = p.parse_args()
    make_portrait(args.source, args.target, args.out, max_dim=args.max_dim, seg=args.seg)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
