# src/portrait.py
"""
Source-anchored portrait: rearrange a source image's OWN pixels so it resembles the
target's subject, while keeping the source's size, color histogram, contrast and
brightness exactly.

Design (budget-driven, gap-free):
  - Split the source into two pools by count: the dominant background (e.g. sky), and
    everything else -- the "object pool". Mask R-CNN instance detection protects real
    objects from being absorbed into the background (a blue book held against a blue
    wall stays in the object pool).
  - The target's subject (background removed, auto-oriented upright) is centered and
    *scaled so its silhouette area equals the number of object pixels*. The remaining
    frame then equals the number of background pixels -- so both pools fit exactly.
  - Object pixels are rearranged (optimal transport + tonal cluster matching) to render
    that centered subject; background pixels are rearranged into a smooth fill around
    it (each border position takes its nearest original background color).
  - Every source pixel is used exactly once and every output position is filled -> no
    empty/jagged gaps; identical histogram / contrast / brightness to the source. The
    background is repositioned (not frozen), so the subject sits cleanly at center.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

from .pixel_rearrange import to_feature, solve_assignment_recursive
from .segment import upright_orientation, subject_mask


def _fill_region(src_pixels, tgt_pixels, space, leaf, k_clusters=6):
    """Rearrange `src_pixels` (a permutation; same length as `tgt_pixels`) to match
    `tgt_pixels`, with tonal cluster sub-matching (dark->hair/shadow, mid->skin)."""
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
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component of a boolean mask. The segmenter can
    return several disconnected instances (the subject plus stray background objects);
    for the centered guide we want just the single main subject, not stray blobs."""
    from scipy.ndimage import label
    comps, n = label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(comps.ravel())
    sizes[0] = 0
    return comps == sizes.argmax()


def dominant_background(src_img: Image.Image, obj_mask: np.ndarray, k: int = 6) -> np.ndarray:
    """
    Boolean (H, W) mask of the single dominant background: the largest *spatially
    connected* patch of the dominant color cluster among the NON-object pixels. Objects
    are excluded (so a same-colored held object is never counted as background), and
    keeping only the connected component yields a coherent region (e.g. the sky) rather
    than every scattered pixel of that color.
    """
    from scipy.cluster.vq import kmeans2
    from scipy.ndimage import label

    h, w = obj_mask.shape
    rgb = np.asarray(src_img, dtype=np.float64).reshape(-1, 3)
    non_obj = ~obj_mask.reshape(-1)
    idx = np.where(non_obj)[0]
    bg = np.zeros(len(rgb), dtype=bool)
    if len(idx) < k:
        bg[idx] = True
        return bg.reshape(h, w)
    feat = to_feature(rgb[idx], "lab")
    _, labels = kmeans2(feat, k, seed=0, minit="++", missing="raise")
    dom = np.bincount(labels).argmax()
    color_mask = np.zeros(len(rgb), dtype=bool)
    color_mask[idx[labels == dom]] = True

    comps, n = label(color_mask.reshape(h, w))      # largest connected patch
    if n == 0:
        return color_mask.reshape(h, w)
    sizes = np.bincount(comps.ravel())
    sizes[0] = 0
    return comps == sizes.argmax()


def _inframe_area(subm, scale, frame):
    """Silhouette pixels that land inside the frame when `subm` is scaled and centered."""
    w, h = frame
    sh, sw = subm.shape
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    m = np.asarray(Image.fromarray((subm * 255).astype(np.uint8)).resize((nw, nh))) > 127
    ys, xs = np.where(m)
    ty, tx = (h - nh) // 2 + ys, (w - nw) // 2 + xs
    return int(((ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)).sum())


def _scaled_subject(target_path, seg, n_obj, frame):
    """
    Return (sub_rgb [nh,nw,3] float, sub_mask [nh,nw] bool): the target's subject,
    background removed and auto-oriented upright, scaled so that the part landing
    inside the frame (when centered) is at least `n_obj` pixels -- a slight OVERSHOOT.

    We deliberately overshoot, then erode down to the exact budget elsewhere: growing
    a region (dilation) produces blobby halos, whereas trimming an oversized, frame-
    filling subject (erosion) stays clean. The subject is allowed to exceed the frame
    and crop, so a large object budget yields a bigger centered subject, not a halo.
    """
    w, h = frame
    img = Image.open(target_path).convert("RGB")
    krot = upright_orientation(target_path)
    if krot:
        img = img.rotate(90 * krot, expand=True)

    # person-only: the target guide should be the subject, never nearby furniture
    tmask = _largest_component(subject_mask(img, method=seg, person_only=True))
    x0, y0, x1, y1 = _bbox(tmask)
    sub = np.asarray(img, dtype=np.float64)[y0:y1, x0:x1]
    subm = tmask[y0:y1, x0:x1]
    sh, sw = sub.shape[:2]
    if int(subm.sum()) == 0:
        return sub, subm

    # smallest scale whose in-frame silhouette area >= n_obj (binary search on the mask)
    lo, hi = 0.02, 3.0 * max(w / sw, h / sh)
    if _inframe_area(subm, hi, frame) < n_obj:             # frame can't be over-filled
        scale = hi
    else:
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if _inframe_area(subm, mid, frame) >= n_obj:
                hi = mid
            else:
                lo = mid
        scale = hi

    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    sub_img = Image.fromarray(sub.astype(np.uint8)).resize((nw, nh))
    subm_img = Image.fromarray((subm * 255).astype(np.uint8)).resize((nw, nh))
    return np.asarray(sub_img, dtype=np.float64), (np.asarray(subm_img) > 127)


def _place_centered(sub_rgb, sub_mask, frame):
    """Center the (already scaled) subject in the frame. Returns (pos, guide):
    pos   = (H,W) bool mask of subject positions,
    guide = (H,W,3) holding the subject's colors at those positions (0 elsewhere)."""
    w, h = frame
    nh, nw = sub_mask.shape
    px, py = (w - nw) // 2, (h - nh) // 2
    pos = np.zeros((h, w), dtype=bool)
    guide = np.zeros((h, w, 3), dtype=np.float64)
    ys, xs = np.where(sub_mask)
    ty, tx = py + ys, px + xs
    inside = (ty >= 0) & (ty < h) & (tx >= 0) & (tx < w)
    ty, tx, ys, xs = ty[inside], tx[inside], ys[inside], xs[inside]
    pos[ty, tx] = True
    guide[ty, tx] = sub_rgb[ys, xs]
    return pos, guide


def _fix_count(pos: np.ndarray, n_target: int) -> np.ndarray:
    """
    Force boolean mask `pos` to have EXACTLY `n_target` True pixels, so the subject
    region consumes precisely the object-pixel budget (and the rest is exactly the
    background budget) -- guaranteeing a gap-free, fully-used permutation.

    Too many -> erode the boundary; too few -> grow into the nearest background.
    """
    from scipy.ndimage import distance_transform_edt
    area = int(pos.sum())
    if area == n_target or area == 0:
        return pos
    if area > n_target:
        d = distance_transform_edt(pos).reshape(-1)        # depth inside subject
        flat = np.where(pos.reshape(-1))[0]
        drop = flat[np.argsort(d[flat])[: area - n_target]]  # shallowest (edge) first
        out = pos.reshape(-1).copy(); out[drop] = False
        return out.reshape(pos.shape)
    d = distance_transform_edt(~pos).reshape(-1)           # distance to subject
    flat = np.where(~pos.reshape(-1))[0]
    add = flat[np.argsort(d[flat])[: n_target - area]]     # nearest background first
    out = pos.reshape(-1).copy(); out[add] = True
    return out.reshape(pos.shape)


def _smooth_background(bg_pixels, bg_mask, src_rgb, bg_idx, frame, space, leaf):
    """
    Rearrange the source's background pixels (e.g. sky) to fill every background
    output position (`bg_idx`) smoothly. Each position takes a guide color equal to
    the NEAREST original background color, so the sky's own gradient is reproduced and
    the central positions vacated by the subject blend seamlessly into it.
    """
    from scipy.ndimage import distance_transform_edt
    w, h = frame
    grid = src_rgb.reshape(h, w, 3)
    _, (iy, ix) = distance_transform_edt(~bg_mask, return_indices=True)
    guide = grid[iy, ix].reshape(-1, 3)[bg_idx]
    perm = solve_assignment_recursive(
        to_feature(bg_pixels, space), to_feature(guide, space), leaf=leaf)
    return bg_pixels[perm]


def make_portrait(
    source_path: str,
    target_path: str,
    out_path: str | None = None,
    max_dim: int = 1600,
    space: str = "lab",
    leaf: int = 64,
    seg: str = "maskrcnn",
) -> Image.Image:
    # 1. output = the source (capped, aspect preserved -> proportions kept)
    src = Image.open(source_path).convert("RGB")
    if max(src.size) > max_dim:
        s = max_dim / max(src.size)
        src = src.resize((max(1, round(src.size[0] * s)), max(1, round(src.size[1] * s))))
    w, h = src.size
    src_rgb = np.asarray(src, dtype=np.float64).reshape(-1, 3)

    # 2. objects (protected) -> 3. dominant background among non-objects
    obj = subject_mask(src, method=seg)
    bg_keep = dominant_background(src, obj)
    obj_pos = np.where((~bg_keep).reshape(-1))[0]          # subject-pool pixels
    bg_pos = np.where(bg_keep.reshape(-1))[0]              # background pixels
    n_obj = len(obj_pos)
    if n_obj == 0 or len(bg_pos) == 0:
        if out_path:
            src.save(out_path)
        return src

    # 4. target subject scaled to the object-pixel budget, centered, count made exact
    sub_rgb, sub_mask = _scaled_subject(target_path, seg, n_obj, (w, h))
    placed_pos, guide = _place_centered(sub_rgb, sub_mask, (w, h))
    subject_pos = _fix_count(placed_pos, n_obj)

    # subject guide colors: nearest placed-subject color (covers any grown positions)
    from scipy.ndimage import distance_transform_edt
    valid = placed_pos & subject_pos
    if valid.any():
        _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
        guide = guide[iy, ix]
    guide = guide.reshape(-1, 3)

    subj_idx = np.where(subject_pos.reshape(-1))[0]        # |subj_idx| == n_obj
    out_bg_idx = np.where(~subject_pos.reshape(-1))[0]     # |out_bg_idx| == len(bg_pos)

    # 5. rearrange: object pixels -> centered face; background pixels -> smooth fill
    out = np.empty_like(src_rgb)
    out[subj_idx] = _fill_region(src_rgb[obj_pos], guide[subj_idx], space, leaf)
    out[out_bg_idx] = _smooth_background(
        src_rgb[bg_pos], bg_keep, src_rgb, out_bg_idx, (w, h), space, leaf)

    out_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3))
    if out_path:
        out_img.save(out_path)
    return out_img


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Rearrange a source image's own pixels so it resembles the target's subject.")
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default="results/portrait.png")
    p.add_argument("--max-dim", type=int, default=1600, help="cap on output's longer edge")
    p.add_argument("--seg", choices=["maskrcnn", "deeplab", "grabcut"], default="maskrcnn",
                   help="object detection (maskrcnn = instances incl. held items)")
    args = p.parse_args()
    make_portrait(args.source, args.target, args.out, max_dim=args.max_dim, seg=args.seg)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
