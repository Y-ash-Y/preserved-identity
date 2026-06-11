# src/pixel_rearrange.py
"""
Identity-aware pixel rearrangement.

Core idea: take a SOURCE image's pixels (the "bag of pixels") and permute them
so the result resembles a TARGET image. Every source pixel is used exactly once,
so the output has the *same color histogram* as the source -- it is literally the
same pixels, shuffled in space to take the shape of the target.

This is an optimal-assignment (optimal-transport) problem: match each source pixel
to a target position, minimizing total color distance, subject to a one-to-one
(bijection) constraint.

Matching is done in CIELAB color space by default, so "distance" is perceptual
(closer to how the eye judges color similarity) rather than raw RGB.

Optionally, a small `recolor` blend nudges the rearranged pixels toward the exact
target colors for a crisper resemblance (the "slight recoloring" knob).
"""

from __future__ import annotations
import numpy as np
from PIL import Image


def load_rgb(path: str, size: tuple[int, int]) -> np.ndarray:
    """Load an image, resize to (W, H), return float RGB array of shape (N, 3)."""
    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float64).reshape(-1, 3)


def output_dims(target_path: str, long_edge: int | None) -> tuple[int, int]:
    """
    Decide the output grid (W, H) from the target's native aspect ratio.

    long_edge : length of the longer side; None keeps the target's native size.
    The source is later resized to this same grid so that every output position
    is filled by one source pixel (a true bijection over essentially all source
    pixels), while the target's proportions are preserved (no squaring).
    """
    with Image.open(target_path) as img:
        w, h = img.size
    if long_edge is None:
        return w, h
    if w >= h:
        return long_edge, max(1, round(h * long_edge / w))
    return max(1, round(w * long_edge / h)), long_edge


def to_feature(rgb: np.ndarray, space: str) -> np.ndarray:
    """Map (N,3) RGB [0,255] to the color space used for distance computation."""
    if space == "rgb":
        return rgb
    if space == "lab":
        from skimage.color import rgb2lab
        return rgb2lab((rgb / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    raise ValueError(f"unknown color space: {space!r}")


def solve_assignment_exact(src_feat: np.ndarray, tgt_feat: np.ndarray) -> np.ndarray:
    """
    Exact optimal one-to-one assignment via the Hungarian algorithm.

    Returns `perm` such that source[perm[p]] is the source pixel placed at target
    position p. Total color distance is globally minimized.

    O(N^3) -- only practical for small images (N up to a few thousand pixels).
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    cost = cdist(tgt_feat, src_feat)            # (N_target, N_source)
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=np.int64)
    perm[rows] = cols
    return perm


def solve_assignment_sorted(src_feat: np.ndarray, tgt_feat: np.ndarray) -> np.ndarray:
    """
    Fast approximate assignment: match by sorted lightness (first feature channel:
    L in Lab, or R in RGB). Scales to full-resolution images instantly. Baseline.
    """
    key_s, key_t = src_feat[:, 0], tgt_feat[:, 0]
    src_order = np.argsort(key_s)
    tgt_rank = np.argsort(np.argsort(key_t))
    return src_order[tgt_rank]


def solve_assignment_recursive(
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    leaf: int = 64,
) -> np.ndarray:
    """
    Recursive median-split assignment: a scalable approximation of the exact
    Hungarian result, in ~O(N log N) plus small exact solves at the leaves.

    Returns `perm` (target position p -> source index), always a true bijection
    (the source multiset is preserved exactly).

    Idea: recursively partition both clouds. At each node, pick the color axis of
    greatest variance among the target pixels, split the target positions at their
    median into two equal halves, and split the source pixels by the same axis so
    counts match -- pairing the low half with the low half. Once a block is <= `leaf`
    pixels, solve it exactly with the Hungarian algorithm. This keeps each source
    pixel in exactly one block (a global bijection) while respecting all color
    dimensions, getting within a few percent of optimal at a fraction of the cost.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    n_total = len(src_feat)
    perm = np.empty(n_total, dtype=np.int64)
    stack = [(np.arange(n_total), np.arange(n_total))]
    while stack:
        si, ti = stack.pop()
        n = len(ti)
        if n <= leaf:
            cost = cdist(tgt_feat[ti], src_feat[si])
            rows, cols = linear_sum_assignment(cost)
            perm[ti[rows]] = si[cols]
            continue
        axis = int(np.argmax(tgt_feat[ti].var(axis=0)))
        ts = ti[np.argsort(tgt_feat[ti, axis], kind="stable")]
        ss = si[np.argsort(src_feat[si, axis], kind="stable")]
        mid = n // 2
        stack.append((ss[:mid], ts[:mid]))
        stack.append((ss[mid:], ts[mid:]))
    return perm


def rearrange(
    source_path: str,
    target_path: str,
    out_path: str | None = None,
    size: int | None = 1024,
    method: str = "recursive",
    space: str = "lab",
    recolor: float = 0.0,
    leaf: int = 64,
) -> Image.Image:
    """
    Rearrange source pixels to resemble target.

    size    : length of the output's longer edge (target aspect ratio is preserved).
              None uses the target's native resolution. The source is resized to the
              same grid, so essentially every source pixel is used exactly once.
    method  : "recursive" (scalable median-split OT, default; handles megapixels),
              "exact" (Hungarian, optimal but only small images), or
              "sorted" (fastest, lightness only).
    space   : "lab" (perceptual, recommended) or "rgb".
    recolor : 0.0 = pure shuffle (source histogram preserved exactly).
              >0  = blend that fraction toward true target colors (slight recoloring).
    leaf    : block size at which the recursive method solves exactly.
    """
    w, h = output_dims(target_path, size)
    src_rgb = load_rgb(source_path, (w, h))
    tgt_rgb = load_rgb(target_path, (w, h))
    src_feat = to_feature(src_rgb, space)
    tgt_feat = to_feature(tgt_rgb, space)

    if method == "exact":
        perm = solve_assignment_exact(src_feat, tgt_feat)
    elif method == "sorted":
        perm = solve_assignment_sorted(src_feat, tgt_feat)
    elif method == "recursive":
        perm = solve_assignment_recursive(src_feat, tgt_feat, leaf=leaf)
    else:
        raise ValueError(f"unknown method: {method!r}")

    out = src_rgb[perm]                          # rearranged pixels, in target order
    if recolor > 0.0:
        out = (1.0 - recolor) * out + recolor * tgt_rgb

    out_img = Image.fromarray(
        np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)
    )
    if out_path:
        out_img.save(out_path)
    return out_img
