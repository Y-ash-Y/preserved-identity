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


def _rect_match(si, ti, src_feat, tgt_feat, leaf, perm):
    """Match every target index in `ti` to a distinct source index drawn from `si`
    (requires len(si) >= len(ti)). Recursive median split, rectangular at the leaves.
    Writes results into `perm` and returns the source indices actually used."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    n, m = len(ti), len(si)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n <= leaf or m <= leaf:
        cost = cdist(tgt_feat[ti], src_feat[si])      # n x m, n <= m
        rows, cols = linear_sum_assignment(cost)       # n distinct columns
        perm[ti[rows]] = si[cols]
        return si[cols]

    axis = int(np.argmax(tgt_feat[ti].var(axis=0)))
    ts = ti[np.argsort(tgt_feat[ti, axis], kind="stable")]
    ss = si[np.argsort(src_feat[si, axis], kind="stable")]
    tn = n // 2                                        # left target count
    # split the source at the SAME color value as the target's median, so the two
    # halves cover aligned color ranges; then clamp so each side has source >= target.
    split_val = tgt_feat[ts[tn], axis]
    sm = int(np.searchsorted(src_feat[ss, axis], split_val, side="right"))
    sm = max(tn, min(sm, m - (n - tn)))
    used_l = _rect_match(ss[:sm], ts[:tn], src_feat, tgt_feat, leaf, perm)
    used_r = _rect_match(ss[sm:], ts[tn:], src_feat, tgt_feat, leaf, perm)
    return np.concatenate([used_l, used_r])


def solve_assignment_priority(
    src_feat: np.ndarray,
    tgt_feat: np.ndarray,
    importance: np.ndarray,
    leaf: int = 64,
    top_frac: float = 0.3,
) -> np.ndarray:
    """
    Importance-aware assignment, still a strict permutation.

    The top `top_frac` most important target positions are matched FIRST against the
    *entire* source pool, so they claim the globally best-fitting source pixels. The
    remaining target positions are then matched to whatever source pixels are left.
    This spends the scarce well-matching pixels on the identity-bearing regions.
    """
    N = len(tgt_feat)
    perm = np.empty(N, dtype=np.int64)
    order = np.argsort(-importance, kind="stable")
    n_a = max(1, min(N - 1, int(round(top_frac * N))))
    a_idx = order[:n_a]                                # high-importance positions
    b_idx = order[n_a:]                                # the rest

    used = _rect_match(np.arange(N), a_idx, src_feat, tgt_feat, leaf, perm)
    free = np.setdiff1d(np.arange(N), used, assume_unique=False)
    sub = solve_assignment_recursive(src_feat[free], tgt_feat[b_idx], leaf=leaf)
    perm[b_idx] = free[sub]
    return perm


def apply_recolor(out, tgt_rgb, recolor: float, recolor_mode: str, w: int, h: int):
    """Shift rearranged pixels `out` (N,3 RGB) toward target colors.

    "blend" = linear RGB blend; "luma" = adopt the target's lightness/contrast while
    keeping the source's chroma (strong resemblance per unit of recolor).
    """
    if recolor <= 0.0:
        return out
    if recolor_mode == "blend":
        return (1.0 - recolor) * out + recolor * tgt_rgb
    if recolor_mode == "luma":
        from skimage.color import rgb2lab, lab2rgb
        r = rgb2lab((out / 255.0).reshape(h, w, 3))
        t = rgb2lab((tgt_rgb / 255.0).reshape(h, w, 3))
        a_c = recolor * 0.33                       # chroma shifts less than lightness
        r[..., 0] = (1 - recolor) * r[..., 0] + recolor * t[..., 0]
        r[..., 1] = (1 - a_c) * r[..., 1] + a_c * t[..., 1]
        r[..., 2] = (1 - a_c) * r[..., 2] + a_c * t[..., 2]
        return (lab2rgb(r) * 255.0).reshape(-1, 3)
    raise ValueError(f"unknown recolor_mode: {recolor_mode!r}")


def rearrange(
    source_path: str,
    target_path: str,
    out_path: str | None = None,
    size: int | None = 1024,
    method: str = "recursive",
    space: str = "lab",
    recolor: float = 0.0,
    recolor_mode: str = "luma",
    leaf: int = 64,
    priority: float = 0.0,
) -> Image.Image:
    """
    Rearrange source pixels to resemble target.

    size     : length of the output's longer edge (target aspect ratio is preserved).
               None uses the target's native resolution. The source is resized to the
               same grid, so essentially every source pixel is used exactly once.
    method   : "recursive" (scalable median-split OT, default; handles megapixels),
               "exact" (Hungarian, optimal but only small images), or
               "sorted" (fastest, lightness only).
    space    : "lab" (perceptual, recommended) or "rgb".
    recolor  : 0.0 = pure shuffle (source histogram preserved exactly).
               >0  = recolor strength toward the target (see recolor_mode).
    recolor_mode : "luma" (default) keeps the source's colors but adopts the target's
               lightness/contrast structure -- strong resemblance per unit of recolor,
               best when the source palette is poor. "blend" is a plain linear blend
               toward the target's RGB.
    leaf     : block size at which the recursive method solves exactly.
    priority : 0.0 = uniform matching. >0 = the top `priority` fraction of important
               (face/saliency) target positions claim the best source pixels first.
               Stays a strict permutation; overrides `method` with the priority solver.
    """
    w, h = output_dims(target_path, size)
    src_rgb = load_rgb(source_path, (w, h))
    tgt_rgb = load_rgb(target_path, (w, h))
    src_feat = to_feature(src_rgb, space)
    tgt_feat = to_feature(tgt_rgb, space)

    if priority > 0.0:
        from .saliency import importance_map
        imp = importance_map(target_path, (w, h))
        perm = solve_assignment_priority(src_feat, tgt_feat, imp, leaf=leaf, top_frac=priority)
    elif method == "exact":
        perm = solve_assignment_exact(src_feat, tgt_feat)
    elif method == "sorted":
        perm = solve_assignment_sorted(src_feat, tgt_feat)
    elif method == "recursive":
        perm = solve_assignment_recursive(src_feat, tgt_feat, leaf=leaf)
    else:
        raise ValueError(f"unknown method: {method!r}")

    out = apply_recolor(src_rgb[perm], tgt_rgb, recolor, recolor_mode, w, h)

    out_img = Image.fromarray(
        np.clip(out, 0, 255).astype(np.uint8).reshape(h, w, 3)
    )
    if out_path:
        out_img.save(out_path)
    return out_img
