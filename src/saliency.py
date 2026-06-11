# src/saliency.py
"""
Importance map for the target image.

Tells the rearranger which target positions matter most, so the scarce well-matching
source pixels get spent there (the identity-bearing regions) instead of on flat
background. Combines a center bias (the subject is usually central) with local edge
energy (faces/features are high-detail) and optionally a detected face box.
"""

from __future__ import annotations
import numpy as np
from PIL import Image


def detect_face_boxes(target_path: str, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Best-effort face detection, robust to 90-degree rotations of the target.

    Returns boxes as (x0, y0, x1, y1) in the (W, H) = `size` grid. Empty if none.
    """
    w, h = size
    try:
        from facenet_pytorch import MTCNN
    except Exception:
        return []
    img = Image.open(target_path).convert("RGB").resize((w, h))
    mtcnn = MTCNN(keep_all=True, device="cpu")
    boxes: list[tuple[int, int, int, int]] = []
    for k in range(4):  # try each 90-degree rotation; map detections back
        rot = img.rotate(90 * k, expand=True)
        det, _ = mtcnn.detect(rot)
        if det is None:
            continue
        rw, rh = rot.size
        for (x0, y0, x1, y1) in det:
            # undo the rotation on the box corners
            pts = [(x0, y0), (x1, y1)]
            mapped = []
            for (x, y) in pts:
                for _ in range((4 - k) % 4):
                    x, y = y, rw - x  # inverse of a 90-degree expand rotation
                    rw, rh = rh, rw
                mapped.append((x, y))
                rw, rh = rot.size
            xs = [p[0] for p in mapped]; ys = [p[1] for p in mapped]
            boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
        if boxes:
            break
    return boxes


def importance_map(
    target_path: str,
    size: tuple[int, int],
    center_w: float = 0.4,
    edge_w: float = 0.6,
    use_face: bool = False,
    face_boost: float = 2.0,
) -> np.ndarray:
    """
    Return a flat (N,) importance array in [0, 1], N = W*H, in raster order.
    """
    from skimage.color import rgb2gray
    from skimage.filters import sobel
    from scipy.ndimage import gaussian_filter

    w, h = size
    img = np.asarray(Image.open(target_path).convert("RGB").resize((w, h)), dtype=np.float64) / 255.0

    edges = sobel(rgb2gray(img))
    edges = gaussian_filter(edges, sigma=max(1.0, min(w, h) / 256))
    edges /= edges.max() + 1e-9

    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    center = np.exp(-1.5 * r2)

    imp = center_w * center + edge_w * edges

    if use_face:
        for (x0, y0, x1, y1) in detect_face_boxes(target_path, size):
            imp[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] *= face_boost

    imp /= imp.max() + 1e-9
    return imp.reshape(-1)
