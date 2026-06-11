# src/segment.py
"""
Segmentation helpers for the region-routed portrait pipeline.

  - find the target face (robust to 90-degree rotations) and produce an upright,
    face-cropped portrait canvas;
  - split an image into foreground (object) vs background pixels via GrabCut.

Only offline tools: facenet-pytorch MTCNN for face detection, OpenCV GrabCut for
foreground extraction.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

_MTCNN = None
_DEEPLAB = None


def deeplab_foreground(pil_img: Image.Image, classes=(15,)) -> np.ndarray:
    """
    Subject mask via a pretrained DeepLabV3 (ResNet-50) segmentation network.
    `classes` are Pascal-VOC label ids; 15 = person. Returns bool (H, W).

    Downloads ~160 MB of weights on first use (needs internet once); raises on failure
    so callers can fall back to GrabCut.
    """
    global _DEEPLAB
    import torch
    from torchvision import transforms as T
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

    if _DEEPLAB is None:
        _DEEPLAB = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()

    w, h = pil_img.size
    inp = pil_img.resize((520, 520))
    x = T.functional.to_tensor(inp)
    x = T.functional.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    with torch.no_grad():
        seg = _DEEPLAB(x)["out"][0].argmax(0).cpu().numpy()
    mask = np.isin(seg, classes).astype(np.uint8) * 255
    mask = np.asarray(Image.fromarray(mask).resize((w, h))) > 127
    return mask


def foreground_mask(pil_img: Image.Image, method: str = "grabcut", rect=None) -> np.ndarray:
    """Foreground/subject mask. method="deeplab" (clean, needs weights) or "grabcut"
    (offline). Falls back to GrabCut if DeepLab is unavailable or degenerate."""
    if method == "deeplab":
        try:
            m = deeplab_foreground(pil_img)
            if 0.02 < m.mean() < 0.99:
                return m
        except Exception:
            pass
    return grabcut_mask(pil_img, rect=rect)


def _mtcnn():
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN
        _MTCNN = MTCNN(keep_all=True, device="cpu", min_face_size=20)
    return _MTCNN


def best_face(path: str, work: int = 1100):
    """
    Detect the most confident face across the 4 right-angle rotations.

    Returns (k, box, (rw, rh)) where k is the number of 90-degree CCW rotations that
    make the face upright, box = (x0, y0, x1, y1) in that rotated & `work`-scaled
    image, and (rw, rh) its size. Returns None if no face is found.
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((work, work))
    best = None
    for k in range(4):
        rot = img.rotate(90 * k, expand=True)
        boxes, probs = _mtcnn().detect(rot)
        if boxes is None:
            continue
        i = int(np.argmax(probs))
        if best is None or probs[i] > best[0]:
            best = (float(probs[i]), k, [int(v) for v in boxes[i]], rot.size)
    if best is None:
        return None
    _, k, box, size = best
    return k, box, size


def upright_face_crop(path: str, size: int = 512, margin: float = 0.6):
    """
    Auto-orient the image so the face is upright and crop to the face plus `margin`
    (fraction of the box size) for hair/neck/shoulders.

    Returns (crop_img, face_rect) where crop_img is a PIL RGB image with longer edge
    `size`, and face_rect = (x0, y0, x1, y1) locating the face within the crop.
    """
    found = best_face(path)
    img = Image.open(path).convert("RGB")
    if found is None:                       # fallback: no face -> center square, no rect
        w, h = img.size
        s = min(w, h)
        crop = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        crop = _resize_long(crop, size)
        return crop, None

    k, box, work_size = found
    rot = img.rotate(90 * k, expand=True)   # full-res upright image
    sx = rot.size[0] / work_size[0]
    sy = rot.size[1] / work_size[1]
    x0, y0, x1, y1 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
    bw, bh = x1 - x0, y1 - y0
    mx, my = bw * margin, bh * margin
    cx0, cy0 = max(0, x0 - mx), max(0, y0 - my)
    cx1, cy1 = min(rot.size[0], x1 + mx), min(rot.size[1], y1 + my)
    crop = rot.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
    face_rect = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)

    scale = size / max(crop.size)
    new = (max(1, round(crop.size[0] * scale)), max(1, round(crop.size[1] * scale)))
    crop = crop.resize(new)
    face_rect = tuple(v * scale for v in face_rect)
    return crop, face_rect


def _resize_long(img: Image.Image, size: int) -> Image.Image:
    scale = size / max(img.size)
    return img.resize((max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale))))


def grabcut_mask(pil_img: Image.Image, rect=None, iters: int = 5) -> np.ndarray:
    """
    Foreground mask via GrabCut. Returns a boolean array of shape (H, W); True = object.

    rect : (x0, y0, x1, y1) probable-foreground box. Defaults to the centered 80%.
    Falls back to a center ellipse if GrabCut yields a degenerate mask.
    """
    import cv2

    arr = np.asarray(pil_img.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    h, w = arr.shape[:2]
    if rect is None:
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(arr, mask, (x0, y0, max(1, x1 - x0), max(1, y1 - y0)),
                    bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    except Exception:
        fg = np.zeros((h, w), bool)

    frac = fg.mean()
    if frac < 0.05 or frac > 0.97:          # degenerate -> ellipse fallback
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rxv, ryv = max(1, (x1 - x0) / 2), max(1, (y1 - y0) / 2)
        fg = ((xx - cx) / rxv) ** 2 + ((yy - cy) / ryv) ** 2 <= 1.0
    return fg
