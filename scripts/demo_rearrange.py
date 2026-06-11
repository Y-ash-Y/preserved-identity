"""Synthetic proof-of-concept for pixel rearrangement.

Generates a 'tree' source image and a 'face' target image with totally different
color palettes, then rearranges the tree's pixels to resemble the face.
Verifies the output is a true permutation (identical color histogram to source).
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pixel_rearrange import rearrange  # noqa: E402

SIZE = 48
os.makedirs("results", exist_ok=True)


def make_tree(path):
    img = Image.new("RGB", (SIZE, SIZE), (135, 206, 235))   # sky blue
    d = ImageDraw.Draw(img)
    d.rectangle([21, 30, 27, 47], fill=(101, 67, 33))        # brown trunk
    d.ellipse([8, 6, 40, 34], fill=(34, 139, 34))            # green canopy
    d.ellipse([14, 2, 34, 22], fill=(60, 179, 60))           # lighter canopy
    img.save(path)


def make_face(path):
    img = Image.new("RGB", (SIZE, SIZE), (240, 240, 235))    # light background
    d = ImageDraw.Draw(img)
    d.ellipse([12, 10, 36, 44], fill=(222, 184, 135))        # skin
    d.pieslice([12, 4, 36, 30], 180, 360, fill=(40, 26, 13)) # dark hair
    d.ellipse([18, 22, 22, 26], fill=(20, 20, 20))           # left eye
    d.ellipse([26, 22, 30, 26], fill=(20, 20, 20))           # right eye
    img.save(path)


def histogram(path):
    arr = np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)
    return np.sort(arr.sum(axis=1))   # sorted pixel-brightness multiset


src, tgt = "results/_demo_source_tree.png", "results/_demo_target_face.png"
make_tree(src)
make_face(tgt)

for recolor in (0.0, 0.25):
    out = f"results/demo_rearranged_recolor{recolor}.png"
    rearrange(src, tgt, out, size=(SIZE, SIZE), method="exact", recolor=recolor)
    print(f"  wrote {out}")

same = np.array_equal(histogram(src), histogram("results/demo_rearranged_recolor0.0.png"))
print(f"\nStrict shuffle preserves source histogram exactly: {same}")
print("Open results/_demo_target_face.png  vs  results/demo_rearranged_recolor0.0.png")
