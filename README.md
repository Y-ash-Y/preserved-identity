# preserved-identity

Rearrange one image's pixels so it looks like another — **using only the source
image's own pixels**.

Give it a **source** image (the bag of pixels, e.g. a tree or a group photo) and a
**target** image (the shape to reproduce, e.g. a face). Each source pixel is placed
exactly once, so the output has the *same color histogram as the source* — it is
literally the same pixels, relocated in space to take the target's form (dark pixels
drift to the hair, bright pixels to highlights, and so on).

## How it works

It's an **optimal-assignment / optimal-transport** problem: every source pixel is
matched one-to-one to a target position, minimizing total color distance. Matching
is done in perceptual **CIELAB** color space. An optional `--recolor` knob blends a
small fraction toward the exact target colors for a crisper resemblance.

The default `recursive` solver is a median-split approximation of optimal transport:
it recursively partitions both images by their highest-variance color axis and solves
small blocks exactly with the Hungarian algorithm. This gets within a few percent of
the optimal assignment while scaling to **megapixel images in a couple of seconds**
(1024×1024 ≈ 2s), where exact Hungarian (O(N³)) is only feasible up to ~64px.

## Quick start

```bash
pip install -r requirements.txt

python -m src.cli \
    --source data/source.jpg \
    --target data/target.jpg \
    --out results/out.png \
    --size 1024 --space lab --recolor 0.3
```

Options:

| flag        | meaning                                                                          |
|-------------|----------------------------------------------------------------------------------|
| `--size`    | output's longer edge; target aspect ratio is preserved. `<=0` = target's native. |
| `--method`  | `recursive` (scalable OT, default), `exact` (optimal, ≤~64px), or `sorted`.       |
| `--space`   | `lab` (perceptual, recommended) or `rgb`.                                         |
| `--recolor` | `0` = pure shuffle; `>0` recolor strength toward target (default 0.85).            |
| `--recolor-mode` | `luma` (adopt target lightness/contrast, keep source colors; default) or `blend` (linear RGB blend). |
| `--leaf`    | block size at which `recursive` solves exactly (default 64).                      |
| `--priority`| `0` = uniform; `>0` = top fraction of important (face/saliency) positions claim the best source pixels first. |

The output grid matches the **target's aspect ratio**, and the source is resized to
that grid — so essentially *every* source pixel is used exactly once. Use `--size 0`
to render at the target's native resolution (e.g. a 12 MP image in ~18s).

`--priority 0.3` spends the best-matching source pixels on the identity-bearing
regions (face/edges via `src/saliency.py`) at the cost of background fidelity, while
staying a strict permutation.

### Choosing a good source (this matters most)

Because the output is a *pure rearrangement*, its colors are **locked to the source's
palette** — no algorithm can create colors the source lacks. For a good result the
source should have a color/brightness distribution that **overlaps the target**: plenty
of mid-tones and some dark pixels for hair/shadows, skin-like tones for skin, etc.
A mostly-white source can never reproduce dark hair; a dark source can never make a
bright sky. Pick a source whose palette resembles the target's, and the resemblance
improves dramatically.

## Source-anchored portrait (the main pipeline)

`src/portrait.py` rearranges a source image's **own pixels** so its **subject** takes
on the **target's face**, while keeping the source's identity intact:

- The output **is the source** — same size, same color histogram, same contrast,
  brightness and intensities. Every source pixel is used exactly once (a true
  permutation); nothing is recolored, duplicated, or dropped.
- The **source background is left untouched**. Only the subject's pixels rearrange.
- The **target is just a guide**: its face is auto-oriented upright and fitted into the
  subject's bounding box, telling the subject pixels where to go.

**Smart, semantic foreground detection** is the key idea. Segmentation uses **Mask
R-CNN instance masks** (it knows *person*, *book*, *bottle*, …), not color thresholding
— so a **blue book held against a blue wall** is kept as foreground while the wall stays
background. Color-based methods can't make that distinction; this is where the project
goes beyond a plain pixel shuffle. Falls back to DeepLabV3 → GrabCut if unavailable.

```bash
python -m src.portrait \
    --source data/source.jpeg \
    --target data/target.jpg \
    --out results/portrait.png \
    --max-dim 1600 --seg maskrcnn
```

Within the subject, pixels are matched to the guide by optimal transport with tonal
cluster sub-matching (dark → hair/shadow, mid → skin), so the rearrangement reads as a
face. The result is genuinely *your* source image — same pixels, same palette — with the
subject morphed toward the target.

## Structure

- `src/pixel_rearrange.py` — the optimal-transport assignment engine + plain rearrange CLI.
- `src/segment.py` — face detection / auto-orient, Mask R-CNN / DeepLabV3 / GrabCut masks.
- `src/portrait.py` — the source-anchored portrait pipeline (main entry point).
- `src/saliency.py` — target importance map for the plain `--priority` mode.
- `src/cli.py` — command-line interface for plain whole-image rearrangement.
- `scripts/demo_rearrange.py` — synthetic tree→face proof-of-concept.
- `data/` — example `source.jpeg` / `target.jpg`.
- `results/` — output images.

## Two modes

- **Plain rearrange** (`src.cli`) — rearrange a whole source into a whole target at any
  resolution, with optional recolor. Good for the abstract "image-from-image" effect.
- **Source-anchored portrait** (`src.portrait`) — the finished product: keep the source
  exactly (size, palette, background) and morph only its subject toward the target face.
