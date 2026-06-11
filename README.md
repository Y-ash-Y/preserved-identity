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
    --size 512 --space lab --recolor 0.3
```

Options:

| flag        | meaning                                                                        |
|-------------|--------------------------------------------------------------------------------|
| `--size`    | working square resolution (SxS). `recursive` handles megapixels.               |
| `--method`  | `recursive` (scalable OT, default), `exact` (optimal, ≤~64px), or `sorted`.     |
| `--space`   | `lab` (perceptual, recommended) or `rgb`.                                       |
| `--recolor` | `0` = pure shuffle; `>0` blends toward target colors.                           |
| `--leaf`    | block size at which `recursive` solves exactly (default 64).                    |

## Structure

- `src/pixel_rearrange.py` — the assignment engine (load, match, recolor, save).
- `src/cli.py` — command-line interface.
- `scripts/demo_rearrange.py` — synthetic tree→face proof-of-concept.
- `data/` — example `source.jpg` / `target.jpg`.
- `results/` — output images.

## Status & next steps

Working: Lab-space matching with a recolor blend, via three solvers — exact Hungarian
(optimal reference), the scalable `recursive` median-split OT (default, megapixel-capable),
and a `sorted` lightness baseline. Proven on synthetic and real images up to 1024px.

Possible next steps: edge/structure-aware cost, alternative target aspect ratios
(currently squared), and a batch/gallery mode.
