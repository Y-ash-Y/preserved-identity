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

## Quick start

```bash
pip install -r requirements.txt

python -m src.cli \
    --source data/source.jpg \
    --target data/target.jpg \
    --out results/out.png \
    --size 64 --space lab --recolor 0.3
```

Options:

| flag        | meaning                                                              |
|-------------|---------------------------------------------------------------------|
| `--size`    | working square resolution (SxS). `exact` is practical up to ~64–80. |
| `--method`  | `exact` (Hungarian, best quality) or `sorted` (fast, for large).    |
| `--space`   | `lab` (perceptual, recommended) or `rgb`.                           |
| `--recolor` | `0` = pure shuffle; `>0` blends toward target colors.               |

## Structure

- `src/pixel_rearrange.py` — the assignment engine (load, match, recolor, save).
- `src/cli.py` — command-line interface.
- `scripts/demo_rearrange.py` — synthetic tree→face proof-of-concept.
- `data/` — example `source.jpg` / `target.jpg`.
- `results/` — output images.

## Status & next steps

Working: exact Hungarian matching in Lab space with a recolor blend, proven on
synthetic and real images at small resolution.

Planned: scale beyond ~64px via **sliced optimal transport** / **Sinkhorn** (the
`sorted` method is the current fast fallback but ignores hue).
