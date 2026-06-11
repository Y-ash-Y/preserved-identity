# src/cli.py
"""Command-line entry point for pixel rearrangement.

Example:
    python -m src.cli --source data/source.jpg --target data/target.jpg \
        --out results/out.png --size 64 --space lab --recolor 0.3
"""
import argparse
from .pixel_rearrange import rearrange


def main():
    p = argparse.ArgumentParser(description="Rearrange a source image's pixels to resemble a target.")
    p.add_argument("--source", required=True, help="image whose pixels are reused")
    p.add_argument("--target", required=True, help="image whose shape to reproduce")
    p.add_argument("--out", default="results/out.png")
    p.add_argument("--size", type=int, default=1024,
                   help="output longer-edge length (target aspect preserved); <=0 = native")
    p.add_argument("--method", choices=["recursive", "exact", "sorted"], default="recursive",
                   help="recursive=scalable OT (default); exact=Hungarian (small); sorted=fastest")
    p.add_argument("--space", choices=["lab", "rgb"], default="lab",
                   help="color space for matching distance")
    p.add_argument("--recolor", type=float, default=0.0,
                   help="0=pure shuffle; >0 blends that fraction toward target colors")
    p.add_argument("--leaf", type=int, default=64,
                   help="block size at which recursive method solves exactly")
    args = p.parse_args()

    rearrange(args.source, args.target, args.out,
              size=(args.size if args.size > 0 else None), method=args.method,
              space=args.space, recolor=args.recolor, leaf=args.leaf)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
