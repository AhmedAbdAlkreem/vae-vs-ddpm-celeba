#!/usr/bin/env python3
"""
Single-image / single-sample inference.

    python inference.py --reconstruct path/to/face.jpg
    python inference.py --generate ddpm
    python inference.py --generate vae
"""
import argparse
import os

from torchvision.utils import save_image

from configs.config import Config
from src.engine.inferencer import GenAIInferencer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruct", type=str, default=None,
                         help="Path to an image to encode+decode through the VAE")
    parser.add_argument("--generate", choices=["vae", "ddpm"], default=None,
                         help="Generate a single new image from the given model")
    parser.add_argument("--out_dir", type=str, default="outputs/inference")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(args.out_dir, exist_ok=True)
    inferencer = GenAIInferencer(cfg)

    if args.reconstruct:
        original, recon = inferencer.reconstruct_image(args.reconstruct)
        save_image(original, os.path.join(args.out_dir, "original.png"))
        save_image(recon, os.path.join(args.out_dir, "reconstructed.png"))
        print(f"Saved original + reconstruction to {args.out_dir}")

    if args.generate:
        img = inferencer.generate_one(source=args.generate)
        path = os.path.join(args.out_dir, f"generated_{args.generate}.png")
        save_image(img, path)
        print(f"Saved generated image to {path}")

    if not args.reconstruct and not args.generate:
        parser.print_help()


if __name__ == "__main__":
    main()
