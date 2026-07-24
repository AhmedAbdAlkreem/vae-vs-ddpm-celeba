"""Turns a list of image frames (e.g. from denoising.py) into an animated GIF."""
import numpy as np


def save_gif(frames, save_path, duration_ms=80):
    import imageio

    uint8_frames = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in frames]
    imageio.mimsave(save_path, uint8_frames, duration=duration_ms / 1000.0, loop=0)
