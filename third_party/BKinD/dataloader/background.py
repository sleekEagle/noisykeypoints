"""noisykeypoints: where the moving things are, found from a long stretch of video (background subtraction).

Per-frame noise changes from frame to frame, while the arena (fixed camera) stays the same. So the
per-pixel median over many frames of a video is a clean picture of the empty arena: the mice are
elsewhere in most frames and the noise averages away. A frame's foreground = where it differs from
that background, after a light blur. Used as a loss mask (never as a network input), so the model is
only asked to explain the parts of the picture that really changed.

The background is computed from the same (noisy) frames the model trains on, never from clean data.
"""
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from dataloader.noise import apply_noise


def video_frames(video_dir):
    return sorted(glob.glob(os.path.join(video_dir, 'image_*.png')))


def load_gray(path, noise=None, size=256):
    """Frame as float [0, 1] grey at `size`, corrupted exactly as the loader does (resize, then noise)."""
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    im = apply_noise(im, noise, path) if noise else im
    return np.asarray(im, dtype=np.float32)[..., 0] / 255.0


def compute_background(video_dir, noise=None, n_frames=300, size=256):
    """Per-pixel median of n_frames frames spread evenly over the video."""
    paths = video_frames(video_dir)
    idx = np.linspace(0, len(paths) - 1, min(n_frames, len(paths))).round().astype(int)
    stack = np.stack([load_gray(paths[i], noise, size) for i in idx])
    return np.median(stack, axis=0)


def foreground(frame, background, sigma=2.0, tau=0.1):
    """Soft mask in [0, 1]: blurred |frame - background| / tau, clipped. sigma in pixels at 256 px."""
    return np.clip(gaussian_filter(np.abs(frame - background), sigma) / tau, 0, 1)


def background_cache(root, noise=None, n_frames=300, cache_dir=None):
    """Backgrounds of every video under `root` (dict video name -> array), cached as .npy per noise
    setting so each is computed once."""
    key = 'clean' if not noise else '_'.join(f'{k}{v}' for k, v in sorted(noise.items()))
    cache_dir = cache_dir or os.path.join(root, '..', 'backgrounds', key)
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for v in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        f = os.path.join(cache_dir, v + '.npy')
        if os.path.exists(f):
            out[v] = np.load(f)
        else:
            out[v] = compute_background(os.path.join(root, v), noise, n_frames)
            np.save(f, out[v])
    return out
