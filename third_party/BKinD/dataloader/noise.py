"""noisykeypoints: image corruption applied on the fly in the data loaders.

Noise is added AFTER the frame is resized to the network's input size, and only to images, never to
labels. Each frame gets its own fixed noise pattern: the random generator is seeded from
(noise seed, video name, frame number), so a frame looks the same in every epoch and every run, as
if it were a recorded noisy video, while different frames (including the two frames of a training
pair) get independent noise. The result is rounded to uint8, as if stored as PNG.

Config (YAML), e.g.:
    noise: {type: poisson_gaussian, photons: 20, read: 0.03, seed: 0}
null / missing = clean frames.

Types:
  poisson_gaussian  camera sensor noise. y = Poisson(x * photons) / photons + Normal(0, read),
                    x in [0, 1]. Fewer photons = more grain (stronger in bright areas); read =
                    std of the sensor's electronic noise.
"""
import os
import zlib

import numpy as np
from PIL import Image


def frame_key(path):
    """(video name, frame number) from .../<video>/image_XXXXXXXX.png."""
    video = os.path.basename(os.path.dirname(path))
    frame = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
    return video, frame


def rng_for(cfg, path):
    video, frame = frame_key(path)
    return np.random.default_rng([int(cfg.get('seed', 0)), zlib.crc32(video.encode()), frame])


def poisson_gaussian(x, rng, photons, read):
    return rng.poisson(x * photons) / photons + rng.normal(0.0, read, x.shape)


NOISES = {'poisson_gaussian': poisson_gaussian}


def apply_noise(img, cfg, path):
    """Corrupt a PIL image (already at network input size) with the noise in `cfg`; returns a PIL
    image of the same mode. Grayscale content stored as RGB gets the same noise in all channels."""
    if not cfg:
        return img
    params = {k: v for k, v in cfg.items() if k not in ('type', 'seed')}
    fn = NOISES[cfg['type']]
    a = np.asarray(img, dtype=np.float64) / 255.0
    gray = a.ndim == 2 or (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]))
    x = a if a.ndim == 2 else (a[..., 0] if gray else a)
    y = fn(x, rng_for(cfg, path), **params)
    y = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
    if a.ndim == 3 and gray:
        y = np.repeat(y[..., None], 3, axis=2)
    return Image.fromarray(y, mode=img.mode)
