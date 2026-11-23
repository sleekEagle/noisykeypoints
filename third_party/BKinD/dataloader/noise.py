"""noisykeypoints: image corruption applied on the fly in the data loaders.

Noise is added AFTER the frame is resized to the network's input size, and only to images, never to
labels. Each frame gets its own fixed noise pattern: the random generator is seeded from
(noise seed, video name, frame number), so a frame looks the same in every epoch and every run, as
if it were a recorded noisy video, while different frames (including the two frames of a training
pair) get independent noise. The result is rounded to uint8, as if stored as PNG.

Config (YAML), either explicit parameters or a calibrated level (see NOISE_LEVELS below):
    noise: {type: poisson_gaussian, photons: 20, read: 0.03, seed: 0}
    noise: {type: motion_blur, level: 3, seed: 0}
    noise: {type: poisson_gaussian, level: 2, frozen: true}
null / missing = clean frames.

frozen: true  = "frozen noise" control: the generator is seeded from (seed, video) only, so every
                frame of a video gets the SAME noise pattern (same blur direction, same grain).

Types (x = clean image in [0, 1]):
  poisson_gaussian  camera sensor noise. y = Poisson(x * photons) / photons + Normal(0, read).
                    photons = how many photons a white pixel collects (fewer = more grain, stronger
                    in bright areas); read = std of the sensor's electronic noise.
  low_light         dark scene, no gain: y = Poisson(x * brightness * photons) / photons
                    + Normal(0, read). Brightness and contrast drop, and the few photons left make
                    the image grainy. Defaults photons=1000, read=0.002 (a clean sensor at full light,
                    SSIM ~0.9), so the damage comes from the darkness.
  speckle           multiplicative speckle, as in ultrasound: y = x * G, G ~ Gamma(looks, 1/looks)
                    (mean 1, variance 1/looks). Fewer looks = stronger. Independent per pixel (no
                    grain size); not realistic for natural video, used as a bridge to echo.
  motion_blur       straight-line blur of `length` pixels (at input resolution) in a random
                    direction per frame, as from camera shake during a slow shutter.
  h264              H.264 compression of the training PAIR: frames A and B are encoded together as
                    a 2-frame clip (A = I-frame, B = P-frame predicted from A) by libx264 at a fixed
                    quantiser `qp` (0-51, higher = coarser), then decoded. Needs PyAV (`pip install
                    av`). Deterministic and has no random seed, so `seed` and `frozen` do nothing.
                    Pair-level: use apply_noise_pair (apply_noise on a single frame encodes it alone,
                    as an I-frame).
"""
import math
import os
import zlib

import numpy as np
from PIL import Image


def frame_key(path):
    """(video name, frame number) from .../<video>/image_XXXXXXXX.png."""
    video = os.path.basename(os.path.dirname(path))
    frame = int(os.path.splitext(os.path.basename(path))[0].split('_')[-1])
    return video, frame


def poisson_from_uniform(u, lam):
    """Exact Poisson(lam) samples by inverse CDF from uniforms u in (0, 1): the smallest k with
    P(X <= k) >= u. Starts from the normal approximation and steps to the exact value."""
    from scipy.special import gammaln, ndtri, pdtr
    shape = np.shape(lam)
    lam = np.asarray(lam, np.float64).ravel()
    u = np.asarray(u, np.float64).ravel()
    k = np.maximum(0.0, np.floor(lam + np.sqrt(lam) * ndtri(u)))
    todo = np.arange(lam.size)                                  # pixels not yet settled
    for _ in range(200):
        kk, ll, uu = k[todo], lam[todo], u[todo]
        cdf = pdtr(kk, ll)                                      # P(X <= k)
        pmf = np.where(ll > 0, np.exp(kk * np.log(np.where(ll > 0, ll, 1.0)) - ll - gammaln(kk + 1)),
                       (kk == 0).astype(np.float64))
        up = cdf < uu                                           # k too small
        down = ~up & (kk >= 1) & (cdf - pmf >= uu)              # P(X <= k-1) >= u: k too big
        k[todo[up]] += 1
        k[todo[down]] -= 1
        todo = todo[up | down]
        if todo.size == 0:
            return k.reshape(shape)
    raise RuntimeError('poisson_from_uniform did not converge')


class FrozenRNG:
    """Random generator for frozen noise. numpy's poisson() uses a variable number of random draws
    per pixel, depending on the pixel's brightness, so two frames that differ anywhere (a moving
    mouse, JPEG grain) fall out of step and get different noise from there on. Here poisson() uses
    exactly one uniform per pixel (inverse CDF), so the same seed gives the same noise pattern on
    every frame wherever the frames agree. Other draws (normal, gamma, uniform) don't depend on the
    image, so they stay in step and are passed through unchanged."""

    def __init__(self, gen):
        self.gen = gen

    def poisson(self, lam):
        lam = np.asarray(lam, np.float64)
        u = np.clip(self.gen.random(lam.shape), 1e-12, 1 - 1e-12)
        return poisson_from_uniform(u, lam)

    def __getattr__(self, name):
        return getattr(self.gen, name)


def rng_for(cfg, path):
    video, frame = frame_key(path)
    key = [int(cfg.get('seed', 0)), zlib.crc32(video.encode())]
    if cfg.get('frozen', False):
        return FrozenRNG(np.random.default_rng(key))
    key.append(frame)
    return np.random.default_rng(key)


def poisson_gaussian(x, rng, photons, read):
    return rng.poisson(x * photons) / photons + rng.normal(0.0, read, x.shape)


def low_light(x, rng, brightness, photons=1000.0, read=0.002):
    return poisson_gaussian(x * brightness, rng, photons, read)


def speckle(x, rng, looks):
    return x * rng.gamma(looks, 1.0 / looks, x.shape)


def line_kernel(length, angle):
    """Normalised kernel for a straight blur of `length` pixels at `angle` radians. Sub-pixel lengths
    work: the line is sampled densely and spread onto pixels with bilinear weights."""
    r = length / 2.0
    size = 2 * int(math.ceil(r)) + 3
    c = size // 2
    k = np.zeros((size, size), np.float64)
    n = max(2, int(math.ceil(length * 8)))
    for t in np.linspace(-r, r, n):
        px, py = c + t * math.cos(angle), c + t * math.sin(angle)
        x0, y0 = int(math.floor(px)), int(math.floor(py))
        fx, fy = px - x0, py - y0
        k[y0, x0] += (1 - fx) * (1 - fy)
        k[y0, x0 + 1] += fx * (1 - fy)
        k[y0 + 1, x0] += (1 - fx) * fy
        k[y0 + 1, x0 + 1] += fx * fy
    return k / k.sum()


def motion_blur(x, rng, length):
    import cv2
    k = line_kernel(float(length), rng.uniform(0.0, math.pi))
    return cv2.filter2D(x.astype(np.float32), -1, k.astype(np.float32),
                        borderType=cv2.BORDER_REFLECT).astype(np.float64)


def h264_frames(frames, qp):
    """Encode a list of float images in [0, 1] (2-D grayscale or H x W x 3 RGB, even sizes) as one
    H.264 clip with libx264 at a constant quantiser, decode it, and return the decoded float images.
    The first frame is an I-frame, the rest P-frames (no B-frames, no scene-cut I-frames)."""
    import io
    import av
    gray = frames[0].ndim == 2
    fmt = 'gray' if gray else 'rgb24'
    u8 = [np.clip(np.round(f * 255.0), 0, 255).astype(np.uint8) for f in frames]
    buf = io.BytesIO()
    with av.open(buf, 'w', format='h264') as c:
        s = c.add_stream('libx264', rate=30)
        s.width, s.height = u8[0].shape[1], u8[0].shape[0]
        s.pix_fmt = 'yuv420p'
        s.options = {'qp': str(int(qp)), 'preset': 'medium',
                     'x264-params': 'scenecut=0:bframes=0:threads=1:keyint=250'}
        for f in u8:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format=fmt).reformat(format='yuv420p')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    buf.seek(0)
    with av.open(buf, 'r', format='h264') as c:
        out = [f.to_ndarray(format=fmt).astype(np.float64) / 255.0 for f in c.decode(video=0)]
    assert len(out) == len(frames), (len(out), len(frames))
    return out


def h264(x, rng, qp):
    return h264_frames([x], qp)[0]


NOISES = {'poisson_gaussian': poisson_gaussian, 'low_light': low_light, 'speckle': speckle,
          'motion_blur': motion_blur, 'h264': h264}
PAIR_NOISES = {'h264': lambda xs, rngs, qp: h264_frames(xs, qp)}

# Calibrated levels (scripts/calibrate_noise.py, CalMS21 at 256x256): for each type, the setting
# whose mean SSIM against the clean frame is closest to the level's target. Level 0 = clean.
# Calibrated 2026-11-16 on 200 frames (20 per no-cable train video), results/noise_calibration/calms21/.
# Motion blur (SSIM >= ~0.46 even at 64 px) and H.264 (SSIM >= ~0.51 even at qp 51) keep the coarse
# image, so they cannot reach SSIM 0.4 / 0.2 and only have levels 1-2 (decided 2026-11-16).
SSIM_TARGETS = [1.0, 0.8, 0.6, 0.4, 0.2]
NOISE_LEVELS = {
    'poisson_gaussian': [None, {'photons': 370.0, 'read': 0.00162}, {'photons': 130.0, 'read': 0.00462},
                         {'photons': 51.0, 'read': 0.0118}, {'photons': 17.0, 'read': 0.0353}],
    'low_light': [None, {'brightness': 0.66}, {'brightness': 0.45}, {'brightness': 0.31},
                  {'brightness': 0.17}],
    'speckle': [None, {'looks': 130.0}, {'looks': 44.0}, {'looks': 18.0}, {'looks': 5.8}],
    'motion_blur': [None, {'length': 5.0}, {'length': 12.0}, None, None],
    'h264': [None, {'qp': 39}, {'qp': 46}, None, None],
}


# Benchmark levels (decided 2026-11-23, replacing the levels above for the paper): levels 1-4 =
# SSIM 0.99 / 0.98 / 0.95 / 0.90, around where BKinD breaks. Measured on the same 200 CalMS21 frames
# at 256 px (results/noise_calibration/calms21/bench_2026-11-23.json). Runs use these as explicit
# parameters (scripts/run_specs.py: bench(kind, ssim, seed)), named CalMS21_<kind>_ssim<NN>_s<seed>.
# `level: N` in a config still means NOISE_LEVELS above, so earlier runs keep their meaning.
# Low light is not in this table (user, 2026-11-23): at SSIM >= 0.9 it is either sensor noise with
# gain (same maths) or plain dimming without it. It keeps its old SSIM-0.6 run as a "severe" result.
BENCH_SSIM = [99, 98, 95, 90]
BENCH_LEVELS = {  # measured SSIM in the comments
    'poisson_gaussian': [{'photons': p, 'read': 0.6 / p} for p in (10000.0, 5000.0, 1900.0, 880.0)],
    # 0.990 / 0.980 / 0.950 / 0.901
    'speckle': [{'looks': v} for v in (3600.0, 1750.0, 650.0, 300.0)],      # 0.990 / 0.980 / 0.950 / 0.901
    'motion_blur': [{'length': v} for v in (0.86, 1.27, 2.1, 3.18)],          # 0.990 / 0.981 / 0.951 / 0.900
    'h264': [{'qp': v} for v in (18, 23, 29, 34)],                            # 0.990 / 0.980 / 0.949 / 0.896
}


def params_for(cfg):
    """Noise function parameters from a config: a calibrated `level`, or explicit parameters."""
    if 'level' in cfg:
        p = NOISE_LEVELS[cfg['type']][int(cfg['level'])]
        if p is None:
            raise ValueError(f"no calibrated level {cfg['level']} for noise type {cfg['type']}")
        return dict(p)
    return {k: v for k, v in cfg.items() if k not in ('type', 'seed', 'frozen')}


def apply_noise(img, cfg, path):
    """Corrupt a PIL image (already at network input size) with the noise in `cfg`; returns a PIL
    image of the same mode. Grayscale content stored as RGB gets the same noise in all channels."""
    return apply_noise_pair([img], cfg, [path])[0]


def apply_noise_pair(imgs, cfg, paths):
    """Corrupt a list of PIL images from one video (e.g. the two frames of a training pair) with the
    noise in `cfg`. Per-frame types corrupt each frame on its own; pair types (h264) see them all."""
    if not cfg or cfg.get('level', None) == 0:
        return list(imgs)
    arrs = [np.asarray(im, dtype=np.float64) / 255.0 for im in imgs]
    gray = all(a.ndim == 2 or (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2]))
               for a in arrs)
    xs = [a if a.ndim == 2 else (a[..., 0] if gray else a) for a in arrs]
    ys = corrupt_pair(xs, cfg, [rng_for(cfg, p) for p in paths])
    out = []
    for im, a, y in zip(imgs, arrs, ys):
        y = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
        if a.ndim == 3 and gray:
            y = np.repeat(y[..., None], 3, axis=2)
        out.append(Image.fromarray(y, mode=im.mode))
    return out


def corrupt(x, cfg, rng):
    """Float image in [0, 1] -> corrupted float image (not clipped or rounded)."""
    return NOISES[cfg['type']](x, rng, **params_for(cfg))


def corrupt_pair(xs, cfg, rngs):
    """List of float images in [0, 1] -> corrupted float images (pair types see all frames)."""
    if cfg['type'] in PAIR_NOISES:
        return PAIR_NOISES[cfg['type']](xs, rngs, **params_for(cfg))
    return [corrupt(x, cfg, r) for x, r in zip(xs, rngs)]
