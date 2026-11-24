"""noisykeypoints: where the moving things are, found from a longer stretch of video (background subtraction).

Per-frame noise changes from frame to frame, while the arena (fixed camera) stays the same. So the
per-pixel median over many nearby frames is a clean picture of the arena without the mice: they are
elsewhere in most of those frames, and the noise averages away. A frame's foreground = where it
differs from that background, after a light blur. Used as a loss mask (never as a network input), so
the model is only asked to explain the parts of the picture that really changed.

The background is LOCAL in time: the mice kick the bedding around, so a median over a whole video
smears the bedding texture and then every frame's bedding looks like foreground (checked 2026-11-24:
32% of the picture). Each segment of `seg` frames gets the median of the frames sampled every `step`
frames within +-`window` frames of its middle.

The background is computed from the same (noisy) frames the model trains on, never from clean data.
"""
import glob
import os
import re

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from dataloader.noise import apply_noise


def video_frames(video_dir):
    return sorted(glob.glob(os.path.join(video_dir, 'image_*.png')))


def frame_no(path):
    return int(re.search(r'image_(\d+)\.png$', path).group(1))


def to_gray(im):
    """PIL image -> float [0, 1] grey array (CalMS21 frames are grey stored as RGB)."""
    a = np.asarray(im, dtype=np.float32) / 255.0
    return a if a.ndim == 2 else a.mean(-1)


def load_gray(path, noise=None, size=256):
    """Frame as float [0, 1] grey at `size`, corrupted exactly as the loader does (resize, then noise)."""
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    im = apply_noise(im, noise, path) if noise else im
    return to_gray(im)


def compute_backgrounds(video_dir, noise=None, step=60, window=600, seg=300, size=256):
    """Local backgrounds of one video: dict(seg=seg, bgs=(n_segments, size, size) float16).
    Segment k covers frames [k*seg, (k+1)*seg)."""
    frames = {frame_no(p): p for p in video_frames(video_dir)}
    nums = np.array(sorted(frames))
    # one sample frame near every multiple of `step` (the folder may hold only some frames)
    picks = sorted({int(nums[np.abs(nums - t).argmin()]) for t in range(0, nums[-1] + 1, step)})
    imgs = np.stack([load_gray(frames[f], noise, size) for f in picks])
    picks = np.array(picks)
    n_seg = nums[-1] // seg + 1
    bgs = np.empty((n_seg, size, size), np.float16)
    for k in range(n_seg):
        mid = k * seg + seg / 2
        use = np.abs(picks - mid) <= window
        if use.sum() < 5:  # too few frames nearby: take the 5 nearest
            use = np.argsort(np.abs(picks - mid))[:5]
        bgs[k] = np.median(imgs[use], axis=0)
    return dict(seg=seg, bgs=bgs)


def background_for(entry, frame):
    return entry['bgs'][min(frame // entry['seg'], len(entry['bgs']) - 1)].astype(np.float32)


def foreground(frame, background, sigma=2.0, tau=0.1):
    """Soft mask in [0, 1]: blurred |frame - background| / tau, clipped. sigma in pixels at 256 px."""
    return np.clip(gaussian_filter(np.abs(frame - background), sigma) / tau, 0, 1)


def background_cache(root, noise=None, step=60, window=600, seg=300, cache_dir=None):
    """Local backgrounds of every video under `root` (dict video name -> compute_backgrounds entry),
    cached as .npz per noise setting so each is computed once."""
    key = 'clean' if not noise else '_'.join(f'{k}{v}' for k, v in sorted(noise.items()))
    key += f'_st{step}_w{window}_seg{seg}'
    cache_dir = cache_dir or os.path.join(root, '..', 'backgrounds', key)
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for v in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        f = os.path.join(cache_dir, v + '.npz')
        if not os.path.exists(f):
            e = compute_backgrounds(os.path.join(root, v), noise, step, window, seg)
            tmp = f'{f}.{os.getpid()}.tmp.npz'  # write then rename: runs sharing a cache never see half a file
            np.savez(tmp, **e)
            os.replace(tmp, f)
        z = np.load(f)
        out[v] = dict(seg=int(z['seg']), bgs=z['bgs'])
    return out
