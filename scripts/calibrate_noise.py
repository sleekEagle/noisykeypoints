"""Calibrate noise levels so every noise type damages frames by the same amount, measured by SSIM.

For each noise type it sweeps one strength parameter over a wide range on sample frames at the
network's input size, records mean SSIM and PSNR against the clean frame, then (with --targets)
interpolates the setting that hits each target SSIM, rounds it, and re-measures it.

The frames are corrupted exactly as the data loader does it (dataloader/noise.py: same per-frame
seed, rounding to uint8). SSIM: Gaussian window 11, sigma 1.5, k1 0.01, k2 0.03 (Wang et al. 2004,
via piq), on the grayscale frame in [0, 1]; mean over frames.

Usage (repo root):
    python scripts/calibrate_noise.py                      # sweep only, writes curves
    python scripts/calibrate_noise.py --targets 0.8 0.6 0.4 0.25
Outputs go to results/noise_calibration/<dataset>/.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import piq
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'BKinD'))
from dataloader import noise as N  # noqa: E402

# One strength parameter per type; the other parameters stay at these values.
# poisson_gaussian: read noise is a fixed 0.6 electrons, i.e. read = 0.6 / photons
# (the pilot setting photons 20, read 0.03 lies on this line).
SWEEPS = {
    'poisson_gaussian': ('photons', np.geomspace(0.5, 5000, 25), lambda v: {'photons': v, 'read': 0.6 / v}),
    'low_light': ('brightness', np.geomspace(0.002, 1.0, 25), lambda v: {'brightness': v}),
    'speckle': ('looks', np.geomspace(0.2, 2000, 25), lambda v: {'looks': v}),
    'motion_blur': ('length', np.geomspace(0.5, 64, 25), lambda v: {'length': v}),
    # integer quantiser; the solved value is rounded to the nearest integer
    'h264': ('qp', np.arange(10, 52, 2), lambda v: {'qp': int(round(v))}),
}
INTEGER = {'h264'}

# Pair types (h264) compress each sampled frame together with its training-pair partner, in its real
# role (frame A = I-frame, frame B = P-frame). Filled in main(): path -> (role, partner path, clean).
PARTNER = {}


def pair_partner(path, gap=6, win_gap=7):
    """The other frame of the training pair (mouse_dataset: pairs (i, i+gap) for i = 0, 7, 14, ...)."""
    d, f = os.path.split(path)
    n = int(os.path.splitext(f)[0].split('_')[-1])
    start = n - n % win_gap
    if n not in (start, start + gap):
        return None, None
    other = start + gap if n == start else start
    return (0 if n == start else 1), os.path.join(d, f'image_{other:08d}.png')


def sample_frames(data, per_video):
    videos = sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d)))
    paths = []
    for v in videos:
        files = sorted(f for f in os.listdir(os.path.join(data, v)) if f.startswith('image_'))
        idx = np.linspace(0, len(files) - 1, per_video).round().astype(int)
        paths += [os.path.join(data, v, files[i]) for i in idx]
    return paths


def load(path, size):
    im = Image.open(path).convert('L')
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float64) / 255.0


def measure(clean, paths, cfg):
    """Mean SSIM and PSNR over frames, corrupting each frame like the data loader does."""
    noisy = []
    for x, p in zip(clean, paths):
        if cfg['type'] in N.PAIR_NOISES:
            role, q, xq = PARTNER[p]
            xs, ps = ([x, xq], [p, q]) if role == 0 else ([xq, x], [q, p])
            y = N.corrupt_pair(xs, cfg, [N.rng_for(cfg, pp) for pp in ps])[role]
        else:
            y = N.corrupt(x, cfg, N.rng_for(cfg, p))
        noisy.append(np.clip(np.round(y * 255.0), 0, 255) / 255.0)
    a = torch.from_numpy(np.stack(clean)[:, None]).float()
    b = torch.from_numpy(np.stack(noisy)[:, None]).float()
    ssim = piq.ssim(b, a, data_range=1.0, reduction='none', downsample=False).numpy()
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3)).numpy()
    psnr = 10 * np.log10(1.0 / np.maximum(mse, 1e-12))
    return float(ssim.mean()), float(ssim.std()), float(psnr.mean())


def round_sig(v, sig=2):
    return float(f'{v:.{sig}g}')


def solve(curve, target):
    """Parameter value whose SSIM equals target, interpolating in log(parameter)."""
    vals = np.array([c['value'] for c in curve])
    ssim = np.array([c['ssim'] for c in curve])
    order = np.argsort(ssim)
    s, lv = ssim[order], np.log(vals[order])
    if not (s[0] <= target <= s[-1]):
        return None
    return float(np.exp(np.interp(target, s, lv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=r'D:\datasets\keypoint\CalMS21_bkind\train')
    ap.add_argument('--name', default='calms21')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--per-video', type=int, default=20)
    ap.add_argument('--targets', type=float, nargs='*', default=None,
                    help='target mean SSIM for levels 1, 2, ... (level 0 = clean)')
    ap.add_argument('--types', nargs='*', default=list(SWEEPS))
    a = ap.parse_args()

    out = os.path.join(ROOT, 'results', 'noise_calibration', a.name)
    os.makedirs(out, exist_ok=True)
    paths = sample_frames(a.data, a.per_video)
    clean = [load(p, a.size) for p in paths]
    for p in paths:
        role, q = pair_partner(p)
        PARTNER[p] = (role, q, load(q, a.size))
    print(f'{len(paths)} frames from {a.data} at {a.size}px')

    curves_file = os.path.join(out, 'curves.json')
    curves = json.load(open(curves_file)) if os.path.exists(curves_file) else {}
    for t in a.types:
        if t in curves and a.targets is not None:
            continue  # reuse the sweep
        pname, values, make = SWEEPS[t]
        curves[t] = []
        for v in values:
            ssim, sd, psnr = measure(clean, paths, {'type': t, **make(float(v))})
            curves[t].append({'value': float(v), 'ssim': ssim, 'ssim_std': sd, 'psnr': psnr})
            print(f'{t:17s} {pname}={v:9.4g}  SSIM {ssim:.3f} (sd {sd:.3f})  PSNR {psnr:5.1f} dB')
        json.dump(curves, open(curves_file, 'w'), indent=1)

    if a.targets is None:
        return

    # Merge with levels already calibrated for other types (same targets), so --types can redo one.
    levels_file = os.path.join(out, 'levels.json')
    levels = {}
    if os.path.exists(levels_file):
        old = json.load(open(levels_file))
        if old['targets'] == a.targets and old['size'] == a.size:
            levels = old['levels']
    for t in a.types:
        pname, _, make = SWEEPS[t]
        levels[t] = [None]
        for i, target in enumerate(a.targets, 1):
            v = solve(curves[t], target)
            if v is None:
                levels[t].append(None)
                continue
            if t in INTEGER:  # try the two neighbouring integers, keep the closer SSIM
                cands = [make(c) for c in sorted({int(np.floor(v)), int(np.ceil(v))})]
            else:
                cands = [{k: round_sig(x, 3) for k, x in make(round_sig(v)).items()}]
            best = None
            for params in cands:
                ssim, sd, psnr = measure(clean, paths, {'type': t, **params})
                if best is None or abs(ssim - target) < abs(best['ssim'] - target):
                    best = {'params': params, 'ssim': ssim, 'ssim_std': sd, 'psnr': psnr}
            levels[t].append(best)
    levels = {t: levels[t] for t in SWEEPS if t in levels}
    json.dump({'targets': a.targets, 'size': a.size, 'frames': len(paths), 'levels': levels},
              open(levels_file, 'w'), indent=1)
    lines = [f'Noise levels for {a.name} at {a.size}px, matched on mean SSIM over {len(paths)} frames',
             f'targets (levels 1..{len(a.targets)}): {a.targets}']
    for t, lv in levels.items():
        lines.append(f'\n{t}')
        for i, l in enumerate(lv[1:], 1):
            lines.append(f'  level {i}: out of range' if l is None else
                         f"  level {i}: {l['params']}  SSIM {l['ssim']:.3f} (sd {l['ssim_std']:.3f})"
                         f"  PSNR {l['psnr']:.1f} dB")
    open(os.path.join(out, 'levels.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    grid(clean, paths, levels, a, out)


def grid(clean, paths, levels, a, out):
    """One frame, rows = noise types, columns = clean + levels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = len(paths) // 2
    x, p = clean[k], paths[k]
    types = list(levels)
    n = len(a.targets) + 1
    fig, axes = plt.subplots(len(types), n, figsize=(2.2 * n, 2.4 * len(types)))
    for r, t in enumerate(types):
        for c in range(n):
            ax = axes[r, c]
            ax.axis('off')
            if c == 0:
                ax.imshow(x, cmap='gray', vmin=0, vmax=1)
                ax.set_title(f'{t}\nclean', fontsize=8)
                continue
            lv = levels[t][c]
            if lv is None:
                continue
            cfg = {'type': t, **lv['params']}
            if t in N.PAIR_NOISES:
                role, q, xq = PARTNER[p]
                xs, ps = ([x, xq], [p, q]) if role == 0 else ([xq, x], [q, p])
                y = N.corrupt_pair(xs, cfg, [N.rng_for(cfg, pp) for pp in ps])[role]
            else:
                y = N.corrupt(x, cfg, N.rng_for(cfg, p))
            y = np.clip(np.round(y * 255), 0, 255) / 255
            ax.imshow(y, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'L{c} SSIM {lv["ssim"]:.2f}\n{lv["psnr"]:.1f} dB', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'grid.png'), dpi=110)
    print('wrote', os.path.join(out, 'grid.png'))


if __name__ == '__main__':
    main()
