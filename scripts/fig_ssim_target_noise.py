"""Figure: BKinD's reconstruction target (1 - local SSIM structure map between frames A and B) under noise.

BKinD's decoder draws to_pred = blur(1 - cs_map(A, B)) (loss/ssim_loss.py compute_ssim: only SSIM's
contrast-structure term, 11x11 Gaussian window sigma 1.5, k2 = 0.03). This script computes that map on
CalMS21 training pairs (frames i, i+6), clean and corrupted exactly as the loader does, and shows that
independent per-frame grain creates a "fake change" where nothing moves, strongest in flat regions,
while blur, H.264 and frozen noise do not. A k2 = 0.1 column shows a noise-tolerant variant.

Regions come from the MARS keypoints (both mice, frames A and B, scaled to 256 px):
  mice  = within 10 px of a keypoint,   still = more than 30 px from every keypoint.
Writes <out>/ssim_target_noise.png, <out>/ssim_target_noise.json. CPU only.

    python scripts/fig_ssim_target_noise.py --data /root/data/CalMS21_bkind/train --out /workspace/fig_ssim
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.noise import BENCH_LEVELS, apply_noise_pair  # noqa: E402
from piq.functional import gaussian_filter  # noqa: E402

W, H, S = 1024, 570, 256
B = BENCH_LEVELS
# name: (label, noise config, k2)
CONDS = {
    'clean': ('clean', None, 0.03),
    'pg98': ('sensor 0.98', dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][1]), 0.03),
    'pg95': ('sensor 0.95', dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][2]), 0.03),
    'sp95': ('speckle 0.95', dict(type='speckle', seed=0, **B['speckle'][2]), 0.03),
    'blur95': ('blur 0.95', dict(type='motion_blur', seed=0, **B['motion_blur'][2]), 0.03),
    'h264_95': ('H.264 0.95', dict(type='h264', **B['h264'][2]), 0.03),
    'pg95_frozen': ('sensor 0.95\nfrozen', dict(type='poisson_gaussian', seed=0, frozen=True,
                                                **B['poisson_gaussian'][2]), 0.03),
    'pg95_k2': ('sensor 0.95\nk2 = 0.1', dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][2]), 0.1),
}
CLEAN_K2 = {'clean_k2': ('clean\nk2 = 0.1', None, 0.1)}


def frame_no(path):
    return int(re.search(r'image_(\d+)\.png', path).group(1))


def gray(im):
    return np.asarray(im.convert('L'), dtype=np.float32) / 255.0


def bkind_target(a, b, k2=0.03, k1=0.01):
    """Same maths as loss/ssim_loss.compute_ssim on grey frames in [0, 1] (all 3 channels equal there)."""
    x = F.pad(torch.from_numpy(a)[None, None], (10,) * 4, mode='replicate')
    y = F.pad(torch.from_numpy(b)[None, None], (10,) * 4, mode='replicate')
    k = gaussian_filter(11, 1.5, dtype=torch.float32)[None]
    mu1, mu2 = F.conv2d(x, k), F.conv2d(y, k)
    s1 = F.conv2d(x * x, k) - mu1 ** 2
    s2 = F.conv2d(y * y, k) - mu2 ** 2
    s12 = F.conv2d(x * y, k) - mu1 * mu2
    c2 = k2 ** 2
    cs = (2 * s12 + c2) / (s1 + s2 + c2)
    t = 1.0 - cs
    t = F.conv2d(t, gaussian_filter(11, 1.0, dtype=torch.float32)[None])
    return t[0, 0].numpy()


def local_std(a):
    x = F.pad(torch.from_numpy(a)[None, None], (5,) * 4, mode='replicate')
    k = gaussian_filter(11, 1.5, dtype=torch.float32)[None]
    mu = F.conv2d(x, k)
    return torch.sqrt(torch.clamp(F.conv2d(x * x, k) - mu ** 2, min=0))[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=150)
    ap.add_argument('--show', type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conds = dict(CONDS, **CLEAN_K2)

    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    kp = np.load(os.path.join(a.data, 'keypoints.npz'))
    rng = np.random.default_rng(0)
    pairs = []
    while len(pairs) < a.pairs:
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(f): os.path.join(a.data, v, f) for f in os.listdir(os.path.join(a.data, v))
                  if f.startswith('image_')}
        starts = [f for f in frames if f + 6 in frames]
        f = starts[rng.integers(len(starts))]
        pairs.append((v, f, frames[f], frames[f + 6]))

    yy, xx = np.mgrid[0:S, 0:S]
    edges = np.geomspace(0.003, 0.3, 13)
    stats = {n: dict(still=[], mice=[], tex_sum=np.zeros(len(edges) - 1), tex_n=np.zeros(len(edges) - 1))
             for n in conds}
    shows = []  # (motion amount, row dict)
    for j, (v, f, pa, pb) in enumerate(pairs):
        ims = [Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR) for p in (pa, pb)]
        clean_a = gray(ims[0])
        tex = local_std(clean_a)
        pts = np.concatenate([kp[f'{v}/keypoints'][f], kp[f'{v}/keypoints'][f + 6]], 0)  # (4, xy, 7)
        px, py = (pts[:, 0] * S / W).reshape(-1), (pts[:, 1] * S / H).reshape(-1)
        d2 = np.min((xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2, axis=-1)
        mice, still = d2 < 10 ** 2, d2 > 30 ** 2
        bins = np.digitize(tex, edges) - 1
        row = dict(A=clean_a)
        for n, (_, cfg, k2) in conds.items():
            na, nb = apply_noise_pair(ims, cfg, [pa, pb])
            t = bkind_target(gray(na), gray(nb), k2)
            st = stats[n]
            st['still'].append(float(t[still].mean()))
            st['mice'].append(float(t[mice].mean()))
            for i in range(len(edges) - 1):
                m = still & (bins == i)
                st['tex_sum'][i] += t[m].sum()
                st['tex_n'][i] += m.sum()
            row[n] = t
            if n == 'pg95':
                row['A_noisy'] = gray(na)
        shows.append((float(np.abs(clean_a - gray(ims[1]))[mice].mean()), row))
        if j % 25 == 0:
            print(f'{j}/{len(pairs)}', flush=True)

    summary = {}
    for n, st in stats.items():
        still, mice = np.mean(st['still']), np.mean(st['mice'])
        summary[n] = dict(label=conds[n][0].replace('\n', ' '), still=still, mice=mice,
                          contrast=mice / max(still, 1e-6),
                          still_by_texture=(st['tex_sum'] / np.maximum(st['tex_n'], 1)).tolist())
        print(f"{n:12s} still {still:.3f}  mice {mice:.3f}  mice/still {mice / max(still, 1e-6):.1f}")
    json.dump(dict(pairs=a.pairs, texture_bin_edges=edges.tolist(), stats=summary),
              open(f'{a.out}/ssim_target_noise.json', 'w'), indent=1)

    # --- figure ---
    shows = [r for _, r in sorted(shows, key=lambda s: -s[0])[:a.show]]  # pairs with clear motion
    map_cols = list(CONDS)
    cols = ['A', 'A_noisy'] + map_cols
    titles = ['frame A', 'frame A\nsensor 0.95'] + [CONDS[n][0] for n in map_cols]
    fig = plt.figure(figsize=(1.6 * len(cols), 1.75 * a.show + 3.6))
    gs = fig.add_gridspec(a.show + 1, len(cols), height_ratios=[1] * a.show + [2.0], hspace=0.25, wspace=0.05)
    for r, row in enumerate(shows):
        for c, (k, t) in enumerate(zip(cols, titles)):
            ax = fig.add_subplot(gs[r, c])
            if k in ('A', 'A_noisy'):
                ax.imshow(row[k], cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(row[k], cmap='magma', vmin=0, vmax=0.8)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(t, fontsize=8)
    half = len(cols) // 2
    ax = fig.add_subplot(gs[a.show, :half])
    centers = np.sqrt(edges[:-1] * edges[1:])
    for n in ['clean', 'pg98', 'pg95', 'sp95', 'blur95', 'h264_95', 'pg95_frozen', 'pg95_k2', 'clean_k2']:
        y = np.array(summary[n]['still_by_texture'])
        ok = stats[n]['tex_n'] > 500
        ax.plot(centers[ok], y[ok], marker='o', ms=3, ls='--' if 'k2' in n else '-',
                label=summary[n]['label'])
    ax.set_xscale('log')
    ax.set_xlabel('local texture of the clean frame (std of grey level, 11x11 window)', fontsize=8)
    ax.set_ylabel('target where nothing moves', fontsize=8)
    ax.legend(fontsize=6, ncol=2)
    ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[a.show, half + 1:])
    names = ['clean', 'pg98', 'pg95', 'sp95', 'blur95', 'h264_95', 'pg95_frozen', 'pg95_k2', 'clean_k2']
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [summary[n]['still'] for n in names], 0.4, label='still floor (> 30 px from mice)')
    ax.bar(xs + 0.2, [summary[n]['mice'] for n in names], 0.4, label='on the mice (< 10 px)')
    ax.set_xticks(xs)
    ax.set_xticklabels([conds[n][0] for n in names], fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('mean target', fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)
    fig.suptitle(f"BKinD's target (1 - local SSIM structure between frames A and B), {a.pairs} CalMS21 pairs",
                 fontsize=10)
    fig.savefig(f'{a.out}/ssim_target_noise.png', dpi=160, bbox_inches='tight')
    print('wrote', f'{a.out}/ssim_target_noise.png')


if __name__ == '__main__':
    main()
