"""Quick check of multi-frame change targets: BKinD's target (and the 4x-shrunk one) computed on the mean of
k frames centred on A and on B (dataloader/mouse_dataset.py `temporal`). Same metrics as
check_feature_target.py (mouse/floor ratio, auc, corr with clean).

    python scripts/check_temporal_target.py --data /root/data/CalMS21_bkind/train \
        --frames /root/data/CalMS21_allframes/train --out /workspace/temporal_target
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'third_party', 'BKinD'))
from compare_change_targets import NOISES, S, W, H, auc, bkind_target, down, gray, up  # noqa: E402
from dataloader.background import frame_no, video_frames  # noqa: E402
from dataloader.noise import apply_noise_pair  # noqa: E402


def mean_frame(frames_dir, video, f, k, noise):
    r = k // 2
    paths = [os.path.join(frames_dir, video, f'image_{g:08d}.png') for g in range(f - r, f + r + 1)]
    paths = [p for p in paths if os.path.exists(p)]
    ims = apply_noise_pair([Image.open(p).convert('RGB') for p in paths], noise, paths)
    return np.mean([gray(im) for im in ims], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=100)
    ap.add_argument('--noises', nargs='+', default=['clean', 'pg95', 'pg90', 'sp95'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kp = dict(np.load(os.path.join(a.data, 'keypoints.npz')))
    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:S, 0:S]
    cands = ['bkind', 'ds4', 't3', 't3_ds4', 't5_ds4']
    acc = {(n, c): [] for n in a.noises for c in cands}
    for j in range(a.pairs):
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(p): p for p in video_frames(os.path.join(a.data, v))}
        f = int(rng.choice([f for f in frames if f + 6 in frames]))
        pts = np.concatenate([kp[f'{v}/keypoints'][f], kp[f'{v}/keypoints'][f + 6]], 0)
        px, py = (pts[:, 0] * S / W).reshape(-1), (pts[:, 1] * S / H).reshape(-1)
        d2 = np.min((xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2, axis=-1)
        mice, still = d2 < 10 ** 2, d2 > 30 ** 2
        clean = {}
        for n in a.noises:
            m = {k: (mean_frame(a.frames, v, f, k, NOISES[n]), mean_frame(a.frames, v, f + 6, k, NOISES[n]))
                 for k in (1, 3, 5)}
            maps = dict(bkind=bkind_target(*m[1]), ds4=up(bkind_target(down(m[1][0], 4), down(m[1][1], 4)), 4),
                        t3=bkind_target(*m[3]), t3_ds4=up(bkind_target(down(m[3][0], 4), down(m[3][1], 4)), 4),
                        t5_ds4=up(bkind_target(down(m[5][0], 4), down(m[5][1], 4)), 4))
            if n == 'clean':
                clean = maps
            for c, t in maps.items():
                x, y = clean[c].ravel(), t.ravel()
                acc[(n, c)].append(dict(mice=float(t[mice].mean()), still=float(t[still].mean()),
                                        auc=auc(t[mice], t[still], rng),
                                        corr=float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else float('nan')))
        if (j + 1) % 25 == 0:
            print(f'{j + 1}/{a.pairs} pairs', flush=True)
    lines = [f"{'noise':8s} {'target':8s} {'ratio':>7s} {'auc':>6s} {'corr':>6s} {'mice':>7s} {'still':>7s}"]
    for n in a.noises:
        for c in cands:
            r = acc[(n, c)]
            m, s_ = np.mean([x['mice'] for x in r]), np.mean([x['still'] for x in r])
            lines.append(f"{n:8s} {c:8s} {m / max(s_, 1e-4):7.1f} {np.mean([x['auc'] for x in r]):6.3f} "
                         f"{np.nanmean([x['corr'] for x in r]):6.3f} {m:7.4f} {s_:7.4f}")
    print('\n'.join(lines))
    open(f'{a.out}/temporal_target.txt', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
