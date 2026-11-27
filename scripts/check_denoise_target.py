"""Can cleaning up the ct4 change map remove leftover grain without losing the white mouse? (CPU only)

ct4 = BKinD's 1 - SSIM change map computed on frames averaged down 4x (loss/change_target.py coarse).
Candidates, all from the noisy frames only:
  bkind        BKinD's full-resolution map (reference)
  bkind_open5  BKinD's map + grey opening 5 px (can morphology alone replace the 4x averaging?)
  ct4          the ct4 map
  ct4_open5 / ct4_open9   ct4 + grey opening (min filter then max filter) with a 5 / 9 px square
  ct4_med5     ct4 + 5 px median filter
  ct4_floor    ct4 minus its own per-image median (the typical fake change on the floor), clipped at 0

Per noise condition and candidate, over random training pairs (frames i, i+6), separately per mouse
(MARS keypoints, < 10 px from a keypoint of that mouse in A or B) against the still floor (> 30 px
from every keypoint):
  ratio_b / ratio_w   mean map on the black / white mouse divided by mean on the still floor
  auc_w               chance a white-mouse pixel beats a still-floor pixel
  keep_w              white-mouse mean relative to the same candidate on clean frames (signal kept)

    python scripts/check_denoise_target.py --data /root/data/CalMS21_bkind/train --out /workspace/denoise_target
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import grey_opening, median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_change_targets import (BENCH_LEVELS, S, W, H, auc, bkind_target, down, load_pair, up)  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.background import frame_no, video_frames  # noqa: E402

B = BENCH_LEVELS
NOISES = {
    'clean': None,
    'pg98': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][1]),
    'pg95': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][2]),
    'pg90': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][3]),
    'sp95': dict(type='speckle', seed=0, **B['speckle'][2]),
    'sp90': dict(type='speckle', seed=0, **B['speckle'][3]),
}
CANDS = ['bkind', 'bkind_open5', 'ct4', 'ct4_open5', 'ct4_open9', 'ct4_med5', 'ct4_floor']
KP = None


def candidates(a, b):
    bk = bkind_target(a, b)
    c4 = up(bkind_target(down(a, 4), down(b, 4)), 4)
    return dict(bkind=bk, bkind_open5=grey_opening(bk, size=5), ct4=c4,
                ct4_open5=grey_opening(c4, size=5), ct4_open9=grey_opening(c4, size=9),
                ct4_med5=median_filter(c4, size=5), ct4_floor=np.maximum(c4 - np.median(c4), 0))


def init_worker(kp_path):
    global KP
    KP = dict(np.load(kp_path))


def pair_job(args):
    j, v, f, pa, pb = args
    rng = np.random.default_rng(j)
    yy, xx = np.mgrid[0:S, 0:S]
    kp = KP[f'{v}/keypoints']                     # (frames, mouse, xy, 7) in 1024 x 570 pixels

    def near(pts):
        px, py = (pts[:, 0] * S / W).reshape(-1), (pts[:, 1] * S / H).reshape(-1)
        return np.min((xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2, axis=-1)
    d_black = near(np.stack([kp[f][0], kp[f + 6][0]]))
    d_white = near(np.stack([kp[f][1], kp[f + 6][1]]))
    black, white = d_black < 10 ** 2, d_white < 10 ** 2
    still = np.minimum(d_black, d_white) > 30 ** 2
    black &= ~white
    white &= ~black
    res = {}
    for n, noise in NOISES.items():
        a, b = load_pair(pa, pb, noise)
        for c, t in candidates(a, b).items():
            res[(n, c)] = dict(black=float(t[black].mean()) if black.any() else np.nan,
                               white=float(t[white].mean()) if white.any() else np.nan,
                               still=float(t[still].mean()),
                               auc_w=auc(t[white], t[still], rng) if white.sum() > 10 else np.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=150)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    rng = np.random.default_rng(0)
    jobs = []
    for j in range(a.pairs):
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(p): p for p in video_frames(os.path.join(a.data, v))}
        starts = [f for f in frames if f + 6 in frames]
        f = int(starts[rng.integers(len(starts))])
        jobs.append((j, v, f, frames[f], frames[f + 6]))
    with Pool(a.workers, initializer=init_worker, initargs=(os.path.join(a.data, 'keypoints.npz'),)) as pool:
        results = list(pool.imap_unordered(pair_job, jobs))

    table = {}
    for n in NOISES:
        for c in CANDS:
            r = [x[(n, c)] for x in results]
            m = {k: float(np.nanmean([x[k] for x in r])) for k in ('black', 'white', 'still', 'auc_w')}
            m['ratio_b'] = m['black'] / max(m['still'], 1e-4)
            m['ratio_w'] = m['white'] / max(m['still'], 1e-4)
            table[(n, c)] = m
    lines = [f"{'noise':6s} {'candidate':12s} {'ratio_b':>8s} {'ratio_w':>8s} {'auc_w':>6s} {'keep_w':>7s} "
             f"{'black':>7s} {'white':>7s} {'still':>7s}"]
    for n in NOISES:
        for c in CANDS:
            t = table[(n, c)]
            keep = t['white'] / max(table[('clean', c)]['white'], 1e-6)
            lines.append(f"{n:6s} {c:12s} {t['ratio_b']:8.1f} {t['ratio_w']:8.1f} {t['auc_w']:6.3f} {keep:7.2f} "
                         f"{t['black']:7.4f} {t['white']:7.4f} {t['still']:7.4f}")
        lines.append('')
    print('\n'.join(lines))
    open(f'{a.out}/denoise_target.txt', 'w').write('\n'.join(lines) + '\n')
    json.dump({f'{n}/{c}': v for (n, c), v in table.items()}, open(f'{a.out}/denoise_target.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
