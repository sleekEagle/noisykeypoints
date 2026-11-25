"""Quick check of the CoTracker3-feature change target (loss/change_target.py feature_target) against
BKinD's target and the 4x-shrunk target, on random CalMS21 training pairs (GPU for the features).

Per noise condition and target: ratio = mean target on the mice (< 10 px from a MARS keypoint in A or B)
/ on still floor (> 30 px from all), auc = chance a mouse pixel beats a still-floor pixel,
corr = correlation with the same target on clean frames. Same definitions as compare_change_targets.py.

    python scripts/check_feature_target.py --data /root/data/CalMS21_bkind/train --out /workspace/feature_target
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'third_party', 'BKinD'))
from compare_change_targets import NOISES, S, W, H, auc, bkind_target, down, load_pair, up  # noqa: E402
from dataloader.background import frame_no, video_frames  # noqa: E402
from loss.change_target import feature_target  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def feat(a, b, size):
    x = np.stack([a, b])[:, None].repeat(3, 1)  # grey -> 3 channels
    x = torch.from_numpy(((x - MEAN) / STD).astype(np.float32)).cuda()
    return feature_target(x[:1], x[1:], size)[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=100)
    ap.add_argument('--sizes', type=int, nargs='+', default=[256, 384])
    ap.add_argument('--noises', nargs='+', default=['clean', 'pg98', 'pg95', 'pg90', 'sp95', 'blur95'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kp = dict(np.load(os.path.join(a.data, 'keypoints.npz')))
    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:S, 0:S]
    cands = ['bkind', 'ds4'] + [f'feat{s}' for s in a.sizes]
    acc = {(n, c): [] for n in a.noises for c in cands}
    for j in range(a.pairs):
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(p): p for p in video_frames(os.path.join(a.data, v))}
        f = int(rng.choice([f for f in frames if f + 6 in frames]))
        pts = np.concatenate([kp[f'{v}/keypoints'][f], kp[f'{v}/keypoints'][f + 6]], 0)
        px, py = (pts[:, 0] * S / W).reshape(-1), (pts[:, 1] * S / H).reshape(-1)
        d2 = np.min((xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2, axis=-1)
        mice, still = d2 < 10 ** 2, d2 > 30 ** 2
        clean_maps = {}
        for n in a.noises:
            ia, ib = load_pair(frames[f], frames[f + 6], NOISES[n])
            maps = dict(bkind=bkind_target(ia, ib), ds4=up(bkind_target(down(ia, 4), down(ib, 4)), 4))
            for s in a.sizes:
                maps[f'feat{s}'] = feat(ia, ib, s)
            if n == 'clean':
                clean_maps = maps
            for c, t in maps.items():
                x, y = clean_maps[c].ravel(), t.ravel()
                acc[(n, c)].append(dict(mice=float(t[mice].mean()), still=float(t[still].mean()),
                                        auc=auc(t[mice], t[still], rng),
                                        corr=float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else float('nan')))
        if (j + 1) % 25 == 0:
            print(f'{j + 1}/{a.pairs} pairs', flush=True)
    lines = [f"{'noise':8s} {'target':8s} {'ratio':>7s} {'auc':>6s} {'corr':>6s} {'mice':>7s} {'still':>7s}"]
    table = {}
    for n in a.noises:
        for c in cands:
            r = acc[(n, c)]
            m, s_ = np.mean([x['mice'] for x in r]), np.mean([x['still'] for x in r])
            t = table[f'{n}/{c}'] = dict(mice=m, still=s_, ratio=m / max(s_, 1e-4), auc=np.mean([x['auc'] for x in r]),
                                         corr=np.nanmean([x['corr'] for x in r]))
            lines.append(f"{n:8s} {c:8s} {t['ratio']:7.1f} {t['auc']:6.3f} {t['corr']:6.3f} {m:7.4f} {s_:7.4f}")
    print('\n'.join(lines))
    open(f'{a.out}/feature_target.txt', 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(pairs=a.pairs, table=table), open(f'{a.out}/feature_target.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
