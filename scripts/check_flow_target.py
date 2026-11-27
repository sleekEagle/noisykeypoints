"""Would optical flow (RAFT) be a more noise-robust change target than ct4? (no training)

For random training pairs (frames i, i+6, 256 px, noised as the loader does), computes
  flow   |RAFT flow A -> B| in pixels (torchvision raft_large, 12 iterations)
  ct4    BKinD's change map on frames averaged down 4x (the current best target), for reference
and, separately per mouse (MARS keypoints, < 10 px from that mouse in A or B) against the still floor
(> 30 px from every keypoint):
  ratio_b / ratio_w   mean map on the black / white mouse / mean on the still floor
  auc_b / auc_w       chance a mouse pixel beats a still-floor pixel
  corr                correlation of the noisy map with the same candidate's clean map
  epe_w / epe_b       flow only: mean endpoint error (px) of noisy vs clean flow on that mouse

    python scripts/check_flow_target.py --data /root/data/CalMS21_bkind/train --out /workspace/flow_target
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_change_targets import S, W, H, auc, bkind_target, down, load_pair, up  # noqa: E402
from check_denoise_target import NOISES  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.background import frame_no, video_frames  # noqa: E402

NOISE_NAMES = ['clean', 'pg95', 'pg90', 'sp90']


def masks(kp, f):
    yy, xx = np.mgrid[0:S, 0:S]

    def near(pts):
        px, py = (pts[:, 0] * S / W).reshape(-1), (pts[:, 1] * S / H).reshape(-1)
        return np.min((xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2, axis=-1)
    db, dw = near(np.stack([kp[f][0], kp[f + 6][0]])), near(np.stack([kp[f][1], kp[f + 6][1]]))
    black, white = (db < 100) & ~(dw < 100), (dw < 100) & ~(db < 100)
    return black, white, np.minimum(db, dw) > 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=150)
    ap.add_argument('--iters', type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()

    kpz = np.load(os.path.join(a.data, 'keypoints.npz'))
    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    rng = np.random.default_rng(0)
    rows = {(n, c): [] for n in NOISE_NAMES for c in ('flow', 'ct4')}
    for j in range(a.pairs):
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(p): p for p in video_frames(os.path.join(a.data, v))}
        starts = [f for f in frames if f + 6 in frames]
        f = int(starts[rng.integers(len(starts))])
        black, white, still = masks(kpz[f'{v}/keypoints'], f)
        prng = np.random.default_rng(j)
        maps, flows = {}, {}
        for n in NOISE_NAMES:
            ga, gb = load_pair(frames[f], frames[f + 6], NOISES[n])
            x = torch.from_numpy(np.stack([ga, gb]))[:, None].repeat(1, 3, 1, 1).to(dev) * 2 - 1
            with torch.no_grad():
                fl = net(x[:1], x[1:], num_flow_updates=a.iters)[-1][0].permute(1, 2, 0).cpu().numpy()
            flows[n] = fl
            maps[(n, 'flow')] = np.linalg.norm(fl, axis=-1)
            maps[(n, 'ct4')] = up(bkind_target(down(ga, 4), down(gb, 4)), 4)
        for n in NOISE_NAMES:
            for c in ('flow', 'ct4'):
                t = maps[(n, c)]
                r = dict(black=t[black].mean() if black.any() else np.nan,
                         white=t[white].mean() if white.any() else np.nan, still=t[still].mean(),
                         auc_b=auc(t[black], t[still], prng) if black.sum() > 10 else np.nan,
                         auc_w=auc(t[white], t[still], prng) if white.sum() > 10 else np.nan,
                         corr=np.corrcoef(maps[('clean', c)].ravel(), t.ravel())[0, 1])
                if c == 'flow':
                    e = np.linalg.norm(flows[n] - flows['clean'], axis=-1)
                    r.update(epe_b=e[black].mean() if black.any() else np.nan,
                             epe_w=e[white].mean() if white.any() else np.nan, epe_still=e[still].mean())
                rows[(n, c)].append(r)
        if (j + 1) % 25 == 0:
            print(f'{j + 1}/{a.pairs} pairs', flush=True)

    lines = [f"{'noise':6s} {'target':5s} {'ratio_b':>8s} {'ratio_w':>8s} {'auc_b':>6s} {'auc_w':>6s} {'corr':>6s} "
             f"{'epe_b':>6s} {'epe_w':>6s} {'epe_st':>6s}"]
    table = {}
    for n in NOISE_NAMES:
        for c in ('flow', 'ct4'):
            m = {k: float(np.nanmean([r[k] for r in rows[(n, c)]])) for k in rows[(n, c)][0]}
            m['ratio_b'], m['ratio_w'] = m['black'] / max(m['still'], 1e-4), m['white'] / max(m['still'], 1e-4)
            table[f'{n}/{c}'] = m
            e = (f"{m['epe_b']:6.2f} {m['epe_w']:6.2f} {m['epe_still']:6.2f}" if c == 'flow' else '')
            lines.append(f"{n:6s} {c:5s} {m['ratio_b']:8.1f} {m['ratio_w']:8.1f} {m['auc_b']:6.3f} {m['auc_w']:6.3f} "
                         f"{m['corr']:6.3f} {e}")
        lines.append('')
    print('\n'.join(lines))
    open(f'{a.out}/flow_target.txt', 'w').write('\n'.join(lines) + '\n')
    json.dump(table, open(f'{a.out}/flow_target.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
