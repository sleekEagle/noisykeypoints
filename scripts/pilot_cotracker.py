"""CoTracker3 pilot: do point tracks stay on the mice, and do they survive our noise?

For each clip from scripts/pilot_cotracker_clips.py (64 consecutive frames, 256 px, as BKinD sees them),
a 40 x 40 grid of points (6.4 px apart) is placed on frame 0 and tracked by CoTracker3 (offline model,
torch.hub facebookresearch/co-tracker) through the clean clip and through noisy copies. Noise is the
loader's: per-frame types are seeded by (seed, video, frame), so each frame looks exactly as in
training; H.264 encodes the whole 64-frame clip (one I-frame, then P-frames). Points go through the
tracker in chunks of 400 (memory), so far-apart points don't share attention; the same for every condition.

Measured per condition (positions in pixels at 256 px):
  agreement   how far noisy tracks are from the clean tracks (separately for tracks that move / stay
              still in the clean clip), and how many clean-still tracks look "moving" under noise.
  mask        a track is "moving" if it gets > 4 px from its start. Precision: share of moving tracks
              that start within 15 px of a MARS keypoint. Recall (per mouse): share of MARS keypoints
              that move > 4 px in the clip and have a moving track starting within 8 px.
  vs MARS     for each moving MARS keypoint, the nearest grid track (<= 5 px): error of its 6-frame
              displacement (the training-pair gap) against MARS's. MARS itself is a tracker, so this
              is only a rough check; compare conditions with each other.

Writes <out>/metrics.json, <out>/summary.txt, <out>/tracks.npz, <out>/tracks_grid.png.

    python scripts/pilot_cotracker.py --clips /root/data/calms21_clips \
        --kp /root/data/CalMS21_bkind/train/keypoints.npz --out /workspace/cotracker_pilot
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.noise import BENCH_LEVELS, apply_noise_pair  # noqa: E402

W, H, S = 1024, 570, 256
B = BENCH_LEVELS
CONDS = {
    'clean': None,
    'pg98': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][1]),
    'pg95': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][2]),
    'pg90': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][3]),
    'sp95': dict(type='speckle', seed=0, **B['speckle'][2]),
    'blur95': dict(type='motion_blur', seed=0, **B['motion_blur'][2]),
    'h264_95': dict(type='h264', **B['h264'][2]),
}
MOVE, NEAR, REC, MATCH, GAP = 4.0, 15.0, 8.0, 5.0, 6


def load_clip(root, c):
    paths = [os.path.join(root, c['video'], f'image_{f:08d}.png') for f in range(c['start'], c['start'] + c['length'])]
    return paths, [Image.open(p).convert('RGB') for p in paths]


def track(model, frames, queries, chunk):
    v = torch.from_numpy(np.stack([np.asarray(f, np.float32) for f in frames])).permute(0, 3, 1, 2)[None].cuda()
    trs, viss = [], []
    for i in range(0, queries.shape[1], chunk):
        with torch.no_grad():
            tr, vis = model(v, queries=queries[:, i:i + chunk])
        trs.append(tr[0].float().cpu())
        viss.append(vis[0].float().cpu())
    return torch.cat(trs, 1).numpy(), (torch.cat(viss, 1).numpy() > 0.5)  # (T, N, 2), (T, N)


def max_disp(tr, vis):
    d = np.linalg.norm(tr - tr[:1], axis=-1)
    return np.where(vis, d, 0).max(0)


def mars_clip(kp, c):
    k = kp[f"{c['video']}/keypoints"][c['start']:c['start'] + c['length']].astype(np.float64)  # (T, 2, xy, 7)
    k = k.transpose(0, 1, 3, 2) * np.array([S / W, S / H])  # (T, mouse, 7, xy) at 256 px
    return k


def clip_metrics(tr, vis, tr_c, vis_c, k):
    out = {}
    dm, dm_c = max_disp(tr, vis), max_disp(tr_c, vis_c)
    moving, moving_c = dm > MOVE, dm_c > MOVE
    still_c = dm_c < 2.0
    both = vis & vis_c
    err = np.linalg.norm(tr - tr_c, axis=-1)
    mean_err = np.where(both, err, 0).sum(0) / np.maximum(both.sum(0), 1)
    out['agree_moving_median_px'] = float(np.median(mean_err[moving_c])) if moving_c.any() else np.nan
    out['agree_still_median_px'] = float(np.median(mean_err[still_c])) if still_c.any() else np.nan
    out['agree_frac_under_2px'] = float((mean_err < 2).mean())
    out['still_turned_moving'] = float(moving[still_c].mean()) if still_c.any() else np.nan
    out['still_maxdisp_mean_px'] = float(dm[still_c].mean()) if still_c.any() else np.nan
    out['frac_moving'] = float(moving.mean())
    out['moving_agree_with_clean'] = float((moving == moving_c).mean())
    # mask vs MARS (frame 0 positions)
    start = tr[0]
    allk = k[0].reshape(-1, 2)
    d_any = np.linalg.norm(start[:, None] - allk[None], axis=-1).min(1)
    out['precision'] = float((d_any[moving] < NEAR).mean()) if moving.any() else np.nan
    for m, name in enumerate(('black', 'white')):
        kd = np.linalg.norm(k[:, m] - k[:1, m], axis=-1).max(0)  # (7,) MARS max disp
        mv = kd > MOVE
        if mv.any():
            d = np.linalg.norm(k[0, m][mv][:, None] - start[None], axis=-1)  # (kp, N)
            hit = ((d < REC) & moving[None]).any(1)
            out[f'recall_{name}'] = float(hit.mean())
        else:
            out[f'recall_{name}'] = np.nan
        # 6-frame displacement error vs MARS for matched tracks
        errs = []
        for j in np.where(mv)[0]:
            d = np.linalg.norm(start - k[0, m, j], axis=-1)
            n = int(np.argmin(d))
            if d[n] > MATCH:
                continue
            t = np.arange(len(tr) - GAP)
            ok = vis[t, n] & vis[t + GAP, n]
            dt = (tr[t + GAP, n] - tr[t, n]) - (k[t + GAP, m, j] - k[t, m, j])
            errs += list(np.linalg.norm(dt, axis=-1)[ok])
        out[f'pair_disp_err_{name}_px'] = float(np.median(errs)) if errs else np.nan
        out[f'mars_pair_disp_{name}_px'] = float(np.median(np.linalg.norm(k[GAP:, m] - k[:-GAP, m], axis=-1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True)
    ap.add_argument('--kp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grid', type=int, default=40)
    ap.add_argument('--chunk', type=int, default=400)
    ap.add_argument('--conds', nargs='+', default=list(CONDS))
    ap.add_argument('--max-clips', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    clips = json.load(open(os.path.join(a.clips, 'clips.json')))[:a.max_clips]
    kp = np.load(a.kp)
    kp = {f"{v}/keypoints": kp[f"{v}/keypoints"] for v in {c['video'] for c in clips}}
    model = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').cuda().eval()
    g = (np.arange(a.grid) + 0.5) * S / a.grid
    gx, gy = np.meshgrid(g, g)
    q = np.stack([np.zeros(gx.size), gx.ravel(), gy.ravel()], 1)
    queries = torch.from_numpy(q).float()[None].cuda()

    tracks, per = {}, {n: [] for n in a.conds}
    t0 = time.time()
    for ci, c in enumerate(clips):
        paths, frames = load_clip(a.clips, c)
        k = mars_clip(kp, c)
        res = {}
        for n in a.conds:
            fr = apply_noise_pair(frames, CONDS[n], paths) if CONDS[n] else frames
            res[n] = track(model, fr, queries, a.chunk)
            tracks[f'{ci}/{n}/tracks'] = res[n][0].astype(np.float16)
            tracks[f'{ci}/{n}/vis'] = res[n][1]
        for n in a.conds:
            m = clip_metrics(*res[n], *res['clean'], k)
            m.update(clip=ci, video=c['video'], start=c['start'])
            per[n].append(m)
        print(f"clip {ci + 1}/{len(clips)}  {time.time() - t0:.0f} s  clean moving {per['clean'][-1]['frac_moving']:.3f}"
              f"  prec {per['clean'][-1]['precision']:.2f}", flush=True)

    keys = [k for k in per['clean'][0] if k not in ('clip', 'video', 'start')]
    summary = {n: {k: float(np.nanmean([m[k] for m in per[n]])) for k in keys} for n in a.conds}
    json.dump(dict(clips=clips, per_clip=per, summary=summary,
                   settings=dict(grid=a.grid, chunk=a.chunk, move=MOVE, near=NEAR, rec=REC, match=MATCH, gap=GAP)),
              open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    cols = ['frac_moving', 'precision', 'recall_black', 'recall_white', 'agree_moving_median_px',
            'agree_still_median_px', 'still_turned_moving', 'moving_agree_with_clean',
            'pair_disp_err_black_px', 'pair_disp_err_white_px']
    lines = ['cond      ' + ' '.join(f'{c[:14]:>14s}' for c in cols)]
    for n in a.conds:
        lines.append(f'{n:9s} ' + ' '.join(f'{summary[n][c]:14.3f}' for c in cols))
    lines.append(f"MARS 6-frame displacement (median px): black {summary['clean']['mars_pair_disp_black_px']:.2f}"
                 f"  white {summary['clean']['mars_pair_disp_white_px']:.2f}")
    open(os.path.join(a.out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    np.savez_compressed(os.path.join(a.out, 'tracks.npz'), **tracks)

    # figure: a few clips x conditions, last frame with trails of moving tracks
    show = [n for n in ('clean', 'pg95', 'pg90', 'sp95', 'h264_95') if n in a.conds]
    rows = list(range(0, len(clips), max(1, len(clips) // 4)))[:4]
    fig, axs = plt.subplots(len(rows), len(show), figsize=(3 * len(show), 3 * len(rows)), squeeze=False)
    for r, ci in enumerate(rows):
        paths, frames = load_clip(a.clips, clips[ci])
        k = mars_clip(kp, clips[ci])
        for cc, n in enumerate(show):
            fr = apply_noise_pair(frames, CONDS[n], paths) if CONDS[n] else frames
            tr = tracks[f'{ci}/{n}/tracks'].astype(np.float32)
            vis = tracks[f'{ci}/{n}/vis']
            mv = max_disp(tr, vis) > MOVE
            ax = axs[r, cc]
            ax.imshow(np.asarray(fr[-1])[..., 0], cmap='gray', vmin=0, vmax=255)
            ax.scatter(tr[-1, ~mv, 0], tr[-1, ~mv, 1], s=0.5, c='cyan', alpha=0.4)
            for i in np.where(mv)[0]:
                ax.plot(tr[:, i, 0], tr[:, i, 1], lw=0.5, c='red', alpha=0.6)
            ax.scatter(k[-1, 0, :, 0], k[-1, 0, :, 1], s=6, c='yellow', marker='x')
            ax.scatter(k[-1, 1, :, 0], k[-1, 1, :, 1], s=6, c='lime', marker='x')
            ax.set_xlim(0, S), ax.set_ylim(S, 0), ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{n}: moving {mv.mean():.2f}", fontsize=9)
    fig.suptitle('CoTracker3: moving tracks (red trails), still (cyan), MARS black (yellow x) / white (green x)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'tracks_grid.png'), dpi=130)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
