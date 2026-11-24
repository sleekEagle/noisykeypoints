"""Check the background-subtraction masks (third_party/BKinD/dataloader/background.py) on CalMS21.

For clean frames and noisy ones, builds each video's background from its own (noisy) frames, then on
random training pairs measures:
  - how much of the picture the mask covers,
  - recall per mouse: share of the tracked MARS keypoints (frame A) inside the mask
    (mouse 0 = black resident, mouse 1 = white intruder), for the mask and, for comparison, for the
    two-frame motion weight used by the warp method (|blur(A) - blur(B)|, sigma 4, tau 0.05),
  - false alarms: mean mask value far (> 25 px) from every keypoint,
  - agreement (IoU) of the noisy mask with the clean one.
Writes <out>/masks.json and <out>/masks_<noise>.png. CPU only.

    python scripts/check_motion_masks.py --data /root/data/CalMS21_bkind/train --out /workspace/mask_check
"""
import argparse
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.background import background_for, compute_backgrounds, foreground, load_gray, video_frames  # noqa: E402
from dataloader.noise import BENCH_LEVELS  # noqa: E402

NOISES = {'clean': None,
          'pg90': dict(type='poisson_gaussian', seed=0, **BENCH_LEVELS['poisson_gaussian'][3]),
          'sp90': dict(type='speckle', seed=0, **BENCH_LEVELS['speckle'][3]),
          'blur90': dict(type='motion_blur', seed=0, **BENCH_LEVELS['motion_blur'][3])}
W, H, S = 1024, 570, 256  # MARS keypoints are in 1024 x 570 pixels; frames are 256 x 256


def frame_no(path):
    return int(re.search(r'image_(\d+)\.png', path).group(1))


def bg_job(args):
    video_dir, noise_name, window = args
    return os.path.basename(video_dir), noise_name, compute_backgrounds(video_dir, NOISES[noise_name], window=window)


def two_frame_weight(a, b):
    return np.clip(np.abs(gaussian_filter(a, 4.0) - gaussian_filter(b, 4.0)) / 0.05, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=200)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--window', type=int, default=600, help='local background: +- frames')
    ap.add_argument('--noises', nargs='+', default=list(NOISES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
    kp = np.load(os.path.join(a.data, 'keypoints.npz'))

    with Pool(a.workers) as pool:
        bgs = {}
        for v, n, bg in pool.imap_unordered(bg_job, [(os.path.join(a.data, v), n, a.window) for v in videos for n in a.noises]):
            bgs[(v, n)] = bg
    for n in a.noises:
        Image.fromarray((bgs[(videos[0], n)]['bgs'][0].astype(np.float32) * 255).astype(np.uint8)).save(
            f'{a.out}/background_{videos[0]}_{n}.png')

    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(a.pairs):
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(p): p for p in video_frames(os.path.join(a.data, v))}
        starts = [f for f in frames if f + 6 in frames]
        f = starts[rng.integers(len(starts))]
        pairs.append((v, frames[f], frames[f + 6], f))

    yy, xx = np.mgrid[0:S, 0:S]
    stats, rows = {}, {n: [] for n in a.noises}
    clean_masks = []
    for n in a.noises:
        rec = {'mask': [[], []], 'two_frame': [[], []]}
        area, far, iou = [], [], []
        for j, (v, pa, pb, f) in enumerate(pairs):
            A, B = load_gray(pa, NOISES[n]), load_gray(pb, NOISES[n])
            bg = background_for(bgs[(v, n)], f)
            mA, mB = foreground(A, bg), foreground(B, bg)
            m = np.maximum(mA, mB)
            w2 = two_frame_weight(A, B)
            pts = kp[f'{v}/keypoints'][f]  # (mouse, xy, 7)
            px = np.clip(np.round(pts[:, 0] * S / W).astype(int), 0, S - 1)
            py = np.clip(np.round(pts[:, 1] * S / H).astype(int), 0, S - 1)
            for mouse in (0, 1):
                rec['mask'][mouse].append(float((mA[py[mouse], px[mouse]] > 0.5).mean()))
                rec['two_frame'][mouse].append(float((w2[py[mouse], px[mouse]] > 0.5).mean()))
            d2 = np.min((xx[..., None] - px.reshape(-1)) ** 2 + (yy[..., None] - py.reshape(-1)) ** 2, axis=-1)
            far.append(float(m[d2 > 25 ** 2].mean()))
            area.append(float((m > 0.5).mean()))
            if n == 'clean':
                clean_masks.append(m > 0.5)
            else:
                c, k = clean_masks[j], m > 0.5
                iou.append(float((c & k).sum() / max(1, (c | k).sum())))
            if j < 4:
                rows[n].append(np.concatenate([A, bg, mA, m, w2], 1))
        stats[n] = dict(mask_area=np.mean(area), false_alarm_far=np.mean(far),
                        recall_black_mask=np.mean(rec['mask'][0]), recall_white_mask=np.mean(rec['mask'][1]),
                        recall_black_two_frame=np.mean(rec['two_frame'][0]),
                        recall_white_two_frame=np.mean(rec['two_frame'][1]),
                        iou_with_clean=np.mean(iou) if iou else 1.0)
        Image.fromarray((np.concatenate(rows[n], 0) * 255).astype(np.uint8)).save(f'{a.out}/masks_{n}.png')
        print(f'window {a.window}', n, ' '.join(f'{k} {v:.3f}' for k, v in stats[n].items()), flush=True)
    json.dump(dict(window=a.window, stats=stats), open(f'{a.out}/masks.json', 'w'), indent=1)
    print('pictures: columns = frame A, background, mask of A, mask of the pair, two-frame weight')


if __name__ == '__main__':
    main()
