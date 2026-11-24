"""Precompute CoTracker3 point tracks for the warp model's track loss (step 1d; dataloader/tracks.py).

Every-frame video (scripts/prepare_calms21_bkind.py --all-frames, 256 px) is cut into windows of 63
frames = 9 training pairs (i, i+6), i = 0, 7, 14, ... (the mouse loader's pairs), so every pair lies
inside one window. In each window a 32 x 32 grid of points (8 px apart) is placed on two frames (15 and 47)
and tracked through the whole window by CoTracker3 offline (forward and backward; default resolution
384 x 512, 1024 points per call). Frames are corrupted exactly as the loader does (dataloader/noise.py,
seeded by video and frame number), so the tracker sees the same noisy frames the model trains on.

A track "moves" if it gets > --move px from its query position anywhere in the window. For each pair,
the moving tracks visible in both frames are saved (positions in A and B). Output:
<out>/<noise key>/<video>.npz (format in dataloader/tracks.py) and <out>/<noise key>/settings.json.
Finished videos are skipped, so a stopped run can be restarted.

    python scripts/precompute_tracks.py --frames /root/data/CalMS21_allframes/train --out /workspace/tracks
    python scripts/precompute_tracks.py ... --bench pg 95      # noise of run_specs.bench('pg', 95, seed)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'third_party', 'BKinD'))
sys.path.insert(0, HERE)
from dataloader.noise import apply_noise_pair  # noqa: E402
from dataloader.tracks import tracks_key  # noqa: E402

WIN, GAP, STRIDE = 63, 6, 7


class Windows(torch.utils.data.Dataset):
    """Window w of a video: (start frame, uint8 frames (T, 256, 256, 3)), noised as in training."""

    def __init__(self, video_dir, n_frames, noise):
        self.dir, self.n, self.noise = video_dir, n_frames, noise
        self.starts = list(range(0, n_frames - GAP, WIN))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        paths = [os.path.join(self.dir, f'image_{f:08d}.png') for f in range(s, min(s + WIN, self.n))]
        ims = [Image.open(p).convert('RGB') for p in paths]
        if self.noise:
            ims = apply_noise_pair(ims, self.noise, paths)
        return s, np.stack([np.asarray(im) for im in ims])


def track_window(model, frames, grid, query_frames, chunk):
    t = len(frames)
    v = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2)[None].float()
    qs = [np.concatenate([np.full((len(grid), 1), min(q, t - 1)), grid], 1) for q in query_frames]
    q = torch.from_numpy(np.concatenate(qs)).float()[None].cuda()
    trs, viss = [], []
    for i in range(0, q.shape[1], chunk):
        with torch.no_grad():
            tr, vis = model(v, queries=q[:, i:i + chunk])
        trs.append(tr[0].float().cpu())
        viss.append(vis[0].float().cpu())
    return torch.cat(trs, 1).numpy(), torch.cat(viss, 1).numpy() > 0.5, q[0, :, 0].long().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, help='folder of every-frame video folders')
    ap.add_argument('--out', required=True)
    ap.add_argument('--bench', nargs=2, default=None, metavar=('KIND', 'SSIM'),
                    help="noise of run_specs.bench(KIND, SSIM, 1), e.g. pg 95; default clean")
    ap.add_argument('--videos', nargs='+', default=None)
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--query-frames', type=int, nargs='+', default=[15, 47])
    ap.add_argument('--chunk', type=int, default=1024)
    ap.add_argument('--move', type=float, default=4.0)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()

    noise = None
    if a.bench:
        import run_specs
        noise = run_specs.bench(a.bench[0], int(a.bench[1]), 1)['overrides']['noise']
        if noise['type'] == 'h264':
            raise SystemExit('h264 is encoded per training pair in the loader; tracks for it are not supported')
    out = os.path.join(a.out, tracks_key(noise))
    os.makedirs(out, exist_ok=True)
    settings = dict(noise=noise, window=WIN, grid=a.grid, query_frames=a.query_frames, chunk=a.chunk,
                    move=a.move, model='cotracker3_offline (torch.hub facebookresearch/co-tracker)')
    json.dump(settings, open(os.path.join(out, 'settings.json'), 'w'), indent=1)
    print('noise:', noise, '->', out, flush=True)

    model = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').cuda().eval()
    g = (np.arange(a.grid) + 0.5) * 256 / a.grid
    gx, gy = np.meshgrid(g, g)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)

    videos = a.videos or sorted(d for d in os.listdir(a.frames) if os.path.isdir(os.path.join(a.frames, d)))
    t0 = time.time()
    for vi, video in enumerate(videos):
        dst = os.path.join(out, video + '.npz')
        if os.path.exists(dst):
            print(f'{video}: done already', flush=True)
            continue
        vdir = os.path.join(a.frames, video)
        n = len([f for f in os.listdir(vdir) if f.startswith('image_')])
        assert os.path.exists(os.path.join(vdir, f'image_{n - 1:08d}.png')), f'{video}: frames not 0..{n - 1}'
        ds = Windows(vdir, n, noise)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=a.workers, prefetch_factor=4)
        frames_out, counts, pa, pb = [], [], [], []
        tv = time.time()
        for wi, (s, fr) in enumerate(dl):
            s, fr = int(s), fr.numpy()
            tr, vis, qf = track_window(model, fr, grid, a.query_frames, a.chunk)
            start_pos = tr[qf, np.arange(tr.shape[1])]
            d = np.linalg.norm(tr - start_pos[None], axis=-1)
            moving = np.where(vis, d, 0).max(0) > a.move
            for i in range(0, len(fr) - GAP, STRIDE):
                keep = moving & vis[i] & vis[i + GAP]
                frames_out.append(s + i)
                counts.append(int(keep.sum()))
                pa.append(tr[i, keep])
                pb.append(tr[i + GAP, keep])
            if wi % 50 == 0:
                print(f'{video} window {wi + 1}/{len(ds)}  {time.time() - tv:.0f} s  '
                      f'moving points per pair {np.mean(counts[-9:]):.0f}', flush=True)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        tmp = dst + '.tmp.npz'
        np.savez_compressed(tmp, frames=np.array(frames_out, np.int32), offsets=offsets,
                            a=np.concatenate(pa).astype(np.float16), b=np.concatenate(pb).astype(np.float16))
        os.replace(tmp, dst)
        print(f'{video}: {len(frames_out)} pairs, median {int(np.median(counts))} moving points per pair, '
              f'{time.time() - tv:.0f} s (total {time.time() - t0:.0f} s, video {vi + 1}/{len(videos)})', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
