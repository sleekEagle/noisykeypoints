"""Test-time clean-up of a trained BKinD model's keypoints with CoTracker3 (CalMS21; no training).

Every frame of the 10 no-cable videos is cut into windows of --win consecutive frames (noised as given by
--noise, exactly as the loader does). In each window:
  1. the detector (BKinD's keypoint net) gives keypoints and a confidence (heatmap peak) on every frame;
  2. for each keypoint, its --anchors most confident frames in the window are anchors: CoTracker3
     (offline, torch.hub facebookresearch/co-tracker) tracks the detected position at each anchor through
     the whole window (forward and backward), on the same noisy frames;
  3. the per-frame estimate is a mix of the detection and the anchor tracks.
Confidence is divided by its median over the video for that keypoint (keypoints differ in peak height),
so cn = 1 is a typical frame. Variants written (all in [-1, 1], eval_calms21.py --keypoints format):
  raw      the detector alone
  track1   the track from the single most confident anchor
  tracks   mean of the anchor tracks, weighted by anchor cn x visibility
  fuse     weighted mean of the detection (weight cn) and the anchor tracks (weight anchor cn x visibility)
  gate     the detection where cn >= --gate, else `tracks`
Also jitter (mean |p(t+1) - 2p(t) + p(t-1)|, px at 1024 x 570, inside windows only) per variant, and how
often each keypoint's detection is far (> 8 px at 256) from `tracks`, by confidence quartile (does low
confidence mark bad detections?).

    python scripts/track_refine.py --run CalMS21_ct4_pg_ssim90_s1 --out /workspace/track_refine/ct4_pg90
    python scripts/track_refine.py --run CalMS21_clean_s1 --noise pg90 --out /workspace/track_refine/clean_pg90
then score each variant with eval_calms21.py --run <run> --keypoints <out>/<variant>_all.npz --out <out>/eval_<variant>.
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
sys.path.insert(0, HERE)
from time_model import NO_CABLE, S, W, H, apply_noise_pair, detect, load_bkind, to_tensor, video_frames  # noqa: E402
from eval_calms21 import eval_noise_spec  # noqa: E402

VARIANTS = ['raw', 'track1', 'tracks', 'fuse', 'gate']


class Windows(torch.utils.data.Dataset):
    def __init__(self, paths, win, noise):
        starts = list(range(0, len(paths), win))
        if len(starts) > 1 and len(paths) - starts[-1] < win // 2:
            starts.pop()  # a short last piece joins the window before it
        self.wins = [paths[s:e] for s, e in zip(starts, starts[1:] + [len(paths)])]
        self.noise = noise

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, i):
        ims = [Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR) for p in self.wins[i]]
        if self.noise:
            ims = apply_noise_pair(ims, self.noise, self.wins[i])
        return torch.from_numpy(np.stack([np.asarray(im) for im in ims]))  # (T, S, S, 3) uint8


def to_px(k):
    return (k + 1) / 2 * S


def to_unit(p):
    return p / S * 2 - 1


def run_video(a, model, tracker, paths, noise, device):
    dl = torch.utils.data.DataLoader(Windows(paths, a.win, noise), batch_size=None, num_workers=a.workers)
    det, conf, trk, vis, aconf = [], [], [], [], []
    for v in dl:
        T = v.shape[0]
        x = torch.stack([to_tensor(f) for f in v.numpy()]).to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            k, c = detect(model, x)
        k, c = k.float(), c.float()  # (T, K, 2) in [-1, 1], (T, K)
        A = min(a.anchors, T)
        top = c.argsort(0, descending=True)[:A].T  # (K, A) frame index of each anchor
        K = k.shape[1]
        kk = torch.arange(K, device=device)[:, None].expand(K, A)
        q = torch.cat([top[..., None].float(), to_px(k[top, kk])], -1).reshape(1, K * A, 3)
        with torch.no_grad():
            tr, vi = tracker(v.permute(0, 3, 1, 2)[None].float().to(device), queries=q)
        det.append(k.cpu().numpy())
        conf.append(c.cpu().numpy())
        trk.append(to_unit(tr[0].float()).reshape(T, K, A, 2).cpu().numpy())
        vis.append((vi[0].float() > 0.5).reshape(T, K, A).cpu().numpy())
        aconf.append(np.broadcast_to(c[top, kk].cpu().numpy()[None], (T, K, A)))
    return det, conf, trk, vis, aconf


def fuse(det, conf, trk, vis, aconf, med, gate):
    """Per window -> dict variant -> (T, K, 2)."""
    cn = conf / med  # (T, K)
    an = aconf / med[None, :, None]  # (T, K, A)
    w = an * vis + 1e-6
    tracks = (w[..., None] * trk).sum(2) / w.sum(2)[..., None]
    wf = np.concatenate([cn[..., None], an * vis], -1) + 1e-6  # (T, K, 1 + A)
    allp = np.concatenate([det[:, :, None], trk], 2)
    return dict(raw=det, track1=trk[:, :, 0], tracks=tracks,
                fuse=(wf[..., None] * allp).sum(2) / wf.sum(2)[..., None],
                gate=np.where((cn >= gate)[..., None], det, tracks)), cn


def jitter(windows):
    acc = []
    for p in windows:
        p = np.stack([(p[..., 0] + 1) / 2 * W, (p[..., 1] + 1) / 2 * H], -1)
        if len(p) >= 3:
            acc.append(np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1).ravel())
    acc = np.concatenate(acc)
    return dict(jitter_px=float(acc.mean()), jitter_median_px=float(np.median(acc)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--runs', default='/workspace/runs')
    ap.add_argument('--noise', default='train', help="'train' (the run's own noise), 'clean', or a cell such as pg90")
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', default='/root/data/CalMS21_allframes/train', help='every-frame folder')
    ap.add_argument('--win', type=int, default=32)
    ap.add_argument('--anchors', type=int, default=3)
    ap.add_argument('--gate', type=float, default=1.0, help='gate: keep detections with cn >= this')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None, help='frames per video (quick test)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = 'cuda'
    model, cfg = load_bkind(a.run, a.runs, device)
    noise = cfg.get('noise') if a.noise == 'train' else eval_noise_spec(a.noise)
    if noise and noise.get('type') == 'h264':
        raise SystemExit('h264 not supported (it encodes frames together)')
    print(f'noise {noise}', flush=True)
    tracker = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline').to(device).eval()
    out = {k: [] for k in VARIANTS}
    wins = {k: [] for k in VARIANTS}
    meta, far = [], []
    for vname in sorted(os.listdir(a.frames)):
        if vname[:8] not in NO_CABLE:
            continue
        paths, frames = video_frames(os.path.join(a.frames, vname))
        paths, frames = paths[:a.limit], frames[:a.limit]
        t0 = time.time()
        det, conf, trk, vis, aconf = run_video(a, model, tracker, paths, noise, device)
        med = np.median(np.concatenate(conf), 0)  # (K,)
        for w in zip(det, conf, trk, vis, aconf):
            res, cn = fuse(*w, med, a.gate)
            for k in VARIANTS:
                out[k].append(res[k])
                wins[k].append(res[k])
            d = np.linalg.norm((res['raw'] - res['tracks']) * S / 2, axis=-1)  # px at 256
            far.append(np.stack([cn.ravel(), d.ravel()], 1))
        meta.append((vname, np.array(frames)))
        print(f'{vname}: {len(frames)} frames, {time.time() - t0:.0f} s', flush=True)
    videos = np.concatenate([[v] * len(f) for v, f in meta])
    frames = np.concatenate([f for _, f in meta])
    rep = {}
    for k in VARIANTS:
        kp = np.concatenate(out[k])
        np.savez_compressed(f'{a.out}/{k}_all.npz', kpts=kp, videos=videos, frames=frames,
                            method=f'track_refine_{k}')
        rep[k] = jitter(wins[k])
    far = np.concatenate(far)
    qs = np.quantile(far[:, 0], [0.25, 0.5, 0.75])
    bins = np.digitize(far[:, 0], qs)
    rep['far_from_tracks_by_conf_quartile'] = [float((far[bins == b, 1] > 8).mean()) for b in range(4)]
    rep['median_px_from_tracks_by_conf_quartile'] = [float(np.median(far[bins == b, 1])) for b in range(4)]
    rep.update(run=a.run, noise=noise, win=a.win, anchors=a.anchors, gate=a.gate)
    json.dump(rep, open(f'{a.out}/report.json', 'w'), indent=1)
    print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    main()
