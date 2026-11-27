"""Can SAM 2 give better mouse masks than background subtraction, and do they survive our noise? (no training)

On the 24 pilot clips (scripts/pilot_cotracker_clips.py: 64 consecutive frames, 256 px), for each noise
condition (noise exactly as the loader adds it; H.264 encodes the whole clip):
  prompt   background subtraction on the (noisy) frames against the clean cached local background
           (dataloader/background.py; median of many noisy frames is ~ the clean one), threshold 0.5,
           the two largest blobs. Prompt frame = the first frame with two blobs (else the frame with the
           largest blob). Each blob's box prompts one SAM 2 object; SAM 2 (sam2.1-hiera-large, video
           predictor) tracks them forward and backward through the clip. No MARS labels are used.
  sam      union of the SAM 2 object masks
  bgsub    the background-subtraction mask itself (threshold 0.5), for comparison
Per mask, over all frames:
  cov_b / cov_w   share of MARS keypoints of the black / white mouse inside the mask
  floor           share of still-floor pixels (> 30 px from every MARS keypoint) inside the mask
  area            share of the picture inside the mask
  iou_clean       IoU with the same method's mask on the clean clip
Writes <out>/sam_masks.{txt,json} and <out>/sam_masks.png.

    python scripts/check_sam_masks.py --clips /root/data/calms21_clips \
        --data /root/data/CalMS21_bkind/train --out /workspace/sam_masks
"""
import argparse
import json
import os
import sys
import tempfile
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import label  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pilot_cotracker import load_clip, mars_clip, S  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.noise import BENCH_LEVELS, apply_noise_pair  # noqa: E402
from dataloader import background as bgsub  # noqa: E402

B = BENCH_LEVELS
CONDS = {
    'clean': None,
    'pg98': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][1]),
    'pg95': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][2]),
    'pg90': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][3]),
    'sp95': dict(type='speckle', seed=0, **B['speckle'][2]),
    'sp90': dict(type='speckle', seed=0, **B['speckle'][3]),
    'blur90': dict(type='motion_blur', seed=0, **B['motion_blur'][3]),
    'h264_90': dict(type='h264', **B['h264'][3]),
}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def bg_masks(frames, bg):
    return np.stack([bgsub.foreground(bgsub.to_gray(f), bg, 2.0, 0.1) > 0.5 for f in frames])  # (T, S, S)


def prompts(fg):
    """Boxes (x0, y0, x1, y1) of the two largest blobs on the prompt frame."""
    best = None
    for t in range(len(fg)):
        lab, n = label(fg[t])
        if n == 0:
            continue
        areas = np.bincount(lab.ravel())[1:]
        order = np.argsort(areas)[::-1]
        keep = [i + 1 for i in order[:2] if areas[i] >= 30]
        if not keep:
            continue
        if best is None or len(keep) > len(best[1]) or (len(keep) == len(best[1]) and areas[order[0]] > best[2]):
            best = (t, keep, areas[order[0]], lab)
        if len(keep) == 2:
            break
    if best is None:
        return None, []
    t, keep, _, lab = best
    boxes = []
    for i in keep:
        ys, xs = np.where(lab == i)
        boxes.append(np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32))
    return t, boxes


def sam_masks(pred, frames, t0, boxes, tmp):
    T = len(frames)
    for i in range(T):  # init_state needs a JPEG folder; the frames themselves are replaced below
        frames[i].save(os.path.join(tmp, f'{i:05d}.jpg'), quality=95)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        st = pred.init_state(tmp)
        size = pred.image_size
        x = np.stack([(np.asarray(f.convert('RGB').resize((size, size)), np.float32) / 255 - MEAN) / STD
                      for f in frames])  # exact (noisy) PNG frames, SAM 2's own preprocessing
        st['images'] = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(st['images'].device)
        st['cached_features'] = {}  # init_state warmed up on the JPEG of frame 0
        for k, b in enumerate(boxes):
            pred.add_new_points_or_box(st, frame_idx=t0, obj_id=k, box=b)
        out = np.zeros((T, S, S), bool)
        for rev in (False, True):
            for f, _, logits in pred.propagate_in_video(st, start_frame_idx=t0, reverse=rev):
                out[f] |= (logits[:, 0] > 0).any(0).cpu().numpy()
        pred.reset_state(st)
    return out


def metrics(m, k, still):
    T = len(m)
    res = {}
    for j, name in enumerate(('b', 'w')):
        pts = np.clip(np.round(k[:, j]).astype(int), 0, S - 1)  # (T, 7, xy)
        res[f'cov_{name}'] = float(m[np.arange(T)[:, None], pts[..., 1], pts[..., 0]].mean())
    res['floor'] = float(m[still].mean())
    res['area'] = float(m.mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', required=True)
    ap.add_argument('--data', required=True, help='training frames folder (keypoints.npz, background cache)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--conds', nargs='+', default=list(CONDS))
    ap.add_argument('--max-clips', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    pred = SAM2VideoPredictor.from_pretrained('facebook/sam2.1-hiera-large')
    clips = json.load(open(os.path.join(a.clips, 'clips.json')))[:a.max_clips]
    kpz = np.load(os.path.join(a.data, 'keypoints.npz'))
    kp = {f"{v}/keypoints": kpz[f"{v}/keypoints"] for v in {c['video'] for c in clips}}
    bgs = bgsub.background_cache(a.data, None, 60, 600, 300)
    yy, xx = np.mgrid[0:S, 0:S]
    rows = {(n, meth): [] for n in a.conds for meth in ('sam', 'bgsub')}
    show = {}
    t_start = time.time()
    for ci, c in enumerate(clips):
        paths, frames = load_clip(a.clips, c)
        k = mars_clip(kp, c)  # (T, mouse, 7, xy) at 256 px
        still = np.stack([np.min((xx[..., None] - kt[..., 0].ravel()) ** 2 + (yy[..., None] - kt[..., 1].ravel()) ** 2,
                                 -1) > 30 ** 2 for kt in k.astype(np.float32)])  # (T, S, S)
        clean = {}
        for n in a.conds:
            fr = apply_noise_pair(frames, CONDS[n], paths) if CONDS[n] else frames
            bg = bgsub.background_for(bgs[c['video']], c['start'])
            fgm = bg_masks(fr, bg)
            t0, boxes = prompts(fgm)
            with tempfile.TemporaryDirectory() as tmp:
                sm = sam_masks(pred, list(fr), t0, boxes, tmp) if boxes else np.zeros_like(fgm)
            for meth, m in (('sam', sm), ('bgsub', fgm)):
                r = metrics(m, k, still)
                if n == 'clean':
                    clean[meth] = m
                ref = clean.get(meth)
                r['iou_clean'] = float((m & ref).sum() / max((m | ref).sum(), 1)) if ref is not None else np.nan
                r['n_obj'] = len(boxes)
                rows[(n, meth)].append(r)
            if ci % 6 == 0 and n in ('clean', 'pg90', 'sp90', 'h264_90'):
                show[(ci, n)] = (np.asarray(fr[32])[..., 0], sm[32], fgm[32], k[32])
        print(f"clip {ci + 1}/{len(clips)} {time.time() - t_start:.0f} s  clean sam cov b/w "
              f"{rows[('clean', 'sam')][-1]['cov_b']:.2f}/{rows[('clean', 'sam')][-1]['cov_w']:.2f}  "
              f"bgsub {rows[('clean', 'bgsub')][-1]['cov_b']:.2f}/{rows[('clean', 'bgsub')][-1]['cov_w']:.2f}",
              flush=True)

    cols = ['cov_b', 'cov_w', 'floor', 'area', 'iou_clean', 'n_obj']
    table, lines = {}, [f"{'noise':8s} {'mask':6s} " + ' '.join(f'{c:>9s}' for c in cols)]
    for n in a.conds:
        for meth in ('sam', 'bgsub'):
            m = {c: float(np.nanmean([r[c] for r in rows[(n, meth)]])) for c in cols}
            table[f'{n}/{meth}'] = m
            lines.append(f'{n:8s} {meth:6s} ' + ' '.join(f'{m[c]:9.3f}' for c in cols))
        lines.append('')
    print('\n'.join(lines))
    open(os.path.join(a.out, 'sam_masks.txt'), 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(summary=table, per_clip={f'{n}/{m}': v for (n, m), v in rows.items()}),
              open(os.path.join(a.out, 'sam_masks.json'), 'w'), indent=1)

    keys = sorted(show)
    cis = sorted({ci for ci, _ in keys})
    ns = [n for n in ('clean', 'pg90', 'sp90', 'h264_90') if n in a.conds]
    fig, axs = plt.subplots(len(cis), len(ns), figsize=(3 * len(ns), 3 * len(cis)), squeeze=False)
    for r, ci in enumerate(cis):
        for cc, n in enumerate(ns):
            ax = axs[r, cc]
            if (ci, n) in show:
                img, sm, fgm, kk = show[(ci, n)]
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                ax.contour(sm, levels=[0.5], colors='red', linewidths=0.8)
                ax.contour(fgm, levels=[0.5], colors='cyan', linewidths=0.6)
                ax.scatter(kk[0, :, 0], kk[0, :, 1], s=5, c='yellow', marker='x')
                ax.scatter(kk[1, :, 0], kk[1, :, 1], s=5, c='lime', marker='x')
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(n, fontsize=9)
    fig.suptitle('SAM 2 mask (red) vs background subtraction (cyan); MARS black (yellow) / white (green)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, 'sam_masks.png'), dpi=130)


if __name__ == '__main__':
    main()
