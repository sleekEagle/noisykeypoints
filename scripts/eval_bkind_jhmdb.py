"""Score a trained BKinD model on JHMDB (layout from scripts/prepare_jhmdb_bkind.py).

1. Extract BKinD's K discovered keypoints for every train and val frame.
2. Fit a linear map with no bias from discovered keypoints to the 15 GT joints on
   train frames (the same method as BKinD's regressor/regression.py, fitted per
   joint on frames where that joint is inside the image), then predict val.
3. Report PCK@0.1/0.2 (threshold = alpha * max side of the GT joint box), mean
   error and jitter, plus a visualization.

Tuning mode (--tune): for choosing settings without touching the test set. 20% of the
TRAIN clips (every 5th clip of each action, by name) are held out; the regression is fitted
on the other train clips and scored on the held-out ones. Writes metrics_tune.{json,txt}.
--keypoints reuses a saved keypoints.npz instead of running the model.

Sanity modes: --source gt (GT joints as "keypoints"; PCK should be ~100%) and
--source random (random keypoints; gives a floor).

Example:
  python scripts/eval_bkind_jhmdb.py --checkpoint third_party/BKinD/checkpoint/JHMDB_split1/checkpoint.pth.tar
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 320, 240  # JHMDB frame size
# Joint order as commonly given for JHMDB (README_joint_positions); used only for labels.
JOINTS = ['neck', 'belly', 'face', 'r_shoulder', 'l_shoulder', 'r_hip', 'l_hip',
          'r_elbow', 'l_elbow', 'r_knee', 'l_knee', 'r_wrist', 'l_wrist',
          'r_ankle', 'l_ankle']


def load_clips(data_root):
    meta = json.load(open(os.path.join(data_root, 'clips.json')))
    return [dict(name=k, **v) for k, v in sorted(meta.items())]


def load_gt(jhmdb, clip):
    """GT joints of one clip as (T, 15, 2) pixels, plus a (T, 15) inside-image mask."""
    pos = sio.loadmat(os.path.join(jhmdb, 'joint_positions', clip['action'], clip['clip'],
                                   'joint_positions.mat'))['pos_img']
    gt = pos.transpose(2, 1, 0)  # (T, 15, 2)
    inside = (gt[..., 0] >= 0) & (gt[..., 0] < W) & (gt[..., 1] >= 0) & (gt[..., 1] < H)
    return gt, inside


def extract_bkind(clips, data_root, ckpt_path, nkpts, image_size, batch):
    """BKinD keypoints per clip as (T, K, 2) in [-1, 1] coordinates, plus confidence (T, K)."""
    import torch
    import torchvision.transforms as T
    sys.path.insert(0, os.path.join(REPO, 'third_party', 'BKinD'))
    from model.kpt_detector import Model

    model = Model(nkpts, pretrained=False, output_shape=(image_size // 4, image_size // 4))
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    md = model.state_dict()
    loaded = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
    missing = [k for k in md if k not in loaded]
    if missing:
        raise RuntimeError(f'{len(missing)} detector weights not in checkpoint, e.g. {missing[:3]}')
    md.update(loaded)
    model.load_state_dict(md)
    model = model.cuda().eval()
    print(f'loaded {len(loaded)} tensors from {ckpt_path}')

    tf = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    kpts, conf = {}, {}
    with torch.no_grad():
        for i, c in enumerate(clips):
            d = os.path.join(data_root, c['split'], c['name'])
            files = sorted(os.listdir(d))
            assert len(files) == c['nframes'], c['name']
            ks, cs = [], []
            for s in range(0, len(files), batch):
                x = torch.stack([tf(Image.open(os.path.join(d, f)).convert('RGB'))
                                 for f in files[s:s + batch]]).cuda()
                out = model(x)
                ks.append(torch.stack(out[0], dim=2).cpu().numpy())
                cs.append(out[5].cpu().numpy())
            kpts[c['name']], conf[c['name']] = np.concatenate(ks), np.concatenate(cs)
            if i % 100 == 0:
                print(f'  extracted {i}/{len(clips)} clips')
    return kpts, conf


def to_norm(p):
    return np.stack([p[..., 0] / W * 2 - 1, p[..., 1] / H * 2 - 1], -1)


def to_pix(p):
    return np.stack([(p[..., 0] + 1) / 2 * W, (p[..., 1] + 1) / 2 * H], -1)


def fit_predict(Xtr, Ytr, Mtr, Xte):
    """Linear, no bias, per joint on frames where that joint is inside the image.
    X: (N, K, 2) normalized keypoints; Y: (N, 15, 2) normalized joints; M: (N, 15)."""
    Xtr, Xte = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    pred = np.zeros((len(Xte), Ytr.shape[1], 2))
    for j in range(Ytr.shape[1]):
        A, *_ = np.linalg.lstsq(Xtr[Mtr[:, j]], Ytr[Mtr[:, j], j], rcond=None)
        pred[:, j] = Xte @ A
    return pred


def jitter(tracks):
    """Mean |second difference| over all points and frames; tracks: list of (T, P, 2) pixels."""
    a = [np.linalg.norm(np.diff(t, n=2, axis=0), axis=-1).ravel() for t in tracks if len(t) > 2]
    return float(np.concatenate(a).mean())


def visualize(clips, data_root, kpts, pred, gts, out_path, n=12):
    val = [c for c in clips if c['split'] == 'val']
    pick = [val[i] for i in np.linspace(0, len(val) - 1, n).astype(int)]
    colors = [tuple(int(255 * v) for v in col) for col in
              [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1),
               (1, .5, 0), (.5, 0, 1), (0, .5, 0), (1, 1, 1)]]
    tiles = []
    for c in pick:
        t = c['nframes'] // 2
        d = os.path.join(data_root, 'val', c['name'])
        im = Image.open(os.path.join(d, sorted(os.listdir(d))[t])).convert('RGB')
        left, right = im.copy(), im.copy()
        dl, dr = ImageDraw.Draw(left), ImageDraw.Draw(right)
        for k, (x, y) in enumerate(to_pix(kpts[c['name']][t])):
            dl.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colors[k % len(colors)], outline=(0, 0, 0))
        for (gx, gy), (px, py) in zip(gts[c['name']][t], pred[c['name']][t]):
            dr.line([gx, gy, px, py], fill=(255, 255, 0))
            dr.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(0, 255, 0))
            dr.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 0, 0))
        dl.text((4, 4), c['action'], fill=(255, 255, 255))
        dr.text((4, 4), 'green=GT red=regressed', fill=(255, 255, 255))
        tile = Image.new('RGB', (2 * W, H)); tile.paste(left, (0, 0)); tile.paste(right, (W, 0))
        tiles.append(tile)
    cols = 2
    grid = Image.new('RGB', (cols * 2 * W, ((len(tiles) + cols - 1) // cols) * H))
    for i, tl in enumerate(tiles):
        grid.paste(tl, ((i % cols) * 2 * W, (i // cols) * H))
    grid.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', help='BKinD checkpoint (needed for --source bkind)')
    p.add_argument('--source', default='bkind', choices=['bkind', 'gt', 'random'])
    p.add_argument('--data', default=r'D:\datasets\keypoint\JHMDB_bkind\split1')
    p.add_argument('--jhmdb', default=r'D:\datasets\keypoint\JHMDB')
    p.add_argument('--out', default=os.path.join(REPO, 'results', 'jhmdb_split1', 'bkind_clean'))
    p.add_argument('--nkpts', type=int, default=10)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--tune', action='store_true', help='score on held-out TRAIN clips (for tuning)')
    p.add_argument('--keypoints', help='reuse this keypoints.npz instead of running the model')
    a = p.parse_args()

    clips = load_clips(a.data)
    gts, masks = {}, {}
    for c in clips:
        gts[c['name']], masks[c['name']] = load_gt(a.jhmdb, c)

    conf = None
    if a.source == 'bkind' and a.keypoints:
        z = np.load(a.keypoints)
        kpts = {c['name']: z[f'kpts/{c["name"]}'] for c in clips}
        conf = {c['name']: z[f'conf/{c["name"]}'] for c in clips}
        assert kpts[clips[0]['name']].shape[1] == a.nkpts, 'set --nkpts to match the saved keypoints'
    elif a.source == 'bkind':
        kpts, conf = extract_bkind(clips, a.data, a.checkpoint, a.nkpts, a.image_size, a.batch)
    elif a.source == 'gt':
        kpts = {k: to_norm(v) for k, v in gts.items()}
    else:
        rng = np.random.default_rng(0)
        kpts = {c['name']: rng.uniform(-1, 1, (c['nframes'], a.nkpts, 2)) for c in clips}
    if a.source != 'bkind':
        a.out = os.path.join(os.path.dirname(a.out), f'sanity_{a.source}')
    os.makedirs(a.out, exist_ok=True)

    tr = [c for c in clips if c['split'] == 'train']
    te = [c for c in clips if c['split'] == 'val']
    if a.tune:  # hold out every 5th train clip of each action; never look at the test set
        hold = set()
        for act in sorted(set(c['action'] for c in tr)):
            hold |= {c['name'] for c in [c for c in tr if c['action'] == act][::5]}
        te = [c for c in tr if c['name'] in hold]
        tr = [c for c in tr if c['name'] not in hold]
    cat = lambda d, cs: np.concatenate([d[c['name']] for c in cs])
    pred_n = fit_predict(cat(kpts, tr), to_norm(cat(gts, tr)), cat(masks, tr), cat(kpts, te))
    pred_px = to_pix(pred_n)

    # Split predictions back into clips for per-clip use.
    pred, s = {}, 0
    for c in te:
        pred[c['name']] = pred_px[s:s + c['nframes']]; s += c['nframes']

    gt_te, m_te = cat(gts, te), cat(masks, te)
    err = np.linalg.norm(pred_px - gt_te, axis=-1)  # (N, 15) pixels
    # Box of the in-image GT joints per frame.
    big = np.where(m_te[..., None], gt_te, np.nan)
    size = np.nanmax(np.nanmax(big, 1) - np.nanmin(big, 1), -1)  # (N,)
    ok = m_te & np.isfinite(size)[:, None]
    rel = err / size[:, None]
    actions = np.concatenate([[c['action']] * c['nframes'] for c in te])

    res = {
        'mode': 'tune (held-out train clips)' if a.tune else 'test (official test clips)',
        'source': a.source, 'checkpoint': a.checkpoint, 'nkpts': a.nkpts,
        'train_frames': int(sum(c['nframes'] for c in tr)), 'val_frames': int(len(err)),
        'scored_joints': int(ok.sum()),
        'PCK@0.2': float((rel[ok] <= 0.2).mean()), 'PCK@0.1': float((rel[ok] <= 0.1).mean()),
        'mean_err_px': float(err[ok].mean()), 'mean_err_rel_box': float(rel[ok].mean()),
        'per_joint_PCK@0.2': {JOINTS[j]: float((rel[ok[:, j], j] <= 0.2).mean()) for j in range(15)},
        'per_action_PCK@0.2': {act: float((rel[ok & (actions == act)[:, None]] <= 0.2).mean())
                               for act in sorted(set(actions))},
        'jitter_kpts_px': jitter([to_pix(kpts[c['name']]) for c in te]),
        'jitter_gt_px': jitter([gts[c['name']] for c in te]),
    }
    stem = 'metrics_tune' if a.tune else 'metrics'
    json.dump(res, open(os.path.join(a.out, stem + '.json'), 'w'), indent=1)
    lines = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
             for k, v in res.items() if not isinstance(v, dict)]
    lines += ['per joint PCK@0.2: ' + ', '.join(f'{k} {v:.2f}' for k, v in res['per_joint_PCK@0.2'].items())]
    lines += ['per action PCK@0.2: ' + ', '.join(f'{k} {v:.2f}' for k, v in res['per_action_PCK@0.2'].items())]
    open(os.path.join(a.out, stem + '.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    if a.source == 'bkind' and not a.keypoints:
        np.savez_compressed(os.path.join(a.out, 'keypoints.npz'),
                            **{f'kpts/{k}': v for k, v in kpts.items()},
                            **{f'conf/{k}': v for k, v in conf.items()})
    if a.source != 'random' and not a.tune:
        visualize(clips, a.data, kpts, pred, gts, os.path.join(a.out, 'vis_val.png'))


if __name__ == '__main__':
    main()
