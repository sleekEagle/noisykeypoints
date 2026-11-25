"""Score trained CalMS21 runs (BKinD or our warp method) against the MARS mouse keypoints.

Development score: it uses only the training videos (the 10 no-cable videos BKinD trains on, as
built by scripts/prepare_calms21_bkind.py), so the CalMS21 test set stays untouched.

1. Run the keypoint network on every frame whose number is a multiple of --stride (default 7 = the
   first frame of every training pair). Frames are corrupted exactly as the run's training corrupted
   frame A of a pair (noise from the run's config; --eval-noise clean scores on clean frames).
2. Fit a linear map without bias (BKinD's regressor: Ridge alpha 0, no intercept) from the K
   discovered keypoints to the 14 MARS keypoints (2 mice x 7), both in [-1, 1] image coordinates.
   Cross-validated by video: 5 folds of 2 videos, fit on 8 videos, predict the other 2.
3. Report per mouse (black resident = mouse 0, white intruder = mouse 1):
     nerr     mean error / mouse length (median nose-to-tail-base distance of that mouse in that video)
     PCK@0.2  share of points with error <= 0.2 mouse lengths
     err_px   mean error in original 1024 x 570 pixels
   and floors computed on the same folds: 'mean pose' (predict the training mean in every frame; what
   a collapsed model gives) and, with --source gt, MARS keypoints as input (should be ~0).
4. 'On a mouse': for each discovered keypoint, the share of frames where it lies within 0.25 mouse
   lengths of one of that mouse's 7 MARS keypoints; a keypoint counts as on the black / white mouse if
   that share is >= 0.5 (the by-eye count from the sample pictures, made objective).
MARS keypoints come from a tracker (not human labels); say so wherever these numbers are used.

Writes <out>/metrics.{json,txt}, <out>/keypoints.npz, and appends one line to <summary>.

    python scripts/eval_calms21.py --run CalMS21_clean_s1 --runs /workspace/runs \\
        --data /root/data/CalMS21_bkind/train
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BK = os.path.join(REPO, 'third_party', 'BKinD')
sys.path.insert(0, BK)
from dataloader.noise import apply_noise_pair  # noqa: E402

W, H, S = 1024, 570, 256
PARTS = ['nose', 'l_ear', 'r_ear', 'neck', 'l_hip', 'r_hip', 'tail_base']
MICE = ['black', 'white']


def frame_no(name):
    m = re.match(r'image_(\d+)\.png$', name)
    return int(m.group(1)) if m else None


def list_frames(data, stride):
    """[(video, frame, path, partner path)] for frames f % stride == 0 that have their pair partner f + 6."""
    out = []
    for v in sorted(d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d))):
        frames = {frame_no(f): os.path.join(data, v, f) for f in os.listdir(os.path.join(data, v)) if frame_no(f) is not None}
        for f in sorted(frames):
            if f % stride == 0 and f + 6 in frames:
                out.append((v, f, frames[f], frames[f + 6]))
    return out


def run_config(run, runs_dir):
    """The run's training config: noise and method (from config/kaggle_<run>.yaml, written by kaggle_bkind)."""
    import yaml
    for p in (os.path.join(BK, 'config', f'kaggle_{run}.yaml'), os.path.join(runs_dir, run, 'config.yaml')):
        if os.path.exists(p):
            return yaml.safe_load(open(p)), p
    return config_from_name(run), 'run name'


def config_from_name(run):
    """Noise from the run name (for runs trained elsewhere, whose config is not on this machine),
    using the same helpers as scripts/run_specs.py. None if the name is not understood."""
    sys.path.insert(0, os.path.join(REPO, 'scripts'))
    import run_specs as rs
    frozen = '_frozen' in run
    m = re.search(r'_(pg|sp|blur|h264)_ssim(\d+)(_frozen)?_s(\d+)$', run)
    if m:
        return dict(noise=rs.bench(m.group(1), int(m.group(2)), 1, frozen=frozen)['overrides']['noise'])
    m = re.search(r'_(pg|ll|sp|blur|h264)_L(\d)(_frozen)?_s(\d+)$', run)
    if m:
        return dict(noise=rs.level(m.group(1), int(m.group(2)), 1, frozen=frozen)['overrides']['noise'])
    if re.search(r'_pg20_s\d+$', run):
        return dict(noise=dict(type='poisson_gaussian', photons=20, read=0.03, seed=0))
    if re.search(r'_clean_s\d+$', run):
        return dict(noise=None)
    return None


class Frames:
    """Frame A of each item (noised as in training). With input_frames = dict(k, step, frames), a stack of
    k frames, oldest first, current frame last, built like dataloader/mouse_dataset.py _context."""
    def __init__(self, items, noise, input_frames=None):
        self.items, self.noise = items, noise
        self.ctx = input_frames if input_frames and int(input_frames.get('k', 1)) > 1 else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        v, f, pa, pb = self.items[i]
        ims = [Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR) for p in (pa, pb)]
        stack = [apply_noise_pair(ims, self.noise, [pa, pb])[0] if self.noise else ims[0]]
        if self.ctx:
            k, step = int(self.ctx['k']), int(self.ctx.get('step', 2))
            paths = []
            for g in range(f - step * (k - 1), f, step):
                p = os.path.join(self.ctx['frames'], v, f'image_{max(g, 0):08d}.png')
                paths.append(p if os.path.exists(p) else (paths[-1] if paths else pa))
            c = [Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR) for p in paths]
            stack = (apply_noise_pair(c, self.noise, paths) if self.noise else c) + stack
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return torch.cat([(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1) - mean) / std
                          for im in stack])


def extract(ckpt_path, items, noise, nkpts, batch, workers, input_frames=None):
    """Keypoints (N, K, 2) in [-1, 1] as (x, y), and confidence (N, K). BKinD and the warp model share
    the same encoder + globalNet + soft-argmax, so BKinD's keypoint-only Model loads both."""
    import torch
    from model.kpt_detector import Model
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()}
    n_frames = sd['encoder.conv1.weight'].shape[1] // 3 if 'encoder.conv1.weight' in sd else 1  # model/multiframe.py
    if n_frames > 1 and int((input_frames or {}).get('k', 1)) != n_frames:
        raise RuntimeError(f'checkpoint takes {n_frames} frames but the config has input_frames {input_frames}')
    model = Model(nkpts, pretrained=False, output_shape=(S // 4, S // 4), n_frames=n_frames)
    md = model.state_dict()
    loaded = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
    missing = [k for k in md if k not in loaded]
    if missing:
        raise RuntimeError(f'{len(missing)} detector weights not in checkpoint, e.g. {missing[:3]}')
    model.load_state_dict(loaded)
    dev = os.environ.get('EVAL_DEVICE', 'cuda')  # EVAL_DEVICE=cpu when training fills the GPU
    model = model.to(dev).eval()
    print(f'loaded {len(loaded)} tensors from {ckpt_path} (epoch {ck.get("epoch")}, method {ck.get("method", "bkind")})',
          flush=True)
    dl = torch.utils.data.DataLoader(Frames(items, noise, input_frames if n_frames > 1 else None),
                                     batch_size=batch, num_workers=workers)
    ks, cs = [], []
    t0 = time.time()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16 if dev == 'cuda' else torch.bfloat16):
        for i, x in enumerate(dl):
            out = model(x.to(dev, non_blocking=True))
            ks.append(torch.stack(out[0], dim=-1).float().cpu().numpy())
            cs.append(out[5].float().cpu().numpy())
            if i % 50 == 0:
                print(f'  {i * batch}/{len(items)} frames, {time.time() - t0:.0f} s', flush=True)
    return np.concatenate(ks), np.concatenate(cs), ck.get('epoch'), ck.get('method', 'bkind')


def to_norm(p):
    return np.stack([p[..., 0] / W * 2 - 1, p[..., 1] / H * 2 - 1], -1)


def to_pix(p):
    return np.stack([(p[..., 0] + 1) / 2 * W, (p[..., 1] + 1) / 2 * H], -1)


def cross_val(X, Y, videos, folds, bias=False):
    """Predict Y (N, P, 2) from X (N, K, 2), fold by video. bias=False: BKinD's regressor (no intercept)."""
    names = sorted(set(videos))
    pred = np.zeros_like(Y)
    Xf, Yf = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    if bias:
        Xf = np.concatenate([Xf, np.ones((len(Xf), 1))], 1)
    for i in range(folds):
        test = np.isin(videos, names[i::folds])
        A, *_ = np.linalg.lstsq(Xf[~test], Yf[~test], rcond=None)
        pred[test] = (Xf[test] @ A).reshape(-1, *Y.shape[1:])
    return pred


def scores(pred_px, gt_px, length):
    """pred/gt (N, 2 mice, 7, 2) pixels, length (N, 2) -> metrics per mouse."""
    err = np.linalg.norm(pred_px - gt_px, axis=-1)  # (N, 2, 7)
    rel = err / length[..., None]
    res = {}
    for m, name in enumerate(MICE):
        res[name] = dict(nerr=float(rel[:, m].mean()), pck02=float((rel[:, m] <= 0.2).mean()),
                         err_px=float(err[:, m].mean()),
                         per_part_nerr={p: float(rel[:, m, j].mean()) for j, p in enumerate(PARTS)})
    res['both'] = dict(nerr=float(rel.mean()), pck02=float((rel <= 0.2).mean()), err_px=float(err.mean()))
    return res


def on_mouse(kpts_px, gt_px, length, thresh=0.25):
    """Share of frames each discovered keypoint lies within thresh * length of a keypoint of each mouse."""
    d = np.linalg.norm(kpts_px[:, :, None, None] - gt_px[:, None], axis=-1).min(-1)  # (N, K, 2 mice)
    share = (d <= thresh * length[:, None, :]).mean(0)  # (K, 2)
    return share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs='+', required=True, help='run names (folders under --runs)')
    ap.add_argument('--runs', default='/workspace/runs')
    ap.add_argument('--data', default='/root/data/CalMS21_bkind/train')
    ap.add_argument('--source', default='model', choices=['model', 'gt', 'random'])
    ap.add_argument('--eval-noise', default='train', choices=['train', 'clean'])
    ap.add_argument('--stride', type=int, default=7)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--nkpts', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--summary', default=None, help='TSV to append one line per run (default <runs>/eval_summary.tsv)')
    a = ap.parse_args()
    summary = a.summary or os.path.join(a.runs, 'eval_summary.tsv')

    items = list_frames(a.data, a.stride)
    z = np.load(os.path.join(a.data, 'keypoints.npz'))
    kp = {f'{v}/keypoints': z[f'{v}/keypoints'] for v in {v for v, *_ in items}}  # npz re-reads on every access
    videos = np.array([v for v, *_ in items])
    gt_px = np.stack([kp[f'{v}/keypoints'][f] for v, f, *_ in items]).transpose(0, 1, 3, 2)  # (N, 2, 7, 2)
    # mouse length = median over the video of the nose-to-tail-base distance, per mouse
    raw_len = np.linalg.norm(gt_px[:, :, 0] - gt_px[:, :, 6], axis=-1)  # (N, 2)
    length = np.zeros_like(raw_len)
    for v in set(videos):
        sel = videos == v
        length[sel] = np.median(raw_len[sel], axis=0)
    Y = to_norm(gt_px).reshape(len(items), 14, 2)
    print(f'{len(items)} frames from {len(set(videos))} videos; median mouse length (px): '
          f'black {np.median(length[:, 0]):.0f}, white {np.median(length[:, 1]):.0f}', flush=True)

    floor = scores(to_pix(cross_val(np.zeros((len(Y), 1, 2)), Y, videos, a.folds, bias=True)
                          ).reshape(gt_px.shape), gt_px, length)
    print(f"mean-pose floor: nerr black {floor['black']['nerr']:.3f} white {floor['white']['nerr']:.3f}", flush=True)

    for run in a.run:
        cfg, cfg_path = run_config(run, a.runs)
        noise = (cfg or {}).get('noise') if a.eval_noise == 'train' else None
        tag = run if a.source == 'model' else f'sanity_{a.source}'
        out = os.path.join(a.runs, run, 'eval' + ('' if a.eval_noise == 'train' else '_clean')) \
            if a.source == 'model' else os.path.join(a.runs, 'eval_sanity', a.source)
        os.makedirs(out, exist_ok=True)
        epoch, method, conf = None, a.source, None
        if a.source == 'model':
            if cfg is None:
                print(f'WARNING: no config for {run}; scoring on clean frames', flush=True)
            print(f'== {run}: noise {noise} ({cfg_path})', flush=True)
            X, conf, epoch, method = extract(os.path.join(a.runs, run, 'checkpoint.pth.tar'), items, noise,
                                             a.nkpts, a.batch, a.workers, (cfg or {}).get('input_frames'))
        elif a.source == 'gt':
            X = Y
        else:
            X = np.random.default_rng(0).uniform(-1, 1, (len(Y), a.nkpts, 2))
        pred_px = to_pix(cross_val(X, Y, videos, a.folds)).reshape(gt_px.shape)
        res = scores(pred_px, gt_px, length)
        share = on_mouse(to_pix(X), gt_px, length)
        res.update(run=tag, method=method, epoch=epoch, noise=noise, eval_noise=a.eval_noise,
                   frames=len(items), stride=a.stride, folds=a.folds,
                   floor_mean_pose={m: floor[m]['nerr'] for m in ('black', 'white', 'both')},
                   kpts_on_black=int((share[:, 0] >= 0.5).sum()), kpts_on_white=int((share[:, 1] >= 0.5).sum()),
                   on_mouse_share=share.tolist(),
                   kpt_spread=float(np.std(to_pix(X), axis=0).mean()))  # ~0 = keypoints never move
        json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
        line = (f"{tag}\tepoch {epoch}\teval {a.eval_noise}\t"
                f"nerr black {res['black']['nerr']:.3f} white {res['white']['nerr']:.3f} both {res['both']['nerr']:.3f}\t"
                f"PCK@0.2 black {res['black']['pck02']:.3f} white {res['white']['pck02']:.3f}\t"
                f"on black {res['kpts_on_black']} white {res['kpts_on_white']}\t"
                f"spread {res['kpt_spread']:.1f} px\t"
                f"(floor nerr black {floor['black']['nerr']:.3f} white {floor['white']['nerr']:.3f})")
        open(os.path.join(out, 'metrics.txt'), 'w').write(line.replace('\t', '\n') + '\n')
        with open(summary, 'a') as fh:
            fh.write(line + '\n')
        print(line, flush=True)
        if a.source == 'model':
            np.savez_compressed(os.path.join(out, 'keypoints.npz'), kpts=X, conf=conf,
                                videos=videos, frames=np.array([f for _, f, *_ in items]))


if __name__ == '__main__':
    main()
