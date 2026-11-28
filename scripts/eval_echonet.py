"""Score trained EchoNet-Dynamic runs (BKinD loader dataloader/echo_dataset.py) against the tracings.

Data: the prepared folder of scripts/prepare_echonet.py (videos/*.npy or *.npz, videos.json, labels.npz).
Videos flagged 'size', 'no fan' or 'fan area' are left out; landmark scores use videos with labels.
Fit on --fit-videos TRAIN videos (a fixed random subset), score on all usable videos of --split
(VAL for development; TEST only for the final table).

The detector runs on every frame of these videos (112 -> 128 px, noised like the run's training or as
--eval-noise says, as the loader does). Metrics (positions in px at 112, the original frames):
  landmarks   linear map WITHOUT bias (BKinD's regressor) from the K keypoints to the 3 landmarks (apex,
              2 mitral-annulus corners) on the 2 traced frames (ED, ES) of each video; fit on the fit
              videos, predicted on the split. nerr = error / LV length (apex to annulus midpoint of that
              frame); also err_px and PCK@0.1 (share with nerr <= 0.1).
  contour     the same kind of map to the 40 outline points; point nerr, and the symmetric mean distance
              between the predicted and traced outlines (closed polygons, densely resampled) in px and / LV
              length.
  jitter      mean |p(t+1) - 2 p(t) + p(t-1)| of the discovered keypoints over all frames, px at 112.
  EF probe    ridge regression (alpha by 5-fold CV on the fit videos) to the ejection fraction:
                edes   keypoints at the traced ED and ES frames, their difference, and the ES/ED ratio of
                       every keypoint-pair distance;
                traj   label-free: per keypoint and axis the std and range over the video, and each
                       keypoint-pair distance's (max - min) / mean.
              MAE (EF points) and R^2 on the split; floor = predicting the fit videos' mean EF.
Floors: 'mean pose' (bias only: what a collapsed model gives). --source gt (landmarks + outline as the
keypoints; should give ~0) and --source random check the scoring itself.

Writes <runs>/<run>/eval_echo[_<noise>]_<split>/metrics.{json,txt} (+ keypoints.npz of the labelled
frames) and appends one line to <runs>/eval_echo_summary.tsv.

    python scripts/eval_echonet.py --run EchoNet_clean_s1 [--split VAL] [--eval-noise clean]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BK = os.path.join(REPO, 'third_party', 'BKinD')
sys.path.insert(0, BK)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from dataloader.noise import apply_noise_pair  # noqa: E402
from eval_calms21 import eval_noise_spec, run_config  # noqa: E402

P = 112          # original frame size; all positions are reported in px at 112
EXCLUDE = {'size', 'no fan', 'fan area'}
LM_NAMES = ['apex', 'annulus_1', 'annulus_2']


def video_frames(data, name):
    npy = os.path.join(data, 'videos', name + '.npy')
    return np.load(npy, mmap_mode='r') if os.path.exists(npy) else np.load(os.path.join(data, 'videos', name + '.npz'))['frames']


class Frames:
    """Every frame of the given videos, as the loader feeds them (resized to S, noised, normalised)."""
    def __init__(self, data, items, noise, S):
        self.data, self.items, self.noise, self.S = data, items, noise, S

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        name, t = self.items[i]
        im = Image.fromarray(np.asarray(video_frames(self.data, name)[t])).convert('RGB').resize((self.S, self.S), Image.BILINEAR)
        if self.noise:  # same virtual path as dataloader/echo_dataset.py, so the same noise per frame
            im = apply_noise_pair([im], self.noise, [os.path.join(self.data, name, f'image_{t:08d}.png')])[0]
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std


def extract(ckpt_path, data, items, noise, nkpts, S, batch, workers):
    """Keypoints (N, K, 2) in [-1, 1] (x, y) for the items [(video, frame)]."""
    import torch
    from model.kpt_detector import Model
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()}
    model = Model(nkpts, pretrained=False, output_shape=(S // 4, S // 4))
    md = model.state_dict()
    loaded = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
    missing = [k for k in md if k not in loaded]
    if missing:
        raise RuntimeError(f'{len(missing)} detector weights not in checkpoint, e.g. {missing[:3]}')
    model.load_state_dict(loaded)
    dev = os.environ.get('EVAL_DEVICE', 'cuda')
    model = model.to(dev).eval()
    print(f'loaded {ckpt_path} (epoch {ck.get("epoch")})', flush=True)
    dl = torch.utils.data.DataLoader(Frames(data, items, noise, S), batch_size=batch, num_workers=workers)
    ks, t0 = [], time.time()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.float16 if dev == 'cuda' else torch.bfloat16):
        for i, x in enumerate(dl):
            out = model(x.to(dev, non_blocking=True))
            ks.append(torch.stack(out[0], dim=-1).float().cpu().numpy())
            if i % 200 == 0:
                print(f'  {i * batch}/{len(items)} frames, {time.time() - t0:.0f} s', flush=True)
    return np.concatenate(ks), ck.get('epoch')


def fit_predict(Xa, Ya, Xb, bias=False):
    """Least squares from Xa (N, K, 2) to Ya (N, P, 2), applied to Xb. bias=False: BKinD's regressor."""
    A_ = Xa.reshape(len(Xa), -1)
    B_ = Xb.reshape(len(Xb), -1)
    if bias:
        A_ = np.concatenate([A_, np.ones((len(A_), 1))], 1)
        B_ = np.concatenate([B_, np.ones((len(B_), 1))], 1)
    W, *_ = np.linalg.lstsq(A_, Ya.reshape(len(Ya), -1), rcond=None)
    return (B_ @ W).reshape(len(Xb), *Ya.shape[1:])


def to_unit(p):
    return p / P * 2 - 1


def to_px(u):
    return (u + 1) / 2 * P


def resample(poly, n=200):
    """Closed polygon (M, 2) -> n points evenly spaced along its outline."""
    q = np.concatenate([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1)


def contour_distance(a, b):
    """Symmetric mean distance between two closed outlines, px."""
    a, b = resample(a), resample(b)
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return 0.5 * (d.min(1).mean() + d.min(0).mean())


def ridge_cv(Xa, ya, Xb, alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100, 1000), folds=5):
    """Ridge on standardised features; alpha by k-fold CV on (Xa, ya). Returns predictions for Xb."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-8
    Za, Zb = (Xa - mu) / sd, (Xb - mu) / sd
    ym = ya.mean()

    def solve(Z, y, a):
        return np.linalg.solve(Z.T @ Z + a * np.eye(Z.shape[1]), Z.T @ (y - ym))

    idx = np.arange(len(Za)) % folds
    cv = [np.mean([np.abs(Za[idx == f] @ solve(Za[idx != f], ya[idx != f], a) + ym - ya[idx == f]).mean()
                   for f in range(folds)]) for a in alphas]
    a = alphas[int(np.argmin(cv))]
    return Zb @ solve(Za, ya, a) + ym, a


def pair_dists(k):
    """(..., K, 2) -> (..., K(K-1)/2) distances between keypoint pairs."""
    i, j = np.triu_indices(k.shape[-2], 1)
    return np.linalg.norm(k[..., i, :] - k[..., j, :], axis=-1)


def traj_features(track):
    """(T, K, 2) keypoints of one video -> label-free motion features."""
    d = pair_dists(track)
    return np.concatenate([track.std(0).ravel(), (track.max(0) - track.min(0)).ravel(),
                           (d.max(0) - d.min(0)) / (d.mean(0) + 1e-6)])


def edes_features(k_ed, k_es):
    return np.concatenate([k_ed.ravel(), k_es.ravel(), (k_es - k_ed).ravel(),
                           pair_dists(k_es) / (pair_dists(k_ed) + 1e-6)])


def probe_scores(pred, y, floor):
    return dict(mae=float(np.abs(pred - y).mean()), r2=float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                floor_mae=float(np.abs(floor - y).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', nargs='+', required=True)
    ap.add_argument('--runs', default='/workspace/runs')
    ap.add_argument('--data', default='/root/data/echonet')
    ap.add_argument('--split', default='VAL', choices=['VAL', 'TEST'])
    ap.add_argument('--fit-videos', type=int, default=1500, help='TRAIN videos the regressions / probes are fitted on')
    ap.add_argument('--eval-noise', default='train', help="'train' (the run's own noise), 'clean', or a cell such as pg90")
    ap.add_argument('--source', default='model', choices=['model', 'gt', 'random'])
    ap.add_argument('--nkpts', type=int, default=10)
    ap.add_argument('--image-size', type=int, default=128)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    info = json.load(open(os.path.join(a.data, 'videos.json')))
    lab = np.load(os.path.join(a.data, 'labels.npz'))
    usable = lambda r: not set(r['flags']) & EXCLUDE
    labelled = {k.split('/')[0] for k in lab.files if k.endswith('/frames')}
    rng = np.random.default_rng(0)
    train = sorted(v for v, r in info.items() if r['split'] == 'TRAIN' and usable(r) and v in labelled)
    fit = sorted(rng.choice(train, min(a.fit_videos, len(train)), replace=False).tolist())
    ev = sorted(v for v, r in info.items() if r['split'] == a.split and usable(r))
    ev_lab = [v for v in ev if v in labelled]
    print(f'fit on {len(fit)} TRAIN videos, score {len(ev)} {a.split} videos ({len(ev_lab)} labelled)', flush=True)

    def gt(videos):
        lm = np.concatenate([lab[f'{v}/landmarks'] for v in videos])  # (2V, 3, 2) ED, ES per video
        ct = np.concatenate([lab[f'{v}/contour'] for v in videos])
        fr = np.array([lab[f'{v}/frames'] for v in videos])  # (V, 2)
        return lm, ct, fr

    lm_fit, ct_fit, fr_fit = gt(fit)
    lm_ev, ct_ev, fr_ev = gt(ev_lab)
    length = lambda lm: np.linalg.norm(lm[:, 0] - (lm[:, 1] + lm[:, 2]) / 2, axis=-1)  # apex to annulus midpoint
    L_ev = length(lm_ev)
    summary = os.path.join(a.runs, 'eval_echo_summary.tsv')

    for run in a.run:
        cfg, cfg_path = run_config(run, a.runs) if a.source == 'model' else ({}, '')
        noise = (cfg or {}).get('noise') if a.eval_noise == 'train' else eval_noise_spec(a.eval_noise)
        t0 = time.time()
        if a.source == 'model':
            items = [(v, t) for v in fit + ev for t in range(info[v]['frames'])]
            K, epoch = extract(os.path.join(a.runs, run, 'checkpoint.pth.tar'), a.data, items, noise,
                               a.nkpts, a.image_size, a.batch, a.workers)
            tracks, i = {}, 0
            for v in fit + ev:
                n = info[v]['frames']
                tracks[v] = K[i:i + n]
                i += n
            pick = lambda videos, fr: np.concatenate([tracks[v][list(f)] for v, f in zip(videos, fr)])
            X_fit, X_ev = pick(fit, fr_fit), pick(ev_lab, fr_ev)
        else:
            epoch, tracks = None, None
            if a.source == 'gt':
                X_fit = to_unit(np.concatenate([lm_fit, ct_fit], 1))
                X_ev = to_unit(np.concatenate([lm_ev, ct_ev], 1))
            else:
                X_fit = rng.uniform(-1, 1, (len(lm_fit), a.nkpts, 2))
                X_ev = rng.uniform(-1, 1, (len(lm_ev), a.nkpts, 2))
        res = dict(run=run if a.source == 'model' else f'sanity_{a.source}', epoch=epoch, noise=noise,
                   eval_noise=a.eval_noise, split=a.split, fit_videos=len(fit), eval_videos=len(ev),
                   eval_labelled=len(ev_lab))

        # landmarks + contour (labelled frames)
        for name, Yf, Ye in (('landmarks', lm_fit, lm_ev), ('contour', ct_fit, ct_ev)):
            pred = to_px(fit_predict(X_fit, to_unit(Yf), X_ev))
            floor = to_px(fit_predict(np.zeros((len(X_fit), 1, 2)), to_unit(Yf), np.zeros((len(X_ev), 1, 2)), bias=True))
            err, ferr = np.linalg.norm(pred - Ye, axis=-1), np.linalg.norm(floor - Ye, axis=-1)
            r = dict(nerr=float((err / L_ev[:, None]).mean()), err_px=float(err.mean()),
                     pck01=float((err / L_ev[:, None] <= 0.1).mean()),
                     floor_nerr=float((ferr / L_ev[:, None]).mean()))
            if name == 'landmarks':
                r['per_landmark_nerr'] = {n: float((err[:, j] / L_ev).mean()) for j, n in enumerate(LM_NAMES)}
                r['ed_es_nerr'] = [float((err[0::2] / L_ev[0::2, None]).mean()), float((err[1::2] / L_ev[1::2, None]).mean())]
            else:
                cd = np.array([contour_distance(p, g) for p, g in zip(pred, Ye)])
                fcd = np.array([contour_distance(p, g) for p, g in zip(floor, Ye)])
                r.update(dist_px=float(cd.mean()), dist_rel=float((cd / L_ev).mean()), floor_dist_px=float(fcd.mean()))
            res[name] = r

        # jitter and EF probes (need every frame: model only)
        if tracks is not None:
            acc = np.concatenate([np.linalg.norm(np.diff(to_px(tracks[v]), 2, axis=0), axis=-1).ravel()
                                  for v in ev if len(tracks[v]) >= 3])
            res['jitter_px'] = float(acc.mean())
            ef_fit = np.array([info[v]['EF'] for v in fit])
            ef_lab = np.array([info[v]['EF'] for v in ev_lab])
            ef_ev = np.array([info[v]['EF'] for v in ev])
            F = lambda videos, fr: np.stack([edes_features(tracks[v][f[0]], tracks[v][f[1]]) for v, f in zip(videos, fr)])
            p, al = ridge_cv(F(fit, fr_fit), ef_fit, F(ev_lab, fr_ev))
            res['ef_edes'] = dict(probe_scores(p, ef_lab, ef_fit.mean()), alpha=al)
            T = lambda videos: np.stack([traj_features(tracks[v]) for v in videos])
            p, al = ridge_cv(T(fit), ef_fit, T(ev))
            res['ef_traj'] = dict(probe_scores(p, ef_ev, ef_fit.mean()), alpha=al)

        tag = f'eval_echo{"" if a.eval_noise == "train" else "_" + a.eval_noise}_{a.split.lower()}'
        out = os.path.join(a.runs, run, tag) if a.source == 'model' else os.path.join(a.runs, 'eval_echo_sanity', a.source)
        os.makedirs(out, exist_ok=True)
        json.dump(res, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
        lm, ct = res['landmarks'], res['contour']
        line = (f"{res['run']}\tepoch {epoch}\teval {a.eval_noise} {a.split}\t"
                f"landmark nerr {lm['nerr']:.3f} ({lm['err_px']:.2f} px, PCK@0.1 {lm['pck01']:.3f})\t"
                f"contour {ct['dist_px']:.2f} px ({ct['dist_rel']:.3f})\t"
                + (f"jitter {res['jitter_px']:.2f} px\tEF MAE edes {res['ef_edes']['mae']:.2f} traj {res['ef_traj']['mae']:.2f} "
                   f"(floor {res['ef_traj']['floor_mae']:.2f})\t" if 'jitter_px' in res else '')
                + f"(floor landmark nerr {lm['floor_nerr']:.3f}, contour {ct['floor_dist_px']:.2f} px)")
        open(os.path.join(out, 'metrics.txt'), 'w').write(line.replace('\t', '\n') + '\n')
        with open(summary, 'a') as fh:
            fh.write(line + '\n')
        print(line, f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        if a.source == 'model':
            np.savez_compressed(os.path.join(out, 'keypoints.npz'), fit=X_fit, eval=X_ev, eval_videos=np.array(ev_lab),
                                eval_frames=fr_ev)


if __name__ == '__main__':
    main()
