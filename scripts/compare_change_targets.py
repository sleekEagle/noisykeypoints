"""Compare noise-robust "what changed between frames A and B" maps on CalMS21 (CPU only).

Candidates (all computed from the noisy frames only, as a training run would):
  bkind      BKinD's target: blur(1 - SSIM contrast-structure map), k2 0.03 (loss/ssim_loss.py).
  opt1_diff  option 1: |fg(A) - fg(B)|, fg = blurred |frame - local background| / 0.1, clipped
             (dataloader/background.py; background = median of nearby noisy frames, +-600 frames).
  opt1_union option 1': max(fg(A), fg(B)) (where a mouse is in A or B).
  opt2_test  option 2: noise-calibrated change test. The noise variance of d = B - A is learned per
             video as a function of brightness (robust MAD over 40 pairs of that video). Then
             T = gauss_2px(d^2 / var(brightness)) (mean 1 where only noise), and the target is
             clip((T - t0) / t0, 0, 1) with t0 = 1 + 3 * (std of T under pure noise).
  opt3_ds2 / opt3_ds4   option 3: BKinD's target on frames averaged down 2x / 4x, upsampled back.

Per noise condition and candidate, on random training pairs (frames i, i+6):
  ratio   mean target on the mice (< 10 px from a MARS keypoint in A or B) / on the still floor
          (> 30 px from all of them); scale-free, higher = real motion stands out more.
  auc     chance that a random mouse pixel has a higher target than a random still-floor pixel.
  corr    correlation of the noisy map with the same candidate's clean map (1 = noise changes nothing).
Writes <out>/change_targets.{json,png}.

    python scripts/compare_change_targets.py --data /root/data/CalMS21_bkind/train --out /workspace/change_targets
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party', 'BKinD'))
from dataloader.background import background_for, compute_backgrounds, foreground, frame_no, video_frames  # noqa: E402
from dataloader.noise import BENCH_LEVELS, apply_noise_pair  # noqa: E402

W, H, S = 1024, 570, 256
B = BENCH_LEVELS
NOISES = {
    'clean': None,
    'pg98': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][1]),
    'pg95': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][2]),
    'pg90': dict(type='poisson_gaussian', seed=0, **B['poisson_gaussian'][3]),
    'sp95': dict(type='speckle', seed=0, **B['speckle'][2]),
    'blur95': dict(type='motion_blur', seed=0, **B['motion_blur'][2]),
    'h264_95': dict(type='h264', **B['h264'][2]),
}
CANDS = ['bkind', 'opt1_diff', 'opt1_union', 'opt2_test', 'opt3_ds2', 'opt3_ds4']
LABELS = dict(bkind='BKinD target', opt1_diff='opt 1: |fg A - fg B|', opt1_union='opt 1\': fg A or B',
              opt2_test='opt 2: noise test', opt3_ds2='opt 3: SSIM at 1/2', opt3_ds4='opt 3: SSIM at 1/4')
BRIGHT_EDGES = np.linspace(0, 1, 21)
SIG_TEST = 2.0
DATA = None


def gray(im):
    return np.asarray(im.convert('L'), dtype=np.float32) / 255.0


def load_pair(pa, pb, noise):
    ims = [Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR) for p in (pa, pb)]
    na, nb = apply_noise_pair(ims, noise, [pa, pb])
    return gray(na), gray(nb)


def bkind_target(a, b, k2=0.03, k1=0.01):
    """loss/ssim_loss.compute_ssim maths on one grey image pair (numpy, any size >= 11)."""
    def g(x, s):  # 'valid' Gaussian window like F.conv2d with padding 0, truncated at 11 taps
        return gaussian_filter(x, s, truncate=5.0 / s)[5:-5, 5:-5]
    x, y = np.pad(a, 10, mode='edge').astype(np.float64), np.pad(b, 10, mode='edge').astype(np.float64)
    mu1, mu2 = g(x, 1.5), g(y, 1.5)
    s1, s2, s12 = g(x * x, 1.5) - mu1 ** 2, g(y * y, 1.5) - mu2 ** 2, g(x * y, 1.5) - mu1 * mu2
    c2 = k2 ** 2
    t = 1.0 - (2 * s12 + c2) / (s1 + s2 + c2)
    return g(t, 1.0).astype(np.float32)


def down(a, f):
    return a.reshape(S // f, f, S // f, f).mean((1, 3))


def up(a, f):
    return np.asarray(Image.fromarray(a).resize((S, S), Image.BILINEAR))


def noise_curve(video, noise, n=40, seed=0):
    """Robust variance of d = B - A as a function of local brightness, from n pairs of one video."""
    frames = {frame_no(p): p for p in video_frames(os.path.join(DATA, video))}
    starts = sorted(f for f in frames if f + 6 in frames)
    rng = np.random.default_rng(seed)
    ds, ms = [], []
    for f in rng.choice(starts, size=min(n, len(starts)), replace=False):
        a, b = load_pair(frames[f], frames[f + 6], noise)
        ds.append((b - a).ravel())
        ms.append(gaussian_filter((a + b) / 2, 2.0).ravel())
    d, m = np.concatenate(ds), np.concatenate(ms)
    idx = np.clip(np.digitize(m, BRIGHT_EDGES) - 1, 0, len(BRIGHT_EDGES) - 2)
    centers, var = [], []
    for i in range(len(BRIGHT_EDGES) - 1):
        di = d[idx == i]
        if di.size >= 2000:
            mad = np.median(np.abs(di - np.median(di)))
            centers.append((BRIGHT_EDGES[i] + BRIGHT_EDGES[i + 1]) / 2)
            var.append((1.4826 * mad) ** 2)
    var = np.maximum(np.array(var), 2 * (1 / 255.0) ** 2 / 12)  # at least uint8 rounding noise of A and B
    return np.array(centers), var


def prep_job(args):
    video, nname = args
    bg = compute_backgrounds(os.path.join(DATA, video), NOISES[nname], step=60, window=600, seg=300)
    return video, nname, bg, noise_curve(video, NOISES[nname])


# std of T = gauss_sigma(chi2_1 variable) under pure Gaussian noise: sqrt(2 * sum(w^2)), sum(w^2) ~ 1/(4 pi s^2)
T_SD = np.sqrt(2.0 / (4 * np.pi * SIG_TEST ** 2))
T0 = 1.0 + 3.0 * T_SD


def targets(a, b, bg, curve):
    out = dict(bkind=bkind_target(a, b))
    fa, fb = foreground(a, bg, 2.0, 0.1), foreground(b, bg, 2.0, 0.1)
    out['opt1_diff'] = np.abs(fa - fb)
    out['opt1_union'] = np.maximum(fa, fb)
    centers, var = curve
    m = gaussian_filter((a + b) / 2, 2.0)
    v = np.interp(m, centers, var)
    t = gaussian_filter((b - a) ** 2 / v, SIG_TEST)
    out['opt2_test'] = np.clip((t - T0) / T0, 0, 1).astype(np.float32)
    for f in (2, 4):
        out[f'opt3_ds{f}'] = up(bkind_target(down(a, f), down(b, f)), f)
    return out


def auc(pos, neg, rng, k=4000):
    pos = rng.choice(pos, size=min(k, pos.size), replace=False)
    neg = rng.choice(neg, size=min(k, neg.size), replace=False)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


PREP, KP = None, None


def pair_job(args):
    j, v, f, pa, pb = args
    rng = np.random.default_rng(j)
    yy, xx = np.mgrid[0:S, 0:S]
    pts = np.concatenate([KP[f'{v}/keypoints'][f], KP[f'{v}/keypoints'][f + 6]], 0)
    px, py = (pts[:, 0] * S / W).reshape(-1), (pts[:, 1] * S / H).reshape(-1)
    d2 = np.min((xx[..., None] - px) ** 2 + (yy[..., None] - py) ** 2, axis=-1)
    mice, still = d2 < 10 ** 2, d2 > 30 ** 2
    res, maps = {}, {}
    for n in NOISES:
        a, b = load_pair(pa, pb, NOISES[n])
        bg, curve = PREP[(v, n)]
        ts = targets(a, b, background_for(bg, f), curve)
        maps[n] = dict(ts, A=a)
        for c in CANDS:
            t = ts[c]
            res[(n, c)] = dict(mice=float(t[mice].mean()), still=float(t[still].mean()),
                               auc=auc(t[mice], t[still], rng))
    for n in NOISES:
        for c in CANDS:
            x, y = maps['clean'][c].ravel(), maps[n][c].ravel()
            res[(n, c)]['corr'] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float('nan')
    motion = float(np.abs(maps['clean']['A'] - load_pair(pa, pb, None)[1])[mice].mean())
    return j, res, (maps if j < 40 else None), motion


def init_worker(kp_path):
    global KP
    KP = dict(np.load(kp_path))


def main():
    global DATA, PREP
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=150)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    DATA = a.data
    os.makedirs(a.out, exist_ok=True)
    videos = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))

    with Pool(a.workers) as pool:
        prep = {}
        for v, n, bg, curve in pool.imap_unordered(prep_job, [(v, n) for v in videos for n in NOISES]):
            prep[(v, n)] = (bg, curve)
            print('prepared', v, n, flush=True)
    PREP = prep  # read by pair_job in forked workers (not pickled: ~0.5 GB of backgrounds)
    curves = {n: {v: dict(brightness=prep[(v, n)][1][0].tolist(), var=prep[(v, n)][1][1].tolist())
                  for v in videos} for n in NOISES}

    rng = np.random.default_rng(0)
    jobs = []
    for j in range(a.pairs):
        v = videos[rng.integers(len(videos))]
        frames = {frame_no(p): p for p in video_frames(os.path.join(a.data, v))}
        starts = [f for f in frames if f + 6 in frames]
        f = int(starts[rng.integers(len(starts))])
        jobs.append((j, v, f, frames[f], frames[f + 6]))

    results, examples = [], []
    with Pool(a.workers, initializer=init_worker, initargs=(os.path.join(a.data, 'keypoints.npz'),)) as pool:
        for j, res, maps, motion in pool.imap_unordered(pair_job, jobs):
            results.append(res)
            if maps is not None:
                examples.append((motion, maps))
            if len(results) % 25 == 0:
                print(f'{len(results)}/{len(jobs)} pairs', flush=True)

    table = {}
    for n in NOISES:
        for c in CANDS:
            r = [x[(n, c)] for x in results]
            mice, still = np.mean([x['mice'] for x in r]), np.mean([x['still'] for x in r])
            table[f'{n}/{c}'] = dict(mice=mice, still=still, ratio=mice / max(still, 1e-4),
                                     auc=np.mean([x['auc'] for x in r]), corr=np.nanmean([x['corr'] for x in r]))
    lines = [f"{'noise':8s} {'candidate':11s} {'ratio':>7s} {'auc':>6s} {'corr':>6s} {'mice':>7s} {'still':>7s}"]
    for n in NOISES:
        for c in CANDS:
            t = table[f'{n}/{c}']
            lines.append(f"{n:8s} {c:11s} {t['ratio']:7.1f} {t['auc']:6.3f} {t['corr']:6.3f} {t['mice']:7.4f} {t['still']:7.4f}")
    print('\n'.join(lines))
    open(f'{a.out}/change_targets.txt', 'w').write('\n'.join(lines) + '\n')
    json.dump(dict(pairs=a.pairs, table=table, noise_curves=curves), open(f'{a.out}/change_targets.json', 'w'), indent=1)
    make_figure(a.out, table, examples)


def make_figure(out, table, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples = [m for _, m in sorted(examples, key=lambda e: -e[0])[:2]]
    show_noises = ['clean', 'pg98', 'pg95', 'pg90', 'sp95', 'blur95', 'h264_95']
    nrow = len(examples) * (len(CANDS) + 1)
    fig = plt.figure(figsize=(1.5 * len(show_noises) + 0.5, 1.5 * nrow + 4.5))
    gs = fig.add_gridspec(nrow + 1, len(show_noises), height_ratios=[1] * nrow + [3.0], hspace=0.12, wspace=0.04)
    r = 0
    for maps in examples:
        rows = ['A'] + CANDS
        for k in rows:
            scale = 1.0 if k == 'A' else max(np.percentile(maps['clean'][k], 99.5), 1e-3)
            for c, n in enumerate(show_noises):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(maps[n][k] / scale, cmap='gray' if k == 'A' else 'magma', vmin=0, vmax=1)
                ax.set_xticks([]), ax.set_yticks([])
                if r == 0:
                    ax.set_title(n, fontsize=8)
                if c == 0:
                    ax.set_ylabel('frame A' if k == 'A' else LABELS[k], fontsize=6)
            r += 1
    for i, (metric, title) in enumerate([('auc', 'AUC: mouse pixels vs still floor'),
                                         ('corr', 'correlation with own clean map')]):
        ax = fig.add_subplot(gs[nrow, i * 3:(i + 1) * 3 + (1 if i else 0)])
        xs = np.arange(len(show_noises))
        wd = 0.8 / len(CANDS)
        for k, c in enumerate(CANDS):
            ax.bar(xs + (k - len(CANDS) / 2 + 0.5) * wd, [table[f'{n}/{c}'][metric] for n in show_noises], wd,
                   label=LABELS[c])
        ax.set_xticks(xs)
        ax.set_xticklabels(show_noises, fontsize=7, rotation=30)
        ax.set_ylim(0.5 if metric == 'auc' else 0, 1.0)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, ncol=2)
    fig.suptitle('Change between frames A and B: candidate targets (maps scaled by their clean 99.5th percentile)',
                 fontsize=9)
    fig.savefig(f'{out}/change_targets.png', dpi=150, bbox_inches='tight')
    print('wrote', f'{out}/change_targets.png')


if __name__ == '__main__':
    main()
