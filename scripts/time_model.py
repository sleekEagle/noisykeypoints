"""A small time model on top of a trained BKinD model's keypoints (CalMS21; trained after BKinD, BKinD frozen).

1. extract: run the trained detector on every frame of the 10 no-cable videos (noised as in that run's
   training) -> <out>/raw/<video>.npz with kpts (T, K, 2) in [-1, 1] (x, y), conf (T, K), frames (T,).
2. train: a transformer over a window of 2r+1 frames of keypoints (tokens = frames, each holding all K
   keypoints with confidence) outputs a correction for the centre frame's keypoints (last layer starts
   at zero, so it starts as "no change"). It is trained with BKinD's own reconstruction loss (epoch-0
   form: perceptual loss on the run's change target), through BKinD's frozen decoder: the refined
   keypoints of A and B replace the detector's in the decoder input. Only the time model learns.
3. apply: refined keypoints for every frame -> <out>/refined/..., plus all-frame npz files for
   eval_calms21.py --keypoints, and jitter (mean acceleration, px at 1024 x 570) raw vs refined.

    python scripts/time_model.py --run CalMS21_ct4_pg_ssim95_s1 --out /workspace/time_model/ct4_pg95 all
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
BK = os.path.join(HERE, '..', 'third_party', 'BKinD')
sys.path.insert(0, BK)
from dataloader.noise import apply_noise_pair  # noqa: E402
from dataloader.mouse_dataset import generate_pair_images  # noqa: E402
from model.unsupervised_model import Model  # noqa: E402

S, W, H = 256, 1024, 570
NO_CABLE = ['mouse001', 'mouse002', 'mouse051', 'mouse052', 'mouse055', 'mouse056', 'mouse058', 'mouse053',
            'mouse054', 'mouse057']
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Args(dict):
    __getattr__ = dict.get


def load_bkind(run, runs_dir, device):
    cp = os.path.join(BK, 'config', f'kaggle_{run}.yaml')
    cfg = yaml.safe_load(open(cp)) if os.path.exists(cp) else {}  # old Kaggle runs (e.g. clean_s1): clean, 10 kpts
    if not cfg:
        print(f'no config {cp}: assuming clean frames, 10 keypoints', flush=True)
    ck = torch.load(os.path.join(runs_dir, run, 'checkpoint.pth.tar'), map_location='cpu', weights_only=False)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in ck['state_dict'].items()}
    if sd['encoder.conv1.weight'].shape[1] != 3:
        raise SystemExit('multi-frame models are not supported here')
    model = Model(cfg.get('nkpts', 10), pretrained=False, output_shape=(S // 4, S // 4))
    model.load_state_dict(sd)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'loaded {run} (epoch {ck.get("epoch")})', flush=True)
    return model, Args(cfg)


def to_tensor(im):
    return (torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1) - MEAN) / STD


def load_frames(paths, noise):
    ims = [Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR) for p in paths]
    if noise:
        ims = apply_noise_pair(ims, noise, paths)
    return [to_tensor(im) for im in ims]


def detect(model, x):
    """BKinD's keypoints (u_x, u_y in [-1, 1]) and confidence of a batch of frames (as in Model.forward)."""
    out = model.kptNet(model.encoder(x))[1][-1]
    hm = F.softmax(out.view(out.shape[0], model.K, -1), dim=2).view_as(out)
    u_x, u_y, _ = model._mapTokpt(hm)
    return torch.stack([u_x, u_y], -1), hm.amax((2, 3))


def recon_from_kpts(model, x_res, ka, kb):
    """BKinD's decoder output from A's encoder features and keypoints of A and B (as in Model.forward)."""
    conds, h, w = [], model.output_shape[0] // 16, model.output_shape[1] // 16
    for std in [0.1, 0.1, 0.01, 0.01, 0.001]:
        h, w = h * 2, w * 2
        conds.append(torch.cat([model._kptTomap(kb[..., 0], kb[..., 1], H=h, W=w, inv_std=std),
                                model._kptTomap(ka[..., 0], ka[..., 1], H=h, W=w, inv_std=std)], 1))
    return model.decoder(x_res, conds)


# ---------------------------------------------------------------- 1. extract
class AllFrames(torch.utils.data.Dataset):
    def __init__(self, items, noise):
        self.items, self.noise = items, noise

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return load_frames([self.items[i][2]], self.noise)[0]


def video_frames(d):
    fs = {}
    for f in os.listdir(d):
        if f.startswith('image_') and f.endswith('.png'):
            fs[int(f[6:-4])] = os.path.join(d, f)
    return [fs[k] for k in sorted(fs)], sorted(fs)


def extract(a, model, cfg, device):
    os.makedirs(f'{a.out}/raw', exist_ok=True)
    noise = cfg.get('noise')
    if noise and noise.get('type') == 'h264':
        raise SystemExit('h264 runs encode pairs together; not supported here')
    for v in sorted(os.listdir(a.frames)):
        if v[:8] not in NO_CABLE or os.path.exists(f'{a.out}/raw/{v}.npz'):
            continue
        paths, frames = video_frames(os.path.join(a.frames, v))
        dl = torch.utils.data.DataLoader(AllFrames([(v, f, p) for f, p in zip(frames, paths)], noise),
                                         batch_size=64, num_workers=a.workers)
        ks, cs, t0 = [], [], time.time()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for x in dl:
                k, c = detect(model, x.to(device))
                ks.append(k.float().cpu().numpy())
                cs.append(c.float().cpu().numpy())
        np.savez_compressed(f'{a.out}/raw/{v}.npz', kpts=np.concatenate(ks), conf=np.concatenate(cs),
                            frames=np.array(frames))
        print(f'extract {v}: {len(frames)} frames, {time.time() - t0:.0f} s', flush=True)


def load_tracks(out, kind='raw'):
    return {f[:-4]: dict(np.load(f'{out}/{kind}/{f}')) for f in sorted(os.listdir(f'{out}/{kind}'))}


# ---------------------------------------------------------------- the time model
class TimeNet(nn.Module):
    def __init__(self, k, r, d=128, layers=3, heads=4):
        super().__init__()
        self.k, self.r = k, r
        self.inp = nn.Linear(3 * k, d)
        self.pos = nn.Parameter(torch.zeros(1, 2 * r + 1, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, 2 * k)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, win):
        """win (B, 2r+1, K, 3): x, y, conf per frame. Returns the refined centre keypoints (B, K, 2)."""
        b = win.shape[0]
        centre = win[:, self.r, :, :2]
        rel = torch.cat([win[..., :2] - centre[:, None], win[..., 2:]], -1)  # positions relative to the centre
        h = self.body(self.inp(rel.reshape(b, 2 * self.r + 1, -1)) + self.pos)
        return centre + self.out(h[:, self.r]).view(b, self.k, 2)


def window(track, i, r):
    """(2r+1, K, 3) around frame index i, clamped at the video's ends."""
    idx = np.clip(np.arange(i - r, i + r + 1), 0, len(track['frames']) - 1)
    return np.concatenate([track['kpts'][idx], track['conf'][idx][..., None]], -1).astype(np.float32)


# ---------------------------------------------------------------- 2. train
class Pairs(torch.utils.data.Dataset):
    def __init__(self, pairs, tracks, r, noise):
        self.pairs, self.tracks, self.r, self.noise = pairs, tracks, r, noise
        self.index = {v: {int(f): i for i, f in enumerate(t['frames'])} for v, t in tracks.items()}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pa, pb = self.pairs[i]
        v = os.path.basename(os.path.dirname(pa))
        fa, fb = (int(os.path.basename(p)[6:-4]) for p in (pa, pb))
        xa, xb = load_frames([pa, pb], self.noise)
        t = self.tracks[v]
        return xa, xb, window(t, self.index[v][fa], self.r), window(t, self.index[v][fb], self.r)


def train(a, model, cfg, device):
    from loss.compute_loss import computeLoss
    tracks = load_tracks(a.out)
    pairs = [p for p in generate_pair_images(a.data) if os.path.basename(os.path.dirname(p[0])) in tracks]
    cfg.update(gpu=0, amp=True)
    loss_module = computeLoss(cfg)
    for p in loss_module.loss_model.parameters():
        p.requires_grad_(False)
    net = TimeNet(model.K, a.r).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    dl = torch.utils.data.DataLoader(Pairs(pairs, tracks, a.r, cfg.get('noise')), batch_size=a.batch, shuffle=True,
                                     num_workers=a.workers, drop_last=True)
    log = open(f'{a.out}/train_log.txt', 'a')
    print(f'{len(pairs)} pairs, window {2 * a.r + 1} frames, {sum(p.numel() for p in net.parameters())} params',
          file=log, flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        tot = n = 0
        for i, (xa, xb, wa, wb) in enumerate(dl):
            xa, xb, wa, wb = (t.to(device, non_blocking=True) for t in (xa, xb, wa, wb))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                x_res = model.encoder(xa)
            ka, kb = net(wa), net(wb)
            with torch.autocast('cuda', dtype=torch.float16):
                recon = recon_from_kpts(model, x_res, ka, kb)
            mask = torch.ones_like(xa[:, :1])
            loss = loss_module.update_loss(xa, xb, mask, (recon.float(),), epoch=0)
            delta = ((ka - wa[:, a.r, :, :2]) ** 2).mean() + ((kb - wb[:, a.r, :, :2]) ** 2).mean()
            loss = loss + a.delta_weight * delta
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item(), n + 1
            if i % 200 == 0:
                print(f'epoch {ep} step {i}/{len(dl)} loss {tot / n:.4f} delta {delta.item() ** 0.5 * S / 2:.2f} px '
                      f'{time.time() - t0:.0f} s', file=log, flush=True)
        torch.save(dict(state_dict=net.state_dict(), r=a.r, k=model.K, epoch=ep + 1), f'{a.out}/time_model.pth')
        print(f'epoch {ep} done: loss {tot / n:.4f}', file=log, flush=True)


# ---------------------------------------------------------------- 3. apply + jitter
def apply(a, device):
    ck = torch.load(f'{a.out}/time_model.pth', map_location='cpu')
    net = TimeNet(ck['k'], ck['r']).to(device).eval()
    net.load_state_dict(ck['state_dict'])
    tracks = load_tracks(a.out)
    res, allk = {}, {'raw': [], 'refined': []}
    os.makedirs(f'{a.out}/refined', exist_ok=True)
    for v, t in tracks.items():
        wins = np.stack([window(t, i, ck['r']) for i in range(len(t['frames']))])
        with torch.no_grad():
            ref = np.concatenate([net(torch.from_numpy(wins[j:j + 1024]).to(device)).cpu().numpy()
                                  for j in range(0, len(wins), 1024)])
        np.savez_compressed(f'{a.out}/refined/{v}.npz', kpts=ref, conf=t['conf'], frames=t['frames'])
        for kind, k in (('raw', t['kpts']), ('refined', ref)):
            allk[kind].append((v, t['frames'], k, t['conf']))
    for kind, lst in allk.items():
        px = [np.stack([(k[..., 0] + 1) / 2 * W, (k[..., 1] + 1) / 2 * H], -1) for _, f, k, _ in lst]
        # jitter = mean |p(t+1) - 2 p(t) + p(t-1)| over consecutive frames, px at 1024 x 570
        acc = np.concatenate([np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1).ravel() for p in px])
        res[kind] = dict(jitter_px=float(acc.mean()), jitter_median_px=float(np.median(acc)))
        np.savez_compressed(f'{a.out}/{kind}_all.npz', kpts=np.concatenate([k for *_, k, _ in lst]),
                            conf=np.concatenate([c for *_, c in lst]),
                            videos=np.concatenate([[v] * len(f) for v, f, *_ in lst]),
                            frames=np.concatenate([f for _, f, *_ in lst]), method=f'time_model_{kind}')
    shift = np.concatenate([np.linalg.norm((r[2] - k[2]) * [W / 2, H / 2], axis=-1).ravel()
                            for r, k in zip(allk['refined'], allk['raw'])])
    res['mean_shift_px'] = float(shift.mean())
    json.dump(res, open(f'{a.out}/jitter.json', 'w'), indent=1)
    print(json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['extract', 'train', 'apply', 'all'])
    ap.add_argument('--run', required=True)
    ap.add_argument('--runs', default='/workspace/runs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', default='/root/data/CalMS21_allframes/train', help='every-frame folder')
    ap.add_argument('--data', default='/root/data/CalMS21_bkind/train', help='BKinD training pairs')
    ap.add_argument('--r', type=int, default=8, help='window = 2r+1 frames')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--delta-weight', type=float, default=0.0)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = 'cuda'
    torch.manual_seed(0)
    if a.step in ('extract', 'train', 'all'):
        model, cfg = load_bkind(a.run, a.runs, device)
    if a.step in ('extract', 'all'):
        extract(a, model, cfg, device)
    if a.step in ('train', 'all'):
        train(a, model, cfg, device)
    if a.step in ('apply', 'all'):
        apply(a, device)


if __name__ == '__main__':
    main()
