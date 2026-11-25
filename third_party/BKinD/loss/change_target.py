"""noisykeypoints: alternative "what changed between frame A and frame B" targets for BKinD's decoder.

BKinD's decoder redraws to_pred = blur(1 - SSIM contrast-structure map of A vs B) (loss/ssim_loss.py).
With noise that changes between A and B that map is noise everywhere, which is what breaks BKinD.
Config option `change_target` picks the target (default None = BKinD's):

  {type: tracks, sigma: 4, peak: 0.5, full: 4}
      Built only from precomputed CoTracker3 tracks of the moving points (dataloader/tracks.py; needs the
      `tracks` config). A Gaussian blob (sigma px) is drawn at each point's position in A and in B, with
      height peak * min(displacement A->B / full px, 1); blobs are combined by max. No pixel differences.
  {type: coarse, factor: 4}
      Control without a tracker: BKinD's own target computed on frames averaged down `factor`x, then
      stretched back to full size (bilinear). Averaging cuts per-pixel noise by `factor`.
  {type: feature, size: 384, gain: 1.0}
      Change in CoTracker3's frozen feature encoder (fnet, stride 4, 128 channels, unit length per spot):
      target = gain * clamp(1 - cos(f_A, f_B), 0, 1) at each spot, stretched back to full size. Frames are
      resized to size x size before the encoder (CoTracker3 runs at 384 x 512). Loaded once with torch.hub.
"""
import torch
import torch.nn.functional as F

from loss.ssim_loss import compute_ssim

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_FNET = {}


def cotracker_encoder(device):
    """CoTracker3 offline's frozen feature encoder (torch.hub facebookresearch/co-tracker)."""
    if device not in _FNET:
        pred = torch.hub.load('facebookresearch/co-tracker', 'cotracker3_offline')
        fnet = pred.model.fnet.to(device).eval()
        for p in fnet.parameters():
            p.requires_grad_(False)
        _FNET[device] = fnet
    return _FNET[device]


@torch.no_grad()
def feature_target(inputs, tr_inputs, size=384, gain=1.0):
    """inputs, tr_inputs: (B, 3, H, W) ImageNet-normalised frames. Returns (B, 3, H, W)."""
    fnet = cotracker_encoder(inputs.device)
    mean = torch.tensor(_MEAN, device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=inputs.device).view(1, 3, 1, 1)
    x = torch.cat([inputs.float(), tr_inputs.float()]) * std + mean           # [0, 1]
    x = F.interpolate(x, size=(int(size), int(size)), mode='bilinear', align_corners=False)
    f = F.normalize(fnet(2 * x - 1).float(), dim=1)
    fa, fb = f.chunk(2)
    t = (1 - (fa * fb).sum(1, keepdim=True)).clamp(0, 1) * gain
    t = F.interpolate(t, size=inputs.shape[-2:], mode='bilinear', align_corners=False)
    return t.repeat(1, 3, 1, 1)


def coarse_target(inputs, tr_inputs, factor=4):
    a = F.avg_pool2d(inputs.float(), int(factor))
    b = F.avg_pool2d(tr_inputs.float(), int(factor))
    t = compute_ssim(a, b)
    return F.interpolate(t, size=inputs.shape[-2:], mode='bilinear', align_corners=False)


def track_target(tracks, size, sigma=4.0, peak=0.5, full=4.0, chunk=64):
    """tracks (B, N, 5): x_A, y_A, x_B, y_B in [-1, 1] (align_corners=True), valid. Returns (B, 3, H, W)."""
    h, w = size
    tracks = tracks.float()
    scale = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], device=tracks.device)
    pa = (tracks[:, :, 0:2] + 1) * scale
    pb = (tracks[:, :, 2:4] + 1) * scale
    amp = peak * ((pb - pa).norm(dim=-1) / full).clamp(max=1.0) * tracks[:, :, 4]
    pts = torch.cat([pa, pb], 1)                      # (B, 2N, 2) pixels
    amp = torch.cat([amp, amp], 1)                    # (B, 2N)
    ys = torch.arange(h, device=tracks.device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.arange(w, device=tracks.device, dtype=torch.float32).view(1, 1, 1, w)
    out = torch.zeros(tracks.shape[0], 1, h, w, device=tracks.device)
    for i in range(0, pts.shape[1], chunk):
        p = pts[:, i:i + chunk]
        d2 = (xs - p[:, :, 0, None, None]) ** 2 + (ys - p[:, :, 1, None, None]) ** 2
        blob = amp[:, i:i + chunk, None, None] * torch.exp(-d2 / (2 * sigma ** 2))
        out = torch.maximum(out, blob.max(1, keepdim=True)[0])
    return out.repeat(1, 3, 1, 1)


def change_target(args, inputs, tr_inputs, tracks=None):
    cfg = getattr(args, 'change_target', None)
    if not cfg:
        return compute_ssim(inputs.float(), tr_inputs.float())
    kind = cfg.get('type')
    if kind == 'coarse':
        return coarse_target(inputs, tr_inputs, cfg.get('factor', 4))
    if kind == 'feature':
        return feature_target(inputs, tr_inputs, cfg.get('size', 384), float(cfg.get('gain', 1.0)))
    if kind == 'tracks':
        if tracks is None:
            raise ValueError('change_target tracks needs the `tracks` config (precomputed tracks)')
        return track_target(tracks, inputs.shape[-2:], float(cfg.get('sigma', 4.0)),
                            float(cfg.get('peak', 0.5)), float(cfg.get('full', 4.0)))
    raise ValueError(f'unknown change_target type {kind}')
