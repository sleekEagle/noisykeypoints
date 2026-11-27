"""noisykeypoints: noise-adaptive weighting of BKinD's change target at several resolutions.

With `multi_scale: {factors: [1, 2, 4], sigma: global | image}` the decoder has one head per factor;
head k redraws BKinD's 1 - SSIM change map computed on frames averaged down factors[k] x (loss/
change_target.py). Each head's loss L_k is weighted by a learned uncertainty (Kendall et al. 2018):

    loss = mean_k ( L_k * exp(-s_k) + s_k ),   s_k = log sigma_k^2

The optimum is exp(s_k) = L_k, so a head whose target is mostly unexplainable noise (full resolution
under grain) gets a large sigma and counts less; on clean frames the fine head keeps its weight.

With `normalize: true` the total weight is fixed at 1 (as for ct4's single head) and sigma only decides
the split: loss = sum_k p_k * L_k with p = softmax(-s) (no gradient through p), plus the uncertainty loss
above on the detached L_k, which trains s alone. p_k is then proportional to 1 / L_k. Without it the
weights exp(-s) grew to ~30 (2026-11-27), so the change loss swamped the rotation and separation losses.

  sigma: global   one learned s_k per head, shared by all images (n_scales numbers).
  sigma: image    s_k predicted per image pair by a small CNN on frames A and B (it can see how grainy
                  the pair is). The last layer starts at zero, so both modes start with s = 0.
"""
import torch
import torch.nn as nn


class GlobalSigma(nn.Module):
    def __init__(self, n_scales):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_scales))

    def forward(self, a, b):
        return self.log_var.float().unsqueeze(0).expand(a.shape[0], -1)


class ImageSigma(nn.Module):
    def __init__(self, n_scales, width=32):
        super().__init__()
        w = width
        layers, c = [], 6
        for out in (w // 2, w, 2 * w, 2 * w):
            layers += [nn.Conv2d(c, out, 3, stride=2, padding=1), nn.ReLU(inplace=True)]
            c = out
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(c, n_scales)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, a, b):
        # a, b: (B, 3, H, W) frames A and B, ImageNet-normalised. B - A shows the grain directly.
        return self.head(self.features(torch.cat([a, b - a], 1))).float()


def make_scale_sigma(kind, n_scales):
    if kind is None or n_scales <= 1:
        return None
    if kind == 'global':
        return GlobalSigma(n_scales)
    if kind == 'image':
        return ImageSigma(n_scales)
    raise ValueError(f'unknown multi_scale sigma {kind}')


def multi_scale_args(args):
    """(factors, sigma kind) from the config, or ((), None) when multi_scale is off."""
    cfg = getattr(args, 'multi_scale', None)
    if not cfg:
        return (), None
    return tuple(int(f) for f in cfg.get('factors', (1, 2, 4))), cfg.get('sigma', 'global')


def multi_scale_normalize(args):
    cfg = getattr(args, 'multi_scale', None)
    return bool(cfg and cfg.get('normalize', False))
