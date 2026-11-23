"""noisykeypoints: dense motion network for the warping method (reports/warp_method_design.md).

Adapted from the First Order Motion Model (Siarohin et al., NeurIPS 2019),
https://github.com/AliaksandrSiarohin/first-order-model (MIT licence): modules/util.py (Hourglass,
AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian) and modules/dense_motion.py. Changes:
plain BatchNorm instead of synchronised BatchNorm, no occlusion map (our decoder may only move pixels),
explicit align_corners=True everywhere (matches BKinD's keypoint coordinates, linspace(-1, 1)), and the
geometric parts (sparse motions, warping) run in float32 under mixed precision.

Coordinates: (x, y) in [-1, 1], x = column, y = row, as in BKinD and FOMM.
Keypoints are dicts: 'value' (N, K, 2) and 'jacobian' (N, K, 2, 2).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coordinate_grid(h, w, device, dtype=torch.float32):
    """(h, w, 2) grid of (x, y) in [-1, 1] (pixel centres at the ends, i.e. align_corners=True)."""
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    return torch.stack([x.view(1, w).expand(h, w), y.view(h, 1).expand(h, w)], dim=2)


def kp2gaussian(kp_value, h, w, variance):
    """Gaussian blob around each keypoint: (N, K, 2) -> (N, K, h, w)."""
    grid = make_coordinate_grid(h, w, kp_value.device, kp_value.dtype).view(1, 1, h, w, 2)
    diff = grid - kp_value.view(*kp_value.shape[:2], 1, 1, 2)
    return torch.exp(-0.5 * (diff ** 2).sum(-1) / variance)


def safe_inverse_2x2(m, min_det=1e-3):
    """Inverse of (..., 2, 2) matrices, with |det| kept >= min_det so near-singular Jacobians can't
    blow up. Computed in float32."""
    m = m.float()
    a, b, c, d = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
    det = a * d - b * c
    det = torch.where(det.abs() < min_det, torch.full_like(det, min_det) * torch.sign(det + 1e-12), det)
    inv = torch.stack([torch.stack([d, -b], -1), torch.stack([-c, a], -1)], -2)
    return inv / det[..., None, None]


class AntiAliasInterpolation2d(nn.Module):
    """Gaussian blur, then keep every (1/scale)-th pixel (FOMM). scale 1 = unchanged."""

    def __init__(self, channels, scale):
        super().__init__()
        self.scale = scale
        if scale == 1.0:
            return
        sigma = (1 / scale - 1) / 2
        size = 2 * round(sigma * 4) + 1
        self.ka = size // 2
        self.kb = self.ka - 1 if size % 2 == 0 else self.ka
        g = torch.arange(size, dtype=torch.float32)
        g = torch.exp(-(g - (size - 1) / 2) ** 2 / (2 * sigma ** 2))
        kernel = g.view(-1, 1) * g.view(1, -1)
        kernel = kernel / kernel.sum()
        self.register_buffer('weight', kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1))
        self.groups = channels
        self.step = int(1 / scale)

    def forward(self, x):
        if self.scale == 1.0:
            return x
        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight.to(out.dtype), groups=self.groups)
        return out[:, :, ::self.step, ::self.step]


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        return self.pool(F.relu(self.norm(self.conv(x))))


class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        return F.relu(self.norm(self.conv(F.interpolate(x, scale_factor=2))))


class Hourglass(nn.Module):
    """U-Net (FOMM): num_blocks halvings, then doublings with skip connections."""

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super().__init__()
        ch = lambda i: min(max_features, block_expansion * (2 ** i))
        self.down_blocks = nn.ModuleList(
            [DownBlock2d(in_features if i == 0 else ch(i), ch(i + 1)) for i in range(num_blocks)])
        self.up_blocks = nn.ModuleList(
            [UpBlock2d((1 if i == num_blocks - 1 else 2) * ch(i + 1), ch(i)) for i in range(num_blocks)[::-1]])
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        outs = [x]
        for block in self.down_blocks:
            outs.append(block(outs[-1]))
        out = outs.pop()
        for block in self.up_blocks:
            out = torch.cat([block(out), outs.pop()], dim=1)
        return out


class DenseMotionNetwork(nn.Module):
    """Blends K keypoint motions + 1 background (identity) motion into one dense flow (FOMM).

    forward(source_image, kp_driving, kp_source) -> dict
      'deformation': (N, h, w, 2), for each pixel of the driving frame, where to sample the source
                     frame (at scale_factor x the input resolution)
      'mask':        (N, K+1, h, w), softmax weights of the K+1 motions (channel 0 = background)
    """

    def __init__(self, num_kp, num_channels=3, block_expansion=64, num_blocks=5, max_features=1024,
                 scale_factor=0.25, kp_variance=0.01):
        super().__init__()
        self.K = num_kp
        self.kp_variance = kp_variance
        self.hourglass = Hourglass(block_expansion, (num_kp + 1) * (num_channels + 1), num_blocks, max_features)
        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)
        self.down = AntiAliasInterpolation2d(num_channels, scale_factor)

    def sparse_motions(self, h, w, kp_driving, kp_source):
        """(N, K+1, h, w, 2): for each motion, where each driving pixel comes from in the source.
        Keypoint k: z_S = p_S + J_S J_D^-1 (z - p_D); motion 0 is the identity."""
        n = kp_driving['value'].shape[0]
        grid = make_coordinate_grid(h, w, kp_driving['value'].device).view(1, 1, h, w, 2)
        rel = grid - kp_driving['value'].view(n, self.K, 1, 1, 2)
        jac = kp_source['jacobian'] @ safe_inverse_2x2(kp_driving['jacobian'])  # (N, K, 2, 2)
        rel = (jac.view(n, self.K, 1, 1, 2, 2) @ rel.unsqueeze(-1)).squeeze(-1)
        to_source = rel + kp_source['value'].view(n, self.K, 1, 1, 2)
        return torch.cat([grid.expand(n, 1, h, w, 2), to_source], dim=1)

    def forward(self, source_image, kp_driving, kp_source):
        with torch.autocast('cuda', enabled=False):  # geometry in float32
            src = self.down(source_image.float())
            n, c, h, w = src.shape
            kp_d = {k: v.float() for k, v in kp_driving.items()}
            kp_s = {k: v.float() for k, v in kp_source.items()}
            heat = kp2gaussian(kp_d['value'], h, w, self.kp_variance) - kp2gaussian(kp_s['value'], h, w, self.kp_variance)
            heat = torch.cat([torch.zeros(n, 1, h, w, device=heat.device), heat], dim=1).unsqueeze(2)
            motions = self.sparse_motions(h, w, kp_d, kp_s)  # (N, K+1, h, w, 2)
            deformed = F.grid_sample(src.unsqueeze(1).expand(n, self.K + 1, c, h, w).reshape(n * (self.K + 1), c, h, w),
                                     motions.reshape(n * (self.K + 1), h, w, 2), align_corners=True)
            deformed = deformed.view(n, self.K + 1, c, h, w)
            net_in = torch.cat([heat, deformed], dim=2).view(n, -1, h, w)
        mask = self.mask(self.hourglass(net_in))  # may run in float16 under autocast
        with torch.autocast('cuda', enabled=False):
            mask = F.softmax(mask.float(), dim=1)
            flow = (motions.permute(0, 1, 4, 2, 3) * mask.unsqueeze(2)).sum(1).permute(0, 2, 3, 1)
        return {'deformation': flow, 'mask': mask}
