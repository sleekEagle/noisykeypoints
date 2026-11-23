"""noisykeypoints: losses for the warping model (reports/warp_method_design.md).

Adapted from the First Order Motion Model (Siarohin et al., NeurIPS 2019),
https://github.com/AliaksandrSiarohin/first-order-model (MIT licence), modules/model.py: the random
affine + thin-plate-spline Transform, the image pyramid perceptual loss and the keypoint / Jacobian
equivariance losses. BKinD's VGG-19 wrapper (model/vggforLoss.py) provides the perceptual features;
inputs are already ImageNet-normalised by the data loader, as that wrapper expects.

All loss maths is float32; only the VGG network runs in float16 under mixed precision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dense_motion import AntiAliasInterpolation2d, make_coordinate_grid, safe_inverse_2x2
from model.vggforLoss import VGG_feat
from loss.separation_loss import separation_loss

PYRAMID_SCALES = (1.0, 0.5, 0.25, 0.125)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gaussian_blur(x, sigma):
    """Separable Gaussian blur of (N, 1, H, W), reflection padding."""
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))


def motion_weight(source, driving, sigma=4.0, tau=0.05, floor=0.05):
    """Where things move between frames A (source) and B (driving): a fixed (no-gradient) per-pixel
    weight in [floor, 1], shape (N, 1, H, W).

    Like BKinD's target, it is large where A and B differ, so the small moving mice count for more
    than the large still background. Unlike BKinD it is computed on heavily blurred frames (Gaussian,
    sigma px at 256 px): blurring averages per-frame grain away (sigma 4 averages ~200 pixels, so noise
    of 0.024 at SSIM 0.90 drops to ~0.002), while a moving mouse is a large, smooth change that survives.
    `tau` is the blurred change (in [0, 1] intensity) that counts as full motion; `floor` keeps a small
    weight on still pixels so the background is not ignored completely."""
    with torch.no_grad():
        mean = torch.tensor(IMAGENET_MEAN, device=source.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=source.device).view(1, 3, 1, 1)
        gray = lambda x: (x.float() * std + mean).mean(1, keepdim=True)  # back to [0, 1] grey
        diff = (gaussian_blur(gray(source), sigma) - gaussian_blur(gray(driving), sigma)).abs()
        return floor + (1 - floor) * (diff / tau).clamp(max=1)


class Transform:
    """Random smooth warp of a batch (FOMM): affine (std sigma_affine around the identity) plus
    thin-plate-spline bending (std sigma_tps at points_tps x points_tps control points).
    transform_frame(img): output pixel z shows img at warp_coordinates(z)."""

    def __init__(self, bs, device, sigma_affine=0.05, sigma_tps=0.005, points_tps=5):
        self.bs = bs
        self.theta = torch.eye(2, 3, device=device).unsqueeze(0) + sigma_affine * torch.randn(bs, 2, 3, device=device)
        self.tps = bool(sigma_tps and points_tps)
        if self.tps:
            self.control_points = make_coordinate_grid(points_tps, points_tps, device).view(1, -1, 2)
            self.control_params = sigma_tps * torch.randn(bs, 1, points_tps ** 2, device=device)

    def warp_coordinates(self, coords):
        """coords (bs, P, 2) -> warped (bs, P, 2)."""
        theta = self.theta.unsqueeze(1)
        out = (theta[:, :, :, :2] @ coords.unsqueeze(-1)).squeeze(-1) + theta[:, :, :, 2]
        if self.tps:
            dist = (coords.view(coords.shape[0], -1, 1, 2) - self.control_points.view(1, 1, -1, 2)).abs().sum(-1)
            out = out + ((dist ** 2) * torch.log(dist + 1e-6) * self.control_params).sum(2).unsqueeze(-1)
        return out

    def transform_frame(self, frame):
        n, _, h, w = frame.shape
        grid = make_coordinate_grid(h, w, frame.device).view(1, h * w, 2).expand(n, h * w, 2)
        grid = self.warp_coordinates(grid).view(n, h, w, 2)
        return F.grid_sample(frame.float(), grid, padding_mode='reflection', align_corners=True)

    def jacobian(self, coords):
        """d warp / d coords at coords (bs, K, 2) -> (bs, K, 2, 2). coords must require grad."""
        new = self.warp_coordinates(coords)
        gx = torch.autograd.grad(new[..., 0].sum(), coords, create_graph=True)[0]
        gy = torch.autograd.grad(new[..., 1].sum(), coords, create_graph=True)[0]
        return torch.stack([gx, gy], dim=-2)


class WarpLoss:
    """Reconstruction (multi-scale VGG perceptual, L1) + keypoint and Jacobian equivariance
    (+ optional BKinD separation). Weights come from the config (args)."""

    def __init__(self, args):
        self.args = args
        self.amp = bool(getattr(args, 'amp', False))
        self.w_perc = float(getattr(args, 'warp_perceptual_weight', 10.0))
        self.w_eq_value = float(getattr(args, 'warp_equivariance_value', 10.0))
        self.w_eq_jac = float(getattr(args, 'warp_equivariance_jacobian', 10.0))
        self.w_sep = float(getattr(args, 'warp_separation_weight', 0.0))
        self.sigma_affine = float(getattr(args, 'warp_sigma_affine', 0.05))
        self.sigma_tps = float(getattr(args, 'warp_sigma_tps', 0.005))
        self.points_tps = int(getattr(args, 'warp_points_tps', 5))
        # step 1b: weight the reconstruction loss by where things move (motion_weight); off = step 1
        self.motion = bool(getattr(args, 'warp_motion_weight', False))
        self.motion_sigma = float(getattr(args, 'warp_motion_sigma', 4.0))
        self.motion_tau = float(getattr(args, 'warp_motion_tau', 0.05))
        self.motion_floor = float(getattr(args, 'warp_motion_floor', 0.05))
        self.last_weight = None  # the latest weight map, for the sample pictures
        self.vgg = torch.nn.DataParallel(VGG_feat()).cuda(args.gpu).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.pyramid = nn.ModuleList([AntiAliasInterpolation2d(3, s) for s in PYRAMID_SCALES]).cuda(args.gpu)
        self.separation = separation_loss(args.nkpts).cuda(args.gpu)

    def make_transform(self, bs, device):
        return Transform(bs, device, self.sigma_affine, self.sigma_tps, self.points_tps)

    def perceptual(self, pred, target, weight=None):
        """weight (N, 1, H, W) or None: with a weight, each feature layer's error is a weighted mean
        over positions (the weight shrunk to the layer's size by area averaging), per image."""
        total = 0.0
        for down in self.pyramid:
            p, t = down(pred), down(target.float())
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                fp, ft = self.vgg(p), self.vgg(t)
            for a, b in zip(fp, ft):
                err = (a.float() - b.float().detach()).abs()
                if weight is None:
                    total = total + err.mean()
                else:
                    w = F.adaptive_avg_pool2d(weight, err.shape[-2:])
                    total = total + ((err.mean(1, keepdim=True) * w).sum((1, 2, 3)) / w.sum((1, 2, 3))).mean()
        return total

    def __call__(self, out, driving, transform, source=None):
        """Returns (total loss, dict of the parts for logging). source (frame A) is needed for the
        motion weight."""
        weight = None
        if self.motion:
            assert source is not None, 'warp_motion_weight needs frame A (source)'
            weight = motion_weight(source, driving, self.motion_sigma, self.motion_tau, self.motion_floor)
            self.last_weight = weight
        parts = {'perceptual': self.w_perc * self.perceptual(out['prediction'], driving, weight)}
        if transform is not None and 'kp_transformed' in out:
            kp_d, kp_t = out['kp_driving'], out['kp_transformed']
            # positions: warping the keypoints found in T(B) must land on the keypoints of B
            parts['equivariance_value'] = self.w_eq_value * (
                kp_d['value'] - transform.warp_coordinates(kp_t['value'])).abs().mean()
            if self.w_eq_jac:
                jac_t = transform.jacobian(kp_t['value']) @ kp_t['jacobian']
                rel = safe_inverse_2x2(kp_d['jacobian']) @ jac_t
                eye = torch.eye(2, device=rel.device).view(1, 1, 2, 2)
                parts['equivariance_jacobian'] = self.w_eq_jac * (eye - rel).abs().mean()
        if self.w_sep:
            for kp in (out['kp_source'], out['kp_driving']):
                parts['separation'] = parts.get('separation', 0.0) + self.w_sep * self.separation(
                    (kp['value'][..., 0], kp['value'][..., 1])).mean()
        return sum(parts.values()), parts
