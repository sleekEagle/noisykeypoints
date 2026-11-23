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
        self.vgg = torch.nn.DataParallel(VGG_feat()).cuda(args.gpu).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)
        self.pyramid = nn.ModuleList([AntiAliasInterpolation2d(3, s) for s in PYRAMID_SCALES]).cuda(args.gpu)
        self.separation = separation_loss(args.nkpts).cuda(args.gpu)

    def make_transform(self, bs, device):
        return Transform(bs, device, self.sigma_affine, self.sigma_tps, self.points_tps)

    def perceptual(self, pred, target):
        total = 0.0
        for down in self.pyramid:
            p, t = down(pred), down(target.float())
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                fp, ft = self.vgg(p), self.vgg(t)
            for a, b in zip(fp, ft):
                total = total + (a.float() - b.float().detach()).abs().mean()
        return total

    def __call__(self, out, driving, transform):
        """Returns (total loss, dict of the parts for logging)."""
        parts = {'perceptual': self.w_perc * self.perceptual(out['prediction'], driving)}
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
