"""noisykeypoints: keypoint-driven warping model, step 1 of our method (reports/warp_method_design.md).

BKinD's keypoint network (ResNet-50 encoder + globalNet heatmaps + soft argmax) is kept unchanged, so
keypoints come out the same way and get_keypoints() works with the existing scoring code. New:
  - a Jacobian head: 4 maps per keypoint, read out under the keypoint's heatmap (FOMM),
  - FOMM's dense motion network (model/dense_motion.py) that blends the keypoints' local motions,
  - a move-only decoder: the predicted target frame is the source frame resampled along the flow.
    Nothing is painted: no generator, no occlusion in-painting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_updated import resnetbank50all as resnetbank50
from .globalNet import globalNet
from .dense_motion import DenseMotionNetwork


class WarpModel(nn.Module):
    def __init__(self, n_kps=10, pretrained=True, output_shape=(64, 64), kp_variance=0.01,
                 motion_scale=0.25, block_expansion=64, num_blocks=5, max_features=1024):
        super().__init__()
        self.K = n_kps
        self.output_shape = output_shape
        self.kptNet = globalNet([2048, 1024, 512, 256], output_shape, n_kps)
        # Same initialisation as BKinD's Model for the keypoint head (before the encoder is built, so
        # the ImageNet weights of the encoder are untouched).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Jacobian head on globalNet's last feature map (256 channels). Zero weights and an identity
        # bias: every Jacobian starts as "no rotation, no stretch".
        self.jacobian = nn.Conv2d(256, 4 * n_kps, kernel_size=3, padding=1)
        nn.init.zeros_(self.jacobian.weight)
        self.jacobian.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 1.0]).repeat(n_kps))

        self.dense_motion = DenseMotionNetwork(n_kps, 3, block_expansion, num_blocks, max_features,
                                               scale_factor=motion_scale, kp_variance=kp_variance)
        self.encoder = resnetbank50(pretrained=pretrained)

    def detect(self, x):
        """Keypoints of a batch of frames: dict with 'value' (N, K, 2) as (x, y) in [-1, 1],
        'jacobian' (N, K, 2, 2), 'heatmap' (N, K, H, W) (softmax) and 'confidence' (N, K)."""
        fms, outs = self.kptNet(self.encoder(x))
        logits = outs[-1]
        n, k, h, w = logits.shape
        heat = F.softmax(logits.float().view(n, k, h * w), dim=2).view(n, k, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device)
        ys = torch.linspace(-1.0, 1.0, h, device=x.device)
        u_x = (heat.sum(2) * xs).sum(2)
        u_y = (heat.sum(3) * ys).sum(2)
        jmaps = self.jacobian(fms[-1]).float()
        if jmaps.shape[-2:] != (h, w):
            jmaps = F.interpolate(jmaps, size=(h, w), mode='bilinear', align_corners=True)
        jac = (heat.unsqueeze(2) * jmaps.view(n, k, 4, h, w)).sum((-1, -2)).view(n, k, 2, 2)
        return {'value': torch.stack([u_x, u_y], dim=-1), 'jacobian': jac, 'heatmap': heat,
                'confidence': heat.amax(dim=(2, 3))}

    def get_keypoints(self, x):
        kp = self.detect(x)['value']
        return kp[..., 0], kp[..., 1]

    def forward(self, source, driving, transformed=None):
        """Predict `driving` (frame B) by moving the pixels of `source` (frame A).
        `transformed` = a randomly warped copy of `driving`, for the equivariance loss."""
        kp_s = self.detect(source)
        kp_d = self.detect(driving)
        motion = self.dense_motion(source, kp_d, kp_s)
        with torch.autocast('cuda', enabled=False):
            flow = motion['deformation'].permute(0, 3, 1, 2)  # (N, 2, h, w)
            flow = F.interpolate(flow, size=source.shape[-2:], mode='bilinear', align_corners=True)
            pred = F.grid_sample(source.float(), flow.permute(0, 2, 3, 1), align_corners=True,
                                 padding_mode='border')
        out = {'prediction': pred, 'kp_source': kp_s, 'kp_driving': kp_d,
               'deformation': motion['deformation'], 'mask': motion['mask']}
        if transformed is not None:
            out['kp_transformed'] = self.detect(transformed)
        return out
