"""noisykeypoints: soft penalty for keypoints that sit off the moving foreground.

The foreground mask comes from background subtraction (dataloader/background.py, config `motion_mask`):
where frame A or B differs from the per-pixel median of nearby frames. The penalty is the share of each
keypoint's heatmap (softmax over pixels, sums to 1) that lies on background, averaged over keypoints and
over the heatmaps of A and B:

    penalty = mean_k sum_p heatmap_k(p) * (1 - fg(p))          in [0, 1]

It uses the whole heatmap, not only the keypoint position, so a spread-out "dead" keypoint still gets a
gradient that moves its mass onto foreground pixels (the mask is flat 0 far from the mice, so a penalty
read at the keypoint position alone would give no gradient there).
"""
import torch.nn.functional as F


def fg_penalty(heatmaps, fg):
    """heatmaps: list of (B, K, h, w) softmax heatmaps; fg: (B, 1, H, W) foreground mask in [0, 1].
    Returns the (B,) penalty."""
    out = 0
    for hm in heatmaps:
        hm = hm.float()
        bg = 1 - F.adaptive_avg_pool2d(fg.float(), hm.shape[-2:]).clamp(0, 1)      # (B, 1, h, w)
        out = out + (hm * bg).sum((2, 3)).mean(1)
    return out / len(heatmaps)
