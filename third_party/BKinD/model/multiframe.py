"""noisykeypoints: let the keypoint network see several frames (config `input_frames: {k, step, frames}`).

The encoder's first conv takes 3*k channels: k frames stacked oldest first, the current frame last.
Only *earlier* frames are added (t - step*(k-1), ..., t - step, t), so frame A's stack holds nothing
from after A, and the decoder (which takes its appearance from A's encoder features) cannot see frame B.
The extra input channels start at zero, so at the start the network is exactly single-frame BKinD.
"""
import torch
import torch.nn as nn


def inflate_input(encoder, n_frames):
    if n_frames <= 1:
        return
    old = encoder.conv1
    new = nn.Conv2d(3 * n_frames, old.out_channels, old.kernel_size, old.stride, old.padding,
                    bias=old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, -3:] = old.weight
        if old.bias is not None:
            new.bias.copy_(old.bias)
    encoder.conv1 = new


def n_input_frames(args):
    cfg = getattr(args, 'input_frames', None)
    return int(cfg.get('k', 1)) if cfg else 1
