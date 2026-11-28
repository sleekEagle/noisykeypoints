"""noisykeypoints: EchoNet-Dynamic loader for BKinD (frame pairs from the prepared videos).

The data folder is the output of scripts/prepare_echonet.py: videos/<name>.npz (frames uint8 (T, 112, 112),
grey, black outside the fan) and videos.json (split, fps, flags). Videos flagged 'size', 'no fan' or
'fan area' are left out.

Pairs: frame t with t + gap, gap = round(frame_gap_s * fps) (0.1 s = 5 frames at 50 fps); a pair starts
every `pair_stride` frames. Frames are resized to image_size (112 -> 128, bilinear, as BKinD's loaders),
then noised (dataloader/noise.py; None = clean), then BKinD's rotations and transforms as in
mouse_dataset.py. Noise is seeded from a virtual path <root>/<video>/image_<t:08d>.png, so each frame is
noised the same way in every epoch.
"""
import json
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image

from dataloader.data_utils import RandomRotation
from dataloader.noise import apply_noise_pair

EXCLUDE = {'size', 'no fan', 'fan area'}


def echo_pairs(root, split, frame_gap_s=0.1, pair_stride=10):
    """[(video, t, t + gap)] for the usable videos of `split` (TRAIN / VAL / TEST)."""
    info = json.load(open(os.path.join(root, 'videos.json')))
    pairs = []
    for name, r in sorted(info.items()):
        if r['split'] != split or set(r['flags']) & EXCLUDE:
            continue
        gap = max(1, int(round(frame_gap_s * r['fps'])))
        pairs += [(name, t, t + gap) for t in range(0, r['frames'] - gap, pair_stride)]
    return pairs


class EchoDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, image_size=(128, 128), split='TRAIN',
                 frame_gap_s=0.1, pair_stride=10, noise=None, rotation=True):
        self.root, self.split = root, split
        self.samples = echo_pairs(root, split, frame_gap_s, pair_stride)
        self.transform, self.target_transform = transform, target_transform
        self._image_size = image_size
        self.noise = noise
        self.rotation = rotation
        if rotation:
            self.transform_rot = RandomRotation(180)
        print(f'EchoNet {split}: {len(self.samples)} pairs from {len({s[0] for s in self.samples})} videos '
              f'(gap {frame_gap_s} s, a pair every {pair_stride} frames)', flush=True)

    def _path(self, name, t):  # virtual frame path: the noise seed and the sample names use it
        return os.path.join(self.root, name, f'image_{t:08d}.png')

    def __getitem__(self, index):
        name, t0, t1 = self.samples[index]
        frames = np.load(os.path.join(self.root, 'videos', name + '.npz'))['frames']
        height, width = self._image_size[:2]
        paths = [self._path(name, t0), self._path(name, t1)]
        ims = [Image.fromarray(frames[t]).convert('RGB').resize((width, height), Image.BILINEAR) for t in (t0, t1)]
        image0, image1 = apply_noise_pair(ims, self.noise, paths)

        rot_image1 = self.target_transform(TF.rotate(image1, 90))
        rot_image2 = self.target_transform(TF.rotate(image1, 180))
        rot_image3 = self.target_transform(TF.rotate(image1, -90))
        image0 = self.transform(image0)
        image1 = self.target_transform(image1)

        mask = torch.ones((1, height, width))
        return image0, image1, mask, mask, rot_image1, rot_image2, rot_image3, paths[0], paths[1]

    def __len__(self):
        return len(self.samples)
