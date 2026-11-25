from __future__ import print_function, absolute_import

import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
import torchvision

import torchvision.transforms.functional as TF

from dataloader.data_utils import *
from dataloader.noise import apply_noise_pair
from dataloader import background as bgsub
from dataloader.tracks import TrackStore


def generate_pair_images(root, gap=6, win_gap = 7):
    images = []

    root = os.path.expanduser(root)

    no_cable_list = ['mouse001', 'mouse002', 'mouse051', 'mouse052','mouse055',
                    'mouse056', 'mouse058', 'mouse053', 'mouse054', 'mouse057']

    for video in sorted(os.listdir(root)):

        if video[:8] in no_cable_list:
            img_root = os.path.join(root, video)
            # noisykeypoints: pair frames by the frame number in the file name (image_XXXXXXXX.png)
            # instead of by position in the folder listing, so a folder may hold only the frames
            # that are used (scripts/prepare_calms21_bkind.py). Same pairs when all frames exist.
            frames = {int(os.path.splitext(f)[0].split('_')[-1]): f
                      for f in os.listdir(img_root) if f.startswith('image_')}
            for index in range(0, max(frames) + 1 - gap, win_gap):
                if index not in frames or index + gap not in frames:
                    continue
                im0 = os.path.join(img_root, frames[index])
                im1 = os.path.join(img_root, frames[index+gap])

                item = [im0, im1]

                images.append(item)

    return images


class MouseDataset(data.Dataset):
    """
    DataLoader for CalMS21 dataset
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, image_size=[128, 128], rotation=True,
                 simplified=False, crop_box=True, noise=None, motion_mask=None, tracks=None, temporal=None):

        samples = generate_pair_images(root)
        self.noise = noise  # noisykeypoints: see dataloader/noise.py (None = clean)
        # noisykeypoints: motion_mask = dict(sigma=2, tau=0.1, step=60, window=600, seg=300) or None.
        # The background of each stretch of video is the per-pixel median of nearby (noisy) frames; the
        # loss mask of a pair is where frame A or B differs from it (dataloader/background.py).
        # None = BKinD's all-ones mask.
        self.motion_mask = dict(motion_mask) if motion_mask else None
        self.backgrounds = None
        if self.motion_mask:
            m = self.motion_mask
            self.backgrounds = bgsub.background_cache(root, noise, int(m.get('step', 60)),
                                                      int(m.get('window', 600)), int(m.get('seg', 300)))
            print(f'motion mask: backgrounds of {len(self.backgrounds)} videos ready', flush=True)

        # noisykeypoints: tracks = dict(dir=..., max_points=256) or None. Precomputed CoTracker3 tracks of
        # the moving points of each pair (dataloader/tracks.py), for the warp model's track loss.
        self.tracks = None
        if tracks:
            self.tracks = TrackStore(tracks['dir'], noise, tracks.get('max_points', 256))
            print(f'tracks: {self.tracks.dir}', flush=True)

        # noisykeypoints: temporal = dict(k=3, frames=<every-frame folder>) or None. Also returns the mean of
        # the k frames centred on A and on B (each noised as in training), for the change target only
        # (loss/change_target.py); the network still sees single frames. Neighbours past the video's ends
        # are left out.
        self.temporal = dict(temporal) if temporal else None

        self.root = root
        self.loader = loader

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.crop_box = crop_box

        # Parameters for transformation
        self._image_size = image_size

        # Rotation loss
        self.rotation = rotation
        if rotation:
            self.transform_rot = RandomRotation(180)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        img_path0, img_path1 = self.samples[index]

        im0 = self.loader(img_path0)
        im1 = self.loader(img_path1)

        height, width = self._image_size[:2]

        image0 = torchvision.transforms.Resize((height, width))(im0)
        image1 = torchvision.transforms.Resize((height, width))(im1)

        # noisykeypoints: corrupt both frames after resizing, before the rotations and transforms.
        # Done as a pair so that H.264 can encode A and B together (A = I-frame, B = P-frame).
        image0, image1 = apply_noise_pair([image0, image1], self.noise, [img_path0, img_path1])

        if self.backgrounds is not None:  # noisykeypoints: where A or B differs from the background
            bg = bgsub.background_for(self.backgrounds[os.path.basename(os.path.dirname(img_path0))],
                                      bgsub.frame_no(img_path0))
            sigma, tau = float(self.motion_mask.get('sigma', 2.0)), float(self.motion_mask.get('tau', 0.1))
            masks = [bgsub.foreground(bgsub.to_gray(im), bg, sigma, tau) for im in (image0, image1)]
            motion = torch.from_numpy(np.maximum(*masks)).float().unsqueeze(0)

        # Create 3 rotations
        deg = 90
        rot_image1 = TF.rotate(image1, deg)
        rot_image1 = self.target_transform(rot_image1)

        deg = 180
        rot_image2 = TF.rotate(image1, deg)
        rot_image2 = self.target_transform(rot_image2)

        deg = -90
        rot_image3 = TF.rotate(image1, deg)
        rot_image3 = self.target_transform(rot_image3)

        if self.transform is not None:
            image0 = self.transform(image0)
        if self.target_transform is not None:
            image1 = self.target_transform(image1)

        mask = torch.ones((1, height, width))
        loss_mask = motion if self.backgrounds is not None else mask  # noisykeypoints

        out = (image0, image1, loss_mask, mask, rot_image1, rot_image2, rot_image3, img_path0, img_path1)
        if self.tracks is not None:  # noisykeypoints: (max_points, 5) = A xy, B xy in [-1, 1], valid
            out = out + (self.tracks.get(os.path.basename(os.path.dirname(img_path0)), bgsub.frame_no(img_path0)),)
        if self.temporal is not None:  # noisykeypoints: last two items = k-frame means around A and B
            out = out + tuple(self._frame_mean(p, height, width) for p in (img_path0, img_path1))
        return out

    def _frame_mean(self, path, height, width):
        """noisykeypoints: mean of the k noisy frames centred on `path`, normalised like the frames."""
        video, f = os.path.basename(os.path.dirname(path)), bgsub.frame_no(path)
        r = int(self.temporal.get('k', 3)) // 2
        paths = [os.path.join(self.temporal['frames'], video, f'image_{g:08d}.png') for g in range(f - r, f + r + 1)]
        paths = [p for p in paths if os.path.exists(p)]
        ims = [torchvision.transforms.Resize((height, width))(self.loader(p)) for p in paths]
        ims = apply_noise_pair(ims, self.noise, paths)
        mean = np.mean([np.asarray(im, np.float32) for im in ims], 0) / 255.0
        x = torch.from_numpy(mean).permute(2, 0, 1)
        return TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
