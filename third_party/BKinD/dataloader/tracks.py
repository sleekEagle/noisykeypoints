"""noisykeypoints: precomputed CoTracker3 point tracks for the warp model's track loss (step 1d).

scripts/precompute_tracks.py tracks a grid of points through every-frame CalMS21 video (clean or with the
run's noise, exactly as the loader corrupts the frames) and saves, for every training pair (frame i, i+6),
the points that move: their positions in frame A and in frame B, in pixels at 256 px (pixel j's centre
at coordinate j). Files: <root>/<noise key>/<video>.npz with
    frames  (P,)    pair start frame i
    offsets (P+1,)  points of pair p are rows offsets[p]:offsets[p+1]
    a, b    (M, 2)  float16 (x, y) in frame A / frame B

TrackStore returns, for a pair, a (max_points, 5) float tensor: x_A, y_A, x_B, y_B in [-1, 1]
(align_corners=True, like the model's coordinates) and a 0/1 valid flag; a random subset if there are
more points than max_points, zero rows (valid 0) when there are fewer.
"""
import os

import numpy as np
import torch

SIZE = 256


def tracks_key(noise):
    """Folder name of a noise setting: 'clean' or e.g. 'photons1900.0_read0.000315..._seed0_typepoisson_gaussian'
    (same rule as dataloader/background.py, without the background settings)."""
    return 'clean' if not noise else '_'.join(f'{k}{v}' for k, v in sorted(noise.items()))


class TrackStore:
    def __init__(self, root, noise=None, max_points=256):
        self.dir = os.path.join(root, tracks_key(noise))
        if not os.path.isdir(self.dir):
            raise FileNotFoundError(f'no precomputed tracks in {self.dir} (scripts/precompute_tracks.py)')
        self.max_points = int(max_points)
        self.videos = {}

    def _video(self, video):
        if video not in self.videos:
            path = os.path.join(self.dir, video + '.npz')
            if not os.path.exists(path):
                raise FileNotFoundError(f'no tracks for {video}: {path}')
            z = np.load(path)
            self.videos[video] = dict(index={int(f): p for p, f in enumerate(z['frames'])},
                                      offsets=z['offsets'], a=z['a'].astype(np.float32),
                                      b=z['b'].astype(np.float32))
        return self.videos[video]

    def get(self, video, frame):
        v = self._video(video)
        out = torch.zeros(self.max_points, 5)
        p = v['index'].get(int(frame))
        if p is None:
            return out
        s, e = int(v['offsets'][p]), int(v['offsets'][p + 1])
        idx = np.arange(s, e)
        if len(idx) > self.max_points:
            idx = np.random.choice(idx, self.max_points, replace=False)
        n = len(idx)
        if n:
            pts = np.concatenate([v['a'][idx], v['b'][idx]], 1) * (2.0 / (SIZE - 1)) - 1.0
            out[:n, :4] = torch.from_numpy(pts)
            out[:n, 4] = 1.0
        return out
