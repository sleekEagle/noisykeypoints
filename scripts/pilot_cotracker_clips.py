"""Pick short every-frame clips of CalMS21 training videos for the CoTracker3 pilot.

Each video is cut into `--chunks` equal parts. In each part, the `--length`-frame window in which the
mice move most (MARS keypoints: sum over frames of the median step of the 14 keypoints, each step
capped at 30 px so tracker glitches don't win) becomes one clip.
Writes <out>/clips.json and <out>/files.txt (frame paths relative to --data, for tar -T).

    python scripts/pilot_cotracker_clips.py --data D:/datasets/keypoint/CalMS21_bkind_allframes/train \
        --kp D:/datasets/keypoint/CalMS21_bkind/train/keypoints.npz --out results/cotracker_pilot
"""
import argparse
import json
import os

import numpy as np

VIDEOS = ['mouse001_task1_annotator1', 'mouse052_task1_annotator1', 'mouse055_task1_annotator1',
          'mouse058_task1_annotator1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--kp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--videos', nargs='+', default=VIDEOS)
    ap.add_argument('--chunks', type=int, default=6)
    ap.add_argument('--length', type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kp = np.load(a.kp)
    clips, files = [], []
    for v in a.videos:
        k = kp[f'{v}/keypoints']  # (T, mouse, xy, 7) in 1024x570 px
        n = len(k)
        step = np.linalg.norm(k[1:] - k[:-1], axis=2).reshape(n - 1, -1)
        path = np.minimum(np.median(step, axis=1), 30.0)
        edges = np.linspace(0, n - a.length, a.chunks + 1).astype(int)
        for c in range(a.chunks):
            starts = np.arange(edges[c], edges[c + 1], 8)
            # robust to MARS glitches: per frame, median step over the 14 keypoints, capped at 30 px
            motion = [float(path[s:s + a.length - 1].sum()) for s in starts]
            s = int(starts[int(np.argmax(motion))])
            clips.append(dict(video=v, start=s, length=a.length, mars_motion_px=max(motion)))
            files += [f'{v}/image_{f:08d}.png' for f in range(s, s + a.length)]
    missing = [f for f in files if not os.path.exists(os.path.join(a.data, f))]
    assert not missing, missing[:5]
    json.dump(clips, open(os.path.join(a.out, 'clips.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'files.txt'), 'w', newline='\n').write('\n'.join(files) + '\n')
    for c in clips:
        print(c)
    print(len(clips), 'clips,', len(files), 'frames')


if __name__ == '__main__':
    main()
