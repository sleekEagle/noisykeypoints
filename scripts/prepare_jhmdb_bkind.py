"""Build the folder layout BKinD's `custom_dataset` loader expects from raw JHMDB.

Output:  <out>/split<k>/{train,val}/<action>__<clip>/image_XXXXX.jpg
  - train = official training clips of split k, val = official test clips.
  - Each clip is trimmed to the number of frames in its joint file (the frames
    zip has 1-2 extra trailing frames in most clips; joint frame t = image t+1).
  - Frames are hard links to the raw frames (no extra disk space). Never write
    into these files in place; noisy versions must be written as new files.
Also writes <out>/split<k>/clips.json listing each clip's action, split and
frame count, for the evaluation code.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import scipy.io as sio


def read_split(splits_dir, k):
    """Return {clip_name: 'train' | 'val'} for split k (1=train, 2=test)."""
    out = {}
    for f in sorted(os.listdir(splits_dir)):
        if not f.endswith(f'_test_split{k}.txt'):
            continue
        for line in open(os.path.join(splits_dir, f)):
            if line.strip():
                name, flag = line.split()
                out[name[:-4]] = {'1': 'train', '2': 'val'}[flag]
    return out


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jhmdb', default=r'D:\datasets\keypoint\JHMDB')
    p.add_argument('--out', default=r'D:\datasets\keypoint\JHMDB_bkind')
    p.add_argument('--split', type=int, default=1, choices=[1, 2, 3])
    p.add_argument('--workers', type=int, default=32, help='files copied at the same time')
    a = p.parse_args()

    split = read_split(os.path.join(a.jhmdb, 'splits'), a.split)
    root = os.path.join(a.out, f'split{a.split}')
    if os.path.exists(root):
        # A folder without clips.json is a half-finished (interrupted) build.
        done = os.path.exists(os.path.join(root, 'clips.json'))
        raise SystemExit(f'{root} already exists' + ('' if done else
                         ' but is INCOMPLETE (no clips.json)') + '; delete it first to rebuild.')

    def listdir(*p):  # skip Mac junk such as .AppleDouble
        return sorted(d for d in os.listdir(os.path.join(*p)) if not d.startswith('.'))

    # Copying is limited by waiting on storage (slow on Kaggle's /kaggle/input), not by the CPU,
    # so many files are copied at once in threads.
    pool = ThreadPoolExecutor(max_workers=a.workers)
    jobs = []
    meta = {}
    for action in listdir(a.jhmdb, 'joint_positions'):
        for clip in listdir(a.jhmdb, 'joint_positions', action):
            pos = sio.loadmat(os.path.join(a.jhmdb, 'joint_positions', action, clip,
                                           'joint_positions.mat'))['pos_img']
            n = pos.shape[2]
            src = os.path.join(a.jhmdb, 'frames', action, clip)
            frames = sorted(os.listdir(src))
            assert len(frames) >= n, (clip, len(frames), n)
            part = split[clip]
            dst = os.path.join(root, part, f'{action}__{clip}')
            os.makedirs(dst)
            for f in frames[:n]:
                jobs.append(pool.submit(link_or_copy, os.path.join(src, f), os.path.join(dst, f)))
            meta[f'{action}__{clip}'] = {'action': action, 'clip': clip,
                                         'split': part, 'nframes': n}
        print(f'{action}: queued', flush=True)

    for j in jobs:
        j.result()  # re-raises any copy error
    pool.shutdown()

    # clips.json is written last, so its presence means the layout is complete.
    json.dump(meta, open(os.path.join(root, 'clips.json'), 'w'), indent=1)
    for part in ('train', 'val'):
        c = [m for m in meta.values() if m['split'] == part]
        print(f'{part}: {len(c)} clips, {sum(m["nframes"] for m in c)} frames')


if __name__ == '__main__':
    main()
