"""Extract CalMS21 Task 1 video frames into the folder layout BKinD's mouse loader expects.

Output:  <out>/<split>/<video>/image_XXXXXXXX.png   (XXXXXXXX = frame number in the video)
  - Frames are read straight out of task1_videos_seq.zip (no need to unzip 74 GB). Each .seq
    video stores every frame as a grayscale JPEG (1024x570).
  - Frames are shrunk to size x size (default 256x256, squashed like BKinD's loader does, with
    the same PIL bilinear resize), because noise is added after shrinking. Saved as PNG, never
    JPEG, so later noise is not altered by compression.
  - By default only the frames BKinD trains on are saved: frame i and i+gap for i = 0, stride,
    2*stride, ... (gap 6, stride 7 as in BKinD's mouse_dataset.py). --all-frames saves every frame.
  - Default videos: the 10 train videos without the head cable that BKinD trains on.
Also writes, per split:
  - keypoints.npz: MARS keypoints (T, 2 mice, 2 xy, 7 parts) in pixels of the 1024x570 frames,
    scores (T, 2, 7) and behaviour labels (T,) for each extracted video.
  - videos.json, written LAST: per-video frame counts (video vs keypoint file). A split folder
    without videos.json is an incomplete build.
"""
import argparse
import io
import json
import os
import struct
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

NO_CABLE = ['mouse001', 'mouse002', 'mouse051', 'mouse052', 'mouse053',
            'mouse054', 'mouse055', 'mouse056', 'mouse057', 'mouse058']
BEHAVIOURS = {'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}


def read_seq_header(f):
    """Parse the 1024-byte Norpix .seq header (layout from CalMS21's data_reader.py)."""
    h = f.read(1024)
    assert struct.unpack('<i', h[:4])[0] == 0xFEED, 'not a .seq file'
    w, hgt, bits, _, _, fmt, nframes, _, true_size = struct.unpack('<5Ii3I', h[548:584])
    fps = struct.unpack('<d', h[584:592])[0]
    assert fmt in (101, 102), f'expected grayscale JPEG frames, got format {fmt}'
    ts = int(true_size - bits // 8 * w * hgt)  # bytes of timestamp after each frame
    return dict(width=w, height=hgt, nframes=nframes, fps=fps, ts=ts)


def find_timestamp_bytes(zpath, member, header_ts):
    """The header's timestamp length is sometimes wrong (e.g. 512). Find the one for which
    frame 2 starts right after frame 1 (CalMS21's own reader falls back to 16)."""
    with zipfile.ZipFile(zpath) as z, z.open(member) as f:
        f.read(1024)
        size = struct.unpack('<i', f.read(4))[0]
        first = f.read(size - 4 + 1024)
    for ts in dict.fromkeys([header_ts, 16, 10, 8]):
        nxt = first[size - 4 + ts:]
        if len(nxt) >= 6 and 0 < struct.unpack('<i', nxt[:4])[0] < 10**7 and nxt[4:6] == b'\xff\xd8':
            return ts
    raise ValueError(f'{member}: could not find the timestamp length')


def iter_seq_jpegs(zpath, member, ts):
    """Yield the JPEG bytes of every frame, reading the zipped video front to back."""
    with zipfile.ZipFile(zpath) as z, z.open(member) as f:
        f.read(1024)
        while True:
            s = f.read(4)
            if len(s) < 4:
                return
            size = struct.unpack('<i', s)[0]  # counts these 4 bytes too
            if not 4 < size < 10**7:
                raise ValueError(f'implausible frame size {size}')
            data = f.read(size - 4)
            f.read(ts)
            if data[:2] != b'\xff\xd8':
                raise ValueError('frame does not start with a JPEG marker (wrong timestamp length?)')
            yield data


def extract_video(job):
    zpath, member, out_dir, size, gap, stride, all_frames, limit = job
    with zipfile.ZipFile(zpath) as z, z.open(member) as f:
        hdr = read_seq_header(f)
    n = hdr['nframes'] if limit is None else min(limit, hdr['nframes'])
    if all_frames:
        keep = set(range(n))
    else:
        keep = {j for i in range(0, n - gap, stride) for j in (i, i + gap)}
    os.makedirs(out_dir, exist_ok=True)

    ts = find_timestamp_bytes(zpath, member, hdr['ts'])
    count, saved = 0, 0
    for i, jpg in enumerate(iter_seq_jpegs(zpath, member, ts)):
        count = i + 1
        if i in keep:
            im = Image.open(io.BytesIO(jpg)).convert('L')
            assert im.size == (hdr['width'], hdr['height'])
            im.resize((size, size), Image.BILINEAR).save(os.path.join(out_dir, f'image_{i:08d}.png'))
            saved += 1
        if limit is not None and count >= limit:
            break
    return dict(video=os.path.basename(out_dir), seq_frames=count, header_frames=hdr['nframes'],
                fps=hdr['fps'], saved_frames=saved, timestamp_bytes=ts)


def load_mars(json_zip, split, videos):
    """Return {video: (keypoints, scores, labels)} from calms21_task1_<split>.json."""
    name = f'task1_classic_classification/calms21_task1_{split}.json'
    t = time.time()
    with zipfile.ZipFile(json_zip) as z:
        d = json.load(z.open(name))
    out = {}
    for ann in d.values():
        assert ann and all(v['metadata']['vocab'] == BEHAVIOURS for v in ann.values())
        for key, v in ann.items():
            vid = key.split('/')[-1]
            if vid in videos:
                out[vid] = (np.asarray(v['keypoints'], np.float32), np.asarray(v['scores'], np.float32),
                            np.asarray(v['annotations'], np.int8))
    del d
    print(f'loaded MARS keypoints for {len(out)} videos in {time.time() - t:.0f} s')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calms21', default=r'D:\datasets\keypoint\CalMS21')
    p.add_argument('--out', default=r'D:\datasets\keypoint\CalMS21_bkind')
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--videos', nargs='*', help='video name prefixes, e.g. mouse001 (default: the 10 '
                   'no-cable train videos; "all" = every video of the split)')
    p.add_argument('--size', type=int, default=256, help='frames are resized to size x size')
    p.add_argument('--gap', type=int, default=6, help='frame gap of a training pair (BKinD: 6)')
    p.add_argument('--stride', type=int, default=7, help='a pair starts every stride frames (BKinD: 7)')
    p.add_argument('--all-frames', action='store_true', help='save every frame, not only pair frames')
    p.add_argument('--limit', type=int, help='only the first N frames per video (quick test)')
    p.add_argument('--workers', type=int, default=min(10, os.cpu_count()))
    a = p.parse_args()

    root = os.path.join(a.out, a.split)
    if os.path.exists(root):
        done = os.path.exists(os.path.join(root, 'videos.json'))
        raise SystemExit(f'{root} already exists' + ('' if done else
                         ' but is INCOMPLETE (no videos.json)') + '; delete it first to rebuild.')

    zpath = os.path.join(a.calms21, 'task1_videos_seq.zip')
    with zipfile.ZipFile(zpath) as z:
        members = sorted(n for n in z.namelist() if n.endswith('.seq') and f'/{a.split}/' in n)
    prefixes = NO_CABLE if a.videos is None else a.videos
    if prefixes != ['all']:
        members = [m for m in members if os.path.basename(m).startswith(tuple(prefixes))]
    videos = [os.path.basename(m)[:-4] for m in members]
    assert videos, 'no matching videos'
    print(f'{len(videos)} {a.split} videos: {", ".join(v[:8] for v in videos)}')

    mars = load_mars(os.path.join(a.calms21, 'task1_classic_classification.zip'), a.split, set(videos))
    missing = sorted(set(videos) - set(mars))
    assert not missing, f'no MARS keypoints for {missing}'

    os.makedirs(root)
    t = time.time()
    jobs = [(zpath, m, os.path.join(root, v), a.size, a.gap, a.stride, a.all_frames, a.limit)
            for m, v in zip(members, videos)]
    info = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(extract_video, jobs):
            kp = mars[r['video']][0]
            r['keypoint_frames'] = int(len(kp))
            info[r['video']] = r
            flag = '' if a.limit or r['seq_frames'] == len(kp) == r['header_frames'] \
                else '   <-- FRAME COUNTS DIFFER'
            print(f"{r['video']}: video {r['seq_frames']} frames (header {r['header_frames']}), "
                  f"keypoints {len(kp)}, saved {r['saved_frames']}  [{time.time() - t:.0f} s]{flag}",
                  flush=True)

    np.savez_compressed(os.path.join(root, 'keypoints.npz'),
                        **{f'{v}/{k}': x for v, xs in mars.items()
                           for k, x in zip(['keypoints', 'scores', 'labels'], xs)})
    meta = dict(source='CalMS21 task 1', split=a.split, size=a.size, gap=a.gap, stride=a.stride,
                all_frames=a.all_frames, limit=a.limit, original_size=[1024, 570],
                keypoint_order=['nose', 'left ear', 'right ear', 'neck', 'left hip', 'right hip',
                                'tail base'],
                mice=['resident (black)', 'intruder (white)'], behaviours=BEHAVIOURS, videos=info)
    json.dump(meta, open(os.path.join(root, 'videos.json'), 'w'), indent=1)  # last = build complete
    bad = [v for v, r in info.items() if a.limit is None and r['seq_frames'] != r['keypoint_frames']]
    print(f'done in {time.time() - t:.0f} s; {sum(r["saved_frames"] for r in info.values())} frames in {root}')
    if bad:
        print('WARNING: video and keypoint frame counts differ for', bad)


if __name__ == '__main__':
    main()
