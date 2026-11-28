"""Prepare EchoNet-Dynamic for keypoint training: grey frames with everything outside the ultrasound fan
blacked out (the orientation dot and depth marks sit outside it), plus landmark labels from the tracings.

Per video (all automatic; same fan finder as scripts/clean_echo.py): the video's colour map is learnt,
the fan = changing, on-map pixels -> largest region -> convex hull; inside the fan, off-colour blobs of
>= 5 px are inpainted (rare here); frames stay 112 x 112 (not cropped, so the tracings keep their
coordinates).

Output (keep private; EchoNet's research-use licence):
  <out>/videos/<name>.npz    frames uint8 (T, 112, 112) grey, black outside the fan; fan bool (112, 112)
  <out>/videos.json          per video: split, fps, frames, EF/ESV/EDV, fan_frac, overlay px, flags
  <out>/labels.npz           for the 2 traced frames of each video (ED = larger LV area, ES = smaller):
                             <name>/frames (2,) [ED, ES] frame index, <name>/landmarks (2, 3, 2) apex,
                             annulus side 1, annulus side 2 (x, y in px at 112), <name>/contour (2, 2N, 2)
                             LV outline (chord endpoints, one side down and the other back up),
                             <name>/area (2,) LV area in px^2.
  <out>/sheet_*.png          contact sheets (first traced frame with the fan outline and landmarks)

Tracing format (VolumeTracings.csv): per traced frame, the first line is the LV long axis; the others are
chords across the LV, whose endpoints form the outline. Apex = the long-axis end nearer the top (apical
view); annulus = the two outline points in the basal 15% of the long axis that lie farthest apart across
the axis diagonally (towards the base and to each side; the chords are perpendicular to the axis, but the annulus is often slanted, so no single chord
spans it).
Flags: 'size' (not 112 x 112; skipped), 'tracing' (not 21 lines per traced frame or not 2 traced frames;
no labels written), 'fan area' (fan area > 40% off the median), 'trace outside fan' (< 95% of contour
points inside the fan; checked: the tracing itself runs past the image edge, usually at the apex; the
mask cuts no tissue).

    python scripts/prepare_echonet.py --root /workspace/data/echonet_dynamic/EchoNet-Dynamic \
        --out /workspace/data/echonet_dynamic/prepared --workers 32
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_echo import colour_map, find_fan, project, read_video  # noqa: E402

A = None


def clean(frames):
    lut = colour_map(frames, A.std)
    grey, off = project(frames, lut)
    colour = off > A.colour
    fan = find_fan(grey, colour.mean(0) > 0.3, A.std)
    if fan is None:
        return None
    kernel = np.ones((2 * A.grow + 1, 2 * A.grow + 1), np.uint8)
    out, n_over = [], []
    for t in range(len(frames)):
        g = np.clip(np.round(grey[t]), 0, 255).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(colour[t].astype(np.uint8), connectivity=8)
        keep = np.zeros(n, np.uint8)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 5
        over = cv2.dilate(keep[lab], kernel) & fan
        if over.any():
            g = cv2.inpaint(g, over, 3, cv2.INPAINT_TELEA)
        out.append(np.where(fan, g, 0).astype(np.uint8))
        n_over.append(int(over.sum()))
    return np.stack(out), fan, np.array(n_over)


def labels(tr):
    """Tracing rows of one video -> dict or None if the tracing is not the standard 21 lines x 2 frames."""
    frames = sorted(tr.Frame.unique())
    if len(frames) != 2:
        return None
    res = []
    for f in frames:
        r = tr[tr.Frame == f][['X1', 'Y1', 'X2', 'Y2']].values
        if len(r) != 21:
            return None
        axis, chords = r[0].reshape(2, 2), r[1:]
        p1, p2 = chords[:, :2], chords[:, 2:]
        contour = np.concatenate([p1, p2[::-1]])
        x, y = contour[:, 0], contour[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        ia = int(np.argmin(axis[:, 1]))  # apical views: the apex is at the top of the fan
        apex, base = axis[ia], axis[1 - ia]
        u = (base - apex) / max(np.linalg.norm(base - apex), 1e-6)  # along the long axis, apex -> base
        n = np.array([-u[1], u[0]])  # across the axis
        along = (contour - apex) @ u
        near = contour[along >= 0.85 * along.max()]  # outline points in the basal 15%
        d = near - base
        # each corner = the point farthest out diagonally (towards the base and to that side)
        lm = np.stack([apex, near[np.argmax(d @ (u - n))], near[np.argmax(d @ (u + n))]])  # annulus corners
        res.append((f, lm, contour, area))
    res.sort(key=lambda z: -z[3])  # ED (larger area) first
    return dict(frames=np.array([z[0] for z in res]), landmarks=np.stack([z[1] for z in res]),
                contour=np.stack([z[2] for z in res]), area=np.array([z[3] for z in res]))


def work(job):
    name, row, tr = job
    rec = dict(split=row['Split'], fps=int(row['FPS']), frames=int(row['NumberOfFrames']),
               EF=float(row['EF']), ESV=float(row['ESV']), EDV=float(row['EDV']), flags=[])
    if (row['FrameHeight'], row['FrameWidth']) != (112, 112):
        rec['flags'].append('size')
        return name, rec, None, None
    frames = read_video(os.path.join(A.root, 'Videos', name + '.avi'))
    r = clean(frames)
    if r is None:
        rec['flags'].append('no fan')
        return name, rec, None, None
    vid, fan, n_over = r
    np.savez_compressed(os.path.join(A.out, 'videos', name + '.npz'), frames=vid, fan=fan)
    rec.update(frames=len(vid), fan_frac=float(fan.mean()), overlay_px_mean=float(n_over.mean()))
    lab = labels(tr) if tr is not None else None
    if lab is None:
        rec['flags'].append('tracing')
    else:
        pts = np.clip(np.round(lab['contour'].reshape(-1, 2)).astype(int), 0, 111)
        rec['trace_in_fan'] = float(fan[pts[:, 1], pts[:, 0]].mean())
        if rec['trace_in_fan'] < 0.95:
            rec['flags'].append('trace outside fan')
    thumb = None
    if lab is not None:
        im = cv2.cvtColor(vid[lab['frames'][0]], cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(cv2.resize(fan.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnts, -1, (0, 0, 255), 1)
        cv2.polylines(im, [np.round(lab['contour'][0] * 2).astype(np.int32)], True, (0, 255, 0), 1)
        for p, c in zip(lab['landmarks'][0], [(0, 255, 255), (255, 0, 255), (255, 0, 255)]):
            cv2.circle(im, tuple(int(v) for v in np.round(p * 2)), 3, c, -1)
        cv2.putText(im, name[:10], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        thumb = im
    return name, rec, lab, thumb


def init(a):
    global A
    A = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--std', type=float, default=4.0)
    ap.add_argument('--colour', type=int, default=20)
    ap.add_argument('--grow', type=int, default=2)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'videos'), exist_ok=True)
    fl = pd.read_csv(os.path.join(a.root, 'FileList.csv'))
    vt = pd.read_csv(os.path.join(a.root, 'VolumeTracings.csv'))
    vt['name'] = vt.FileName.str.replace('.avi', '', regex=False)
    groups = {n: g for n, g in vt.groupby('name')}
    rows = fl.head(a.limit) if a.limit else fl
    jobs = [(r.FileName, r._asdict(), groups.get(r.FileName)) for r in rows.itertuples(index=False)]
    report, labs, thumbs = {}, {}, []
    with Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        for i, (name, rec, lab, thumb) in enumerate(pool.imap(work, jobs, chunksize=8)):
            report[name] = rec
            if lab is not None:
                for k, v in lab.items():
                    labs[f'{name}/{k}'] = v
            if thumb is not None and len(thumbs) < 240 and i % 20 == 0:
                thumbs.append(thumb)
            if i % 500 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    med = np.median([r['fan_frac'] for r in report.values() if 'fan_frac' in r])
    for r in report.values():
        if 'fan_frac' in r and abs(r['fan_frac'] / med - 1) > 0.4:
            r['flags'].append('fan area')
    np.savez_compressed(os.path.join(a.out, 'labels.npz'), **labs)
    json.dump(report, open(os.path.join(a.out, 'videos.json'), 'w'), indent=1)
    for j in range(0, len(thumbs), 60):
        t = thumbs[j:j + 60] + [np.zeros_like(thumbs[0])] * (-len(thumbs[j:j + 60]) % 10)
        cv2.imwrite(os.path.join(a.out, f'sheet_{j // 60}.png'),
                    np.concatenate([np.concatenate(t[i:i + 10], 1) for i in range(0, len(t), 10)], 0))
    flags = {}
    for r in report.values():
        for f in r['flags']:
            flags[f] = flags.get(f, 0) + 1
    split = pd.Series([r['split'] for r in report.values() if not set(r['flags']) & {'size', 'no fan'}]).value_counts()
    print(f'{len(report)} videos; usable by split {split.to_dict()}; flags {flags}; '
          f'labelled {sum(1 for k in labs if k.endswith("/frames"))}; fan median {med:.3f}', flush=True)


if __name__ == '__main__':
    main()
