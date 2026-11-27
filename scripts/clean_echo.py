"""Remove the burned-in overlays from the HMC-QU echo videos and crop them to the ultrasound fan.

Per video (all automatic, no hand work):
  1. fan     pixels whose intensity changes over the video (std over frames > --std) and that are coloured
             in at most 30% of frames; opened, largest connected region, its convex hull (the fan is convex, so the hull
             fills dark blood pools inside it and drops thin stray bits such as the ECG cursor line).
  0. colours the video's colour map, learnt as brightness -> median colour of the fan pixels (grey in most
             videos, blue -> cyan -> white in some). Intensity = mean of the channels; "coloured" = far
             (> --colour) from the map's colour at that brightness. tint_bgr in the report = colour at 200.
  2. overlay inside the fan, every pixel that is coloured in a frame (the ECG trace, its cursor and the
             yellow depth marks are off the colour map), widened by --grow px, is filled
             from the surrounding tissue (OpenCV Telea inpainting), frame by frame.
  3. output  grey frames, black outside the fan, cropped to the fan's bounding box, as uint8
             <out>/<view>/<video>.npz: frames (T, h, w), fan (h, w), box (x0, y0, x1, y1) in the original
             frame, size (W, H) of the original frame, overlay (T,) inpainted pixels per frame.

Check: <out>/clean_echo.json (per video: size, frames, box, fan area, overlay pixels, outlier flags, and for
the 109 A4C videos with a heart-wall mask the share of mask pixels inside the fan, mask stretched back to
the frame size), <out>/sheets/fan_*.png (frame 1 with the fan outline, 40 videos per sheet) and
<out>/sheets/clean_*.png (the cleaned crops).

These are clean (not noised) derived copies; keep them private (licence not checked).

    python scripts/clean_echo.py --root "D:/datasets/keypoint/HMC-QU Dataset-Kaggle v1.0/HMC-QU Dataset-Kaggle v1.0" \
        --out D:/datasets/keypoint/HMC-QU_clean
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import scipy.io

VIEWS = {'A4C': 'HMC-QU/A4C', 'A2C': 'HMC-QU/A2C', 'A2C_TAU': 'HMC-QU-TAU/2CH_Extension'}


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return np.stack(frames)  # (T, H, W, 3) BGR uint8


def colour_map(frames, std_thresh):
    """The video's colour map as a lookup table: for each brightness (mean of the 3 channels, 0-255), the
    median colour of the changing (fan) pixels with that brightness. Grey videos give grey; some HMC-QU
    videos draw the ultrasound in blue -> cyan -> white. Empty brightness bins are interpolated."""
    f = frames.astype(np.float32)
    inten = f.mean(-1)
    sel = np.broadcast_to((inten.std(0) > std_thresh)[None], inten.shape)
    b = np.clip(inten[sel].astype(int), 0, 255)
    cols = f[sel]
    lut = np.full((256, 3), np.nan, np.float32)
    order = np.argsort(b, kind='stable')
    b, cols = b[order], cols[order]
    edges = np.searchsorted(b, np.arange(257))
    for i in range(256):
        if edges[i + 1] - edges[i] >= 20:
            lut[i] = np.median(cols[edges[i]:edges[i + 1]], axis=0)
    for ch in range(3):  # fill empty bins; with no data at all, grey
        ok = ~np.isnan(lut[:, ch])
        lut[:, ch] = np.interp(np.arange(256), np.where(ok)[0], lut[ok, ch]) if ok.any() else np.arange(256)
    return lut


def project(frames, lut):
    """Intensity (mean of channels) and each pixel's colour distance from the colour map at that intensity."""
    f = frames.astype(np.float32)
    inten = f.mean(-1)
    off = np.linalg.norm(f - lut[np.clip(inten.astype(int), 0, 255)], axis=-1)
    return inten, off


def find_fan(grey, colour_any, std_thresh):
    cand = (grey.std(0) > std_thresh) & ~colour_any
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n < 2:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    pts = np.column_stack(np.where(lab == big)[::-1]).astype(np.int32)  # (N, xy)
    fan = np.zeros(grey.shape[1:], np.uint8)
    cv2.fillConvexPoly(fan, cv2.convexHull(pts), 1)
    return fan.astype(bool)


def clean_video(frames, a):
    v = colour_map(frames, a.std)
    grey, off = project(frames, v)
    colour = off > a.colour  # off the video's own colour map: overlay (ECG, marks, text in another colour)
    fan = find_fan(grey, colour.mean(0) > 0.3, a.std)
    if fan is None:
        return None
    kernel = np.ones((2 * a.grow + 1, 2 * a.grow + 1), np.uint8)
    out, n_over = [], []
    for t in range(len(frames)):
        g = np.clip(np.round(grey[t]), 0, 255).astype(np.uint8)
        # isolated off-colour specks are speckle in tinted videos, not overlay: keep blobs of >= 5 px
        n, lab, st, _ = cv2.connectedComponentsWithStats(colour[t].astype(np.uint8), connectivity=8)
        keep = np.zeros(n, np.uint8)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 5
        over = cv2.dilate(keep[lab], kernel) & fan
        if over.any():
            g = cv2.inpaint(g, over, 3, cv2.INPAINT_TELEA)
        out.append(np.where(fan, g, 0).astype(np.uint8))
        n_over.append(int(over.sum()))
    ys, xs = np.where(fan)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    x0, y0, x1, y1 = box
    return dict(frames=np.stack(out)[:, y0:y1, x0:x1], fan=fan[y0:y1, x0:x1], box=box,
                fan_full=fan, overlay=np.array(n_over), tint=v[200] / max(v[200].max(), 1))


def thumb(img, w=240):
    h = int(round(img.shape[0] * w / img.shape[1]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def sheet(tiles, path, cols=8, w=240, h=180):
    rows = []
    for i in range(0, len(tiles), cols):
        row = [cv2.copyMakeBorder(cv2.resize(t, (w, h)) if t.shape[:2] != (h, w) else t, 0, 0, 0, 0,
                                  cv2.BORDER_CONSTANT) for t in tiles[i:i + cols]]
        row += [np.zeros((h, w, 3), np.uint8)] * (cols - len(row))
        rows.append(np.concatenate(row, 1))
    cv2.imwrite(path, np.concatenate(rows, 0))


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--std', type=float, default=4.0, help='grey-level std over frames that counts as changing')
    ap.add_argument('--colour', type=int, default=20, help='distance from the colour map that counts as coloured')
    ap.add_argument('--grow', type=int, default=2, help='px added around coloured pixels before inpainting')
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'sheets'), exist_ok=True)
    mask_dir = os.path.join(a.root, 'LV Ground-truth Segmentation Masks')
    report, fan_tiles, clean_tiles = {}, [], []
    for view, sub in VIEWS.items():
        os.makedirs(os.path.join(a.out, view), exist_ok=True)
        for path in sorted(glob.glob(os.path.join(a.root, sub, '*.avi')))[:a.limit]:
            name = os.path.splitext(os.path.basename(path))[0]
            frames = read_video(path)
            H, W = frames.shape[1:3]
            r = clean_video(frames, a)
            rec = dict(view=view, size=[W, H], frames=len(frames))
            if r is None:
                rec['error'] = 'no fan found'
                report[f'{view}/{name}'] = rec
                print(f'{view}/{name}: NO FAN', flush=True)
                continue
            np.savez_compressed(os.path.join(a.out, view, name + '.npz'), frames=r['frames'], fan=r['fan'],
                                box=np.array(r['box']), size=np.array([W, H]), overlay=r['overlay'])
            rec.update(box=r['box'], fan_frac=float(r['fan_full'].mean()), tint_bgr=[round(float(x), 3) for x in r['tint']],
                       overlay_px_mean=float(r['overlay'].mean()), overlay_px_max=int(r['overlay'].max()))
            mp = os.path.join(mask_dir, f'Mask_{name}.mat')
            if os.path.exists(mp):
                m = scipy.io.loadmat(mp)['predicted'].max(0)  # union over frames, (224, 224) on the squashed frame
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                rec['mask_in_fan'] = float((m & r['fan_full']).sum() / max(m.sum(), 1))
            report[f'{view}/{name}'] = rec
            outline = frames[0].copy()
            cnts, _ = cv2.findContours(r['fan_full'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(outline, cnts, -1, (0, 0, 255), 2)
            fan_tiles.append(label(cv2.resize(outline, (240, 180)), f'{view} {name}'))
            c = cv2.cvtColor(r['frames'][0], cv2.COLOR_GRAY2BGR)
            clean_tiles.append(label(cv2.resize(c, (240, 180)), f'{view} {name}'))
            print(f"{view}/{name}: {W}x{H} T={len(frames)} box {r['box']} fan {rec['fan_frac']:.3f} "
                  f"overlay {rec['overlay_px_mean']:.0f} px" +
                  (f" mask-in-fan {rec['mask_in_fan']:.3f}" if 'mask_in_fan' in rec else ''), flush=True)

    # outliers: fan area or box far from the median of videos with the same frame size
    by_size = {}
    for k, rec in report.items():
        if 'box' in rec:
            by_size.setdefault(tuple(rec['size']), []).append(k)
    for size, keys in by_size.items():
        med_area = np.median([report[k]['fan_frac'] for k in keys])
        for k in keys:
            flags = []
            if abs(report[k]['fan_frac'] / med_area - 1) > 0.4:  # fan size varies with depth/width settings
                flags.append('fan area')
            if report[k].get('mask_in_fan', 1) < 0.95:
                flags.append('mask outside fan')
            report[k]['flags'] = flags
    json.dump(report, open(os.path.join(a.out, 'clean_echo.json'), 'w'), indent=1)
    for i in range(0, len(fan_tiles), 40):
        sheet(fan_tiles[i:i + 40], os.path.join(a.out, 'sheets', f'fan_{i // 40:02d}.png'))
        sheet(clean_tiles[i:i + 40], os.path.join(a.out, 'sheets', f'clean_{i // 40:02d}.png'))
    flagged = {k: v.get('flags') or v.get('error') for k, v in report.items() if v.get('flags') or v.get('error')}
    mif = [v['mask_in_fan'] for v in report.values() if 'mask_in_fan' in v]
    print(f'{len(report)} videos, {len(flagged)} flagged; masks: {len(mif)}, mask-in-fan min {min(mif):.3f} '
          f'median {np.median(mif):.3f}' if mif else f'{len(report)} videos, {len(flagged)} flagged')
    for k, f in flagged.items():
        print('  FLAG', k, f)


if __name__ == '__main__':
    main()
