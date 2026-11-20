"""Detect the white quadruped in each frame of a run and log its pixel position.
Works on frames downscaled to 960x540 (2x down from 1920x1080)."""
import subprocess, sys, csv
import numpy as np, cv2

import os
VIDEO = os.environ.get('VIDEO', '/home/teiszler/Videos/ddil_tracking/nov09.MP4')
W, H = 960, 540
FPS = 30000 / 1001


def frames(start, dur):
    cmd = ['ffmpeg', '-v', 'error', '-ss', str(start), '-i', VIDEO, '-t', str(dur),
           '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


def candidates(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vf = v.astype(np.float32)
    bg = cv2.blur(vf, (41, 41))  # local background brightness
    contrast = vf - bg
    white = ((s < 70) & (v > 140) & (contrast > 40)).astype(np.uint8)
    green = ((h >= 25) & (h <= 95) & (s > 45)).astype(np.float32)
    greenfrac = cv2.boxFilter(green, -1, (61, 61))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # pass 1: find landing pads via enclosed dark holes and erase them from the white mask
    contours, hier = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for i, c in enumerate(contours):
            if hier[0][i][3] == -1:
                continue
            ha = cv2.contourArea(c)
            if ha < 8:
                continue
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            if min(rw, rh) < 0.4 * max(rw, rh):
                continue  # pad centres are squarish
            box = cv2.boxPoints(((cx, cy), (rw * 2.6 + 4, rh * 2.6 + 4), ang)).astype(np.int32)
            cv2.fillPoly(white, [box], 0)
    contours, hier = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(contours):
        if hier[i][3] != -1:
            continue  # it's a hole
        area = cv2.contourArea(c)
        x, y, w, hh = cv2.boundingRect(c)
        if not (12 <= area <= 600) or max(w, hh) > 45:
            continue
        # holes (landing pad dark centre)
        hole = 0
        j = hier[i][2]
        while j != -1:
            hole += cv2.contourArea(contours[j]); j = hier[j][0]
        if hole > 0.15 * area:
            continue
        # pad-like: near-square, solid filled bbox with dark interior
        if w > 12 and hh > 12:
            cv_ = v[y + hh // 3:y + 2 * hh // 3, x + w // 3:x + 2 * w // 3]
            wf = white[y:y + hh, x:x + w].mean()
            if cv_.size and cv_.mean() < 110 and wf > 0.45:
                continue  # landing pad: white frame, dark centre
        cx, cy = x + w / 2, y + hh / 2
        if greenfrac[int(cy), int(cx)] < 0.55:
            continue  # must be surrounded by grass (rejects path, trailer, buildings)
        peak_v = int(v[y:y + hh, x:x + w].max())
        peak_c = float(contrast[y:y + hh, x:x + w].max())
        if peak_v < 200 or peak_c < 70:
            continue  # sun glints on grass are dimmer and lower-contrast than the robot's white body
        out.append((cx, cy, area, w, hh, peak_c))
    return out


def run(name, start, dur, out_csv):
    pred = (W / 2, H / 2); vel = (0.0, 0.0)
    lost = 0
    rows = []
    for k, img in enumerate(frames(start, dur)):
        cands = candidates(img)
        gate = 45 + 3 * min(lost, 20)  # px at 960 scale; widen slowly while lost
        best = None
        for c in cands:
            d = np.hypot(c[0] - pred[0], c[1] - pred[1])
            if d < gate and (best is None or d < best[0]):
                best = (d, c)
        if best is None and (lost > 30 or k == 0):
            # re-acquire anywhere: strongest robot-like candidate
            strong = [c for c in cands if c[5] >= 85 and c[2] >= 30 and
                      (k == 0 or np.hypot(c[0] - last[0], c[1] - last[1]) < 250)]
            if strong:
                best = (0, max(strong, key=lambda c: c[5] * np.sqrt(c[2])))
                lost = 1  # reset velocity
        if best:
            c = best[1]
            if lost == 0:
                nv = (c[0] - last[0], c[1] - last[1]) if k else (0, 0)
                vel = (0.7 * vel[0] + 0.3 * nv[0], 0.7 * vel[1] + 0.3 * nv[1])
            else:
                vel = (0.0, 0.0)
            last = c[:2]; lost = 0
            pred = (c[0] + vel[0], c[1] + vel[1])
            rows.append((k, k / FPS, c[0] * 2, c[1] * 2, c[2] * 4, 1, len(cands)))
        else:
            lost += 1
            vel = (vel[0] * 0.8, vel[1] * 0.8)
            pred = (pred[0] + vel[0], pred[1] + vel[1])
            rows.append((k, k / FPS, '', '', '', 0, len(cands)))
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 't', 'x', 'y', 'area', 'detected', 'ncand'])
        wr.writerows(rows)
    print(name, len(rows), 'frames, detected', sum(r[5] for r in rows))


if __name__ == '__main__':
    name, start, dur, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    run(name, start, dur, out)
