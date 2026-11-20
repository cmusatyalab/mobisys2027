"""Seconds the gimbal was centred on a distractor: a non-robot robotic-dog detection
within R1 (108 px) of frame centre while the GT robot is > R2 (270 px) from centre
or not detected/out of frame. usage: fpfollow.py names..."""
import sys, numpy as np, pandas as pd
FPS = 30000 / 1001
EP = []
for n in sys.argv[1:]:
    det = pd.read_csv(f'{n}.detr.csv'); run = pd.read_csv(f'run{n}.csv')
    dd = det[(det.cls == 'robotic dog') & (det.score >= 0.3)].copy(); dd['u'] = (dd.x1 + dd.x2) / 2; dd['v'] = (dd.y1 + dd.y2) / 2
    rx = run.set_index('frame')
    fol = np.zeros(len(run), bool); isch = np.zeros(len(run), bool)
    ch = {f: g for f, g in det[det.cls == 'chair'].groupby('frame')}
    for f, g in dd.groupby('frame'):
        r = rx.loc[f]
        if r.detected:
            g = g[np.hypot(g.u - r.x, g.v - r.y) > 2]   # drop the robot's own box
            rob_far = np.hypot(r.x - 960, r.y - 540) > 270
        else:
            rob_far = True
        near = g[np.hypot(g.u - 960, g.v - 540) < 108] if len(g) else g
        if rob_far and len(near):
            fol[f] = True
            b = near.iloc[0]; c = ch.get(f)
            if c is not None:
                ix = np.maximum(0, np.minimum(c.x2, b.x2) - np.maximum(c.x1, b.x1)); iy = np.maximum(0, np.minimum(c.y2, b.y2) - np.maximum(c.y1, b.y1))
                isch[f] = bool((ix * iy > 0.3 * (b.x2 - b.x1) * (b.y2 - b.y1)).any())
    # episodes: merge gaps < 1 s, keep >= 0.5 s
    idx = np.where(fol)[0]; eps = []
    for i in idx:
        if eps and i - eps[-1][1] <= FPS: eps[-1][1] = i
        else: eps.append([i, i])
    for a0, a1 in eps:
        if (a1 - a0 + 1) / FPS >= 0.5:
            cf = isch[a0:a1 + 1][fol[a0:a1 + 1]].mean()
            EP.append((n, round(a0 / FPS, 1), round((a1 + 1) / FPS, 1), round(cf, 2)))
    # first/last time, for notes
    t = np.where(fol)[0] / FPS
    print(f'{n} {fol.sum() / FPS:.1f} ' + (f'{t.min():.0f}-{t.max():.0f}' if len(t) else '-'))

import csv
with open('fp_episodes.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['run', 'start', 'end', 'chairbox_frac']); w.writerows(EP)
