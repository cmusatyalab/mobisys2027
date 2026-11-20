"""Ground-truth robot track for a Nov 17 clip from RF-DETR detections, rejecting
static distractors (chair, bag) that the model also calls "robotic dog".

Each detection is mapped to a world direction (az, el) from the per-frame camera
quaternion (world.py). A stationary object keeps the same direction for the whole
clip; the robot's direction changes as it walks.

1. Anchors: direction clusters (radius ANCHOR_R) that are populated over a long
   time span (>= MIN_SPAN s, >= MIN_FRAMES frames) are static objects. Each anchor
   claims at most one detection per frame (the nearest), so the robot walking
   right past the chair is not swallowed by it.
2. Robot: among unclaimed detections, follow continuity in direction space
   (gate widens while lost); start from the strongest detection nearest the
   frame centre in the first frames.

Output: run<name>.csv with frame,t,x,y,area,detected,ncand (same as track.py),
plus <name>.anchors.csv for inspection.

usage: gt.py <name> [--f 1400 --lag 1]   (reads <name>.detr.csv, <name>.quat.csv)
"""
import argparse, numpy as np, pandas as pd, world

FPS = 30000 / 1001
N = 1799
ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('--f', type=float, default=1400); ap.add_argument('--lag', type=int, default=1)
ap.add_argument('--thr', type=float, default=0.3)
ap.add_argument('--min_dur', type=float, default=4.0)      # s
ap.add_argument('--max_speed', type=float, default=0.5)    # deg/s
a = ap.parse_args()

det = pd.read_csv(f'{a.name}.detr.csv')
q = np.loadtxt(f'{a.name}.quat.csv', delimiter=',', skiprows=1)
d = det[(det.cls == 'robotic dog') & (det.score >= a.thr)].copy().reset_index(drop=True)
d['u'] = (d.x1 + d.x2) / 2; d['v'] = (d.y1 + d.y2) / 2
# The model often adds a secondary "chair" box on the chair (~30-40% of its
# robotic-dog detections) but almost never on the robot (~1%): flag overlaps.
chairs = {f: g for f, g in det[det.cls == 'chair'].groupby('frame')}
def chair_iou(r):
    g = chairs.get(r.frame)
    if g is None:
        return False
    ix = np.maximum(0, np.minimum(g.x2, r.x2) - np.maximum(g.x1, r.x1)); iy = np.maximum(0, np.minimum(g.y2, r.y2) - np.maximum(g.y1, r.y1))
    inter = ix * iy
    return bool((inter / ((g.x2 - g.x1) * (g.y2 - g.y1) + (r.x2 - r.x1) * (r.y2 - r.y1) - inter) > 0.5).any())
d['chairbox'] = [chair_iou(r) for r in d.itertuples()]
R = world.rotmats(q[np.clip(d.frame.values + a.lag, 0, N - 1)])
d['az'], d['el'] = world.pix2dir(R, d.u.values, d.v.values, a.f)


def unit(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


V = unit(d.az.values, d.el.values)

# --- 1. static objects: link detections into tracklets in direction space, then
# call a tracklet static if it lasts >= MIN_DUR s yet moves < MAX_SPEED deg/s.
# (Fixed-radius anchors fail: static directions drift a few degrees per minute
# with the drone's heading/position drift.)
d['trk'] = -1
tracks = []          # each: dict(last_v, last_f, members)
order = d.sort_values('frame').index
for f, g in d.loc[order].groupby('frame'):
    used = set()
    pairs = []
    for ti, t in enumerate(tracks):
        gap = f - t['last_f']
        if gap > 30:
            continue
        gate = 1.0 + 0.1 * gap
        ang = np.degrees(np.arccos(np.clip(V[g.index] @ t['last_v'], -1, 1)))
        for j, an in enumerate(ang):
            if an < gate:
                pairs.append((an, ti, g.index[j]))
    for an, ti, di in sorted(pairs):
        if ti in used or d.at[di, 'trk'] != -1:
            continue
        used.add(ti); d.at[di, 'trk'] = ti
        tracks[ti]['last_v'] = V[di]; tracks[ti]['last_f'] = f; tracks[ti]['m'].append(di)
    for di in g.index:
        if d.at[di, 'trk'] == -1:
            d.at[di, 'trk'] = len(tracks); tracks.append(dict(last_v=V[di], last_f=f, m=[di]))

# robot seed: in the first frame with detections, the strongest one nearest the frame
# centre (each trial starts with the gimbal pitching onto the robot). Its tracklet can
# never be static, even if the robot stands still at the start.
f0 = d.frame.min(); g0 = d[d.frame == f0]
seed = g0.index[np.argmax(g0.score.values - 0.3 * np.hypot(g0.u.values - 960, g0.v.values - 540) / 1100)]
seed_trk = d.at[seed, 'trk']
_m = np.array(tracks[seed_trk]['m'])
print(f"seed: frame {f0} u={d.at[seed,'u']:.0f} v={d.at[seed,'v']:.0f} score={d.at[seed,'score']:.2f} tracklet n={len(_m)} chairfrac={d.chairbox.values[_m].mean():.2f}")

anchors = []
d['static'] = -1
for ti, t in enumerate(tracks):
    if ti == seed_trk:
        continue
    m = np.array(t['m']); fr = d.frame.values[m]
    dur = (fr.max() - fr.min()) / FPS
    cf = d.chairbox.values[m].mean()
    if len(m) >= 30 and cf >= 0.10:      # chair by secondary label (drift-proof)
        c = V[m].mean(0); c /= np.linalg.norm(c)
        anchors.append(dict(trk=ti, n=len(m), span=round(dur, 1), f0=fr.min(), f1=fr.max(), speed=-1, p90=-1, chairfrac=round(cf, 2),
                            az=np.degrees(np.arctan2(c[1], c[0])), el=np.degrees(np.arcsin(c[2]))))
        d.loc[m, 'static'] = len(anchors) - 1
        continue
    if dur < a.min_dur or len(m) < a.min_dur * FPS * 0.3:
        continue
    # speed: direction change over ~1 s steps
    vv = V[m]; step = []
    for i in range(0, len(m) - 1):
        j = np.searchsorted(fr, fr[i] + 30)
        if j < len(m):
            step.append(np.degrees(np.arccos(np.clip(vv[i] @ vv[j], -1, 1))) / ((fr[j] - fr[i]) / FPS))
    sp = np.median(step) if step else 0
    p90 = np.percentile(step, 90) if step else 0
    if sp < a.max_speed and p90 < 4 * a.max_speed:
        k = len(anchors)
        c = vv.mean(0); c /= np.linalg.norm(c)
        anchors.append(dict(trk=ti, n=len(m), span=round(dur, 1), f0=fr.min(), f1=fr.max(), speed=round(sp, 3), p90=round(p90, 3), chairfrac=round(cf, 2),
                            az=np.degrees(np.arctan2(c[1], c[0])), el=np.degrees(np.arcsin(c[2]))))
        d.loc[m, 'static'] = k

# --- 2. robot track: continuity among non-static detections; after a loss of
# > REACQ frames, re-acquire on a stable (>= 1 s) non-static tracklet that is not
# near any static object's recent direction, nearest the last robot direction.
trk_len = d.groupby('trk').size()
SV = V[d.static.values >= 0]; SF = d.frame.values[d.static.values >= 0]; ST = d.trk.values[d.static.values >= 0]
REACQ = 15
# A static tracklet that begins where the robot was last seen (<= 2 deg, within
# 3 s after the last robot frame) and never carries a "chair" box is the robot
# standing still (e.g. sitting on a landing pad), not a distractor: adopt it.
trk_first = d.groupby('trk').frame.min(); trk_chair = d.groupby('trk').chairbox.mean()
byf = {f: g for f, g in d.groupby('frame')}
rows = []; prev = None; lost = 0; cur_trk = None; last_f = None; adopted = set()
for k in range(N):
    g = byf.get(k)
    pick = None
    if k == f0:
        pick = seed
    elif g is not None and len(g) and prev is not None:
        ang_all = np.degrees(np.arccos(np.clip(V[g.index] @ prev, -1, 1)))
        ok_mask = (g.static.values == -1) | (g.trk.values == cur_trk) | np.isin(g.trk.values, list(adopted))
        for j, di in enumerate(g.index):
            t = d.at[di, 'trk']
            if not ok_mask[j] and ang_all[j] < 2.0 and trk_first[t] >= last_f - 30 and trk_first[t] <= last_f + 90 and trk_chair[t] < 0.05:
                ok_mask[j] = True; adopted.add(t)
        gg = g[ok_mask]; ang = ang_all[ok_mask]
        if len(gg):
            if lost <= REACQ:
                j = np.argmin(ang)
                if ang[j] < 3.0 + 0.25 * lost:
                    pick = gg.index[j]
            else:
                ok = []
                for j, di in enumerate(gg.index):
                    if trk_len[d.at[di, 'trk']] < 30 or (d.at[di, 'static'] >= 0 and d.at[di, 'trk'] not in adopted):
                        continue
                    near_static = len(SV) and ((np.abs(SF - k) < 150) & (ST != d.at[di, 'trk']) & ((SV @ V[di]) > np.cos(np.radians(3.0)))).any()
                    if not near_static:
                        ok.append((ang[j], di))
                if ok:
                    pick = min(ok)[1]
    if pick is not None:
        r = d.loc[pick]; prev = V[pick]; lost = 0; cur_trk = d.at[pick, 'trk']; last_f = k
        rows.append((k, k / FPS, r.u, r.v, (r.x2 - r.x1) * (r.y2 - r.y1), 1, 0 if g is None else len(g)))
    else:
        lost += 1
        rows.append((k, k / FPS, '', '', '', 0, 0 if g is None else len(g)))
if adopted:
    print('adopted static tracklets as robot-standing-still:', sorted(int(t) for t in adopted))

pd.DataFrame(rows, columns=['frame', 't', 'x', 'y', 'area', 'detected', 'ncand']).to_csv(f'run{a.name}.csv', index=False)
pd.DataFrame([{k: v for k, v in an.items() if k != 'c'} for an in anchors]).round(2).to_csv(f'{a.name}.anchors.csv', index=False)
det_n = sum(r[5] for r in rows)
print(f'{a.name}: anchors={len(anchors)} ' + ' '.join(f"[az{an['az']:.1f} el{an['el']:.1f} n{an['n']} {an['span']:.0f}s]" for an in anchors)
      + f' | robot detected {det_n}/{N} ({100 * det_n / N:.1f}%) | static-claimed {int((d.static >= 0).sum())} dets')
