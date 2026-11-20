"""AUC of precision curve and % of frames within radius R, with R given as fractions of frame half-height (540 px at 1080p).
AUC over [0,R] = 1 - mean(min(e,R))/R.  Usage: python3 auc.py label=path.csv ...

Gaps in the ground-truth track (frames where the robot was not detected) are classified:
  * out of frame  - the detection just before or just after the gap is within EDGE px of the image border.
                    The robot is treated as lost: error = infinity (counts as a miss at every radius).
  * in frame miss - otherwise (e.g. robot standing on a landing pad near centre): position is interpolated.
"""
import sys, numpy as np, pandas as pd
FR = [0.2, 0.5, 1.0]; HH = 540; W, H = 1920, 1080
EDGE = 150   # px at 1080p
FPS = 30000 / 1001


def gaps(det):
    """list of (start, end_exclusive) index ranges where det is False"""
    out, i, n = [], 0, len(det)
    while i < n:
        if not det[i]:
            j = i
            while j < n and not det[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def near_edge(x, y):
    return x < EDGE or x > W - EDGE or y < EDGE or y > H - EDGE


def errors(d):
    """per-frame error (px) with out-of-frame gaps set to inf; also returns boolean out-of-frame mask"""
    det = d.detected.values.astype(bool)
    x = d.x.interpolate(limit_direction='both').values
    y = d.y.interpolate(limit_direction='both').values
    lost = np.zeros(len(d), bool)
    for s, e in gaps(det):
        before = (d.x.values[s - 1], d.y.values[s - 1]) if s > 0 else None
        after = (d.x.values[e], d.y.values[e]) if e < len(d) else None
        if (before and near_edge(*before)) or (after and near_edge(*after)):
            lost[s:e] = True
    err = np.hypot(x - W / 2, y - H / 2)
    err[lost] = np.inf
    return err, lost


def metrics(d):
    """dict of all per-run metrics for a track DataFrame"""
    e, lost = errors(d)
    det = d.detected.values.astype(bool)
    r = {'detected %': 100 * det.mean(),
         'longest gap s': max([b - a for a, b in gaps(det)] or [0]) / FPS,
         'out of frame s': lost.sum() / FPS,
         'out-of-frame episodes': len([1 for a, b in gaps(det) if lost[a]])}
    for f in FR:
        R = f * HH
        r[f'AUC@{f}'] = 100 * (1 - np.minimum(e, R).mean() / R)
    ed = np.hypot(d.x.dropna() - W / 2, d.y.dropna() - H / 2).values
    for f in FR:
        R = f * HH
        r[f'AUC@{f} (detected only)'] = 100 * (1 - np.minimum(ed, R).mean() / R)
    for f in FR:
        r[f'% within {f}'] = 100 * (e <= f * HH).mean()
    r['% in frame'] = 100 * (~lost).mean()
    ei = e[~lost]   # in-frame frames only
    r['RMSE px (in frame)'] = np.sqrt((ei ** 2).mean()); r['p95 px (in frame)'] = np.percentile(ei, 95)
    return r


if __name__ == '__main__':
    rows = {}
    for arg in sys.argv[1:]:
        lab, p = arg.split('=', 1)
        rows[lab] = metrics(pd.read_csv(p))
    pd.set_option('display.width', 250)
    df = pd.DataFrame(rows)
    df = df.loc[list(next(iter(rows.values())).keys())]
    print(df.round(1).to_string())
