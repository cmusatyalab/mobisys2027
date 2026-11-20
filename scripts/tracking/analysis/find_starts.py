"""Candidate run starts from gimbal pitch (run att.py first).
A start = first pitch change after a quiet (still) period; flagged '<== ~60' when the
gimbal stays active ~50-66 s afterwards (a run lasts 60 s). Always confirm by eye with
frames 1.5 s before/after (before: still, robot off-centre; after: pitching onto robot).
usage: find_starts.py att_<video>.csv ..."""
import sys, numpy as np
for f in sys.argv[1:]:
    d = np.loadtxt(f, delimiter=',', skiprows=1); t = d[:, 0]; p = d[:, 1]
    mv = np.where(np.abs(np.diff(p, prepend=p[0])) > 0.05)[0]; tm = t[mv]
    bursts = []
    for i in mv:
        if bursts and t[i] - t[bursts[-1][1]] < 3: bursts[-1][1] = i
        else: bursts.append([i, i])
    print(f); pe = 0
    for s, e in bursts:
        st = t[s]; q = st - pe; pe = t[e]
        if q < 3: continue
        after = tm[tm >= st]; k = np.where(np.diff(after) >= 10)[0]
        end = after[k[0]] if len(k) else after[-1]
        print(f'   start {st:7.3f} quiet {q:5.1f} active-until {end:6.1f} span {end - st:5.1f}' + ('   <== ~60' if 50 <= end - st <= 66 else ''))
