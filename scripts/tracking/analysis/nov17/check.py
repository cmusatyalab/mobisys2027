"""Visual checks for gt.py output.
  check.py <name> sheet [t1,t2,...]  -> chk_<name>.jpg: frames with robot pick (green circle),
                                       other robotic-dog detections (yellow), frame centre (cyan)
  check.py <name> plot              -> dirs_<name>.png: az/el of detections over time, robot pick in red
"""
import sys, os, numpy as np, pandas as pd, cv2, world
FPS = 30000 / 1001
name, mode = sys.argv[1], sys.argv[2]
det = pd.read_csv(f'{name}.detr.csv'); run = pd.read_csv(f'run{name}.csv')
dd = det[(det.cls == 'robotic dog') & (det.score >= 0.3)]
if mode == 'sheet':
    ts = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else list(np.arange(0, 60, 4))
    cap = cv2.VideoCapture(f"{os.environ.get('CLIPS', '/home/teiszler/Videos/ddil_tracking/nov17')}/{name}.mp4"); ims = []
    for t in ts:
        k = min(int(round(t * FPS)), 1798); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, im = cap.read()
        for _, r in dd[dd.frame == k].iterrows():
            cv2.rectangle(im, (int(r.x1) - 8, int(r.y1) - 8), (int(r.x2) + 8, int(r.y2) + 8), (0, 255, 255), 4)
        r = run.iloc[k]
        if r.detected:
            cv2.circle(im, (int(r.x), int(r.y)), 70, (0, 255, 0), 8)
        cv2.drawMarker(im, (960, 540), (255, 255, 0), cv2.MARKER_CROSS, 60, 3)
        cv2.putText(im, f'{name} t={t:.1f} det={int(r.detected)}', (20, 70), 0, 2.2, (255, 0, 255), 5)
        ims.append(cv2.resize(im, (480, 270)))
    while len(ims) % 4:
        ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(f'chk_{name}.jpg', np.vstack([np.hstack(ims[i:i + 4]) for i in range(0, len(ims), 4)]))
else:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    q = np.loadtxt(f'{name}.quat.csv', delimiter=',', skiprows=1)
    d = dd.copy(); d['u'] = (d.x1 + d.x2) / 2; d['v'] = (d.y1 + d.y2) / 2
    R = world.rotmats(q[np.clip(d.frame.values + 1, 0, 1798)]); d['az'], d['el'] = world.pix2dir(R, d.u.values, d.v.values, 1400)
    rr = run[run.detected == 1]; Rr = world.rotmats(q[np.clip(rr.frame.values + 1, 0, 1798)])
    raz, rel = world.pix2dir(Rr, rr.x.values.astype(float), rr.y.values.astype(float), 1400)
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for a_, col, rv in ((ax[0], 'az', raz), (ax[1], 'el', rel)):
        a_.scatter(d.frame / FPS, d[col], s=3, c='gray'); a_.plot(rr.t, rv, 'r.', ms=1.5); a_.set_ylabel(col); a_.grid(alpha=.4)
    ax[2].scatter(d.frame / FPS, d.u, s=3, c='gray'); ax[2].plot(rr.t, rr.x, 'r.', ms=1.5); ax[2].set_ylabel('u px'); ax[2].grid(alpha=.4)
    ax[0].set_title(f'{name}: gray = all robotic-dog detections, red = ground-truth robot')
    plt.tight_layout(); plt.savefig(f'dirs_{name}.png', dpi=55)
