import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import sys
RUNS=sys.argv[1:] or list('ABC')
FPS=30000/1001; CX,CY=960,540; HALFDIAG=np.hypot(960,540)
res={}; curves={}
fig,ax=plt.subplots(1,2,figsize=(14,4.5))
for r in RUNS:
    d=pd.read_csv(f'run{r}.csv'); cam=pd.read_csv(f'cam{r}.csv')
    det=d.detected.values.astype(bool)
    x=d.x.interpolate(limit_direction='both').values; y=d.y.interpolate(limit_direction='both').values
    ex,ey=x-CX,y-CY; e=np.hypot(ex,ey)
    # max gap
    gaps=np.diff(np.flatnonzero(np.diff(np.r_[1,det.astype(int),1])!=0))[::2] if (~det).any() else [0]
    # camera content shift (px/frame), robust clip of bad correlations
    sx,sy=cam.dx.values,cam.dy.values
    sxs,sys_=uniform_filter1d(sx,15),uniform_filter1d(sy,15)
    cam_speed=np.hypot(sxs,sys_)*FPS
    jitter=np.sqrt(np.mean((sx-sxs)**2+(sy-sys_)**2))*FPS
    cam_acc=np.hypot(np.diff(sxs),np.diff(sys_))*FPS*FPS
    # target scene-frame apparent velocity = image motion minus content shift (smoothed)
    xs,ys=uniform_filter1d(x,15),uniform_filter1d(y,15)
    vtx=np.diff(xs)-sxs[1:]; vty=np.diff(ys)-sys_[1:]
    tgt_speed=np.hypot(vtx,vty)*FPS
    th=np.arange(0,481)
    prec=np.array([(e<=t).mean() for t in th]); curves[r]=prec
    res[r]=dict(coverage=det.mean()*100, max_gap_s=max(gaps)/FPS,
        rmse_px=np.sqrt(np.mean(e**2)), mean_px=e.mean(), median_px=np.median(e), p95_px=np.percentile(e,95), max_px=e.max(),
        rms_x=np.sqrt(np.mean(ex**2)), rms_y=np.sqrt(np.mean(ey**2)),
        in50=(e<=50).mean()*100, in100=(e<=100).mean()*100, in200=(e<=200).mean()*100,
        auc300=prec[:301].mean()*100,
        tgt_speed_px_s=tgt_speed.mean(), cam_speed_px_s=cam_speed.mean(),
        cam_jitter_px_s=jitter, cam_acc_rms=np.sqrt(np.mean(cam_acc**2)),
        lag_s=e.mean()/tgt_speed.mean())
    ax[0].plot(d.t,e,lw=1,label=r)
    ax[1].plot(th,prec*100,label=r)
ax[0].set_xlabel('t (s)'); ax[0].set_ylabel('distance from centre (px, 1080p)'); ax[0].legend(); ax[0].grid()
ax[1].set_xlabel('radius threshold (px)'); ax[1].set_ylabel('% frames within'); ax[1].legend(); ax[1].grid()
plt.tight_layout(); plt.savefig('metrics.png',dpi=70)
pd.set_option('display.width',200); print(pd.DataFrame(res).round(2))
