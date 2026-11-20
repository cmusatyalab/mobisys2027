import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import sys
RUNS=sys.argv[1:] or list('ABC')
FPS=30000/1001
fig,ax=plt.subplots(2,3,figsize=(16,7))
for i,r in enumerate(RUNS):
    d=pd.read_csv(f'run{r}.csv'); c=pd.read_csv(f'cam{r}.csv')
    ex=d.x.interpolate(limit_direction='both').values-960; ey=d.y.interpolate(limit_direction='both').values-540
    # camera pan velocity in image px/s: content shifts opposite to camera rotation
    px=-uniform_filter1d(c.dx.values,9)*FPS; py=-uniform_filter1d(c.dy.values,9)*FPS
    out=[]
    for nm,e,p in (('yaw/x',ex,px),('pitch/y',ey,py)):
        e0=e-e.mean(); p0=p-p.mean(); lags=np.arange(0,int(3*FPS))
        cc=np.array([np.corrcoef(e0[:len(e0)-L],p0[L:])[0,1] for L in lags])
        L=lags[cc.argmax()]
        gain=np.polyfit(e0[:len(e0)-L],p0[L:],1)[0]  # pan px/s per px error = effective P gain (1/s)
        out.append(f'{nm}: delay {L/FPS:.2f}s (r={cc.max():.2f}), eff. gain {gain:.2f}/s')
        ax[0 if nm[0]=='y' else 1,i].plot(lags/FPS,cc); ax[0 if nm[0]=='y' else 1,i].axvline(L/FPS,color='r',ls='--')
        ax[0 if nm[0]=='y' else 1,i].set_title(f'{r} {nm}: peak {L/FPS:.2f}s'); ax[0 if nm[0]=='y' else 1,i].grid()
    # also: how often does the camera move at all (fraction of time pan speed < 10 px/s)
    sp=np.hypot(px,py); print(r,' | '.join(out),f'| camera ~still {100*(sp<10).mean():.0f}% of time')
for a in ax[1]: a.set_xlabel('camera pan lags error by (s)')
plt.tight_layout(); plt.savefig('latency.png',dpi=65)
