import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
RUNS=sys.argv[1:] or list('ABC')
fig,ax=plt.subplots(len(RUNS),1,figsize=(14,9),sharex=True)
for a,r in zip(ax,RUNS):
    d=pd.read_csv(f'run{r}.csv')
    a.plot(d.t,d.x-960,'.',ms=2,label='dx'); a.plot(d.t,d.y-540,'.',ms=2,label='dy')
    m=d.detected==0; a.plot(d.t[m],[0]*m.sum(),'r|',ms=10,label='lost')
    a.set_ylabel(f'run {r} px'); a.set_ylim(-960,960); a.grid(); a.legend(loc='upper right')
plt.tight_layout(); plt.savefig('traj.png',dpi=70)
