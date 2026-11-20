import sys,subprocess,numpy as np,cv2,pandas as pd
run,start=sys.argv[1],float(sys.argv[2]); ts=[float(x) for x in sys.argv[3].split(',')]
d=pd.read_csv(f'run{run}.csv'); ims=[]
for t in ts:
    k=int(round(t*30000/1001)); r=d.iloc[k]
    b=subprocess.run(['ffmpeg','-v','error','-ss',str(start+r.t),'-i',__import__('os').environ.get('VIDEO','/home/teiszler/Videos/ddil_tracking/nov09.MP4'),'-frames:v','1','-f','image2pipe','-vcodec','png','-'],capture_output=True).stdout
    im=cv2.imdecode(np.frombuffer(b,np.uint8),1)
    if r.detected: cv2.circle(im,(int(r.x),int(r.y)),40,(0,0,255),4)
    cv2.drawMarker(im,(960,540),(255,255,0),cv2.MARKER_CROSS,60,3)
    cv2.putText(im,f'{run} t={r.t:.1f} det={int(r.detected)}',(20,60),0,2,(0,255,255),4)
    ims.append(cv2.resize(im,(640,360)))
while len(ims)%3: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(f'ov_{run}.jpg',np.vstack([np.hstack(ims[i:i+3]) for i in range(0,len(ims),3)]))
