import sys, numpy as np, cv2, subprocess
import os
VIDEO=os.environ.get('VIDEO','/home/teiszler/Videos/ddil_tracking/nov09.MP4'); W,H=480,270
name,start=sys.argv[1],float(sys.argv[2])
p=subprocess.Popen(['ffmpeg','-v','error','-ss',str(start),'-i',VIDEO,'-t','60','-vf',f'scale={W}:{H},format=gray','-f','rawvideo','-'],stdout=subprocess.PIPE)
win=cv2.createHanningWindow((W,H),cv2.CV_32F); prev=None; out=[]
while True:
    b=p.stdout.read(W*H)
    if len(b)<W*H: break
    g=np.frombuffer(b,np.uint8).reshape(H,W).astype(np.float32)
    if prev is None: out.append((0,0,1))
    else:
        (dx,dy),r=cv2.phaseCorrelate(prev,g,win); out.append((dx*4,dy*4,r))  # full-res px
    prev=g
np.savetxt(f'cam{name}.csv',np.array(out),delimiter=',',header='dx,dy,resp',comments='')
print(name,len(out))
