"""Run the dog_detr2x RF-DETR checkpoint on every frame of each clip; write all
detections (full-res pixel coords) to <clip>.detr.csv. Optional 2x2 tiling."""
import sys, csv, time, argparse
import cv2, numpy as np, rfdetr
ap = argparse.ArgumentParser()
ap.add_argument('clips', nargs='+'); ap.add_argument('--thr', type=float, default=0.15)
ap.add_argument('--frames', type=int, default=0); ap.add_argument('--tile', action='store_true')
ap.add_argument('--suffix', default='.detr.csv')
ap.add_argument('--labels', default='/root/.local/share/steeleagle/models/coco81_classes.txt')
ap.add_argument('--model', default='/root/.local/share/steeleagle/models/dog_extended_2x.pth')
a = ap.parse_args()
m = rfdetr.RFDETR.from_checkpoint(a.model, device='cuda')
import json
LAB = json.load(open(a.labels)) if a.labels else []
print('resolution', m.model.resolution, flush=True)
def run(rgb, ox=0, oy=0):
    d = m.predict(rgb, threshold=a.thr)
    raw = d.data.get('class_name', [str(c) for c in d.class_id])
    names = [LAB[i] if 0 <= i < len(LAB) else n for i, n in zip(d.class_id.tolist(), raw)]  # same remap as the detection engine
    return [(x1+ox, y1+oy, x2+ox, y2+oy, s, str(n)) for (x1, y1, x2, y2), s, n in zip(d.xyxy.tolist(), d.confidence.tolist(), names)]
for clip in a.clips:
    cap = cv2.VideoCapture(clip); out = []; k = 0; t0 = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok or (a.frames and k >= a.frames): break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
        if a.tile:   # 2x2 overlapping tiles, each ~1060x620
            dets = []
            tw, th = int(W*0.55), int(H*0.575)
            for ty in (0, H-th):
                for tx in (0, W-tw):
                    dets += run(np.ascontiguousarray(rgb[ty:ty+th, tx:tx+tw]), tx, ty)
        else:
            dets = run(rgb)
        for x1, y1, x2, y2, s, n in dets:
            out.append((k, round(x1,1), round(y1,1), round(x2,1), round(y2,1), round(s,4), n))
        k += 1
    with open(clip.rsplit('.',1)[0] + a.suffix, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['frame','x1','y1','x2','y2','score','cls']); w.writerows(out)
    print(clip, k, 'frames', len(out), 'dets', f'{k/(time.time()-t0):.1f} fps', flush=True)
