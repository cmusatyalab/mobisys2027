"""Write per-frame camera quaternion (w,x,y,z, body->NED, Parrot vmeta P3 offset 44)
for each 60 s clip, taken from its source MP4's metadata stream.
usage: clipquat.py cuts.txt [src_dir out_dir]   (lines: source clip start_s)"""
import sys, subprocess, numpy as np, math
FPS = 30000 / 1001
SRC = (sys.argv[2] if len(sys.argv) > 2 else '/home/teiszler/Videos/ddil_tracking/nov17').rstrip('/') + '/'
OUT = (sys.argv[3] if len(sys.argv) > 3 else '/home/teiszler/Videos/ddil_tracking/analysis/nov17').rstrip('/') + '/'
cache = {}
for line in open(sys.argv[1]):
    src, clip, st = line.split(); st = float(st)
    if src not in cache:
        raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', SRC + src + '.MP4', '-map', '0:1', '-c', 'copy', '-f', 'data', '-'], capture_output=True).stdout
        d = np.frombuffer(raw, np.uint8); d = d[:len(d) // 144 * 144].reshape(-1, 144)
        assert (d[:, :2] == np.frombuffer(b'P3', np.uint8)).all()
        cache[src] = d[:, 44:52].copy().view('>i2').astype(float) / 16384
    q = cache[src]; i0 = math.ceil(st * FPS - 1e-6)
    np.savetxt(f'{OUT}{clip}.quat.csv', q[i0:i0 + 1799], delimiter=',', fmt='%.5f', header='w,x,y,z', comments='')
    print(clip, i0, len(q[i0:i0 + 1799]))
