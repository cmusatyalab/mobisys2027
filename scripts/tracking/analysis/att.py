"""Per-frame camera pitch/yaw (deg) from a Parrot Anafi MP4's metadata stream
(P3 records, camera quaternion at byte 44, gimbal base at 28).
usage: att.py video.MP4 out.csv"""
import numpy as np, sys, subprocess
v, out = sys.argv[1], sys.argv[2]; fps = 30000 / 1001
raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', v, '-map', '0:1', '-c', 'copy', '-f', 'data', '-'], capture_output=True).stdout
d = np.frombuffer(raw, np.uint8); d = d[:len(d) // 144 * 144].reshape(-1, 144)
assert (d[:, :2] == np.frombuffer(b'P3', np.uint8)).all()
def e(off):
    w, x, y, z = (d[:, off:off + 8].copy().view('>i2').astype(float) / 16384).T
    return np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))), np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
p, y = e(44); bp, by = e(28)
np.savetxt(out, np.c_[np.arange(len(p)) / fps, p, y, bp, by], delimiter=',', fmt='%.3f', header='t,pitch,yaw,base_pitch,base_yaw', comments='')
