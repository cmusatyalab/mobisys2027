"""Map pixel positions to world directions (azimuth/elevation, degrees) using the
per-frame camera quaternion from the Parrot metadata. A hovering drone sees a
stationary object (chair, bag) at a constant direction even while the gimbal
moves; the robot's direction changes as it walks."""
import numpy as np

W, H = 1920, 1080


def rotmats(q):
    """q: (N,4) w,x,y,z body->NED. Returns (N,3,3)."""
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def pix2dir(R, u, v, f):
    """R: (N,3,3) per-detection rotation; u,v pixel arrays. Camera body frame:
    x forward (optical axis), y right, z down. Returns az, el in degrees."""
    d = np.stack([np.full_like(u, f, dtype=float), u - W / 2, v - H / 2], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    n = np.einsum('nij,nj->ni', R, d)
    az = np.degrees(np.arctan2(n[:, 1], n[:, 0]))
    el = np.degrees(np.arctan2(-n[:, 2], np.hypot(n[:, 0], n[:, 1])))
    return az, el


def dir2pix(R, az, el, f):
    """Inverse of pix2dir for one rotation per row: world direction -> pixel."""
    a, e = np.radians(az), np.radians(el)
    n = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), -np.sin(e)], -1)
    d = np.einsum('nji,nj->ni', R, n)  # R^T n
    return W / 2 + f * d[:, 1] / d[:, 0], H / 2 + f * d[:, 2] / d[:, 0]
