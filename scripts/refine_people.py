# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy"]
# ///
"""Second pass over autolabel's people labels of a video, for a target picked
out by the color of their clothes (e.g. the one person in a red shirt):

    uv run scripts/refine_people.py videos/ric_back_720p.mp4 labels/ric_back_720p.json \\
        --target "red shirt" --hue red

Boxes are linked into runs (frame to frame by overlap, as autolabel's tracks).
For each run, a vision-language model on Ollama is shown a few crops and asked
whether the boxed figure is a real person outdoors: runs it says are
reflections in glass, statues, machines or other objects are dropped. A run
whose upper body is mostly of --hue (saturated) is labeled --target, without
asking: a reflection or a shadow is never that saturated. The rest are
"person". Writes the
labels back in place (or to -o), and prints each run's verdict to check.
"""

import argparse
import base64
import json
import sys
import urllib.request

import cv2
import numpy as np

REAL_QUESTION = (
    "This crop is from a drone video. Is the figure inside the red box a real person, standing or walking "
    "outdoors? Answer no if it is a reflection of a person in a window or glass, a statue, a machine, or any "
    "other object. Answer with only yes or no."
)
# Hue ranges (OpenCV, 0-180) of the target's clothes
HUES = {"red": [(0, 10), (170, 180)], "blue": [(100, 130)], "green": [(40, 80)], "yellow": [(20, 35)]}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def runs_of(boxes, max_gap=15, min_iou=0.1):
    """Each box's run: a list of runs, each {frame: box index}."""
    runs, last = [], []
    for f in sorted(boxes, key=int):
        fi = int(f)
        taken = set()
        for k, (_, *box) in enumerate(boxes[f]):
            best, best_iou = None, min_iou
            for ri, (lf, lbox) in enumerate(last):
                if ri in taken or fi - lf > max_gap or lf == fi:
                    continue
                o = iou(lbox, box)
                if o > best_iou:
                    best, best_iou = ri, o
            if best is None:
                runs.append({})
                last.append(None)
                best = len(runs) - 1
            taken.add(best)
            runs[best][fi] = k
            last[best] = (fi, box)
    return runs


def ask(host, model, crop, timeout=120):
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": REAL_QUESTION,
                                     "images": [base64.b64encode(buf.tobytes()).decode()]}]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["message"]["content"].strip().lower().startswith("yes")
    except Exception as e:
        print(f"  VLM failed: {e}", file=sys.stderr)
        return None


def hue_fraction(bgr, box, W, H, hues):
    """Fraction of the box's upper body (its 15-55% band) that is saturated
    and of one of `hues`."""
    x0, y0, x1, y1 = box
    X0, X1 = int(x0 * W), max(int(x0 * W) + 1, int(x1 * W))
    h = (y1 - y0) * H
    Y0, Y1 = int(y0 * H + 0.15 * h), max(int(y0 * H + 0.15 * h) + 1, int(y0 * H + 0.55 * h))
    patch = cv2.cvtColor(bgr[Y0:Y1, X0:X1], cv2.COLOR_BGR2HSV)
    if patch.size == 0:
        return 0.0
    hue, sat, val = patch[..., 0], patch[..., 1], patch[..., 2]
    mask = np.zeros(hue.shape, bool)
    for lo, hi in hues:
        mask |= (hue >= lo) & (hue <= hi)
    mask &= (sat >= 100) & (val >= 60)
    return float(mask.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("labels")
    parser.add_argument("-o", "--out")
    parser.add_argument("--target", required=True)
    parser.add_argument("--hue", choices=sorted(HUES), required=True)
    parser.add_argument("--min-fraction", type=float, default=0.12,
                        help="least fraction of the upper body of --hue, on average, for --target")
    parser.add_argument("--crops", type=int, default=3)
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    args = parser.parse_args()

    d = json.load(open(args.labels))
    W, H = d["width"], d["height"]
    boxes = d["boxes"]
    runs = runs_of(boxes)
    # Every frame of a run is measured for color; a few are shown to the VLM
    shown = {}
    for ri, run in enumerate(runs):
        fs = sorted(run)
        for f in sorted({fs[int(i * (len(fs) - 1) / max(1, args.crops - 1))] for i in range(args.crops)}):
            shown.setdefault(f, []).append(ri)
    fractions = {ri: [] for ri in range(len(runs))}
    crops = {ri: [] for ri in range(len(runs))}
    by_frame = {}
    for ri, run in enumerate(runs):
        for f, k in run.items():
            by_frame.setdefault(f, []).append((ri, k))
    cap = cv2.VideoCapture(args.video)
    f = 0
    last = max(by_frame) if by_frame else -1
    while f <= last:
        ok, bgr = cap.read()
        if not ok:
            break
        for ri, k in by_frame.get(f, []):
            box = boxes[str(f)][k][1:]
            fractions[ri].append(hue_fraction(bgr, box, W, H, HUES[args.hue]))
            if ri in shown.get(f, []):
                x0, y0, x1, y1 = box
                cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
                # Wide enough to show what the figure stands in front of
                s = max((x1 - x0) * W, (y1 - y0) * H) * 3 + 24
                X0, Y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
                marked = bgr.copy()
                cv2.rectangle(marked, (int(x0 * W) - 2, int(y0 * H) - 2), (int(x1 * W) + 2, int(y1 * H) + 2),
                              (0, 0, 255), 2)
                crop = marked[Y0:int(min(H, cy + s / 2)), X0:int(min(W, cx + s / 2))]
                if crop.size:
                    scale = max(1.0, 224 / min(crop.shape[:2]))
                    crops[ri].append(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
        f += 1
    cap.release()

    labels = {}
    for ri, run in enumerate(runs):
        frac = float(np.mean(fractions[ri])) if fractions[ri] else 0.0
        if frac >= args.min_fraction:
            votes, label = [], args.target
        else:
            votes = [v for v in (ask(args.ollama, args.model, c) for c in crops[ri]) if v is not None]
            label = "person" if not votes or sum(votes) * 2 > len(votes) else None
        labels[ri] = label
        fs = sorted(run)
        print(f"run {ri}: frames {fs[0]}-{fs[-1]} ({len(fs)}): real {sum(votes)}/{len(votes)}, "
              f"{args.hue} {frac:.2f} -> {label or 'dropped'}", file=sys.stderr)

    out = {}
    for ri, run in enumerate(runs):
        if labels[ri] is None:
            continue
        for f, k in run.items():
            out.setdefault(str(f), []).append([labels[ri], *boxes[str(f)][k][1:]])
    d["boxes"] = {f: out[f] for f in sorted(out, key=int)}
    json.dump(d, open(args.out or args.labels, "w"))
    n_target = sum(1 for b in out.values() if any(x[0] == args.target for x in b))
    print(f"wrote {args.out or args.labels}: {len(out)} frames with boxes, {n_target} with a {args.target}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
