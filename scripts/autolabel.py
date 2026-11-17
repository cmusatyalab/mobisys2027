# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "rfdetr>=1.8.3",
#     "rfdetr-plus>=1.1.0",
#     "opencv-python-headless",
#     "numpy",
#     "torch",
#     "torchvision",
# ]
#
# # lambda2's NVIDIA driver is CUDA 12.8: take torch built for it
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# torchvision = { index = "pytorch-cu128" }
# ///
"""First-pass labels of a test video, for `uv run label` to correct.

Run it where the video and a GPU are (lambda2), with uv, which installs what
it needs in its own environment:

    uv run scripts/autolabel.py videos/parkinglot.mp4 -o labels/parkinglot.json

Every frame goes through RF-DETR (2XLarge by default), keeping its people.
The boxes are linked frame to frame into tracks by overlap, and a
vision-language model on Ollama (qwen3-vl by default, as the cliped verifier
uses) is shown a few crops of each track and asked whether it is a
mannequin or a real person, so the track is labeled "mannequin" or "person".
Tracks seen in fewer than --min-track frames are dropped as noise.

For a fine-tuned checkpoint, e.g. the robot dog model, give --weights, its
class names by id in --labels (a JSON list, as the detection plugin's
rfdetr_labels), the classes to keep in --keep, and --no-vlm to label each
track with the class the detector gave it most:

    uv run scripts/autolabel.py nov16_tracking_720p.mp4 -o labels/nov16_tracking.json \
        --weights dog_extended_2x.pth --labels dog_extended.json --keep "robotic dog" \
        --no-vlm --target "robotic dog"

To label one individual among people, give a photo of them in --reference:
the VLM is shown it with each track's crops and asked whether the track is
that person. Tracks it says yes to are labeled --target, the rest "person":

    uv run scripts/autolabel.py tunnel_720p.mp4 -o labels/tunnel_720p.json \
        --reference red_shirt.png --target "red shirt"

The file written is the labels/<name>.json format of mobisys2027/labels.py,
with boxes normalized to the frame.
"""

import argparse
import base64
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

QUESTION = (
    "This crop is from a drone video of a search and rescue test, where a mannequin "
    "is placed among real people. Is the figure in it a mannequin (a dummy) or a real, "
    "living person? Answer with one word: mannequin or person."
)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU tracker: each detection joins the track whose last box it
    overlaps most (IoU >= min_iou), if that track was seen within max_gap
    frames, or starts a new one."""

    def __init__(self, min_iou=0.2, max_gap=15):
        self.min_iou = min_iou
        self.max_gap = max_gap
        self.tracks = []  # each: {"boxes": {frame: (x0,y0,x1,y1,score)}, "last": frame}

    def update(self, frame, boxes):
        live = [t for t in self.tracks if frame - t["last"] <= self.max_gap]
        pairs = sorted(
            ((iou(t["boxes"][t["last"]], b), ti, bi) for ti, t in enumerate(live) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for score, ti, bi in pairs:
            if score < self.min_iou:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            live[ti]["boxes"][frame] = boxes[bi]
            live[ti]["last"] = frame
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                self.tracks.append({"boxes": {frame: b}, "last": frame})


REFERENCE_QUESTION = (
    "Image 1 is a reference photo of a person. Image 2 is a crop from a drone video, possibly from higher up "
    "and at lower resolution. Is the person in image 2 the same person as in image 1? Answer with only yes or no."
)


def chat(host, model, question, images, timeout):
    """The VLM's answer, lowercased, or None if it failed."""
    body = json.dumps({
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": question, "images": images}],
    }).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())["message"]["content"].strip().lower()
    except Exception as e:  # keep going: the user fixes the labels anyway
        print(f"  VLM failed: {e}", file=sys.stderr)
        return None


def encode(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def is_reference(crops, reference, host, model, timeout):
    """Whether the track whose crops these are is the person in `reference`
    (base64 JPEG), by majority of the crops; None if no answer came."""
    yes = no = 0
    for crop in crops:
        answer = chat(host, model, REFERENCE_QUESTION, [reference, encode(crop)], timeout)
        if answer is None:
            continue
        if answer.startswith("yes"):
            yes += 1
        else:
            no += 1
    return None if yes == no == 0 else yes > no


def ask_vlm(crops, host, model, timeout):
    """'mannequin' or 'person' by majority over the crops, or None."""
    votes = {"mannequin": 0, "person": 0}
    for crop in crops:
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        body = json.dumps({
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": QUESTION, "images": [base64.b64encode(buf.tobytes()).decode()]}],
        }).encode()
        req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                answer = json.loads(resp.read())["message"]["content"].strip().lower()
        except Exception as e:  # keep going: the user fixes the labels anyway
            print(f"  VLM failed: {e}", file=sys.stderr)
            continue
        if "mannequin" in answer or "dummy" in answer:
            votes["mannequin"] += 1
        elif "person" in answer or "human" in answer:
            votes["person"] += 1
    if votes["mannequin"] == votes["person"] == 0:
        return None
    return "mannequin" if votes["mannequin"] > votes["person"] else "person"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--size", default="2xlarge", help="RF-DETR size: nano ... 2xlarge")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--min-track", type=int, default=5, help="fewest frames a kept track is seen in")
    parser.add_argument("--crops", type=int, default=3, help="crops of each track shown to the VLM")
    parser.add_argument("--ollama", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--target", default="mannequin")
    parser.add_argument("--limit", type=int, default=0, help="only the first N frames (testing)")
    parser.add_argument("--weights", type=Path, help="an RF-DETR .pth checkpoint instead of the stock model of --size")
    parser.add_argument("--labels", type=Path, help="JSON list of the checkpoint's class names, by class id")
    parser.add_argument("--keep", nargs="+", default=["person"], help="the classes kept (default: person)")
    parser.add_argument("--reference", type=Path,
                        help="a photo of the --target individual: tracks the VLM says are them are --target, "
                             "the rest person")
    parser.add_argument("--no-vlm", action="store_true",
                        help="label each track with its most frequent class, instead of asking the VLM")
    args = parser.parse_args()

    import rfdetr
    import torch

    if not torch.cuda.is_available():
        print("warning: no CUDA; RF-DETR runs on the CPU, slowly", file=sys.stderr)
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES
    except ImportError:
        COCO_CLASSES = {1: "person"}

    classes = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge",
               "xlarge": "RFDETRXLarge", "2xlarge": "RFDETR2XLarge"}
    if args.weights:
        # The checkpoint records its own variant, resolution and classes
        model = rfdetr.RFDETR.from_checkpoint(str(args.weights), trust_checkpoint=True)
    else:
        model = getattr(rfdetr, classes[args.size])()
    labels = json.loads(args.labels.read_text()) if args.labels else None
    reference = None
    if args.reference:
        image = cv2.imread(str(args.reference))
        if image is None:
            raise SystemExit(f"cannot read {args.reference}")
        reference = encode(image)
    keep_classes = {k.lower() for k in args.keep}
    try:
        model.optimize_for_inference()
    except Exception as e:
        print(f"(not optimized for inference: {e})", file=sys.stderr)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = Tracker()
    keep = {}  # frame -> BGR image, only for frames the crops come from (filled in a second pass)
    n = 0
    started = time.time()
    while True:
        ok, bgr = cap.read()
        if not ok or (args.limit and n >= args.limit):
            break
        det = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), threshold=args.conf)
        names = det.data.get("class_name") if hasattr(det, "data") else None
        boxes = []
        for k in range(len(det)):
            class_id = int(det.class_id[k])
            if labels is not None and 0 <= class_id < len(labels):
                name = labels[class_id]
            else:
                name = names[k] if names is not None else COCO_CLASSES.get(class_id, str(class_id))
            if str(name).lower() not in keep_classes:
                continue
            x0, y0, x1, y1 = det.xyxy[k]
            boxes.append((x0 / width, y0 / height, x1 / width, y1 / height, float(det.confidence[k]), str(name)))
        tracker.update(n, boxes)
        n += 1
        if n % 300 == 0:
            print(f"{n} frames, {n / (time.time() - started):.1f} fps, {len(tracker.tracks)} tracks", file=sys.stderr)
    cap.release()

    tracks = [t for t in tracker.tracks if len(t["boxes"]) >= args.min_track]
    print(f"{n} frames; {len(tracks)} tracks of {args.min_track}+ frames", file=sys.stderr)

    # The crops each track shows the VLM: its most confident boxes, spread out
    wanted = {}
    for ti, t in enumerate(tracks if not args.no_vlm else []):
        frames = sorted(t["boxes"], key=lambda f: -t["boxes"][f][4])[: max(1, args.crops * 4)]
        frames = sorted(frames)[:: max(1, len(frames) // args.crops)][: args.crops]
        for f in frames:
            wanted.setdefault(f, []).append(ti)
    cap = cv2.VideoCapture(str(args.video))
    crops = {ti: [] for ti in range(len(tracks))}
    f = 0
    while wanted and f <= max(wanted):
        ok, bgr = cap.read()
        if not ok:
            break
        for ti in wanted.get(f, []):
            x0, y0, x1, y1 = tracks[ti]["boxes"][f][:4]
            pw, ph = (x1 - x0) * 0.3, (y1 - y0) * 0.3
            X0, Y0 = int(max(0, (x0 - pw) * width)), int(max(0, (y0 - ph) * height))
            X1, Y1 = int(min(width, (x1 + pw) * width)), int(min(height, (y1 + ph) * height))
            if X1 > X0 and Y1 > Y0:
                crops[ti].append(bgr[Y0:Y1, X0:X1].copy())
        f += 1
    cap.release()

    boxes = {}
    for ti, t in enumerate(tracks):
        if args.no_vlm:
            label = Counter(box[5] for box in t["boxes"].values()).most_common(1)[0][0]
        elif reference is not None:
            label = args.target if is_reference(crops[ti], reference, args.ollama, args.model, timeout=120) \
                else "person"
        else:
            label = ask_vlm(crops[ti], args.ollama, args.model, timeout=120) or "person"
        first, last = min(t["boxes"]), max(t["boxes"])
        print(f"track {ti}: frames {first}-{last} ({len(t['boxes'])} seen): {label}", file=sys.stderr)
        # Fill the gaps the detector missed inside the track by interpolation
        seen = sorted(t["boxes"])
        for a, b in zip(seen, seen[1:] + [None]):
            boxes.setdefault(a, []).append([label, *(round(float(v), 5) for v in t["boxes"][a][:4])])
            if b is None or b - a <= 1:
                continue
            for g in range(a + 1, b):
                w = (g - a) / (b - a)
                box = [t["boxes"][a][k] + (t["boxes"][b][k] - t["boxes"][a][k]) * w for k in range(4)]
                boxes.setdefault(g, []).append([label, *(round(float(v), 5) for v in box)])

    out = {
        "source": args.video.name,
        "kind": "video",
        "fps": fps,
        "frames": n,
        "width": width,
        "height": height,
        "target": args.target,
        "boxes": {str(f): b for f, b in sorted(boxes.items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    with_target = sum(1 for b in boxes.values() if any(x[0] == args.target for x in b))
    print(f"wrote {args.out}: {len(boxes)} frames with boxes, {with_target} with a {args.target}", file=sys.stderr)


if __name__ == "__main__":
    main()
