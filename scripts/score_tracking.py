"""Score tracking runs from detection_log's box logs: per method and profile,
precision, recall, throughput and latency.

    uv run python scripts/score_tracking.py lab/2.2-rerank/runs/*/*/trial_1 [--labels beagle_tracking_720p]

Each argument is a trial directory, holding detection_log's boxes.csv (in
result/<daemon>/) and fake_drone's frames.csv, which maps frame ids to the
video's frames.

Every frame the method decided (logged by detection_log) is graded on the box
it reported: its best-scoring box of the target's class, if any. Frames the
pipeline skipped while busy are never decided. The box graded is the best of
the class --class (by default the labels' target, e.g. "robotic dog"). A box is correct if its IoU
with the labeled target is at least --iou; otherwise it is a false positive,
put down to the labeled distractor it overlaps most (bag, chair), or "other".

- precision: correct / boxes reported;
- recall: correct / decided frames with the target;
- coverage: correct / every frame of the video with the target, so frames
  skipped count as missed;
- throughput: frames decided per second of video;
- latency: from capture (the frame's timestamp, on the vehicle) to the
  result reaching detection_log. A log on another host than the vehicle's is
  corrected by --clock-offset-ms (vehicle clock minus that host's).
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from mobisys2027 import labels as L

VEHICLE = "heisenberg"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def classify(box, truth, target, threshold):
    """'correct', the distractor class it hits, or 'other'."""
    best, best_iou = None, 0.0
    for cls, *gt in truth:
        overlap = iou(box, gt)
        if overlap > best_iou:
            best, best_iou = cls, overlap
    if best_iou < threshold:
        return "other"
    return "correct" if best == target else best


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else float("nan")


def score(trial, labels, args):
    boxes_csv = next(trial.rglob("boxes.csv"), None)
    frames_csv = next(trial.rglob("frames.csv"), None)
    if boxes_csv is None or frames_csv is None:
        return None
    # The log's host is the result directory it was collected into
    offset_ns = 0 if boxes_csv.parent.name == VEHICLE else args.clock_offset_ms * 1e6
    source = {r["frame_id"]: int(r["source"]) for r in read_csv(frames_csv) if r["source"].isdigit()}
    by_frame = defaultdict(list)
    for row in read_csv(boxes_csv):
        by_frame[row["frame_id"]].append(row)

    target = labels.target
    c = Counter()
    latency_ms = []
    for frame_id, rows in by_frame.items():
        if frame_id not in source:
            c["unmapped"] += 1
            continue
        truth = labels.frame_boxes(source[frame_id])
        c["frames"] += 1
        c["target frames"] += any(b[0] == target for b in truth)
        head = rows[0]
        if head["timestamp_ns"] and head["result_ns"]:
            latency_ms.append((int(head["result_ns"]) + offset_ns - int(head["timestamp_ns"])) / 1e6)
        dets = [r for r in rows if r["class"] == (args.cls or target)]
        if not dets:
            c["none"] += 1
            continue
        best = max(dets, key=lambda r: float(r["score"]))
        box = tuple(float(best[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        what = classify(box, truth, target, args.iou)
        c["reported"] += 1
        if what == "correct":
            c["correct"] += 1
        else:
            c["fp"] += 1
            c[f"fp {what}"] += 1

    video_s = (labels.frames - 1) / labels.fps
    all_target = sum(any(b[0] == target for b in labels.frame_boxes(i)) for i in range(labels.frames))
    return {
        "counts": c,
        "precision": c["correct"] / c["reported"] if c["reported"] else float("nan"),
        "recall": c["correct"] / c["target frames"] if c["target frames"] else float("nan"),
        "coverage": c["correct"] / all_target if all_target else float("nan"),
        "fps": c["frames"] / video_s,
        "lat p50": percentile(latency_ms, 0.5),
        "lat p95": percentile(latency_ms, 0.95),
        "log": boxes_csv,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--labels", default="beagle_tracking_720p")
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--clock-offset-ms", type=float, default=12.0,
                        help="the vehicle's clock minus the other host's (heisenberg - lambda was +12 ms)")
    parser.add_argument("--class", dest="cls", help="the class graded (default: the labels' target)")
    parser.add_argument("--csv", type=Path, help="also write a row per trial to this CSV file")
    args = parser.parse_args()
    labels = L.load(args.labels)

    rows = []
    print(f"{'trial':44s} {'frames':>6s} {'fps':>5s} {'lat p50':>8s} {'p95':>6s} {'boxes':>6s} {'FP':>5s} "
          f"{'precision':>9s} {'recall':>6s} {'coverage':>8s}  false positives")
    for trial in args.trials:
        r = score(trial, labels, args)
        name = "/".join(trial.parts[-3:-1]) if trial.name.startswith("trial") else str(trial)
        if r is None:
            print(f"{name:44s} no boxes.csv or frames.csv")
            continue
        c = r["counts"]
        fps_by = ", ".join(f"{k[3:]} {v}" for k, v in sorted(c.items()) if k.startswith("fp ")) or "none"
        print(f"{name:44s} {c['frames']:6d} {r['fps']:5.1f} {r['lat p50']:6.0f}ms {r['lat p95']:4.0f}ms "
              f"{c['reported']:6d} {c['fp']:5d} {r['precision']:9.3f} {r['recall']:6.3f} {r['coverage']:8.3f}  {fps_by}")
        rows.append({"trial": name, "frames": c["frames"], "fps": round(r["fps"], 2),
                     "latency_p50_ms": round(r["lat p50"], 1), "latency_p95_ms": round(r["lat p95"], 1),
                     "boxes": c["reported"], "correct": c["correct"], "false_positives": c["fp"],
                     "precision": round(r["precision"], 4), "recall": round(r["recall"], 4),
                     "coverage": round(r["coverage"], 4), "fp_breakdown": fps_by})
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
