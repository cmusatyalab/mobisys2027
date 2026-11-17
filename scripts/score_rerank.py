"""Score box_rerank against YOLO alone, from the logs of a run.

    uv run python scripts/score_rerank.py lab/2.2-rerank/<run dir> [...] --labels beagle_tracking_720p

Each argument is a directory searched (recursively) for box_rerank's
candidates.csv and fake_drone's frames.csv of one trial.

Every frame box_rerank decided is graded on the box it got. Frames the
pipeline skipped while busy never reach it: Uriel skips them at the engines,
so fake_drone's frames.csv (which maps frame ids to video frames) gives every
frame a sequence number either way. The frames it decided with candidates are
in candidates.csv; those where YOLO found nothing are only counted, in
box_rerank's 30 s summaries in its log (the box_rerank_log catalog), and got
no box from either.

- YOLO alone: its most confident box, if at least --yolo-conf (0.5, as the
  onboard tracker uses it);
- box_rerank: the box it reported, if any.

A box is correct if its IoU with the labeled target is at least --iou; otherwise it
is a false positive, put down to the labeled distractor it overlaps most
(bag, chair), or "other".

Throughput is the frames decided per second of video. Latency is
approximate: the flow lets one camera frame in at a time, so the gap between
consecutive decided frames is about one frame's time through YOLO and
box_rerank (plus up to one camera interval, 33 ms, waiting for the next
frame; a skipped frame with no candidates widens a gap too). box_rerank's own
time per frame is from candidates.csv.

Also counted: frames where box_rerank vetoed the target (a no learned on the
dog, the failure that matters most), and when each distractor was first
vetoed.
"""

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from mobisys2027 import labels as L

WHO = ("yolo", "rerank")


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


SUMMARY = re.compile(r"last \d+ s: (\d+) frames,")


def summarized_frames(trial_dir):
    """The frames box_rerank decided, from the summaries in its log, or None
    without one. The last summary may be missing a few seconds of frames."""
    log = next(trial_dir.rglob("*box_rerank*.log"), None)
    if log is None:
        return None
    return sum(int(m.group(1)) for m in SUMMARY.finditer(log.read_text(errors="replace")))


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else float("nan")


def score(trial_dir, labels, args):
    candidates = next(trial_dir.rglob("candidates.csv"), None)
    frames = next(trial_dir.rglob("frames.csv"), None)
    if candidates is None or frames is None:
        return None
    by_frame = defaultdict(list)
    for row in read_csv(candidates):
        by_frame[int(row["frame_id"])].append(row)
    source = {int(r["frame_id"]): int(r["source"]) for r in read_csv(frames) if r.get("source", "").isdigit()}
    # The frames decided with candidates: (frame id, index in the video, capture time)
    through = sorted(((f, source.get(f, f), int(rows[0]["timestamp_ns"])) for f, rows in by_frame.items()),
                     key=lambda t: t[2])
    # Every frame decided, from box_rerank's summaries; the ones not in
    # candidates.csv had no candidates
    decided = summarized_frames(trial_dir)
    empty = max(0, decided - len(through)) if decided is not None else 0

    target = labels.target
    counts = {who: Counter() for who in WHO}
    dog_vetoed, first_veto = 0, {}
    t0 = through[0][2] if through else 0
    for frame_id, index, stamp in through:
        truth = labels.frame_boxes(index)
        has_target = any(b[0] == target for b in truth)
        rows = by_frame.get(frame_id, [])
        picks = dict.fromkeys(WHO)
        boxes = {}
        if rows:
            w, h = int(rows[0]["frame_width"]), int(rows[0]["frame_height"])
            boxes = {int(r["candidate"]): (int(r["left"]) / w, int(r["top"]) / h, int(r["right"]) / w,
                                           int(r["bottom"]) / h) for r in rows}
            top = min(rows, key=lambda r: int(r["candidate"]))
            if float(top["score"]) >= args.yolo_conf:
                picks["yolo"] = boxes[int(top["candidate"])]
            picks["rerank"] = next((boxes[int(r["candidate"])] for r in rows if r["reported"] == "1"), None)
        for who, box in picks.items():
            c = counts[who]
            c["frames"] += 1
            c["target frames"] += has_target
            if box is None:
                c["none"] += 1
                continue
            what = classify(box, truth, target, args.iou)
            c["reported"] += 1
            c["correct" if what == "correct" else "fp"] += 1
            if what != "correct":
                c[f"fp {what}"] += 1
        for r in rows:
            if r["verdict"] != "-1":
                continue
            what = classify(boxes[int(r["candidate"])], truth, target, args.iou)
            if what == "correct":
                dog_vetoed += 1
            else:
                first_veto.setdefault(what, (stamp - t0) / 1e9)

    for who in WHO:
        counts[who]["frames"] += empty
        counts[who]["none"] += empty
    gaps_ms = [(b[2] - a[2]) / 1e6 for a, b in zip(through, through[1:])]
    rerank_ms = [float(rows[0]["ms"]) for rows in by_frame.values()]
    timing = {
        "through": len(through) + empty,
        "empty": empty if decided is not None else None,
        "fps": (len(through) + empty) / ((labels.frames - 1) / labels.fps),
        "gap p50": percentile(gaps_ms, 0.5),
        "gap p95": percentile(gaps_ms, 0.95),
        "rerank p50": percentile(rerank_ms, 0.5),
        "rerank p95": percentile(rerank_ms, 0.95),
    }
    return counts, timing, dog_vetoed, first_veto


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--labels", default="beagle_tracking_720p")
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    args = parser.parse_args()
    labels = L.load(args.labels)
    for trial in args.trials:
        result = score(trial, labels, args)
        if result is None:
            print(f"{trial}: no candidates.csv or frames.csv")
            continue
        counts, t, dog_vetoed, first_veto = result
        empty = "no box_rerank log: frames with no candidates not counted" if t["empty"] is None \
            else f"{t['empty']} with no candidates"
        print(f"{trial}: {t['through']} of {labels.frames} frames decided ({empty}), {t['fps']:.1f} fps | "
              f"latency ~{t['gap p50']:.0f} ms (p95 {t['gap p95']:.0f}), of which box_rerank "
              f"{t['rerank p50']:.0f} ms (p95 {t['rerank p95']:.0f})")
        for who in WHO:
            c = counts[who]
            fps = ", ".join(f"{k[3:]} {v}" for k, v in sorted(c.items()) if k.startswith("fp "))
            precision = c["correct"] / c["reported"] if c["reported"] else float("nan")
            recall = c["correct"] / c["target frames"] if c["target frames"] else float("nan")
            print(f"  {who:7s} boxes {c['reported']:5d}: correct {c['correct']:5d}, false positives "
                  f"{c['fp']:4d} ({fps or 'none'}) | no box {c['none']:4d} | precision {precision:.3f}, "
                  f"recall {recall:.3f}")
        vetoes = ", ".join(f"{k} at {s:.1f} s" for k, s in sorted(first_veto.items(), key=lambda kv: kv[1]))
        print(f"  first vetoed: {vetoes or 'nothing'} | the target vetoed in {dog_vetoed} frames")


if __name__ == "__main__":
    main()
