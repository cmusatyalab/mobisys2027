"""One row per SwiftMap trial, from what the runner saved in its directory.

    python stats.py [runs/swiftmap] [--vehicle]

Reads each <profile>/trial_<n>/ (swiftmap_server.log, netsim.log, and the
profile's stats_<profile>_swiftmap.csv) and writes summary.csv next to the
profile directories. Times are seconds from the start of network shaping,
which starts with the flow. keyframes counts every keyframe the server took,
the batch's and any after it. With --vehicle, it also reads gs25456's
framestore log (reachback*.log before the rename), fetched into
runs/swiftmap/vehicle/ by fetch_vehicle_logs.sh.
"""
import csv
import glob
import os
import re
import sys
from datetime import datetime, timedelta

TRANSPORT = re.compile(r"^(\d\d:\d\d:\d\d) \[transport\] (.*)")
COLUMNS = [
    "profile", "trial", "trial_s", "bandwidth_mean_kbps", "time_off_s", "bytes_delivered",
    "first_keyframe_s", "keyframes", "batch", "batch_frames", "batch_closed_s",
    "skipped_while_held", "viewpoints", "sendback_first_s", "sent_back", "sendback_done_s",
    "reachback_cut", "plan_s", "sendback_s", "server_stopped_s", "points",
    "inference_s", "reconstruct_s", "gps_align_s", "nfn_plan_s",
    "vehicle_hold_s", "vehicle_request", "vehicle_fps",
]


def shaping_start(trial_dir):
    """Wall-clock time network shaping started: netsim.log's last write minus
    its last logged offset."""
    path = os.path.join(trial_dir, "netsim.log")
    offsets = re.findall(r"^\[\s*([\d.]+)s\]", open(path).read(), re.M)
    end = datetime.fromtimestamp(os.path.getmtime(path))
    return end - timedelta(seconds=float(offsets[-1]) if offsets else 0.0)


def seconds(clock, start):
    """Seconds from start to an HH:MM:SS clock time on the same day (or the next)."""
    t = datetime.combine(start.date(), datetime.strptime(clock, "%H:%M:%S").time())
    if t < start - timedelta(hours=12):
        t += timedelta(days=1)
    return round((t - start).total_seconds(), 1)


def server_stats(trial_dir, start):
    """Milestones and timings from swiftmap_server.log."""
    row = {"keyframes": 0, "skipped_while_held": 0, "sent_back": 0, "reachback_cut": False}
    timings = {"inference_s": [], "reconstruct_s": [], "gps_align_s": [], "nfn_plan_s": []}
    for line in open(os.path.join(trial_dir, "swiftmap_server.log"), errors="replace"):
        for key, pattern in (("inference_s", r"Inference time: ([\d.]+)s"),
                             ("reconstruct_s", r"\] processing completed in ([\d.]+)s"),
                             ("gps_align_s", r"\[GPS\] alignment completed in ([\d.]+)s"),
                             ("nfn_plan_s", r"\[NFN\] .* in ([\d.]+)s")):
            m = re.search(pattern, line)
            if m:
                timings[key].append(m.group(1))
        m = re.search(r"\[NFN\] (\d+) viewpoints", line)
        if m and "viewpoints" not in row:
            row["viewpoints"] = int(m.group(1))
        m = re.search(r"grew with .*: \d+ maps, (\d+) points", line)
        if m:
            row["points"] = int(m.group(1))
        m = TRANSPORT.match(line)
        if not m:
            continue
        t, msg = seconds(m.group(1), start), m.group(2)
        if msg.startswith("keyframe ") and "(batch" in msg:
            row["keyframes"] += 1
            row.setdefault("first_keyframe_s", t)
        elif msg.endswith("skipped: NFN round running"):
            row["skipped_while_held"] += 1
        elif msg.startswith(("batch full:", "flushed ")) and "batch" not in row:
            row["batch"] = "full" if msg.startswith("batch full") else "flushed"
            row["batch_frames"] = int(re.search(r"(\d+) (keyframes|frame)", msg).group(1))
            row["batch_closed_s"] = t
        elif msg.startswith("reachback frame "):
            row["sent_back"] += 1
            row.setdefault("sendback_first_s", t)
            row["sendback_done_s"] = t
        elif msg.startswith("reachback done:"):
            row["sendback_done_s"] = t
        elif msg.startswith("reachback not done"):
            row["reachback_cut"] = True
            row["sendback_done_s"] = t
        elif msg == "stopping":
            row["server_stopped_s"] = t
    row.update({k: ";".join(v) for k, v in timings.items()})
    if "batch_closed_s" in row and "sendback_first_s" in row:
        row["plan_s"] = round(row["sendback_first_s"] - row["batch_closed_s"], 1)
        row["sendback_s"] = round(row["sendback_done_s"] - row["sendback_first_s"], 1)
    return row


def trial_end(trial_dir):
    """Wall-clock time the trial was collected (its last file write)."""
    return datetime.fromtimestamp(max(os.path.getmtime(os.path.join(trial_dir, f))
                                      for f in os.listdir(trial_dir)))


def vehicle_stats(runs, trial_dir, start):
    """When the framestore held, what it was asked for and the frame rate it
    saw, from the trial's window of runs/vehicle/framestore*.log and reachback*.log."""
    paths = sorted(p for name in ("framestore*.log", "reachback*.log")
                   for p in glob.glob(os.path.join(runs, "vehicle", name)))
    if not paths:
        return {}
    end = (trial_end(trial_dir) - start).total_seconds()
    row, fps = {}, []
    lines = sorted(set(line for p in paths for line in open(p, errors="replace")))
    for line in lines:
        m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        if not m:
            continue
        t = round((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") - start).total_seconds(), 1)
        if not 0 <= t <= end:
            continue
        if "holding keyframes" in line:
            row.setdefault("vehicle_hold_s", t)
        m2 = re.search(r"sending (\d+) frames back for (\d+) targets", line)
        if m2:
            row.setdefault("vehicle_request", f"{m2.group(1)}/{m2.group(2)}")
        m2 = re.search(r"([\d.]+) fps;", line)
        if m2 and t > 5:
            fps.append(float(m2.group(1)))
    if fps:
        row["vehicle_fps"] = round(sum(fps) / len(fps), 2)
    return row


def network_stats(profile_dir, profile, trial):
    """The runner's netsim summary row for this trial."""
    path = os.path.join(profile_dir, f"stats_{profile}_swiftmap.csv")
    if not os.path.exists(path):
        return {}
    for r in csv.DictReader(open(path)):
        if r["trial"] == str(trial):
            return {"trial_s": r["duration_sec"], "bandwidth_mean_kbps": r["bandwidth_mean_kbps"],
                    "time_off_s": r["time_off_sec"], "bytes_delivered": r["total_bytes_delivered"]}
    return {}


def main(runs, vehicle):
    rows = []
    for profile in sorted(os.listdir(runs)):
        profile_dir = os.path.join(runs, profile)
        if not os.path.isdir(profile_dir):
            continue
        for name in sorted(os.listdir(profile_dir)):
            trial_dir = os.path.join(profile_dir, name)
            m = re.fullmatch(r"trial_(\d+)", name)
            if not m or not os.path.exists(os.path.join(trial_dir, "swiftmap_server.log")):
                continue
            start = shaping_start(trial_dir)
            row = {"profile": profile, "trial": int(m.group(1))}
            row.update(network_stats(profile_dir, profile, row["trial"]))
            row.update(server_stats(trial_dir, start))
            if vehicle:
                row.update(vehicle_stats(runs, trial_dir, start))
            rows.append(row)
    out = os.path.join(runs, "summary.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} trial(s) -> {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "swiftmap"),
         "--vehicle" in sys.argv)
