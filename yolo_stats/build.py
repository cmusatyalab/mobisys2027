"""Builds yolo_stats.csv and yolo_stats_per_class.csv from the logs in logs/
and the 1.2.4 pipeline metrics (pipeline_1.2.4.csv). Run: python3 build.py

Sources, all on fermi's HTP:
- accuracy: experiments/accuracy_prec.py (vartether-exps client) on
  summer_training val (235 images, 846 boxes: person, bicycle, car, truck,
  bus) through the gRPC inference-server, at confidence 0.001 and 0.5;
  matched at IoU 0.5. Overall precision counts only the five labeled
  classes. The gRPC server reads 80-class output only, so the coco81
  (COCO + robotic dog) models have no accuracy here.
- inference: experiments/latency.py on okutama_frames_300, the server's
  "inference" stage only (no JPEG decode, preprocessing or postprocessing).
- pipeline: lab 1.2.4-local-yolo26-sizes (parkinglot, YOLO on fermi, nothing
  sent, 3 trials): throughput, end-to-end latency, fermi's power, and the
  boxes each model reported (all classes, its confidence 0.5).
"""
import csv
import re
from pathlib import Path

HERE = Path(__file__).parent
LOGS = HERE / "logs"
CLASSES = ["person", "bicycle", "car", "truck", "bus"]


def parse_accuracy(text):
    """{(model, conf): {"overall": {...}, "<class>": {...}}} from summary text."""
    out, cur = {}, None
    for line in text.splitlines():
        line = re.sub(r"^.*INFO: +", "", line).strip()
        m = re.match(r"##### (?:accuracy )?(\S+) conf=(\S+)", line)
        if m:
            cur = out.setdefault((m.group(1), m.group(2)), {"overall": {}})
            continue
        if cur is None:
            continue
        m = re.match(r"(\w[\w ]*?)\s+AP50=([\d.na]+)\s+mAP50-95=([\d.na]+)"
                     r"(?:\s+recall@[\d.]+=([\d.na]+) \((\d+)/(\d+)\))?"
                     r"(?:\s+precision@[\d.]+=([\d.na]+) \((\d+)/(\d+)\))?", line)
        if m and m.group(1) in CLASSES:
            g = m.groups()
            cur[g[0]] = {"AP50": g[1], "mAP50_95": g[2], "recall": g[3], "tp": g[4],
                         "gt": g[5], "precision": g[6], "pred": g[8]}
            continue
        for key, pat in [("mAP50", r"Overall mAP50:\s+([\d.]+)"),
                         ("mAP50_95", r"Overall mAP50-95:\s+([\d.]+)"),
                         ("recall", r"Overall recall@[\d.]+:\s+([\d.]+)"),
                         ("precision", r"Overall precision@[\d.]+:\s+([\d.]+)"),
                         ("other_class_boxes", r"; (\d+) boxes of classes not in the labels")]:
            m = re.search(pat, line)
            if m:
                cur["overall"][key] = m.group(1)
    return out


def parse_inference(path):
    for line in path.read_text().splitlines():
        m = re.search(r"INFO:\s+inference\s+n=(\d+)\s+min=\s*([\d.]+)ms\s+mean=\s*([\d.]+)ms\s+"
                      r"stdev=\s*([\d.]+)ms\s+median=\s*([\d.]+)ms\s+p95=\s*([\d.]+)ms\s+"
                      r"p99=\s*([\d.]+)ms\s+max=\s*([\d.]+)ms", line)
        if m:
            k = ["n", "min", "mean", "stdev", "median", "p95", "p99", "max"]
            return dict(zip(k, m.groups()))
    return {}


acc = {}
for f in ["accuracy_summaries_2026-11-17.txt", "accuracy_summaries_x_2026-11-18.txt"]:
    acc.update(parse_accuracy((LOGS / f).read_text()))

inference = {p.name[len("latency_"):-len(".log")] + ".so": parse_inference(p)
             for p in LOGS.glob("latency_*.log")}

pipeline = list(csv.DictReader(open(HERE / "pipeline_1.2.4.csv")))

# Order: the current model, the stock COCO models by size, then coco81
order = ["base_s640", "stock_n", "stock_s", "stock_m", "stock_l", "stock_x", "stock_x640",
         "n", "s", "m", "l"]
pipeline.sort(key=lambda r: order.index(r["key"]))

cols = ["model", "family", "size", "input",
        "acc_conf0.001_mAP50", "acc_conf0.001_mAP50_95", "acc_conf0.001_precision", "acc_conf0.001_recall",
        "acc_conf0.001_person_AP50", "acc_conf0.001_person_precision", "acc_conf0.001_person_recall",
        "acc_conf0.5_mAP50", "acc_conf0.5_mAP50_95", "acc_conf0.5_precision", "acc_conf0.5_recall",
        "acc_conf0.5_person_AP50", "acc_conf0.5_person_precision", "acc_conf0.5_person_recall",
        "acc_conf0.5_other_class_boxes",
        "inference_mean_ms", "inference_median_ms", "inference_p95_ms", "inference_p99_ms",
        "inference_min_ms", "inference_max_ms", "inference_stdev_ms",
        "pipeline_throughput_fps", "pipeline_e2e_latency_mean_ms", "pipeline_e2e_latency_median_ms",
        "pipeline_e2e_latency_p95_ms", "pipeline_power_mean_w", "pipeline_power_median_w",
        "pipeline_power_p95_w", "pipeline_frames_processed", "pipeline_person_boxes",
        "pipeline_robotic_dog_boxes", "pipeline_all_boxes", "pipeline_trials", "notes"]

rows, per_class = [], []
for p in pipeline:
    model = p["model"]
    family = "coco81 (COCO + robotic dog)" if "coco81" in model else "stock COCO (80)"
    size = re.search(r"yolo26([nsmlx])|lib([nsmlx])640", model)
    size = size.group(1) or size.group(2)
    inp = "640x384" if "640x384" in model else "640x640"
    r = {"model": model, "family": family, "size": size, "input": inp}
    for conf in ("0.001", "0.5"):
        a = acc.get((model, conf))
        if not a:
            continue
        o, pe = a["overall"], a.get("person", {})
        r.update({f"acc_conf{conf}_mAP50": o.get("mAP50"), f"acc_conf{conf}_mAP50_95": o.get("mAP50_95"),
                  f"acc_conf{conf}_precision": o.get("precision"), f"acc_conf{conf}_recall": o.get("recall"),
                  f"acc_conf{conf}_person_AP50": pe.get("AP50"),
                  f"acc_conf{conf}_person_precision": pe.get("precision"),
                  f"acc_conf{conf}_person_recall": pe.get("recall")})
        if conf == "0.5":
            r["acc_conf0.5_other_class_boxes"] = o.get("other_class_boxes")
        for c in CLASSES:
            if c in a:
                per_class.append({"model": model, "confidence": conf, "class": c, **a[c]})
    i = inference.get(model, {})
    for k in ("mean", "median", "p95", "p99", "min", "max", "stdev"):
        r[f"inference_{k}_ms"] = i.get(k)
    for k_out, k_in in [("throughput_fps", "throughput_fps"), ("e2e_latency_mean_ms", "e2e_latency_mean_ms"),
                        ("e2e_latency_median_ms", "e2e_latency_median_ms"), ("e2e_latency_p95_ms", "e2e_latency_p95_ms"),
                        ("power_mean_w", "power_mean_w"), ("power_median_w", "power_median_w"),
                        ("power_p95_w", "power_p95_w"), ("frames_processed", "frames_processed"),
                        ("person_boxes", "person_boxes"), ("robotic_dog_boxes", "robotic_dog_boxes"),
                        ("all_boxes", "all_boxes"), ("trials", "trials")]:
        r[f"pipeline_{k_out}"] = p[k_in]
    notes = []
    if p["key"] == "base_s640":
        notes.append("the model the lab experiments use now (older build)")
    if "coco81" in model:
        notes.append("no accuracy or inference rows: the gRPC server reads 80-class output only; "
                     "same network as the stock model of its size plus one class")
    r["notes"] = "; ".join(notes)
    rows.append(r)

with open(HERE / "yolo_stats.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols)
    w.writeheader()
    w.writerows(rows)
with open(HERE / "yolo_stats_per_class.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["model", "confidence", "class", "AP50", "mAP50_95", "recall", "tp", "gt",
                           "precision", "pred"])
    w.writeheader()
    w.writerows(per_class)
print(f"{len(rows)} models, {len(per_class)} per-class rows")
