"""Turns the runs in lab/ into the tables and figures of plan.md.

    uv run analyze                 # every section, from lab/**/runs
    uv run analyze --quality FILE  # 1.1-quality's numbers from elsewhere

It writes, in each top-level section's lab/<section>/results/ (1.1's figures
come from lab/1.1-quality/results/1.1-quality.csv, the numbers of the sheet):

- one CSV per top-level section of the plan (1.1-quality.csv,
  1.2-microbenchmarks.csv, 2.1-search.csv), each a tab of the sheet, with
  its subsections one after another, each under its title;
- figs/: each figure as a PDF (vector, Computer Modern, sized for a column)
  and as <name>.tex, the same chart in pgfplots, to \\input into the paper.

The numbers of a run, per trial and then averaged over its trials (with the
standard deviation in the _sd columns):

- throughput_fps: frames reaching the end of the pipeline per second, over
  the span they arrive in;
- latency_*_ms: when a frame arrives at the end of the pipeline, minus when
  the fake drone captured it (its frame log; clocks synced over the tailnet);
- power_*_w: fermi's power draw over the trial;
- bytes: the bytes of the frames that reach lambda2 (rx_metrics, as sent,
  before decoding), each one once; frac_1mbps is that over what a 1 Mbps
  stream sends over the video's length;
- uot: utility of transmission, the share of those bytes that were relevant
  (a frame, crop or tile holding a target);
- precision: the share of the distinct frames, crops or tiles that reached
  lambda2 holding a target; recall: the share of the frames with a target
  (labels/<source>.json) for which one did. A crop or tile holds a target
  if it covers at least OVERLAP of the target's box, or the target covers
  that much of it. Which classes are targets depends on the experiment
  (lab.py's targets: anyone for YOLO and "person", only the mannequin or the
  onager for an image prompt);
- verified_precision and verified_recall: the same for what the verifier on
  lambda2 passed on (sink_metrics), when there is one;
- effective_recall (and verified_effective_recall): recall over only the
  frames the pipeline processed. The fake drone sends a frame only when the
  pipeline has a free input (its frame log's seq); a frame sent while it was
  busy is dropped there, unseen. The gap between recall and
  effective_recall is the targets the pipeline never saw; what
  effective_recall misses, it saw and lost (filtered out, or lost on the way);
- processed_frac: the share of the source's frames the pipeline took in.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from mobisys2027 import lab as LAB
from mobisys2027 import labels as L

LAB_ROOT = L.REPO_ROOT / "lab"
TOPS = ("1.1-quality", "1.2-microbenchmarks", "2.1-search")


def results_dir(top: str) -> Path:
    """Where a top-level section's CSV and figures go: lab/<top>/results/."""
    return LAB_ROOT / top / "results"


def fig_path(name: str, ext: str) -> Path:
    """results/figs/<name>.<ext> of the section the figure belongs to (its
    name starts with the section's number, e.g. 2.1.1.2-...)."""
    top = next((t for t in TOPS if name.startswith(t.split("-")[0]) and name[len(t.split("-")[0])] in ".-"),
               TOPS[-1])
    d = results_dir(top) / "figs"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.{ext}"
OVERLAP = 0.3
MBPS_BYTES = 1_000_000 / 8

SECTIONS = {
    "1.1-quality": "1.1 quality",
    "1.2-microbenchmarks": "1.2 microbenchmarks",
    "2.1-search": "2.1 search",
}


# ---- reading a trial ------------------------------------------------------------------


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find(trial: Path, name: str) -> Path | None:
    """The file `name` the trial collected, from whichever daemon."""
    for p in (trial / "result").glob(f"*/{name}"):
        return p
    return None


def pct(values, q):
    if not values:
        return math.nan
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def mean(values):
    return statistics.fmean(values) if values else math.nan


@dataclass
class Item:
    """A frame, crop or tile that reached a point of the pipeline."""

    frame: int | None  # source frame (video frame id or image index)
    box: tuple | None  # (x0, y0, x1, y1) normalized, or None for the whole frame
    bytes: int
    recv_ns: int
    capture_ns: int | None


class Trial:
    def __init__(self, path: Path):
        self.path = path
        frames = read_csv(find(path, "frames.csv") or Path("-"))
        # capture timestamp -> (source frame, capture ns); seq -> the same
        self.by_ts: dict[int, tuple[int, int]] = {}
        self.by_seq: dict[int, tuple[int, int]] = {}
        # Source frames the pipeline took in (sent with a seq), and all played
        self.processed: set[int] = set()
        self.played: set[int] = set()
        for r in frames:
            if r.get("source") in ("pad", None, ""):
                continue
            src, ts = int(r["source"]), int(r["timestamp_ns"])
            self.by_ts[ts] = (src, ts)
            self.played.add(src)
            if r.get("seq"):
                self.by_seq[int(r["seq"])] = (src, ts)
                self.processed.add(src)
        # What was cut and sent: timestamp (and tile) -> box; patchcore: timestamp -> seq
        self.sent_box: dict[int, list[tuple]] = {}
        self.sent_seq: dict[int, int] = {}
        for r in read_csv(find(path, "sent.csv") or Path("-")):
            ts = int(r["timestamp_ns"])
            if "left" in r:
                w, h = float(r["frame_width"]), float(r["frame_height"])
                box = (float(r["left"]) / w, float(r["top"]) / h, float(r["right"]) / w, float(r["bottom"]) / h)
                self.sent_box.setdefault(ts, []).append(box)
            elif "seq" in r:
                self.sent_seq[ts] = int(r["seq"])
        self.power = [float(r["power_w"]) for r in read_csv(next(path.glob("power_*.csv"), Path("-")))
                      if r.get("power_w")]

    def items(self, name: str) -> list[Item]:
        """The frames the metrics engine logging to `name` saw."""
        out = []
        for r in read_csv(find(self.path, name) or Path("-")):
            ts = int(r["send_ts_ns"])
            src = self.by_ts.get(ts)
            if src is None and ts in self.sent_seq:
                src = self.by_seq.get(self.sent_seq[ts])
            # cliped sends one tile per frame (max_send 1), detection-crop one crop
            boxes = self.sent_box.get(ts)
            box = boxes[0] if boxes else None
            out.append(Item(src[0] if src else None, box, int(r["bytes"]), int(r["recv_ts_ns"]),
                            src[1] if src else None))
        return out


# ---- scoring -----------------------------------------------------------------------------


def holds_target(item: Item, labels: L.Labels, targets) -> bool:
    if item.frame is None:
        return False
    boxes = [b for b in labels.frame_boxes(item.frame) if b[0] in targets]
    if not boxes:
        return False
    if item.box is None:
        return True
    return any(L.overlap(item.box, b[1:]) >= OVERLAP for b in boxes)


def score(items: list[Item], labels: L.Labels | None, targets, processed: set[int] | None = None) -> dict:
    """precision, recall and uot of `items`, each distinct one once; with
    `processed` (the source frames the pipeline took in), effective_recall:
    the recall over only those."""
    if labels is None or not targets:
        return {}
    seen, uniq = set(), []
    for it in items:
        key = (it.frame, it.box)
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    hits = [holds_target(it, labels, targets) for it in uniq]
    target_frames = {f for f in labels.boxes if any(b[0] in targets for b in labels.boxes[f])}
    found = {it.frame for it, hit in zip(uniq, hits) if hit}
    total = sum(it.bytes for it in uniq)
    seen_targets = target_frames & processed if processed else set()
    return {
        "effective_recall": len(found & seen_targets) / len(seen_targets) if seen_targets else math.nan,
        "precision": sum(hits) / len(uniq) if uniq else math.nan,
        "recall": len(found & target_frames) / len(target_frames) if target_frames else math.nan,
        "uot": sum(it.bytes for it, hit in zip(uniq, hits) if hit) / total if total else math.nan,
        "items": len(uniq),
        "tp_items": sum(hits),
    }


def trial_metrics(trial: Trial, labels: L.Labels | None, targets, video_s: float | None) -> dict:
    rx = trial.items("rx.csv")
    sink = trial.items("sink.csv")
    local = trial.items("local.csv")
    end = sink or rx or local
    m: dict = {}
    if end:
        span = (max(i.recv_ns for i in end) - min(i.recv_ns for i in end)) / 1e9
        m["throughput_fps"] = (len(end) - 1) / span if span > 0 else math.nan
        lat = [(i.recv_ns - i.capture_ns) / 1e6 for i in end if i.capture_ns]
        m |= {"latency_mean_ms": mean(lat), "latency_median_ms": pct(lat, 0.5), "latency_p95_ms": pct(lat, 0.95)}
    if local and (sink or rx):  # fusion: the local branch too
        span = (max(i.recv_ns for i in local) - min(i.recv_ns for i in local)) / 1e9
        m["local_throughput_fps"] = (len(local) - 1) / span if span > 0 else math.nan
        lat = [(i.recv_ns - i.capture_ns) / 1e6 for i in local if i.capture_ns]
        m |= {"local_latency_mean_ms": mean(lat), "local_latency_p95_ms": pct(lat, 0.95)}
    if trial.power:
        p = trial.power
        m |= {"power_mean_w": mean(p), "power_median_w": pct(p, 0.5), "power_p95_w": pct(p, 0.95)}
    if rx:
        uniq = {(i.frame, i.box, i.recv_ns): i.bytes for i in rx}
        m["bytes"] = sum(uniq.values())
        if video_s:
            m["frac_1mbps"] = m["bytes"] / (MBPS_BYTES * video_s)
        m |= score(rx, labels, targets, trial.processed)
    if sink and rx:
        v = score(sink, labels, targets, trial.processed)
        if v:
            m["verified_precision"], m["verified_recall"] = v["precision"], v["recall"]
            m["verified_effective_recall"] = v["effective_recall"]
    if trial.played:
        m["processed_frac"] = len(trial.processed & trial.played) / len(trial.played)
    stats = read_csv(next(trial.path.parent.glob("stats_*.csv"), Path("-")))
    n = trial.path.name.split("_")[-1]
    for r in stats:
        if r.get("trial") == n and r.get("total_bytes_delivered"):
            m["netsim_bytes"] = float(r["total_bytes_delivered"])
    return m


# ---- runs -> rows ---------------------------------------------------------------------


@dataclass
class Row:
    section: str
    experiment: str
    source: str
    profile: str
    trials: int
    metrics: dict = field(default_factory=dict)  # name -> (mean, sd)


COLUMNS = ["throughput_fps", "latency_mean_ms", "latency_median_ms", "latency_p95_ms",
           "local_throughput_fps", "local_latency_mean_ms", "local_latency_p95_ms",
           "power_mean_w", "power_median_w", "power_p95_w", "bytes", "netsim_bytes", "frac_1mbps", "uot",
           "precision", "recall", "effective_recall", "processed_frac", "verified_precision", "verified_recall",
           "verified_effective_recall", "items", "tp_items"]


def collect(d: LAB.Dir) -> list[Row]:
    rows = []
    base = LAB.LAB / d.path / "runs"
    for e in d.exps:
        labels = e.source.labels
        video_s = e.source.duration
        for profile in e.profiles:
            run = base / e.name / profile
            trials = sorted(run.glob("trial_*")) if run.is_dir() else []
            per = [trial_metrics(Trial(t), labels, e.targets, video_s) for t in trials]
            per = [p for p in per if p]
            if not per:
                continue
            agg = {}
            for c in COLUMNS:
                vals = [p[c] for p in per if c in p and not math.isnan(p[c])]
                if vals:
                    agg[c] = (mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
            rows.append(Row(d.path, e.name, e.source.name, profile, len(per), agg))
    return rows


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_section_csv(top: str, dirs: list[LAB.Dir], rows_by_dir: dict) -> Path:
    """One tab: each subsection's title, its table, and a blank line."""
    path = results_dir(top) / f"{top}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([SECTIONS.get(top, top)])
        w.writerow([])
        for d in dirs:
            rows = rows_by_dir.get(d.path, [])
            used = [c for c in COLUMNS if any(c in r.metrics for r in rows)]
            w.writerow([d.title])
            if not rows:
                w.writerow(["(no runs yet)"])
                w.writerow([])
                continue
            w.writerow(["experiment", "source", "profile", "trials",
                        *[x for c in used for x in (c, f"{c}_sd")]])
            for r in rows:
                w.writerow([r.experiment, r.source, r.profile, r.trials,
                            *[fmt(v) for c in used for v in r.metrics.get(c, (None, None))]])
            w.writerow([])
    return path


# ---- figures ----------------------------------------------------------------------------


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "savefig.bbox": "tight",
    })
    return plt


COLUMN_W = 3.4  # inches: one column of a two-column paper


def pgf_line_chart(title, xlabel, ylabel, series: dict[str, list[tuple]], *, xdir_reverse=False, ymin=None,
                   ymax=None, xtick_labels=None) -> str:
    """A pgfplots line chart of `series` (name -> [(x, y)])."""
    opts = [f"title={{{title}}}" if title else None, f"xlabel={{{xlabel}}}", f"ylabel={{{ylabel}}}",
            "width=\\columnwidth", "height=0.62\\columnwidth", "grid=major", "legend pos=south west",
            "x dir=reverse" if xdir_reverse else None,
            f"ymin={ymin}" if ymin is not None else None, f"ymax={ymax}" if ymax is not None else None]
    if xtick_labels:
        opts += [f"xtick={{{','.join(str(x) for x, _ in xtick_labels)}}}",
                 f"xticklabels={{{','.join(lbl for _, lbl in xtick_labels)}}}"]
    lines = ["% Generated by `uv run analyze`; needs \\usepackage{pgfplots}",
             "\\begin{tikzpicture}", "\\begin{axis}[" + ", ".join(o for o in opts if o) + "]"]
    for name, pts in series.items():
        coords = " ".join(f"({x:g},{y:g})" for x, y in pts if not math.isnan(y))
        lines += [f"\\addplot+[mark=*, thick] coordinates {{{coords}}};", f"\\addlegendentry{{{name}}}"]
    lines += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def quality_figures(path: Path) -> list[Path]:
    """1.1.1: mAP 50-95 against QP, and against resolution (high to low), for
    each model, from the 1.1-quality numbers. The CSV has columns model,
    axis ("qp" or "resolution"), value (the QP, or the resolution's height),
    map (mAP 50-95)."""
    plt = setup_matplotlib()
    rows = read_csv(path)
    out = []
    for axis, xlabel, reverse in (("qp", "QP", False), ("resolution", "Resolution (p)", True)):
        series: dict[str, list] = {}
        for r in rows:
            if r["axis"].strip().lower() == axis:
                series.setdefault(r["model"].strip(), []).append((float(r["value"]), float(r["map"])))
        if not series:
            continue
        for s in series.values():
            s.sort()
        fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.62))
        for name, pts in series.items():
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", ms=3, label=name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mAP 50-95")
        if reverse:
            ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
        name = f"1.1.1-map-vs-{axis}"
        fig.savefig(fig_path(name, "pdf"))
        plt.close(fig)
        (fig_path(name, "tex")).write_text(pgf_line_chart("", xlabel, "mAP 50-95", series, xdir_reverse=reverse))
        out.append(fig_path(name, "pdf"))
    return out


def recall_curve_figures(d: LAB.Dir) -> list[Path]:
    """2.1.1.2: for each source, as example.gif: the fraction of frames sent
    (line), split into true and false positives (areas), against the
    confidence cutoff, with the recall on the right axis."""
    plt = setup_matplotlib()
    out = []
    by_source: dict[str, list] = {}
    for e in d.exps:
        run = LAB.LAB / d.path / "runs" / e.name / "none"
        trial = next(iter(sorted(run.glob("trial_*"))), None) if run.is_dir() else None
        labels = e.source.labels
        if trial is None or labels is None:
            continue
        conf = int(e.name.rsplit("_c", 1)[1]) / 100
        items = Trial(trial).items("rx.csv")
        s = score(items, labels, e.targets)
        frames = labels.frames
        tp = s.get("tp_items", 0) / frames
        sent = s.get("items", 0) / frames
        by_source.setdefault(e.source.name, []).append((conf, sent, tp, sent - tp, s.get("recall", math.nan)))
    for src, pts in by_source.items():
        pts.sort()
        x = [p[0] for p in pts]
        fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.7))
        ax.fill_between(x, 0, [p[2] for p in pts], color="#8e6cc9", alpha=0.75, label="True positives", lw=0)
        ax.fill_between(x, [p[2] for p in pts], [p[1] for p in pts], color="#9bd49b", alpha=0.6,
                        label="False positives", lw=0)
        ax.plot(x, [p[1] for p in pts], color="#d62728", lw=1.2, label="Transmitted")
        ax.set_xlabel("Cutoff Threshold")
        ax.set_ylabel("Frame Fraction", color="#d62728")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax2 = ax.twinx()
        ax2.plot(x, [p[4] for p in pts], color="#2b3fb3", marker="s", ms=3, lw=1, label="Recall")
        ax2.set_ylabel("Recall", color="#2b3fb3")
        ax2.set_ylim(0, 1.05)
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False,
                  fontsize=7, handlelength=1.2, columnspacing=0.8)
        name = f"2.1.1.2-recall-curve-{src}"
        fig.savefig(fig_path(name, "pdf"))
        plt.close(fig)
        (fig_path(name, "tex")).write_text(recall_curve_tex(pts))
        out.append(fig_path(name, "pdf"))
    return out


def recall_curve_tex(pts) -> str:
    def coords(k):
        return " ".join(f"({p[0]:g},{p[k]:.4f})" for p in pts if not math.isnan(p[k]))

    return "\n".join([
        "% Generated by `uv run analyze`; needs \\usepackage{pgfplots} and \\usepgfplotslibrary{fillbetween}",
        "\\begin{tikzpicture}",
        "\\begin{axis}[width=\\columnwidth, height=0.7\\columnwidth, xmin=0, xmax=1, ymin=0, ymax=1.05,",
        "  xlabel={Cutoff Threshold}, ylabel={Frame Fraction}, ylabel style={red}, axis y line*=left,",
        "  legend style={at={(0.5,1.03)}, anchor=south, legend columns=-1, draw=none, font=\\footnotesize}]",
        f"\\addplot[draw=none, fill=violet!60, name path=tp] coordinates {{{coords(2)}}} \\closedcycle;",
        "\\addlegendentry{True positives}",
        f"\\addplot[draw=none, forget plot, name path=sent] coordinates {{{coords(1)}}};",
        "\\addplot[green!40] fill between[of=tp and sent];",
        "\\addlegendentry{False positives}",
        f"\\addplot[red, thick] coordinates {{{coords(1)}}};",
        "\\addlegendentry{Transmitted}",
        "\\end{axis}",
        "\\begin{axis}[width=\\columnwidth, height=0.7\\columnwidth, xmin=0, xmax=1, ymin=0, ymax=1.05,",
        "  axis y line*=right, axis x line=none, ylabel={Recall}, ylabel style={blue}]",
        f"\\addplot[blue, mark=square*, mark size=1.2pt] coordinates {{{coords(4)}}};",
        "\\end{axis}",
        "\\end{tikzpicture}",
        "",
    ])


def profile_bar_figures(d: LAB.Dir, rows: list[Row], metrics=("recall", "effective_recall", "precision", "uot", "frac_1mbps")) -> list[Path]:
    """For a search subsection: each metric by profile, a group of bars per
    source."""
    plt = setup_matplotlib()
    out = []
    num = d.path.rsplit("/", 1)[-1]
    for metric in metrics:
        data: dict[str, dict[str, tuple]] = {}
        for r in rows:
            if metric in r.metrics:
                data.setdefault(r.source, {})[r.profile] = r.metrics[metric]
        if not data:
            continue
        profiles = [p for p in (*LAB.ALL_PROFILES, "none") if any(p in v for v in data.values())]
        sources = list(data)
        fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.62))
        width = 0.8 / max(1, len(profiles))
        for k, p in enumerate(profiles):
            xs = [i + (k - (len(profiles) - 1) / 2) * width for i in range(len(sources))]
            ys = [data[s].get(p, (math.nan, 0))[0] for s in sources]
            es = [data[s].get(p, (0, 0))[1] for s in sources]
            ax.bar(xs, ys, width, yerr=es, label=p, capsize=1.5, error_kw={"lw": 0.6})
        ax.set_xticks(range(len(sources)), sources)
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(ncol=min(5, len(profiles)), fontsize=6, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, 1.18))
        ax.grid(axis="y", alpha=0.3)
        name = f"{num}-{metric}"
        fig.savefig(fig_path(name, "pdf"))
        plt.close(fig)
        (fig_path(name, "tex")).write_text(bar_tex(metric, sources, profiles, data))
        out.append(fig_path(name, "pdf"))
    return out


def bar_tex(metric, sources, profiles, data) -> str:
    lines = ["% Generated by `uv run analyze`; needs \\usepackage{pgfplots}", "\\begin{tikzpicture}",
             "\\begin{axis}[ybar, width=\\columnwidth, height=0.62\\columnwidth, bar width=4pt,",
             f"  symbolic x coords={{{','.join(sources)}}}, xtick=data, ylabel={{{metric.replace('_', ' ')}}},",
             "  legend style={at={(0.5,1.03)}, anchor=south, legend columns=-1, draw=none, font=\\footnotesize},",
             "  error bars/y dir=both, error bars/y explicit]"]
    for p in profiles:
        pts = " ".join(f"({s},{data[s][p][0]:.4g}) +- (0,{data[s][p][1]:.4g})" for s in sources
                       if p in data[s] and not math.isnan(data[s][p][0]))
        lines += [f"\\addplot coordinates {{{pts}}};", f"\\addlegendentry{{{p}}}"]
    lines += ["\\end{axis}", "\\end{tikzpicture}", ""]
    return "\n".join(lines)


def micro_figures(rows: list[Row]) -> list[Path]:
    """1.2: throughput, latency and power of local, remote and fusion."""
    plt = setup_matplotlib()
    out = []
    names = {"1.2.1-local": "local", "1.2.2-remote": "remote", "1.2.3-fusion": "fusion"}
    for metric, label in (("throughput_fps", "Throughput (fps)"), ("latency_median_ms", "Median latency (ms)"),
                          ("power_mean_w", "Mean power (W)")):
        pts = [(names.get(r.section.rsplit("/", 1)[-1], r.section), *r.metrics[metric]) for r in rows
               if metric in r.metrics]
        if not pts:
            continue
        fig, ax = plt.subplots(figsize=(COLUMN_W * 0.7, COLUMN_W * 0.55))
        ax.bar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], capsize=2, color="#4c72b0")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        name = f"1.2-{metric}"
        fig.savefig(fig_path(name, "pdf"))
        plt.close(fig)
        coords = " ".join(f"({p[0]},{p[1]:.4g}) +- (0,{p[2]:.4g})" for p in pts)
        (fig_path(name, "tex")).write_text("\n".join([
            "% Generated by `uv run analyze`; needs \\usepackage{pgfplots}", "\\begin{tikzpicture}",
            f"\\begin{{axis}}[ybar, width=0.7\\columnwidth, height=0.55\\columnwidth, symbolic x coords={{{','.join(p[0] for p in pts)}}},",
            f"  xtick=data, ylabel={{{label}}}, error bars/y dir=both, error bars/y explicit]",
            f"\\addplot coordinates {{{coords}}};", "\\end{axis}", "\\end{tikzpicture}", ""]))
        out.append(fig_path(name, "pdf"))
    return out


# ---- entry point ----------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quality", type=Path, default=results_dir("1.1-quality") / "1.1-quality.csv",
                        help="1.1-quality's numbers: CSV with model, axis (qp|resolution), value, map "
                             "(default %(default)s)")
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args(argv)

    dirs = LAB.all_dirs()
    rows_by_dir = {d.path: collect(d) for d in dirs}
    written = []
    for top in ("1.2-microbenchmarks", "2.1-search"):
        group = [d for d in dirs if d.path.startswith(top)]
        written.append(write_section_csv(top, group, rows_by_dir))
    if not args.no_figs:
        if args.quality.is_file():
            written += quality_figures(args.quality)
        else:
            print(f"no {args.quality}: skipping the 1.1.1 figures")
        written += micro_figures([r for d in dirs if d.path.startswith("1.2") for r in rows_by_dir[d.path]])
        for d in dirs:
            if d.path.endswith("2.1.1.2-yolo-recall-curve"):
                written += recall_curve_figures(d)
            elif d.path.startswith("2.1"):
                written += profile_bar_figures(d, rows_by_dir[d.path])
    for p in written:
        print(f"wrote {p.relative_to(L.REPO_ROOT)}")


if __name__ == "__main__":
    main()
