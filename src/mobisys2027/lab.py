"""Writes the Mobisys 2027 experiment directories of plan.md into lab/.

    uv run lab            # (re)write every experiment directory
    uv run lab --list     # print what would be written

Each section of the plan with a flow is a directory, nested as the plan's
headers are (lab/2.1-search/2.1.1-yolo/2.1.1.1-yolo-ed/), with an
experiment.toml, its flows, lambda2.toml, and fermi_<experiment>.toml for
each experiment (one per video or dataset, or per confidence for the recall
curve). `uv run experiment -r lab` runs them all; see the README.

Generated, so edit this file and rerun rather than editing the output. What
the experiments measure with:

- fake_drone logs every frame it publishes (mb/frames.csv on fermi): its
  frame id, capture timestamp and Uriel seq;
- cliped, detection-crop and patchcore-score log what they queue to send
  (mb/sent.csv on fermi), with where it was cut from its frame;
- a metrics engine on lambda2 (rx_metrics) sees each frame as it arrives,
  before it is decoded, and logs its size and timestamps (mb/rx.csv), and
  one at the end of the pipeline (sink_metrics, mb/sink.csv) does the same;
- the runner measures fermi's power and netsim's bytes.

`uv run analyze` turns these into results/. The malicious profile's
failpoints come from labels/<source>.json, made with
`uv run label`; until a source is labeled, the failpoints found earlier from
cliped's TP/TN labels are used.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mobisys2027 import labels as L

LAB = L.REPO_ROOT / "lab"
BRANCH = "v4_backend_rework"
FILES = "/root/.local/share/steeleagle/files"
LOG_DIR = "mb"  # in each daemon's files directory: the logs collected after each trial

# Every profile but these is run on the plain experiment; these get GRACE_S of
# grace and black padding after the video, as the separate <source>_lossy experiment
LOSSY = ("intermittent", "malicious")
DEFAULT_PROFILES = ("normal", "limited", "degraded", "intermittent")
ALL_PROFILES = (*DEFAULT_PROFILES, "malicious")
IMAGE_FPS = 10.0
WARNINGS: set[str] = set()  # printed by `uv run lab`
TRIALS = 3
# Grace after the video of a lossy (intermittent, malicious) experiment, for
# reintegration to finish
GRACE_S = 60
# The send queue of every reintegrating engine (detection-crop, cliped,
# patchcore-score): room to wait out an outage. Their reintegration delivers
# one item at a time and resends on the flow's events alone (its producer has
# one token), so there is no resend timer
SEND_QUEUE = {"max_queue": 512}


def pkg(path: str) -> str:
    return f"https://github.com/cmusatyalab/steeleagle_plugins//{path}@{BRANCH}"


# ---- sources ----------------------------------------------------------------------


@dataclass
class Source:
    name: str  # experiment name prefix and label file name
    upload: str  # name in STEELEAGLE_FILES on fermi (a video, or a directory of images)
    kind: str  # "video" or "images"
    frames: int = 0
    fps: float = 30000 / 1001
    target: str = L.MANNEQUIN
    # Failpoints from cliped's TP/TN labels, used until labels/<name>.json exists
    fallback_failpoints: list = field(default_factory=list)

    @property
    def labels(self) -> L.Labels | None:
        return L.try_load(self.name)

    @property
    def duration(self) -> float | None:
        lab = self.labels
        if lab is not None and lab.frames:
            return lab.frames / lab.fps
        return self.frames / self.fps if self.frames else None

    def failpoints(self):
        lab = self.labels
        if lab is not None and lab.target_frames():
            return L.failpoints(lab)
        return self.fallback_failpoints


SOURCES = {
    s.name: s
    for s in [
        Source("parkinglot", "parkinglot.mp4", "video", 2590, fallback_failpoints=[[48.6, 16.8]]),
        Source("traintrack", "traintrack.mp4", "video", 2844, fallback_failpoints=[[21.2, 8.4], [61.1, 21.4]]),
        Source("street", "street.mp4", "video", 3053, fallback_failpoints=[[39.1, 12.6], [68.9, 12.8]]),
        Source("field", "field.mp4", "video"),
        Source("wisard", "wisard", "images", 264, fps=IMAGE_FPS, target=L.PERSON),
        Source("sard", "sard", "images", 1144, fps=IMAGE_FPS, target=L.PERSON),
        # MMLA: four African painted dog clips spliced into zebra and giraffe
        # footage, 5% dog (scripts/prepare_datasets.py's MMLA_CLIPS)
        Source("mmla", "mmla", "images", 4000, fps=IMAGE_FPS, target="dog"),
    ]
}
VIDEOS3 = ("parkinglot", "traintrack", "street")
SEARCH5 = (*VIDEOS3, "wisard", "sard")


# ---- TOML -------------------------------------------------------------------------


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise TypeError(v)


@dataclass
class Service:
    name: str
    plugin: str
    package: str
    args: dict = field(default_factory=dict)
    environment: str = ""
    comment: str = ""
    type: str = "engine"

    def toml(self) -> str:
        out = [f"# {self.comment}"] if self.comment else []
        out += ["[[services]]", f'name = "{self.name}"', f'type = "{self.type}"', "[services.plugin]",
                f'plugin = "{self.plugin}"', f"package = {toml_value(pkg(self.package))}"]
        if self.environment:
            out.append(f'environment = "{self.environment}"')
        if self.args:
            out.append("[services.plugin.args]")
            out += [f"{k} = {toml_value(v)}" for k, v in self.args.items()]
        return "\n".join(out)


def daemon_toml(hostname: str, services: list[Service], comment: str) -> str:
    head = [f"# {comment}", "# Generated by `uv run lab` (src/mobisys2027/lab.py); edit that instead.", "",
            "[daemon]", "port = 9000", "lcs = 9001", f'hostname = "{hostname}"']
    if hostname == "fermi":
        head.append('orchestrator = "lambda2:9000"')
    return "\n".join(head) + "\n\n" + "\n\n".join(s.toml() for s in services) + "\n"


# Timeout of an edge to or between remote engines: room for an item to cross
# intermittent's 0-100 Kbps link, so a slow crossing isn't timed out and sent again
REMOTE_EDGE_TIMEOUT = "30s"
# Inputs each producer may have in a flow at once: one, so a producer's next
# input waits for the last to finish instead of racing it to the engines
FLOW_TOKENS = 1


def flow_toml(edges: list[tuple[str, str]], comment: str) -> str:
    """A flow of the edges (from, to): a node is a vehicle stream
    (vehicle.rgb), a local or remote engine (local.yolo, remote.decoder), or a
    local engine's producer (local.reduce.reintegration)."""
    nodes: list[str] = []
    for a, b in edges:
        for n in (a, b):
            if n not in nodes:
                nodes.append(n)
    producers = [n for n in nodes if n.startswith("vehicle.") or n.count(".") == 2]
    engines = [n for n in nodes if n not in producers]
    out = [f"# {comment}", "# Generated by `uv run lab` (src/mobisys2027/lab.py); edit that instead.", ""]
    for p in producers:
        out += ["[[producers]]", f'name = "{p}"', f"tokens = {FLOW_TOKENS}", ""]
    for e in engines:
        out += ["[[engines]]", f'name = "{e}"', f'location = "{e.split(".")[0]}"', ""]
    for a, b in edges:
        remote = a.startswith("remote.") or b.startswith("remote.")
        out += ["[[edges]]", f'from = "{a}"', f'to = "{b}"', f'timeout = "{REMOTE_EDGE_TIMEOUT if remote else "2s"}"', ""]
    return "\n".join(out).rstrip() + "\n"


# ---- services -----------------------------------------------------------------------


def drone(src: Source, pad: bool) -> Service:
    args: dict = {"no_loop": True, "frame_log": f"{LOG_DIR}/frames.csv"}
    if src.kind == "video":
        # The Tachyon's hardware decoder, as heisenberg's drone uses: avdec_h264
        # can't decode 1080p in real time there
        args = {"video": src.upload, "decoder": "qtic2vdec", **args}
    else:
        args = {"images": src.upload, "fps": src.fps, **args}
    if pad:
        args["pad_after"] = True
    what = f"the images in {src.upload} at {src.fps:g} fps" if src.kind == "images" else src.upload
    return Service("drone", "fake_drone", "drivers/fake_drone", args, type="vehicle",
                   environment="native" if src.kind == "video" else "",
                   comment=f"Replays {what} once{', then black frames until stopped' if pad else ''}; logs each frame"
                           + ("; hardware decoder" if src.kind == "video" else ""))


def qnn(confidence=None, person=True, early_discard=True) -> Service:
    args: dict = {}
    if person:
        args["only_classes"] = ["person"]
    if early_discard:
        args["early_discard"] = True
    if confidence is not None:
        # NMS drops boxes under its own score threshold, so it follows the confidence
        args["confidence"] = confidence
        args["nms_score_threshold"] = confidence
    what = "early discard of frames with no person" if early_discard else "every frame forwarded"
    return Service("yolo", "detection-qnn", "engines/detection-qnn", args,
                   comment=f"YOLO26 on the HTP (detection-qnn, COCO){': ' + what if args else ''}")


def h264_encoder() -> Service:
    return Service("h264_encoder", "encoder", "defaults/h264", {
        "decoder_name": "h264_decoder",
        "converter": "qtivtransform ! video/x-raw(memory:GBM),format=NV12",
        "encoder": "qtic2venc control-rate=disable b-frames=0 idr-interval=30",
    }, environment="native", comment="The Tachyon's hardware H.264 encoder, as in live/")


def webp_encoder() -> Service:
    return Service("webp_encoder", "webp_encoder", "defaults/thumbnail", {"quality": 65},
                   comment="WebP at quality 65")


def metrics(name: str, forward: bool) -> Service:
    args: dict = {"csv_path": f"{LOG_DIR}/{name.split('_')[0]}.csv"}
    if forward:
        args["forward_frames"] = True
    what = ("Logs each frame as it arrives from fermi, before it is decoded, and passes it on"
            if forward else "Logs each frame reaching the end of the pipeline")
    return Service(name, "metrics", "engines/metrics", args, comment=what)


def rfdetr(confidence=None, person=False, crops=False, early_discard=False, name="rfdetr") -> Service:
    args: dict = {"model": "rfdetr", "rfdetr_size": "2xlarge"}
    if person:
        args["only_classes"] = ["person"]
    if confidence is not None:
        args["confidence"] = confidence
    if crops:
        args["letterbox"] = True
    if early_discard:
        args["early_discard"] = True
    # cuda126: torch built for CUDA 12.6; the default environment's (CUDA 13)
    # needs a newer driver than lambda2's (12.8), and falls back to the CPU
    return Service(name, "detection", "engines/detection", args, environment="cuda126",
                   comment="RF-DETR 2XLarge on lambda2's GPU (torch for CUDA 12.6)")


def crop(verifier: str) -> Service:
    return Service("reduce", "detection-crop", "engines/detection-crop", {
        "detector": "yolo", "class": "person", "verifier": verifier, "log_sent": f"{LOG_DIR}/sent.csv",
        **SEND_QUEUE,
    }, comment=f"Crops each frame to yolo's best person; queued until {verifier} has the crop; logs each crop")


H264_DECODER = Service("h264_decoder", "decoder", "defaults/h264")
WEBP_DECODER = Service("webp_decoder", "webp_decoder", "defaults/thumbnail")


# ---- experiments ----------------------------------------------------------------------


@dataclass
class Exp:
    name: str
    source: Source
    profiles: tuple
    fermi: list[Service]
    flow: str = "flow.toml"
    lossy: bool = False  # GRACE_S of grace and black padding
    trials: int = TRIALS
    # The labeled classes that count as found (true positives) when scoring
    targets: tuple = ()

    def toml(self) -> str:
        src = self.source
        lines = ["[[experiments]]", f'name = "{self.name}"', 'vehicle = "drone"', "[experiments.video]",
                 f'video = "{src.upload}"', f'flow = "{self.flow}"', "auto = true"]
        d = src.duration
        # A video's length comes from videos/ (on lambda2); images have none to read
        if src.kind == "images" and d:
            frames = src.labels.frames if src.labels else src.frames
            lines.append(f"duration = {round(d, 2)}  # {frames} frames at {src.fps:g} fps")
        if self.lossy:
            lines.append(f"grace = {GRACE_S}  # for reintegration to finish")
        if self.targets:
            lines.insert(1, f"# scored: a {' or '.join(self.targets)} counts as found")
        lines += ["[experiments.netsim]", f"profiles = {toml_value(list(self.profiles))}",
                  'interface = "tailscale0"', "port = 9001", f"trials = {self.trials}"]
        if "malicious" in self.profiles:
            fp = src.failpoints()
            if not fp:
                raise SystemExit(f"{self.name}: malicious needs failpoints; label {src.name} first")
            lines.append(f"failpoints = {toml_value(fp)}  # while the {src.target} is in view, padded 1 s")
        return "\n".join(lines)


KEEP = {"runs", "results"}  # never deleted when an experiment directory is regenerated


@dataclass
class Dir:
    path: str  # under lab/
    title: str
    flows: dict[str, str]  # file -> flow toml
    lambda2: list[Service]
    exps: list[Exp]
    uploads: list[str] = field(default_factory=list)  # media/ files to push

    def write(self, root: Path) -> Path:
        d = root / self.path
        # Replace the generated files, keeping what runs and analysis left
        if d.exists():
            for p in d.iterdir():
                if p.name in KEEP:
                    continue
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        d.mkdir(parents=True, exist_ok=True)
        for name, text in self.flows.items():
            (d / name).write_text(text)
        (d / "lambda2.toml").write_text(daemon_toml("lambda2", self.lambda2, f"lambda2 (orchestrator) for {self.title}"))
        for e in self.exps:
            (d / f"fermi_{e.name}.toml").write_text(daemon_toml("fermi", e.fermi, f"fermi for {self.title}: {e.name}"))
        sources = []
        for e in self.exps:
            if e.source not in sources:
                sources.append(e.source)
        text = [f"# {self.title}", "# Generated by `uv run lab` (src/mobisys2027/lab.py); edit that instead.",
                "# fermi_<experiment>.toml is each experiment's fermi config; lambda2.toml is shared.", ""]
        text += [e.toml() + "\n" for e in self.exps]
        text.append("# Pushed to every daemon once, before the first experiment")
        for u in self.uploads:
            text += ["[[uploads]]", f'file = "{u}"']
        for s in sources:
            note = ("already in fermi's files directory (videos/ has a copy for its length)" if s.kind == "video"
                    else "a directory of images already in fermi's files directory (scripts/prepare_datasets.py)")
            text += [f"[[uploads]] # {note}", f'file = "{s.upload}"', "no_upload = true"]
        text += ["", "[daemons]", "[daemons.lambda2]", "orchestrator = true", 'prefix = ""',
                 "[daemons.fermi]", 'prefix = "root@fermi:"', "",
                 "[results] # moved off the hosts after each trial",
                 f'fermi = "fermi{FILES}/{LOG_DIR}/"', f'lambda2 = "lambda2{FILES}/{LOG_DIR}/"', ""]
        (d / "experiment.toml").write_text("\n".join(text))
        return d


# What counts as found: yolo and a "person" prompt look for anyone, so a
# mannequin or a person; an image prompt only for what it shows
ANYONE = (L.MANNEQUIN, L.PERSON)


def targets_of(src: Source, wanted: tuple) -> tuple:
    """The classes of `wanted` that `src` labels, or its own target if none
    (e.g. a dataset whose only class is person or dog)."""
    if src.kind == "video":
        return wanted
    return tuple(t for t in wanted if t == src.target) or (src.target,)


def per_source(sources, profiles, fermi_for, *, lossy_split=True, malicious_only=VIDEOS3, targets=ANYONE):
    """An experiment per source: <source> for the profiles not in LOSSY, and
    <source>_lossy, with grace and padding, for those in it. A source not in
    `malicious_only` skips malicious. `targets` are the classes that count as
    found."""
    exps = []
    for name in sources:
        src = SOURCES[name]
        ps = tuple(p for p in profiles if p != "malicious" or name in malicious_only)
        if "malicious" in ps and not src.failpoints():
            WARNINGS.add(f"warning: {name} has no labels to place malicious's failpoints; leaving malicious "
                         f"out until `uv run label {name}` and `uv run lab` are run")
            ps = tuple(p for p in ps if p != "malicious")
        plain = tuple(p for p in ps if p not in LOSSY) if lossy_split else ps
        lossy = tuple(p for p in ps if p in LOSSY) if lossy_split else ()
        found = targets_of(src, targets)
        if plain:
            exps.append(Exp(name, src, plain, fermi_for(src, False), targets=found))
        if lossy:
            exps.append(Exp(f"{name}_lossy", src, lossy, fermi_for(src, True), lossy=True, targets=found))
    return exps


def microbenchmarks() -> list[Dir]:
    src = SOURCES["parkinglot"]
    local = Dir(
        "1.2-microbenchmarks/1.2.1-local", "1.2.1 local: YOLO (QNN) on fermi, nothing sent",
        {"flow.toml": flow_toml([("vehicle.rgb", "local.yolo"), ("local.yolo", "local.sink_metrics")],
                                "rgb -> yolo (QNN) -> sink_metrics, all on fermi")},
        [], [Exp("parkinglot", src, ("normal",),
                 [drone(src, False), qnn(person=False, early_discard=False), metrics("sink_metrics", False)])])
    remote = Dir(
        "1.2-microbenchmarks/1.2.2-remote", "1.2.2 remote: H.264 to lambda2, RF-DETR 2XL there",
        {"flow.toml": flow_toml([
            ("vehicle.rgb", "local.h264_encoder"), ("local.h264_encoder", "remote.rx_metrics"),
            ("remote.rx_metrics", "remote.h264_decoder"), ("remote.h264_decoder", "remote.rfdetr"),
            ("remote.rfdetr", "remote.sink_metrics")],
            "rgb -> h264_encoder ==network==> rx_metrics -> h264_decoder -> rfdetr -> sink_metrics (lambda2)")},
        [metrics("rx_metrics", True), H264_DECODER, rfdetr(), metrics("sink_metrics", False)],
        [Exp("parkinglot", src, ("normal",), [drone(src, False), h264_encoder()])])
    fusion = Dir(
        "1.2-microbenchmarks/1.2.3-fusion", "1.2.3 fusion: YOLO (QNN) on fermi and RF-DETR 2XL on lambda2, fused on fermi",
        {"flow.toml": flow_toml([
            ("vehicle.rgb", "local.yolo"), ("local.yolo", "local.local_metrics"),
            ("vehicle.rgb", "local.h264_encoder"), ("local.h264_encoder", "remote.rx_metrics"),
            ("remote.rx_metrics", "remote.h264_decoder"), ("remote.h264_decoder", "remote.rfdetr"),
            ("remote.rfdetr", "remote.sink_metrics"), ("vehicle.rgb", "local.fusion")],
            "rgb -> yolo -> local_metrics (fermi); rgb -> h264_encoder ==network==> rx_metrics -> h264_decoder "
            "-> rfdetr -> sink_metrics (lambda2); rgb -> fusion (fermi) fuses yolo's and rfdetr's results")},
        [metrics("rx_metrics", True), H264_DECODER, rfdetr(), metrics("sink_metrics", False)],
        [Exp("parkinglot", src, ("normal",), [
            drone(src, False), qnn(person=False, early_discard=False),
            Service("local_metrics", "metrics", "engines/metrics", {"csv_path": f"{LOG_DIR}/local.csv"},
                    comment="Logs each frame yolo finishes"),
            h264_encoder(),
            Service("fusion", "fusion_detection", "engines/fusion", {"local": "yolo", "remote": "rfdetr"},
                    comment="Fuses yolo's and rfdetr's detections of each frame"),
        ])])
    for d in (local, remote, fusion):
        for e in d.exps:
            e.trials = 3
    return [local, remote, fusion]


def yolo() -> list[Dir]:
    base = "2.1-search/2.1.1-yolo"
    rx_h264 = [("local.h264_encoder", "remote.rx_metrics"), ("remote.rx_metrics", "remote.h264_decoder")]
    ed_flow = flow_toml([("vehicle.rgb", "local.yolo"), ("local.yolo", "local.h264_encoder"), *rx_h264],
                        "rgb -> yolo (early discard) -> h264_encoder ==network==> rx_metrics -> h264_decoder (lambda2)")
    ed = Dir(f"{base}/2.1.1.1-yolo-ed", "2.1.1.1 YOLO early discard (person, conf 0.5), H.264",
             {"flow.toml": ed_flow}, [metrics("rx_metrics", True), H264_DECODER],
             per_source(SEARCH5, DEFAULT_PROFILES, lambda s, pad: [drone(s, pad), qnn(0.5), h264_encoder()],
                        lossy_split=False))

    confs = [0.01, *[round(0.1 * k, 1) for k in range(1, 11)]]
    curve_exps = []
    for name in SEARCH5:
        src = SOURCES[name]
        for c in confs:
            curve_exps.append(Exp(f"{name}_c{int(round(c * 100)):03d}", src, ("none",),
                                  [drone(src, False), qnn(c), h264_encoder()], trials=1,
                                  targets=targets_of(src, ANYONE)))
    curve = Dir(f"{base}/2.1.1.2-yolo-recall-curve",
                "2.1.1.2 YOLO early discard recall curve: confidence 0.01, 0.1 ... 1.0, unshaped link",
                {"flow.toml": ed_flow}, [metrics("rx_metrics", True), H264_DECODER], curve_exps)

    crop_rx = [("local.webp_encoder", "remote.rx_metrics"), ("remote.rx_metrics", "remote.webp_decoder")]
    crop_dir = Dir(
        f"{base}/2.1.1.3-yolo-crop", "2.1.1.3 YOLO early discard + crop to the best person, WebP",
        {"flow.toml": flow_toml([("vehicle.rgb", "local.yolo"), ("local.yolo", "local.reduce"),
                                 ("local.reduce", "local.webp_encoder"), *crop_rx],
                                "rgb -> yolo -> reduce (crop) -> webp_encoder ==network==> rx_metrics -> webp_decoder (lambda2)")},
        [metrics("rx_metrics", True), WEBP_DECODER],
        per_source(SEARCH5, DEFAULT_PROFILES,
                   lambda s, pad: [drone(s, pad), qnn(0.5), crop("webp_decoder"), webp_encoder()], lossy_split=False))

    # reduce is a dead end: only its reintegration producer sends crops
    reint_edges = [("vehicle.rgb", "local.yolo"), ("local.yolo", "local.reduce"),
                   ("local.reduce.reintegration", "local.webp_encoder"), *crop_rx]
    reint = Dir(
        f"{base}/2.1.1.4-yolo-reintegration", "2.1.1.4 YOLO early discard + crop + prioritized reintegration, WebP",
        {"flow.toml": flow_toml(reint_edges, "rgb -> yolo -> reduce (a dead end); "
                                             "reduce.reintegration -> webp_encoder ==network==> rx_metrics -> webp_decoder")},
        [metrics("rx_metrics", True), WEBP_DECODER],
        per_source(SEARCH5, ALL_PROFILES, lambda s, pad: [drone(s, pad), qnn(0.5), crop("webp_decoder"), webp_encoder()]))

    # 2.1.1.5 (the same, verified by RF-DETR 2XL) was folded into 2.1.1.4,
    # which took over its profiles: YOLO's crops at conf 0.5 already reach ~1.0
    # precision, so the verifier only lost recall, at every threshold from 0.1
    # to 0.85 (replayed on the quick run's crops). Recall is what leaves YOLO.
    return [ed, curve, crop_dir, reint]


def change_detection() -> list[Dir]:
    def fermi(src, pad):
        return [
            drone(src, pad),
            Service("rescale", "rescale", "engines/rescale", {"resolution": "720p"},
                    comment="patchcore takes 1280x720 frames"),
            Service("embed", "embed", "engines/patchcore-embed", comment="PatchCore's backbone on the HTP"),
            Service("score", "score", "engines/patchcore-score", {
                "threshold_meta": "models/gravel_meta.pt", "faiss_index": "models/gravel.faiss",
                "verifier": "h264_decoder", "log_sent": f"{LOG_DIR}/sent.csv",
                "log_scored": f"{LOG_DIR}/scored.csv", **SEND_QUEUE},
                comment="Queues anomalous frames for its reintegration, each until h264_decoder has it; logs "
                        "each frame scored and each one queued"),
            h264_encoder(),
        ]

    return [Dir(
        "2.1-search/2.1.2-change-detection", "2.1.2 change detection: PatchCore, reintegration, H.264",
        {"flow.toml": flow_toml([
            ("vehicle.rgb", "local.rescale"), ("local.rescale", "local.embed"), ("local.embed", "local.score"),
            ("local.score.reintegration", "local.h264_encoder"),
            ("local.h264_encoder", "remote.rx_metrics"), ("remote.rx_metrics", "remote.h264_decoder")],
            "rgb -> rescale -> embed -> score (a dead end); "
            "score.reintegration -> h264_encoder ==network==> rx_metrics -> h264_decoder")},
        [metrics("rx_metrics", True), H264_DECODER],
        per_source(("field",), ALL_PROFILES, fermi, malicious_only=("field",)),
    )]


def cliped() -> list[Dir]:
    base = "2.1-search/2.1.3-cliped"
    # cliped is a dead end: what it forwards goes nowhere, and only its
    # reintegration producer sends tiles over the network
    edge = [("vehicle.rgb", "local.rescale"), ("local.rescale", "local.cliped"),
            ("local.cliped.reintegration", "local.webp_encoder"), ("local.webp_encoder", "remote.rx_metrics"),
            ("remote.rx_metrics", "remote.webp_decoder")]
    # Without learning there is no verifier (the VLM rejected nearly every small
    # target): what cliped sends is the result
    plain_flow = flow_toml([*edge, ("remote.webp_decoder", "remote.sink_metrics")],
        "rgb -> rescale (720p) -> cliped (a dead end); "
        "cliped.reintegration -> webp_encoder ==network==> rx_metrics -> webp_decoder -> sink_metrics")
    # Learning needs the verifier's answers: its "no"s become cliped's backgrounds
    verified_flow = flow_toml([*edge, ("remote.webp_decoder", "remote.verifier"), ("remote.verifier", "remote.sink_metrics")],
        "rgb -> rescale (720p) -> cliped (a dead end); "
        "cliped.reintegration -> webp_encoder ==network==> rx_metrics -> webp_decoder -> verifier (VLM) -> sink_metrics")

    def make(path, title, prompt: dict, learn: bool, sources, profiles, uploads=(), targets=ANYONE, question=""):
        def fermi(src, pad):
            return [
                drone(src, pad),
                Service("rescale", "rescale", "engines/rescale", {"resolution": "720p"},
                        comment="cliped's tiling is tuned for 720p"),
                Service("cliped", "cliped", "engines/cliped", {
                    **prompt, "verifier": "verifier" if learn else "webp_decoder", "learn": learn,
                    "log_sent": f"{LOG_DIR}/sent.csv", "log_scored": f"{LOG_DIR}/scored.csv", **SEND_QUEUE},
                    comment=f"CLIP edge filter, learning {'on' if learn else 'off'}; a tile is queued until "
                            f"{'the verifier answers it' if learn else 'webp_decoder has it'}; logs each frame "
                            "scored and each tile queued"),
                webp_encoder(),
            ]

        verifier = [Service("verifier", "verifier", "engines/cliped", {**prompt, **({"question": question} if question else {})},
                            comment="The VLM (Ollama) that answers each tile, which cliped learns from")] if learn else []
        return Dir(f"{base}/{path}", title, {"flow.toml": verified_flow if learn else plain_flow},
                   [metrics("rx_metrics", True), WEBP_DECODER, *verifier, metrics("sink_metrics", False)],
                   per_source(sources, profiles, fermi, targets=targets), list(uploads))

    text = {"prompt": "person"}
    mannequin = {"image": "mannequin.jpeg"}
    dog_image = {"image": "dog.jpg"}
    dog_text = {"prompt": "African wild dog"}
    # The verifier's generic image question passes ~4 in 10 zebra tiles for dog.jpg; this one, for an animal,
    # kept 30/30 dog tiles and passed 0/120 onager, zebra, giraffe and empty ones
    animal_question = (
        "Image 1 is a reference photo of an animal, taken from the ground. Image 2 is a low-resolution tile cut from "
        "an aerial drone photo, seen from far above. Animals in the tile are small and will never look exactly like the "
        "reference, so judge whether any animal in image 2 is the same kind of animal as image 1, not the same "
        "individual. Other kinds of animals may appear, and many tiles have no animal at all. Answer with only yes or no.")
    return [
        make("2.1.3.1-cliped-text", '2.1.3.1 cliped, text prompt "person", no learning', text, False, SEARCH5, ALL_PROFILES),
        make("2.1.3.2-cliped-text-learning", '2.1.3.2 cliped, text prompt "person", learning', text, True, SEARCH5,
             ALL_PROFILES),
        # Only the mannequin counts: people are what it should not send
        make("2.1.3.3-cliped-image-mannequin", "2.1.3.3 cliped, image prompt mannequin.jpeg, no learning", mannequin, False,
             VIDEOS3, ALL_PROFILES, ["mannequin.jpeg"], targets=(L.MANNEQUIN,)),
        make("2.1.3.4-cliped-image-mannequin-learning", "2.1.3.4 cliped, image prompt mannequin.jpeg, learning",
             mannequin, True, VIDEOS3, ALL_PROFILES, ["mannequin.jpeg"], targets=(L.MANNEQUIN,)),
        make("2.1.3.5-cliped-wildlife-dog", "2.1.3.5 cliped, image prompt dog.jpg, no learning, MMLA", dog_image, False,
             ("mmla",), DEFAULT_PROFILES, ["dog.jpg"], targets=("dog",)),
        make("2.1.3.6-cliped-wildlife-dog-text", '2.1.3.6 cliped, text prompt "African wild dog", no learning, MMLA',
             dog_text, False, ("mmla",), DEFAULT_PROFILES, targets=("dog",)),
        make("2.1.3.7-cliped-wildlife-dog-learning", "2.1.3.7 cliped, image prompt dog.jpg, learning, MMLA", dog_image,
             True, ("mmla",), DEFAULT_PROFILES, ["dog.jpg"], targets=("dog",), question=animal_question),
        make("2.1.3.8-cliped-wildlife-dog-text-learning", '2.1.3.8 cliped, text prompt "African wild dog", learning, MMLA',
             dog_text, True, ("mmla",), DEFAULT_PROFILES, targets=("dog",)),
    ]


def all_dirs() -> list[Dir]:
    return [*microbenchmarks(), *yolo(), *change_detection(), *cliped()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the directories and experiments, write nothing")
    parser.add_argument("--root", type=Path, default=LAB)
    args = parser.parse_args(argv)
    for d in all_dirs():
        if args.list:
            print(d.path)
            for e in d.exps:
                print(f"  {e.name:24} {','.join(e.profiles):45} {e.trials} trial(s)")
            continue
        out = d.write(args.root)
        print(f"wrote {out.relative_to(L.REPO_ROOT)} ({len(d.exps)} experiments)")
    for w in sorted(WARNINGS):
        print(w)
    for s in SOURCES.values():
        if s.labels is None:
            print(f"note: labels/{s.name}.json is missing: P/R can't be scored"
                  f"{', and the fallback failpoints are used' if s.fallback_failpoints else ''}"
                  f"{'; field needs it for malicious and the grace' if s.name == 'field' else ''}")


if __name__ == "__main__":
    main()
