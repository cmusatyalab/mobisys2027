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
# Seconds the runner waits for the engines to load before starting the video,
# for experiments with a CLIP engine: cliped is listed as running ~10-20 s
# before it can score a frame on fermi (torch import, MobileCLIP, the HTP
# graph), and without it missed the start of every video
WARMUP_S = 30
CLIP_PLUGINS = frozenset({"cliped", "box_rerank"})
# The stream 2.1.1.1 (YOLO early discard) sends, by netsim profile: 480p on
# normal's 2-3 Mbps, 240p on the slower ones
ED_RESOLUTION = {"normal": "480p", "other": "240p"}
# YOLO26-M at 640x384, quantized for the HTP, fine-tuned on COCO plus the
# robotic dog (yolo26-dog-qnn, built 2026-11-17): every experiment's YOLO but
# 1.2.4's; its 81 class names are in the .labels beside it. It replaced the
# stock 640x640 small model (libs640_v26_quant.so)
SEARCH_MODEL = "libcoco81_yolo26m_640x384_quant.so"
# 1.2.4's models, in the daemon's models directory: the stock 640x640 small
# model the experiments used before SEARCH_MODEL, then the YOLO26 sizes
# fine-tuned on COCO plus the robotic dog, at 640x384
YOLO26_SIZES = {
    "base_s640": "libs640_v26_quant.so",
    **{k: f"libcoco81_yolo26{k}_640x384_quant.so" for k in ("n", "s", "m", "l")},
    # The stock COCO models (yolo26-nsmlx-...-2026-11-18), which accuracy.py
    # can score (80 classes): every size at 640x384, and x at 640x640 too
    **{f"stock_{k}": f"lib{k}640x384_v26_quant.so" for k in ("n", "s", "m", "l", "x")},
    "stock_x640": "libx640_v26_quant.so",
}
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
        # person_track (a Parrot ANAFI flight: patrol, then tracking a person in a
        # red hoodie), its first tracking section cut out and encoded at 720p
        # (libx264 CRF 18, 30-frame GOP, no B-frames); the re-encode drops the
        # ANAFI telemetry track, so the fake drone sends no telemetry.
        # 2:30-6:10 of the original; it half-starts tracking at ~3:00, goes back
        # to patrolling, and tracks fully from 5:06 (t = 156 s). The second
        # section (7:00-9:10, person_track_2.mp4) was dropped: the hoodie is in
        # view too briefly before tracking for inference to recognize it
        Source("person_track_1", "person_track_1.mp4", "video", 6594, target=L.PERSON),
        # A Parrot ANAFI search-and-rescue flight (1080p, 225.7 s), from
        # lambda's videos/; pushed by hand to heisenberg, which runs it
        Source("sandr_test", "sandr_test.mp4", "video", 6764, target=L.PERSON),
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
            out += [f"{k} = {toml_value(v)}" for k, v in self.args.items() if not isinstance(v, dict)]

            # A dict arg is a table of its own, after the plain args
            def tables(prefix, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        out.append(f"[{prefix}.{k}]")
                        out.extend(f"{kk} = {toml_value(vv)}" for kk, vv in v.items() if not isinstance(vv, dict))
                        tables(f"{prefix}.{k}", v)

            tables("services.plugin.args", self.args)
        return "\n".join(out)


def daemon_toml(hostname: str, services: list[Service], comment: str, orchestrator: str = "") -> str:
    head = [f"# {comment}", "# Generated by `uv run lab` (src/mobisys2027/lab.py); edit that instead.", "",
            "[daemon]", "port = 9000", "lcs = 9001", f'hostname = "{hostname}"']
    if orchestrator:
        head.append(f'orchestrator = "{orchestrator}:9000"')
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


def qnn(confidence=None, person=True, early_discard=True, model=None) -> Service:
    """detection-qnn running `model`, by default SEARCH_MODEL."""
    args: dict = {}
    if model:
        # A .so in the daemon's models directory; its class names are read from
        # the <model>.labels beside it, if there is one
        args["model"] = model
    if person:
        args["only_classes"] = ["person"]
    if early_discard:
        args["early_discard"] = True
    if confidence is not None:
        # NMS drops boxes under its own score threshold, so it follows the confidence
        args["confidence"] = confidence
        args["nms_score_threshold"] = confidence
    what = "early discard of frames with no person" if early_discard else "every frame forwarded"
    comment = f"YOLO26 on the HTP (detection-qnn, COCO){': ' + what if args else ''}"
    if not model:
        args = {"model": SEARCH_MODEL, **args}
        comment = comment.replace("(detection-qnn, COCO)", "(detection-qnn, YOLO26-M 640x384, COCO + robotic dog)")
    return Service("yolo", "detection-qnn", "engines/detection-qnn", args, comment=comment)


def h264_encoder() -> Service:
    return Service("h264_encoder", "encoder", "defaults/h264", {
        "decoder_name": "h264_decoder",
        "converter": "qtivtransform ! video/x-raw(memory:GBM),format=NV12",
        "encoder": "qtic2venc control-rate=disable b-frames=0 idr-interval=30",
    }, environment="native", comment="The Tachyon's hardware H.264 encoder, as in live/")


def yolo_log() -> Service:
    # A side branch off the camera: it drops each frame at once, so it holds
    # no token, and logs a row for each frame yolo returned a result on. The
    # drone's frame log can't say this: Uriel gives every published frame a
    # seq, admitted or not
    return Service("yolo_log", "detection-log", "engines/detection-log", {
        "detector": "yolo", "images": False, "csv": f"{LOG_DIR}/scored.csv",
    }, comment="Logs each frame yolo processed (frame_id, timestamp_ns, seq, its boxes), for effective recall")


def rescale(resolution: str, comment: str = "") -> Service:
    return Service("rescale", "rescale", "engines/rescale", {"resolution": resolution},
                   comment=comment or f"Rescales each frame to {resolution}")


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
    # fermi's config for a profile, where it differs from `fermi`: written as
    # fermi_<name>_<profile>.toml, which the runner uses for that profile
    fermi_by_profile: dict[str, list[Service]] = field(default_factory=dict)
    # The outage profile's failpoints, [event_time, dead_time] in seconds from
    # the start of the video, with why
    outage: list = field(default_factory=list)
    outage_note: str = ""
    # Commands run at points of each trial ([experiments.hooks]): hook -> argv
    hooks: dict[str, list[str]] = field(default_factory=dict)

    def toml(self) -> str:
        src = self.source
        lines = ["[[experiments]]", f'name = "{self.name}"', 'vehicle = "drone"', "[experiments.video]",
                 f'video = "{src.upload}"', f'flow = "{self.flow}"', "auto = true"]
        d = src.duration
        # A video's length comes from videos/ (on lambda2); images have none to read
        if src.kind == "images" and d:
            frames = src.labels.frames if src.labels else src.frames
            lines.append(f"duration = {round(d, 2)}  # {frames} frames at {src.fps:g} fps")
        if any(s.plugin in CLIP_PLUGINS for s in self.fermi):
            lines.append(f"warmup = {WARMUP_S}  # for CLIP to load before the first frame")
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
        if "outage" in self.profiles:
            if not self.outage:
                raise SystemExit(f"{self.name}: outage needs failpoints")
            lines.append(f"failpoints = {toml_value(self.outage)}  # {self.outage_note}")
        if self.hooks:
            lines.append("[experiments.hooks]")
            lines += [f"{k} = {toml_value(v)}" for k, v in self.hooks.items()]
        return "\n".join(lines)


KEEP = {"runs", "results"}  # never deleted when an experiment directory is regenerated


@dataclass
class Dir:
    path: str  # under lab/
    title: str
    flows: dict[str, str]  # file -> flow toml
    lambda2: list[Service]
    exps: list[Exp]
    uploads: list = field(default_factory=list)  # media/ files to push, or (file, push_as)
    # More results (moved) and catalogs (copied) than each daemon's mb/: key -> (source, comment)
    results: dict[str, tuple[str, str]] = field(default_factory=dict)
    catalogs: dict[str, tuple[str, str]] = field(default_factory=dict)
    # The hosts: the edge daemon that runs the fake drone (<edge>_<experiment>.toml)
    # and the orchestrator (<server>.toml), which runs the remote engines
    edge: str = "fermi"
    server: str = "lambda2"

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
        edge, server = self.edge, self.server
        (d / f"{server}.toml").write_text(daemon_toml(server, self.lambda2, f"{server} (orchestrator) for {self.title}"))
        for e in self.exps:
            (d / f"{edge}_{e.name}.toml").write_text(
                daemon_toml(edge, e.fermi, f"{edge} for {self.title}: {e.name}", server))
            for p, services in e.fermi_by_profile.items():
                (d / f"{edge}_{e.name}_{p}.toml").write_text(
                    daemon_toml(edge, services, f"{edge} for {self.title}: {e.name}, profile {p}", server))
        sources = []
        for e in self.exps:
            if e.source not in sources:
                sources.append(e.source)
        text = [f"# {self.title}", "# Generated by `uv run lab` (src/mobisys2027/lab.py); edit that instead.",
                f"# {edge}_<experiment>.toml is each experiment's {edge} config; {server}.toml is shared.", ""]
        text += [e.toml() + "\n" for e in self.exps]
        text.append("# Pushed to every daemon once, before the first experiment")
        for u in self.uploads:
            if isinstance(u, tuple):  # (file, push_as)
                text += ["[[uploads]]", f'file = "{u[0]}"', f'push_as = "{u[1]}"']
            else:
                text += ["[[uploads]]", f'file = "{u}"']
        for s in sources:
            note = (f"already in {edge}'s files directory (videos/ has a copy for its length)" if s.kind == "video"
                    else f"a directory of images already in {edge}'s files directory (scripts/prepare_datasets.py)")
            text += [f"[[uploads]] # {note}", f'file = "{s.upload}"', "no_upload = true"]
        text += ["", "[daemons]", f"[daemons.{server}]", "orchestrator = true", 'prefix = ""',
                 f"[daemons.{edge}]", f'prefix = "root@{edge}:"', "",
                 "[results] # moved off the hosts after each trial",
                 f'{edge} = "{edge}{FILES}/{LOG_DIR}/"', f'{server} = "{server}{FILES}/{LOG_DIR}/"']
        for k, (src_, why) in self.results.items():
            text += [f"# {why}", f"{k} = {toml_value(src_)}"]
        if self.catalogs:
            text.append("[catalogs] # copied off the hosts after each trial")
            for k, (src_, why) in self.catalogs.items():
                text += [f"# {why}", f"{k} = {toml_value(src_)}"]
        text.append("")
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
    # 1.2.1 with each YOLO26 size: rgb -> yolo -> sink_metrics, plus yolo_log
    # to record what each model detects (a check that it detects at all)
    sizes = Dir(
        "1.2-microbenchmarks/1.2.4-local-yolo26-sizes",
        "1.2.4 local: YOLO26 n/s/m/l (QNN, 640x384, COCO + robotic dog) on fermi, nothing sent",
        {"flow.toml": flow_toml([("vehicle.rgb", "local.yolo"), ("local.yolo", "local.sink_metrics"),
                                 ("vehicle.rgb", "local.yolo_log")],
                                "rgb -> yolo (QNN) -> sink_metrics, all on fermi; rgb -> yolo_log")},
        [], [Exp(f"parkinglot_{tag}", src, ("normal",),
                 [drone(src, False), qnn(person=False, early_discard=False, model=model), yolo_log(),
                  metrics("sink_metrics", False)], trials=3)
             for tag, model in YOLO26_SIZES.items()])
    return [local, remote, fusion, sizes]


def yolo() -> list[Dir]:
    base = "2.1-search/2.1.1-yolo"
    rx_h264 = [("local.h264_encoder", "remote.rx_metrics"), ("remote.rx_metrics", "remote.h264_decoder")]
    # Every YOLO flow also sends each frame it admits to yolo_log, which logs
    # the frames yolo processed (mb/scored.csv), for effective recall
    log_edge = ("vehicle.rgb", "local.yolo_log")
    ed_flow = flow_toml([("vehicle.rgb", "local.yolo"), ("local.yolo", "local.h264_encoder"), *rx_h264, log_edge],
                        "rgb -> yolo (early discard) -> h264_encoder ==network==> rx_metrics -> h264_decoder (lambda2); "
                        "rgb -> yolo_log")
    # Early discard sends whole frames, so the stream is rescaled after yolo
    # (which still sees the full frame) to fit the link: at full resolution an
    # H.264 frame is ~50-200 KB, and with one in flight a 2-3 Mbps link
    # delivered ~1-2 of them a second
    def ed_fermi(src, pad, resolution):
        return [drone(src, pad), qnn(0.5), yolo_log(),
                rescale(resolution, f"Rescales the frames yolo keeps to a {resolution} stream"), h264_encoder()]

    ed_exps = per_source(SEARCH5, DEFAULT_PROFILES, lambda s, pad: ed_fermi(s, pad, ED_RESOLUTION["normal"]),
                         lossy_split=False)
    for e in ed_exps:
        e.fermi_by_profile = {p: ed_fermi(e.source, e.lossy, ED_RESOLUTION.get(p, ED_RESOLUTION["other"]))
                              for p in e.profiles}
    ed = Dir(f"{base}/2.1.1.1-yolo-ed",
             f"2.1.1.1 YOLO early discard (person, conf 0.5), H.264 at {ED_RESOLUTION['normal']} on normal, "
             f"{ED_RESOLUTION['other']} on the rest",
             {"flow.toml": flow_toml(
                 [("vehicle.rgb", "local.yolo"), ("local.yolo", "local.rescale"), ("local.rescale", "local.h264_encoder"),
                  *rx_h264, log_edge],
                 "rgb -> yolo (early discard) -> rescale -> h264_encoder ==network==> rx_metrics -> h264_decoder (lambda2); "
                 "rgb -> yolo_log")},
             [metrics("rx_metrics", True), H264_DECODER], ed_exps)

    confs = [0.01, *[round(0.1 * k, 1) for k in range(1, 11)]]
    curve_exps = []
    for name in SEARCH5:
        src = SOURCES[name]
        for c in confs:
            curve_exps.append(Exp(f"{name}_c{int(round(c * 100)):03d}", src, ("none",),
                                  [drone(src, False), qnn(c), yolo_log(), h264_encoder()], trials=1,
                                  targets=targets_of(src, ANYONE)))
    curve = Dir(f"{base}/2.1.1.2-yolo-recall-curve",
                "2.1.1.2 YOLO early discard recall curve: confidence 0.01, 0.1 ... 1.0, unshaped link",
                {"flow.toml": ed_flow}, [metrics("rx_metrics", True), H264_DECODER], curve_exps)

    crop_rx = [("local.webp_encoder", "remote.rx_metrics"), ("remote.rx_metrics", "remote.webp_decoder")]
    crop_dir = Dir(
        f"{base}/2.1.1.3-yolo-crop", "2.1.1.3 YOLO early discard + crop to the best person, WebP",
        {"flow.toml": flow_toml([("vehicle.rgb", "local.yolo"), ("local.yolo", "local.reduce"),
                                 ("local.reduce", "local.webp_encoder"), *crop_rx, log_edge],
                                "rgb -> yolo -> reduce (crop) -> webp_encoder ==network==> rx_metrics -> webp_decoder (lambda2); "
                                "rgb -> yolo_log")},
        [metrics("rx_metrics", True), WEBP_DECODER],
        per_source(SEARCH5, DEFAULT_PROFILES,
                   lambda s, pad: [drone(s, pad), qnn(0.5), yolo_log(), crop("webp_decoder"), webp_encoder()],
                   lossy_split=False))

    # reduce is a dead end: only its reintegration producer sends crops
    reint_edges = [("vehicle.rgb", "local.yolo"), ("local.yolo", "local.reduce"),
                   ("local.reduce.reintegration", "local.webp_encoder"), *crop_rx, log_edge]
    reint = Dir(
        f"{base}/2.1.1.4-yolo-reintegration", "2.1.1.4 YOLO early discard + crop + prioritized reintegration, WebP",
        {"flow.toml": flow_toml(reint_edges, "rgb -> yolo -> reduce (a dead end); "
                                             "reduce.reintegration -> webp_encoder ==network==> rx_metrics -> webp_decoder; "
                                             "rgb -> yolo_log")},
        [metrics("rx_metrics", True), WEBP_DECODER],
        per_source(SEARCH5, ALL_PROFILES,
                   lambda s, pad: [drone(s, pad), qnn(0.5), yolo_log(), crop("webp_decoder"), webp_encoder()]))

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

    def make(path, title, prompt: dict, learn: bool, sources, profiles, uploads=(), targets=ANYONE, question="",
             edge="fermi", server="lambda2"):
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
                   per_source(sources, profiles, fermi, targets=targets), list(uploads), edge=edge, server=server)

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
        # 2.1.3.2 on the S&R flight, on heisenberg and lambda instead of fermi and
        # lambda2; malicious is left out until sandr_test is labeled
        make("2.1.3.9-sandr-cliped-text-learning",
             '2.1.3.9 cliped, text prompt "person", learning, S&R flight (heisenberg, lambda)', text, True,
             ("sandr_test",), ALL_PROFILES, edge="heisenberg", server="lambda"),
    ]

# The verifier's live directory on lambda2 (in its files directory): the
# headless GCS inference writes prompt.txt there, the verifier reads it
VERIFIER_LIVE = "verifier_live"
# The prompt box_rerank and its verifier start on, before inference names the
# tracked person: one the person in the red hoodie isn't, so inference's check
# of the box it tracks says no and it writes its description instead
START_PROMPT = "moving"
# person_track's clips: (when the red hoodie first appears, the clip's
# length), in seconds of the clip
TRACKING = {"person_track_1": (31.2, 220.0)}
# The length of the inference's sections (its default is 15 s): it analyzes a
# section once it ends
SECTION_S = 5.0
# From the hoodie appearing to box_rerank switching to the prompt the GCS
# inferred. With 15 s sections it was 18.8 s on person_track_1 (2026-11-18):
# the hoodie at 31.2 s, the section ending at ~45.7 s, its analysis taking
# 5.2 s, and box_rerank switching at 50.0 s. With 5 s sections it was 15.7 s:
# the hoodie at 31.2 s, the section ending at ~35 s, its analysis taking 8.3 s
# (the vision request 6.4 s), the prompt written at ~43.3 s, and box_rerank
# switching at 46.9 s
PROMPT_LATENCY_S = 15.7
# The outage starts this long after the prompt reaches box_rerank, and lasts
# to the clip's end
OUTAGE_AFTER_S = 5.0


def inference() -> list[Dir]:
    """3.1: the headless GCS mission inference over the drone's H.264, with the
    cliped tracking framework (box_rerank and its verifier) on the side."""
    edges = [
        ("vehicle.rgb", "local.yolo"), ("vehicle.rgb", "local.box_rerank"), ("vehicle.rgb", "local.box_log"),
        ("vehicle.rgb", "local.h264_encoder"),
        # One copy of the H.264 crosses the link; h264_rx fans it out on lambda2
        ("local.h264_encoder", "remote.h264_rx"),
        ("remote.h264_rx", "remote.inferred_mission"),
        ("remote.h264_rx", "remote.h264_decoder"), ("remote.h264_decoder", "remote.detection"),
        # Reintegration: WebP, to the verifier only; never to the GCS
        ("local.box_rerank.reintegration", "local.webp_encoder"), ("local.webp_encoder", "remote.rx_metrics"),
        ("remote.rx_metrics", "remote.webp_decoder"), ("remote.webp_decoder", "remote.verifier")]
    flow = flow_toml(edges,
        "rgb -> yolo (COCO), box_rerank, box_log (box_rerank's boxes), h264_encoder (fermi); "
        "h264_encoder ==network==> h264_rx -> inferred_mission (headless GCS) and h264_decoder -> detection "
        "(RF-DETR, lambda2); box_rerank.reintegration -> webp_encoder ==network==> rx_metrics -> webp_decoder -> "
        "verifier (lambda2)")

    def fermi(src, pad):
        return [
            drone(src, pad),
            # The stock COCO YOLO26-S at 640x384, at a low confidence so box_rerank sees the unsure boxes too
            qnn(confidence=0.25, early_discard=False, model=YOLO26_SIZES["stock_s"]),
            Service("box_rerank", "box_rerank", "engines/cliped", {
                "detector": "yolo", "verifier": "verifier", "class": "person", "prompt": START_PROMPT,
                "rank": "prompt", "confidence": 0.5, "gate": True, "ignore_new": False,
                "log_candidates": f"{LOG_DIR}/candidates.csv", **SEND_QUEUE},
                comment=f'Picks the tracked person among yolo\'s boxes; starts on "{START_PROMPT}" and follows the '
                        "verifier's prompt heartbeat (ignore_new off), so the prompt inference writes; logs "
                        "every candidate"),
            Service("box_log", "detection-log", "engines/detection-log", {
                "detector": "box_rerank", "images": False, "csv": f"{LOG_DIR}/boxes.csv"},
                comment="Logs the box box_rerank reports on each frame; no images"),
            Service("h264_encoder", "encoder", "defaults/h264", {
                "decoder_name": "h264_decoder",
                "converter": "qtivtransform ! video/x-raw(memory:GBM),format=NV12",
                "encoder": "qtic2venc control-rate=VBR-CFR target-bitrate=1500000 b-frames=0 idr-interval=30",
            }, environment="native",
                comment="The Tachyon's hardware H.264 encoder at 1.5 Mbps (VBR), under outage's 2-3 Mbps"),
            webp_encoder(),
        ]

    exps = []
    for name, (appears, length) in TRACKING.items():
        src = SOURCES[name]
        at = round(appears + PROMPT_LATENCY_S + OUTAGE_AFTER_S, 1)
        exps.append(Exp(name, src, ("outage",), fermi(src, True), lossy=True, trials=1, targets=(L.PERSON,),
                        outage=[[at, round(length - at, 1)]],
                        outage_note=f"{OUTAGE_AFTER_S:g} s after the inferred prompt reaches box_rerank (the hoodie "
                                    f"appears at {appears:g} s, +{PROMPT_LATENCY_S:g} s) to the video's end",
                        hooks={"end": ["sh", "-c", (
                            # The verifier's last prompt into the trial: the crop of the
                            # tracked box the GCS wrote (prompt.jpg), its distractors
                            # (d<N>.jpg), the verifier's text of it (description.txt), and
                            # any text prompt; cleared, so the next trial starts from the args
                            f'd={FILES}/{VERIFIER_LIVE}; mkdir -p "$d"; '
                            f'for p in "$d"/prompt.txt "$d"/prompt.jpg "$d"/prompt.jpeg "$d"/description.txt '
                            f'"$d"/class.txt "$d"/d*.jpg; do [ -e "$p" ] && mv "$p" "$EXPERIMENT_TRIAL_DIR/verifier_${{p##*/}}"; done; '
                            f'rm -f "$d"/replaced-*; true')]}))
    gcs = Service("inferred_mission", "inferred_mission_headless", "defaults/gcs", {
        "vehicle": "drone", "camera": "rgb", "flow": "inference_flow.toml",
        "detector": "detection",
        # The Track detector by its name in the planning flow, a template: {daemon}.{engine}
        "local_detector": "fermi.box_rerank",
        "verifier": "qwen3-vl:30b-a3b-instruct", "verifier_live_dir": f"{FILES}/{VERIFIER_LIVE}",
        "auto_deploy": False,
        "ai": {"inferred_mission": {"section_seconds": SECTION_S}}},
        comment=f"The headless GCS mission inference: {SECTION_S:g} s sections at 1 fps, with RF-DETR's boxes; on Track, "
                "checks the tracked box against the verifier's prompt and, if it isn't it, writes a "
                "crop of it to the verifier's live prompt (prompt.jpg). No deploy: the fake drone can't fly")
    lambda2 = [
        metrics("h264_rx", True), gcs, H264_DECODER,
        rfdetr(name="detection"),
        metrics("rx_metrics", True), WEBP_DECODER,
        Service("verifier", "verifier", "engines/cliped", {
            "prompt": START_PROMPT, "question": "box", "min_side": 224, "live": VERIFIER_LIVE, "log_tiles": True,
            "image_mode": "contrastive"},
            comment="The VLM (Qwen3-VL 30B, Ollama) that answers box_rerank's crops; its prompt follows "
                    f"{VERIFIER_LIVE}/prompt.jpg, the GCS's crop of the tracked box, which it describes by the "
                    "colors and features that set it apart from the distractors the GCS writes beside it "
                    "(d<N>.jpg; image_mode contrastive), not just its kind; the end hook moves them into the "
                    "trial directory"),
    ]
    # The inference's planning flow, which binds Track to box_rerank: the same
    # flow as a Flow Builder template, whose nodes are named by host
    # ({daemon}.{engine}, {vehicle}.{stream}), not local./remote.
    planning = "# The planning flow of 3.1's inference: flow.toml as a template, named by host\n" + flow
    for prefix, host in (("vehicle", "drone"), ("local", "fermi"), ("remote", "lambda2")):
        planning = planning.replace(f'"{prefix}.', f'"{host}.')
    return [Dir("3.1-inference", "3.1 headless GCS mission inference, with cliped tracking (box_rerank), outage",
                {"flow.toml": flow, "planning_flow.toml": planning}, lambda2, exps,
                [("planning_flow.toml", "inference_flow.toml")],
                results={"inference": (f"lambda2{FILES}/inferred_mission_headless/",
                                       "The inference's runs: each section's record, frames and target.png")},
                catalogs={"verifier": (f"lambda2{FILES}/*.verifier",
                                       "The verifier's crops in (in/) and the ones it said yes to (out/)")})]


def all_dirs() -> list[Dir]:
    return [*microbenchmarks(), *yolo(), *change_detection(), *cliped(), *inference()]


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
