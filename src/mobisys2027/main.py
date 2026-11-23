"""Auto-experiment runner for SteelEagle.

    experiment <dir>              run every experiment in dir
    experiment <dir> -e <name>    run only that experiment (repeatable)
    experiment <dir> -l           list the experiments in dir
    experiment <dir> -p <profile> run only that netsim profile (repeatable)
    experiment <dir> --overwrite  replace an earlier run's results in place
    experiment <dir> -t <n>       run only trial n of each profile, as a new trial
                                  (with --overwrite, replacing trial_<n>)
    experiment -r <dir_of_dirs>   run every experiment directory inside

An experiment directory holds an experiment.toml (see example/) and every file
it names. The toml holds a list of experiments ([[experiments]]), each either a
mission or a single video with its flow, run under its own netsim profiles; the
daemons, uploads, results and catalogs are shared by all of them. The uploads
(files and videos) are pushed to every daemon's files directory
(STEELEAGLE_FILES) once, before the directory's first experiment, except those
marked no_upload, which are assumed to already be there. Then, for each
experiment, for each of its netsim profiles, and for each trial, the runner:

  1. restarts the daemons with the experiment's configs for the profile,
  2. for the ambient profile only: gives the trial the link measured with
     iperf3 once, at the start of the run, before any daemon is configured
     (the trial's bandwidth timeline; see measure_ambient_once),
  3. starts the mission (upload, or CompileMission for a mission folder holding
     flow.toml and mission.json or mission.dsl; then load and start; if StartMission fails, the
     daemons get their own configs back and the run aborts) or sets the video's
     flow on the target vehicle, and starts netsim as soon as the vehicle acknowledges it
     (not for none or ambient, which run unattenuated), along with the power
     script (power/tachyon_power.py) on every remote daemon's host, over ssh,
     (after a warmup period, if one is set, for engines that load a model once
     they are running),
  4. waits for the duration (for a video, the video's length unless a
     duration is set), or for ENTER when the duration is 0, and then for the
     grace period, if one is set, with everything still running,
  5. stops the power script, the mission (Hold, then StopMission) and netsim, and
  6. collects the trial into <dir>/runs/<experiment>/<profile>/:
        trial_<n>/result/<key>/              result files, moved off their hosts
        trial_<n>/catalog/<key>/             catalog files, copied
        trial_<n>/timeline.csv               bandwidth netsim allowed over time
        trial_<n>/netsim.log                 netsim's output
        trial_<n>/power_<daemon>.csv         a remote's power draw over time
        trial_<n>/power_<daemon>.log         the power script's messages
        stats_<profile>_<experiment>.csv     totals, bandwidth and power statistics, a row per trial
        timeline_<profile>_<experiment>.csv  every trial's timeline, with a trial column
        power_<profile>_<experiment>.csv     every trial's power timelines, with trial and daemon columns

An experiment's hooks ([experiments.hooks]) are commands run in the
experiment directory during each trial: setup before the mission or flow
starts, start once it has started (step 3), grace when the grace period starts
(step 4), grace_end once the grace period is over, and end once the trial has
been stopped and collected (step 6), however it ended. With a done hook, the
runner polls it while the trial runs: once it exits 0 the trial ends early and
skips the grace period, and in the grace period it ends the wait, so duration
and grace become the most the trial waits.

Once an experiment's last trial is collected, each directory a result names (or
its glob matches) is removed from its host (rm -rf), along with anything left in it.

Restarting the daemons means every remote daemon and then the local
orchestrator is reconfigured, which restarts it, and the runner waits for the
vehicle to be listed and running. A daemon's config is found by its name in
[daemons] and the experiment's name: for daemon "fermi", experiment "track"
and profile "normal", fermi_track_normal.toml is used if it exists; otherwise
the config of the previous profile carries over, or, for the first profiles,
fermi_track.toml, or else fermi.toml.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import fnmatch
import glob
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import grpc
from steeleagle_protocol.v1.services.daemon import (
    daemon_pb2,
    daemon_pb2_grpc,
    transfer_pb2,
    transfer_pb2_grpc,
)
from steeleagle_protocol.v1.services.driver import control_pb2, control_pb2_grpc
from steeleagle_protocol.v1.services.mission import mission_pb2, mission_pb2_grpc
from steeleagle_protocol.v1.services.vehicle import (
    data_pb2,
    data_pb2_grpc,
    upload_pb2,
    upload_pb2_grpc,
)

EXPERIMENT_FILE = "experiment.toml"
# A mission folder (mission.file naming a directory) holds a flow and one
# mission source, JSON or DSL, and is compiled on the vehicle (CompileMission)
# instead of uploaded as a binary; with its map features, a .geojson file, if
# the mission names any (e.g. a patrol's params.Polygon_<name>)
MISSION_FLOW = "flow.toml"
MISSION_JSON = "mission.json"
MISSION_DSL = "mission.dsl"
MISSION_GEOJSON_SUFFIX = ".geojson"


def mission_source(folder: Path) -> Path:
    """A mission folder's mission source: its MISSION_JSON or MISSION_DSL
    (exactly one of them)."""
    found = [folder / n for n in (MISSION_JSON, MISSION_DSL) if (folder / n).is_file()]
    if len(found) != 1:
        raise ValueError(
            f"mission folder {folder} must have exactly one of {MISSION_JSON} or {MISSION_DSL}"
            + (", not both" if found else "")
        )
    return found[0]


def mission_geojson(folder: Path) -> Path | None:
    """A mission folder's map features: its one .geojson file, or None."""
    found = sorted(folder.glob(f"*{MISSION_GEOJSON_SUFFIX}"))
    if len(found) > 1:
        raise ValueError(f"mission folder {folder} has more than one {MISSION_GEOJSON_SUFFIX} file: "
                         + ", ".join(f.name for f in found))
    return found[0] if found else None


VIDEOS_DIR = "videos"  # looked for at and above the experiment directory, before it
# Images and other media the experiments upload (e.g. a prompt image), looked for like videos/
MEDIA_DIR = "media"
NETSIM = Path(__file__).resolve().parent / "netsim" / "netsim.py"
# Run on each remote daemon's host for the length of each trial, to measure
# its power draw; written for the Particle Tachyon
POWER_SCRIPT = Path(__file__).resolve().parent / "power" / "tachyon_power.py"
POWER_INTERVAL = 1.0  # seconds per row of the power timeline
# Profiles that aren't netsim's, and run the trial without it: none just
# runs it unattenuated, and ambient first measures the link with iperf3
NONE = "none"
AMBIENT = "ambient"
NO_NETSIM = {NONE, AMBIENT}
NETSIM_PROFILES = [
    "normal", "limited", "degraded", "intermittent", "dripfeed", "denied", "malicious", "outage", "delay",
    NONE, AMBIENT,
]
# Points of each trial a hook ([experiments.hooks]) can run at: before the
# mission or flow starts, when it starts, when its grace period starts, once
# the grace period is over, and once the trial is stopped and collected; done
# is polled instead, and ends the trial's wait once it exits 0
HOOKS = ("setup", "start", "done", "grace", "grace_end", "end")
# Seconds between polls of the done hook
DONE_POLL_S = 2.0
IPERF_PORT = 5201
IPERF_DURATION = 30.0
ROUTE_HEADER = "x-route-key"
CHUNK_SIZE = 1 << 20  # same chunk size as `eagle transfer-file`
RPC_TIMEOUT = 60.0
READY_TIMEOUT = 300.0
# Where each daemon's own config is saved before an experiment config replaces
# it, until it is restored after the trial. A run that dies before restoring
# leaves it here, and the next run restores that config rather than saving the
# experiment config it finds on the daemon.
ORIGINALS_DIR = Path.home() / ".local" / "state" / "steeleagle-experiment" / "original-configs"
# The services the runner stopped at the end of a trial, which stay stopped
# across the restore (StopService lasts across reconfigures), so it restarts
# them after it. Kept until then, so a run that dies in between leaves them
# for the next run's restore.
STOPPED_FILE = ORIGINALS_DIR / "stopped-services.json"
# A package install downloads the package, which can be slow
INSTALL_TIMEOUT = 600.0
# How long a daemon may report that it failed to start its services before
# the wait for it gives up, in case the report is from before its restart
START_FAILURE_GRACE = 15.0
# How often a wait says what it is still waiting for
PROGRESS_INTERVAL = 15.0
# Daemon files directory (STEELEAGLE_FILES) relative to the home directory,
# for rsyncing a video to a remote that isn't registered
REMOTE_FILES_DIR = ".local/share/steeleagle/files/"
# Where install.sh puts eagle and its tools, which isn't on the PATH of the
# non-interactive shell ssh runs a command in
REMOTE_PATH = 'PATH="$HOME/.local/bin:$PATH"'


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# Seconds a trial is assumed to spend outside its duration and grace (daemon
# restarts, collecting results) until one has been timed
TRIAL_OVERHEAD_GUESS = 60.0
BAR_WIDTH = 30


def fmt_secs(secs: float) -> str:
    secs = max(0, int(round(secs)))
    h, rest = divmod(secs, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Progress:
    """Counts the trials of the run and estimates when it ends. A trial is
    expected to take its duration and grace, plus the overhead of the trials
    timed so far (restarts, collection), or TRIAL_OVERHEAD_GUESS before any.
    A trial run until ENTER (duration 0) counts only its overhead."""

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.pending: list[float] = []  # expected run time (duration + grace) of each trial left
        self.total = 0
        self.done = 0
        self.overheads: list[float] = []
        self.trial_started: float | None = None
        self.trial_expected = 0.0

    def add(self, exp: "Experiment", n: int) -> None:
        """Counts n trials of exp to run."""
        run = exp.phase.warmup + exp.phase.duration + exp.phase.grace
        self.pending += [run] * n
        self.total += n

    @property
    def overhead(self) -> float:
        return sum(self.overheads) / len(self.overheads) if self.overheads else TRIAL_OVERHEAD_GUESS

    def remaining(self) -> float:
        """Estimated seconds until the run ends."""
        left = sum(r + self.overhead for r in self.pending)
        if self.trial_started is not None:
            spent = time.monotonic() - self.trial_started
            left += max(0.0, self.trial_expected + self.overhead - spent)
        return left

    def bar(self, fraction: float) -> str:
        fraction = min(max(fraction, 0.0), 1.0)
        filled = round(fraction * BAR_WIDTH)
        return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + f"] {fraction * 100:5.1f}%"

    def summary(self) -> str:
        """The run's progress: trials done, elapsed, and the estimated end."""
        left = self.remaining()
        elapsed = time.monotonic() - self.started
        frac = elapsed / (elapsed + left) if elapsed + left > 0 else 0.0
        eta = datetime.now() + timedelta(seconds=left)
        return (f"{self.bar(frac)} {self.done}/{self.total} trials, elapsed {fmt_secs(elapsed)}, "
                f"~{fmt_secs(left)} left, done ~{eta:%a %H:%M}")

    def start_trial(self) -> None:
        self.trial_expected = self.pending.pop(0) if self.pending else 0.0
        self.trial_started = time.monotonic()
        log(f"progress {self.summary()}")

    def end_trial(self) -> None:
        if self.trial_started is None:
            return
        spent = time.monotonic() - self.trial_started
        self.overheads.append(max(0.0, spent - self.trial_expected))
        self.trial_started = None
        self.done += 1

    def sleep(self, secs: float, what: str) -> None:
        """Sleeps `secs`, drawing a live bar of the wait on a terminal."""
        if secs <= 0:
            return
        if not sys.stdout.isatty():
            time.sleep(secs)
            return
        end = time.monotonic() + secs
        try:
            while (now := time.monotonic()) < end:
                line = (f"\r  {what} {self.bar(1 - (end - now) / secs)} {fmt_secs(end - now)} left | "
                        f"run: {self.done}/{self.total} trials, ~{fmt_secs(self.remaining())} left")
                cols = shutil.get_terminal_size((120, 20)).columns
                sys.stdout.write(line[: cols - 1].ljust(cols - 1))
                sys.stdout.flush()
                time.sleep(min(1.0, end - now))
        finally:
            sys.stdout.write("\r" + " " * (shutil.get_terminal_size((120, 20)).columns - 1) + "\r")
            sys.stdout.flush()


PROGRESS = Progress()


def default_socket() -> str:
    """The daemon control socket, as `eagle` finds it (XDG runtime dir).
    eagled is a user service, so under sudo it is the invoking user's socket."""
    sudo_uid = os.environ.get("SUDO_UID")
    if os.getuid() == 0 and sudo_uid:
        runtime = f"/run/user/{sudo_uid}"
    else:
        runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return os.path.join(runtime, "steeleagle", "eagled.sock")


# --------------------------------------------------------------------------
# ssh
# --------------------------------------------------------------------------

# Every ssh the runner starts, directly or through rsync, shares one master
# connection per host (OpenSSH connection multiplexing), so a host's password
# is asked for once, when open_ssh connects to it, rather than on every call.
_ssh_dir: str | None = None
_ssh_hosts: set[str] = set()


def ssh_opts() -> list[str]:
    global _ssh_dir
    if _ssh_dir is None:
        # Short, as the sockets' paths must fit in about 100 characters
        _ssh_dir = tempfile.mkdtemp(prefix="se-ssh-", dir="/tmp")
        atexit.register(close_ssh)
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={_ssh_dir}/%C",
        "-o", "ControlPersist=yes",
    ]


def ssh(host: str, command: str) -> list[str]:
    """An ssh command line that runs command on host."""
    _ssh_hosts.add(host)
    return ["ssh", *ssh_opts(), host, command]


def rsync_cmd(host: str) -> list[str]:
    """An rsync command line whose ssh to host shares its master connection."""
    _ssh_hosts.add(host)
    return ["rsync", "-e", shlex.join(["ssh", *ssh_opts()])]


def open_ssh(hosts: list[str]) -> None:
    """Connects to each host up front, asking for its password (or host key
    check) on the terminal now, rather than in the middle of a run. The
    connection stays open in the background for later calls to share."""
    for host in hosts:
        _ssh_hosts.add(host)
        log(f"connecting to {host} over ssh")
        # -f: go to the background once logged in; -N: run no command
        # ControlMaster=auto makes this the master, as none is open yet
        r = subprocess.run(["ssh", *ssh_opts(), "-f", "-N", host])
        if r.returncode:
            log(f"warning: couldn't connect to {host} over ssh (exit {r.returncode}); later calls will retry")


def close_ssh() -> None:
    """Closes the master connections, and removes their sockets."""
    if _ssh_dir is None:
        return
    for host in sorted(_ssh_hosts):
        subprocess.run(
            ["ssh", *ssh_opts(), "-O", "exit", host],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    shutil.rmtree(_ssh_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Experiment config
# --------------------------------------------------------------------------


@dataclass
class Phase:
    kind: str  # "mission" or "video"
    # the mission (a binary, or a folder with MISSION_FLOW and MISSION_JSON
    # or MISSION_DSL, compiled on the vehicle), or the video's flow
    file: Path
    auto: bool
    # seconds to run each trial; 0 waits for ENTER. For a video with no
    # duration set, the video's own length (duration_from_video)
    duration: float
    # the video, for "video": one of the directory's uploads, found here (if it
    # is) for its length
    video: Path | None = None
    # no_upload: the mission or video is not pushed, and is assumed to already
    # be on its daemons (a video's comes from its upload's no_upload)
    no_upload: bool = False
    duration_from_video: bool = False
    # seconds to keep running after the duration (or ENTER), before the trial
    # is stopped, e.g. for reintegration or reachback to finish transmitting
    grace: float = 0.0
    # seconds to wait once the daemons report their services running, before
    # the mission or flow starts: an engine that loads a model after it starts
    # (e.g. CLIP) is only listed as running, and would miss the first frames
    warmup: float = 0.0


@dataclass
class Upload:
    path: Path  # in a videos/ folder at or above the experiment directory, or in it
    name: str  # name it is saved as in each daemon's files directory
    # not pushed: assumed to already be in every daemon's files directory as name
    no_upload: bool = False


@dataclass
class DaemonCfg:
    key: str
    orchestrator: bool
    prefix: str

    @property
    def ssh_host(self) -> str:
        return self.prefix.rstrip(":")

    @property
    def hostname(self) -> str:
        return self.ssh_host.rsplit("@", 1)[-1] or self.key


@dataclass
class Experiment:
    dir: Path
    name: str
    vehicle: str
    phase: Phase
    profiles: list[str]
    interface: str
    port: int
    netsim_args: list[str]
    failpoints: str  # netsim --failpoints value, used only by the malicious and outage profiles
    trials: int  # runs of each profile
    iperf_port: int  # ambient: port iperf3 measures the link on
    iperf_duration: float  # ambient: seconds iperf3 measures the link for
    daemons: dict[str, DaemonCfg]
    configs: dict[str, dict[str, Path]]  # profile -> daemon key -> config file
    # the directory's uploads, shared by its experiments: pushed to every
    # daemon's files directory once, before its first experiment runs
    uploads: list[Upload] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    catalogs: dict[str, str] = field(default_factory=dict)
    # catalogs moved aside on their host after each trial, into a directory of
    # their own in its files directory, to be transferred later, rather than copied
    defer_pull: set[str] = field(default_factory=set)
    # Whether to measure the remote daemons' power draw, and the seconds per
    # row of its timeline
    power: bool = True
    power_interval: float = POWER_INTERVAL
    # Shell commands (argv lists) run in the experiment directory at points of
    # each trial: "setup", "start", "grace", "grace_end" and "end", and "done",
    # polled (see HOOKS)
    hooks: dict[str, list[str]] = field(default_factory=dict)

    @property
    def orchestrator(self) -> DaemonCfg:
        return next(d for d in self.daemons.values() if d.orchestrator)

    @property
    def remotes(self) -> list[DaemonCfg]:
        return [d for d in self.daemons.values() if not d.orchestrator]


def read_toml(exp_dir: Path) -> dict:
    path = exp_dir / EXPERIMENT_FILE
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None


def experiment_tables(exp_dir: Path, raw: dict, names: list[str] | None = None) -> list[dict]:
    """The [[experiments]] tables, in order, each checked for a unique name;
    all of them, or, if names are given, only those."""
    if "experiment" in raw:
        raise ValueError(
            f"{exp_dir}: [experiment] is now a list of [[experiments]] tables, one per "
            f"experiment, with sections [experiments.mission], [experiments.video] and "
            f"[experiments.netsim] (after a [[experiments]] header, [experiment.<x>] makes "
            f"a separate table)"
        )
    if "netsim" in raw:
        raise ValueError(f"{exp_dir}: [netsim] is now set per experiment, as [experiments.netsim]")
    tables = raw.get("experiments")
    if not isinstance(tables, list) or not tables or not all(isinstance(t, dict) for t in tables):
        raise ValueError(f"{exp_dir}: no experiments; add one or more [[experiments]] tables")
    known = []
    for i, t in enumerate(tables, 1):
        name = t.get("name")
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(
                f"{exp_dir}: experiment #{i} needs a name (usable in a file name), not {name!r}"
            )
        known.append(name)
    dups = sorted({n for n in known if known.count(n) > 1})
    if dups:
        raise ValueError(f"{exp_dir}: more than one experiment is named {', '.join(dups)}")
    if names:
        unknown = [n for n in names if n not in known]
        if unknown:
            raise ValueError(
                f"{exp_dir}: no experiment named {', '.join(unknown)}; "
                f"its experiments are {', '.join(known)}"
            )
        tables = [t for t in tables if t["name"] in names]
    return tables


def load_experiments(exp_dir: Path, names: list[str] | None = None) -> list[Experiment]:
    """Loads and checks the experiments in exp_dir's experiment.toml: all of
    them, in order, or, if names are given, only those."""
    raw = read_toml(exp_dir)
    tables = experiment_tables(exp_dir, raw, names)

    daemons = {}
    for key, d in raw.get("daemons", {}).items():
        orch = bool(d.get("orchestrator", False))
        if "config" in d:
            raise ValueError(
                f"{exp_dir}: daemons.{key}.config is no longer supported; name the file "
                f"{key}.toml, {key}_<experiment>.toml or {key}_<experiment>_<profile>.toml instead"
            )
        if "pass" in d:
            raise ValueError(
                f"{exp_dir}: daemons.{key}.pass is no longer supported; ssh and rsync now "
                f"prompt for the password (and host key checks) on the terminal"
            )
        daemons[key] = DaemonCfg(key, orch, d.get("prefix", ""))
    if sum(d.orchestrator for d in daemons.values()) != 1:
        raise ValueError(f"{exp_dir}: exactly one daemon must have orchestrator = true")

    results: dict[str, str] = {}
    for key, value in raw.get("results", {}).items():
        if isinstance(value, dict) and "defer_pull" in value:
            raise ValueError(
                f"{exp_dir}: results.{key}: defer_pull goes under [catalogs], not [results], "
                f"since results are removed from their host once they are pulled"
            )
        if not isinstance(value, str):
            raise ValueError(f"{exp_dir}: results.{key} must be a path")
        results[key] = value
    catalogs: dict[str, str] = {}
    defer_pull: set[str] = set()
    for key, value in raw.get("catalogs", {}).items():
        if isinstance(value, dict):
            unknown = set(value) - {"path", "defer_pull"}
            if unknown or not isinstance(value.get("path"), str):
                raise ValueError(
                    f"{exp_dir}: catalogs.{key} must be a path, or a table "
                    f"{{ path = \"<daemon>/<path>\", defer_pull = true }}"
                    + (f" (unknown keys {', '.join(sorted(unknown))})" if unknown else "")
                )
            if not isinstance(value.get("defer_pull", False), bool):
                raise ValueError(f"{exp_dir}: catalogs.{key}.defer_pull must be true or false")
            catalogs[key] = value["path"]
            if value.get("defer_pull", False):
                defer_pull.add(key)
        elif isinstance(value, str):
            catalogs[key] = value
        else:
            raise ValueError(f"{exp_dir}: catalogs.{key} must be a path or a table")
    for table in (results, catalogs):
        for key, src in table.items():
            host = src.split("/", 1)[0]
            if host not in daemons:
                raise ValueError(f"{exp_dir}: {key} = {src!r} names unknown daemon {host!r}")

    uploads = parse_uploads(exp_dir, raw.get("uploads", []))
    check_results_spare_uploads(exp_dir, results, uploads)
    return [
        parse_experiment(exp_dir, t, daemons, results, catalogs, defer_pull, uploads)
        for t in tables
    ]


def parse_uploads(exp_dir: Path, raw) -> list[Upload]:
    """Parses the top-level uploads: a list of file names, each optionally a
    table { file = "...", push_as = "...", no_upload = true }. Each file is
    looked for in the nearest videos/ folder at or above exp_dir, then in
    exp_dir. A no_upload file need not be here; only its name is used."""
    if not isinstance(raw, list):
        raise ValueError(f"{exp_dir}: uploads must be a list, or [[uploads]] tables")
    uploads = []
    for entry in raw:
        if isinstance(entry, str):
            entry = {"file": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str) or not entry["file"]:
            raise ValueError(
                f"{exp_dir}: invalid uploads entry {entry!r}; want a file name or "
                f'{{ file = "...", push_as = "...", no_upload = true }}'
            )
        unknown = set(entry) - {"file", "push_as", "no_upload"}
        if unknown:
            raise ValueError(f"{exp_dir}: uploads entry {entry!r} has unknown keys {sorted(unknown)}")
        name = entry["file"]
        if "/" in name:
            raise ValueError(f"{exp_dir}: uploads file must be a file name, not {name!r}")
        no_upload = entry.get("no_upload", False)
        if not isinstance(no_upload, bool):
            raise ValueError(f"{exp_dir}: uploads {name}: no_upload must be true or false")
        push_as = entry.get("push_as", "")
        if not isinstance(push_as, str) or "/" in push_as:
            raise ValueError(f"{exp_dir}: uploads {name}: push_as must be a file name, not {push_as!r}")
        if no_upload and push_as:
            raise ValueError(f"{exp_dir}: uploads {name}: push_as does nothing with no_upload; leave it out")
        path = find_video(exp_dir, name)
        if not no_upload and not path.is_file():
            raise ValueError(
                f"{exp_dir}: uploads file {name} is in neither a {VIDEOS_DIR}/ folder nor {exp_dir}"
            )
        uploads.append(Upload(path, push_as or name, no_upload))
    names = [u.name for u in uploads]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"{exp_dir}: uploads has more than one file named {', '.join(dups)}")
    return uploads


def check_results_spare_uploads(exp_dir: Path, results: dict[str, str], uploads: list[Upload]) -> None:
    """Rejects a result that would move an upload off its daemon: uploads are
    pushed only once, so the next trial would run without it."""
    for key, src in results.items():
        path = src.split("/", 1)[1].rstrip("/") if "/" in src else ""
        if REMOTE_FILES_DIR not in path + "/":
            continue
        files_dir = path[: (path + "/").index(REMOTE_FILES_DIR) + len(REMOTE_FILES_DIR)].rstrip("/")
        for u in uploads:
            if fnmatch.fnmatchcase(f"{files_dir}/{u.name}", path) or fnmatch.fnmatchcase(files_dir, path):
                raise ValueError(
                    f"{exp_dir}: results.{key} = {src!r} would move upload {u.name} off its "
                    f"daemon, and uploads are pushed only once; collect it as a catalog instead"
                )


def find_video(exp_dir: Path, name: str) -> Path:
    """Where the video (or other upload) name is: in the nearest videos/ or
    media/ folder at or above exp_dir that has it, or else in exp_dir itself
    (which may not have it either)."""
    for d in (exp_dir.resolve(), *exp_dir.resolve().parents):
        for folder in (VIDEOS_DIR, MEDIA_DIR):
            path = d / folder / name
            if path.is_file():
                return path
    return exp_dir / name


def parse_experiment(
    exp_dir: Path, exp: dict, daemons: dict[str, DaemonCfg],
    results: dict[str, str], catalogs: dict[str, str], defer_pull: set[str],
    uploads: list[Upload],
) -> Experiment:
    """Parses one [[experiments]] table."""
    name = exp["name"]
    where = f"{exp_dir}: experiment {name}"
    vehicle = exp.get("vehicle", "")
    if not vehicle:
        raise ValueError(f"{where}: vehicle is not set")
    if "files" in exp:
        raise ValueError(
            f"{where}: files is no longer set per experiment; declare them once for the "
            f"directory, at the top level, as [[uploads]] tables (pushed once, not every trial)"
        )

    def exp_file(key: str, value: str) -> Path:
        path = exp_dir / value
        if not path.is_file():
            raise ValueError(f"{where}: {key} file {path} does not exist")
        return path

    def flag(sect: dict, key: str) -> bool:
        v = sect.get(key.rsplit(".", 1)[-1], False)
        if not isinstance(v, bool):
            raise ValueError(f"{where}: {key} must be true or false, not {v!r}")
        return v

    def duration(kind: str, sect: dict) -> float | None:
        if "duration" not in sect:
            return None
        d = sect["duration"]
        if not isinstance(d, (int, float)) or isinstance(d, bool) or d < 0:
            raise ValueError(f"{where}: {kind}.duration must be a number >= 0, not {d!r}")
        return float(d)

    def grace(kind: str, sect: dict, key: str = "grace") -> float:
        g = sect.get(key, 0)
        if not isinstance(g, (int, float)) or isinstance(g, bool) or g < 0:
            raise ValueError(f"{where}: {kind}.{key} must be a number >= 0, not {g!r}")
        return float(g)

    def file_name(key: str, value) -> str:
        if not isinstance(value, str) or not value or "/" in value:
            raise ValueError(f"{where}: {key} must be a file name, not {value!r}")
        return value

    phases = []
    sect = exp.get("mission", {})
    if sect.get("file"):
        no_upload = flag(sect, "mission.no_upload")
        if no_upload:
            # Not pushed, so it need not exist here; only its name is used
            mission = exp_dir / file_name("mission.file (with no_upload)", sect["file"])
        else:
            mission = exp_dir / sect["file"]
            if mission.is_dir():
                if not (mission / MISSION_FLOW).is_file():
                    raise ValueError(f"{where}: mission folder {mission} has no {MISSION_FLOW}")
                try:
                    mission_source(mission)
                    mission_geojson(mission)
                except ValueError as e:
                    raise ValueError(f"{where}: {e}") from None
            else:
                mission = exp_file("mission.file", sect["file"])
        phases.append(Phase(
            "mission", mission, flag(sect, "mission.auto"),
            duration("mission", sect) or 0.0, no_upload=no_upload,
            grace=grace("mission", sect), warmup=grace("mission", sect, "warmup"),
        ))
    sect = exp.get("video", {})
    if sect.get("video") or sect.get("flow"):
        if not (sect.get("video") and sect.get("flow")):
            raise ValueError(f"{where}: video needs both video and flow set")
        if isinstance(sect["video"], list):
            raise ValueError(
                f"{where}: video.video is now a single video; for several videos, "
                f"add an experiment for each"
            )
        for key in ("no_upload", "push_as"):
            if key in sect:
                raise ValueError(
                    f"{where}: video.{key} is no longer set per experiment; set it on the "
                    f"video's entry in the top-level [[uploads]]"
                )
        vname = file_name("video.video", sect["video"])
        upload = next((u for u in uploads if vname in (u.path.name, u.name)), None)
        if upload is None:
            raise ValueError(
                f"{where}: video {vname} is not in the top-level uploads; add it as "
                f'[[uploads]] file = "{vname}" (with no_upload = true if it is already '
                f"on the daemons)"
            )
        video, no_upload = upload.path, upload.no_upload
        d = duration("video", sect)
        from_video = d is None
        if from_video:
            # No duration set: the video runs for its own length
            if no_upload and not video.is_file():
                raise ValueError(
                    f"{where}: with no_upload, video {video.name} is not in a {VIDEOS_DIR}/ "
                    f"folder or the experiment directory to find its length; set video.duration"
                )
            try:
                d = video_length(video)
            except ValueError as e:
                raise ValueError(f"{where}: {e}; set video.duration instead") from None
        flow = exp_file("video.flow", sect["flow"])
        try:
            check_flow(flow)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None
        phases.append(Phase(
            "video", flow, flag(sect, "video.auto"), d,
            video=video, no_upload=no_upload, duration_from_video=from_video,
            grace=grace("video", sect), warmup=grace("video", sect, "warmup"),
        ))
    if len(phases) != 1:
        raise ValueError(f"{where}: set exactly one of mission (file) or video (video and flow)")
    phase = phases[0]

    ns = exp.get("netsim", {})
    profiles = ns.get("profiles", ["*"])
    if "*" in profiles:
        profiles = NETSIM_PROFILES
    for p in profiles:
        if p not in NETSIM_PROFILES:
            raise ValueError(f"{where}: unknown netsim profile {p!r}")
    if "interface" not in ns or "port" not in ns:
        raise ValueError(f"{where}: netsim.interface and netsim.port must be set")
    # Optional extra netsim flags, e.g. args = ["--direction", "egress", "--seed", "1"]
    netsim_args = [str(a) for a in ns.get("args", [])]
    if "failpoint_map" in ns:
        raise ValueError(
            f"{where}: netsim.failpoint_map is no longer supported; each experiment has "
            f"one video, so set netsim.failpoints"
        )
    failpoints = parse_failpoints(ns.get("failpoints", []), where)
    trials = ns.get("trials", 1)
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError(f"{where}: netsim.trials must be an integer >= 1, not {trials!r}")
    iperf_port = ns.get("iperf_port", IPERF_PORT)
    if not isinstance(iperf_port, int) or isinstance(iperf_port, bool) or not 1 <= iperf_port <= 65535:
        raise ValueError(f"{where}: netsim.iperf_port must be a port number, not {iperf_port!r}")
    iperf_duration = ns.get("iperf_duration", IPERF_DURATION)
    if not isinstance(iperf_duration, (int, float)) or isinstance(iperf_duration, bool) or iperf_duration < 1:
        raise ValueError(f"{where}: netsim.iperf_duration must be a number >= 1, not {iperf_duration!r}")

    configs = resolve_configs(exp_dir, name, list(daemons), list(profiles))

    pw = exp.get("power", {})
    if not isinstance(pw, dict):
        raise ValueError(f"{where}: power must be a table, [experiments.power]")
    unknown = set(pw) - {"enabled", "interval"}
    if unknown:
        raise ValueError(f"{where}: power has unknown keys {', '.join(sorted(unknown))}")
    power = flag(pw, "power.enabled") if "enabled" in pw else True
    power_interval = pw.get("interval", POWER_INTERVAL)
    if not isinstance(power_interval, (int, float)) or isinstance(power_interval, bool) or power_interval < 0.1:
        raise ValueError(f"{where}: power.interval must be a number >= 0.1, not {power_interval!r}")

    hk = exp.get("hooks", {})
    if not isinstance(hk, dict):
        raise ValueError(f"{where}: hooks must be a table, [experiments.hooks]")
    unknown = set(hk) - set(HOOKS)
    if unknown:
        raise ValueError(
            f"{where}: hooks has unknown keys {', '.join(sorted(unknown))}; "
            f"it takes {', '.join(HOOKS)}"
        )
    hooks = {}
    for key, cmd in hk.items():
        if not isinstance(cmd, list) or not all(isinstance(a, str) for a in cmd):
            raise ValueError(
                f"{where}: hooks.{key} must be a command as a list of strings, "
                f'e.g. ["sh", "-c", "echo hi"], not {cmd!r}'
            )
        if cmd:
            hooks[key] = cmd

    return Experiment(
        exp_dir, name, vehicle, phase, list(profiles), ns["interface"], int(ns["port"]),
        netsim_args, failpoints, trials, iperf_port, float(iperf_duration),
        daemons, configs, uploads, results, catalogs, defer_pull,
        power, float(power_interval), hooks,
    )


def video_length(path: Path) -> float:
    """A video's length in seconds, from ffprobe if it is installed, or else
    from the header of an MP4/MOV file."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            d = float(json.loads(r.stdout)["format"]["duration"])
            if d > 0:
                return d
    except (FileNotFoundError, subprocess.TimeoutExpired, KeyError, ValueError):
        pass
    d = mp4_length(path)
    if d is None or d <= 0:
        raise ValueError(
            f"can't find the length of video {path.name} (install ffprobe, or use an MP4/MOV file)"
        )
    return d


def mp4_length(path: Path) -> float | None:
    """Reads the length from an MP4/MOV file's moov/mvhd box, or None."""

    def boxes(f, end: int):
        """Yields (type, payload offset, payload size) of the boxes up to end."""
        pos = f.tell()
        while pos + 8 <= end:
            f.seek(pos)
            size, kind = struct.unpack(">I4s", f.read(8))
            header = 8
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
                header = 16
            elif size == 0:
                size = end - pos
            if size < header:
                return
            yield kind, pos + header, size - header
            pos += size

    try:
        with open(path, "rb") as f:
            end = f.seek(0, os.SEEK_END)
            f.seek(0)
            for kind, off, size in boxes(f, end):
                if kind != b"moov":
                    continue
                f.seek(off)
                for kind2, off2, size2 in boxes(f, off + size):
                    if kind2 != b"mvhd" or size2 < 20:
                        continue
                    f.seek(off2)
                    version = f.read(4)[0]
                    if version == 1:
                        f.seek(16, os.SEEK_CUR)
                        timescale, duration = struct.unpack(">IQ", f.read(12))
                    else:
                        f.seek(8, os.SEEK_CUR)
                        timescale, duration = struct.unpack(">II", f.read(8))
                    return duration / timescale if timescale else None
    except (OSError, struct.error, IndexError):
        pass
    return None


def resolve_configs(
    exp_dir: Path, exp_name: str, keys: list[str], profiles: list[str],
) -> dict[str, dict[str, Path]]:
    """Picks each daemon's config for each profile, in profile order:
    <key>_<experiment>_<profile>.toml if it exists, else the config the
    previous profile used, else <key>_<experiment>.toml, else <key>.toml."""
    configs: dict[str, dict[str, Path]] = {p: {} for p in profiles}
    for key in keys:
        current: Path | None = None
        for p in profiles:
            specific = exp_dir / f"{key}_{exp_name}_{p}.toml"
            if specific.is_file():
                current = specific
            elif current is None:
                bases = [exp_dir / f"{key}_{exp_name}.toml", exp_dir / f"{key}.toml"]
                current = next((b for b in bases if b.is_file()), None)
                if current is None:
                    raise ValueError(
                        f"{exp_dir}: experiment {exp_name}: no config for daemon {key!r} under "
                        f"profile {p!r}; add {specific.name}, {bases[0].name} or {bases[1].name}"
                    )
            configs[p][key] = current
    return configs


def parse_failpoints(raw, where: str, key: str = "netsim.failpoints") -> str:
    """Turns netsim.failpoints, a list of [event_time, dead_time] pairs in
    seconds, into netsim's "30:10,90:5" form."""
    if not isinstance(raw, list):
        raise ValueError(f"{where}: {key} must be a list of [event_time, dead_time] pairs")
    pairs = []
    for fp in raw:
        if (
            not isinstance(fp, list) or len(fp) != 2
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in fp)
            or fp[0] < 0 or fp[1] <= 0
        ):
            raise ValueError(
                f"{where}: invalid failpoint {fp!r} in {key}; want [event_time >= 0, dead_time > 0] in seconds"
            )
        pairs.append(f"{fp[0]:g}:{fp[1]:g}")
    return ",".join(pairs)

# --------------------------------------------------------------------------
# Daemon client
# --------------------------------------------------------------------------


class Daemon:
    """A client of the local daemon's control socket, like `eagle`. Calls with a
    route key are proxied by the daemon to that vehicle or remote daemon."""

    def __init__(self, socket: str):
        self.socket = socket
        self.channel = grpc.insecure_channel("unix://" + socket)
        self.daemon = daemon_pb2_grpc.DaemonServiceStub(self.channel)
        self.transfer = transfer_pb2_grpc.TransferServiceStub(self.channel)
        self.loader = upload_pb2_grpc.LoaderServiceStub(self.channel)
        self.mission = mission_pb2_grpc.MissionServiceStub(self.channel)
        self.data = data_pb2_grpc.DataServiceStub(self.channel)
        self.control = control_pb2_grpc.ControlServiceStub(self.channel)

    @staticmethod
    def route(key: str | None):
        return ((ROUTE_HEADER, key),) if key else None

    def list_services(self, timeout: float = 10.0) -> daemon_pb2.ListServicesResponse:
        return self.daemon.ListServices(daemon_pb2.ListServicesRequest(), timeout=timeout)

    def get_configuration(self, route: str | None = None) -> str:
        resp = self.daemon.GetConfiguration(
            daemon_pb2.GetConfigurationRequest(), metadata=self.route(route), timeout=RPC_TIMEOUT,
        )
        return resp.config_toml.decode()

    def configure(self, toml_text: str, route: str | None = None) -> None:
        self.daemon.Configure(
            daemon_pb2.ConfigureRequest(config_toml=toml_text),
            metadata=self.route(route), timeout=RPC_TIMEOUT,
        )

    def wait_ready(self, vehicle: str, remotes: list[DaemonCfg], timeout: float = READY_TIMEOUT) -> str:
        """Waits until ListServices answers, every remote is registered and the
        vehicle is running. Returns the vehicle's qualified name. Says what it
        is waiting for every PROGRESS_INTERVAL seconds, and gives up early if
        a daemon keeps reporting that it failed to start its services."""
        start = time.monotonic()
        deadline = start + timeout
        next_progress = start + PROGRESS_INTERVAL
        failing_since: float | None = None
        last = "daemon not reachable"
        while time.monotonic() < deadline:
            failed = None
            try:
                resp = self.list_services()
            except grpc.RpcError as e:
                last = f"ListServices: {e.code().name} {e.details()}"
                if is_start_failure(e.details()):
                    failed = f"orchestrator: {e.details()}"
                resp = None
            if resp is not None:
                missing = []
                for r in remotes:
                    listed = find_remote(resp, r, include_failed=True)
                    if listed is None:
                        missing.append(r.key)
                    elif listed.error:
                        missing.append(f"{r.key} ({listed.error})")
                        if is_start_failure(listed.error):
                            failed = f"remote {r.key}: {listed.error}"
                svc = find_vehicle(resp, vehicle)
                if missing:
                    last = f"remote daemon(s) not available: {', '.join(missing)}"
                elif svc is None:
                    last = f"vehicle {vehicle!r} not listed"
                elif not svc.running:
                    last = f"vehicle {svc.name!r} not running" + (f" ({svc.error})" if svc.error else "")
                else:
                    return svc.name
            now = time.monotonic()
            if failed is None:
                failing_since = None
            elif failing_since is None:
                failing_since = now
            elif now - failing_since >= START_FAILURE_GRACE:
                raise RuntimeError(f"daemon failed to start: {failed}")
            if now >= next_progress:
                log(f"still waiting for the daemons ({now - start:.0f}s): {last}")
                next_progress = now + PROGRESS_INTERVAL
            time.sleep(1)
        raise TimeoutError(f"daemon not ready after {timeout:.0f}s: {last}")

    def list_packages(self, route: str | None = None) -> dict[str, float]:
        """The packages installed on the daemon, or, with a route, that
        remote daemon, including those pending its next restart: the time
        each was last installed (Unix seconds), by uri."""
        resp = self.daemon.ListPackages(
            daemon_pb2.ListPackagesRequest(), metadata=self.route(route), timeout=RPC_TIMEOUT,
        )
        return {p.uri: p.modified.seconds + p.modified.nanos / 1e9 for p in resp.packages}

    def install_package(self, uri: str, route: str | None = None) -> None:
        """Installs a package, which takes effect on the daemon's next restart."""
        self.daemon.InstallPackage(
            daemon_pb2.InstallPackageRequest(uri=uri), metadata=self.route(route), timeout=INSTALL_TIMEOUT,
        )

    def _send_chunks(
        self, call, request_type, path: Path, route: str | None, what: str, timeout: float,
        filename: str = "",
    ) -> None:
        """Streams a file in chunks to an upload-style RPC, which acks each
        chunk before the next is sent. The file is sent as filename, if
        given, or its own name."""
        data = path.read_bytes()
        chunks = [data[i : i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)] or [b""]
        total = len(chunks)
        acks: queue.Queue[int] = queue.Queue()

        def requests():
            for n, chunk in enumerate(chunks, start=1):
                yield request_type(filename=filename or path.name, chunk=chunk, number=n, total=total)
                if acks.get(timeout=RPC_TIMEOUT) != n:
                    return

        received = 0
        for resp in call(requests(), metadata=self.route(route), timeout=timeout):
            received = resp.received
            acks.put(received)
        if received != total:
            raise RuntimeError(f"{what} ended after {received} of {total} chunks were acknowledged")

    def upload_mission(self, path: Path, route: str) -> None:
        self._send_chunks(
            self.loader.UploadMission, upload_pb2.UploadMissionRequest,
            path, route, "mission upload", RPC_TIMEOUT,
        )

    def transfer_file(self, path: Path, route: str | None = None, name: str = "") -> None:
        """Saves a file to the daemon's files directory, or, with a route,
        to that remote daemon's, as name if given. A video can be large, so
        the deadline is for the whole transfer, not a chunk."""
        self._send_chunks(
            self.transfer.TransferFile, transfer_pb2.TransferFileRequest,
            path, route, "file transfer", None, filename=name,
        )

    def compile_mission(self, folder: Path, route: str) -> None:
        """Compiles a mission folder's flow and mission source (JSON or DSL),
        with its map features (.geojson) if it has them, on the vehicle,
        saved there under the folder's name. Compiling can be slow, so it gets
        the install deadline."""
        source = mission_source(folder)
        geojson = mission_geojson(folder)
        self.loader.CompileMission(
            upload_pb2.CompileMissionRequest(
                name=folder.name,
                flow=(folder / MISSION_FLOW).read_bytes(),
                geojson=geojson.read_bytes() if geojson else b"",
                mission=source.read_bytes(),
                json=source.name == MISSION_JSON,
            ),
            metadata=self.route(route), timeout=INSTALL_TIMEOUT,
        )

    def load_mission(self, name: str, route: str) -> None:
        self.loader.LoadMission(upload_pb2.LoadMissionRequest(name=name), metadata=self.route(route), timeout=RPC_TIMEOUT)

    def start_mission(self, route: str) -> None:
        self.mission.StartMission(mission_pb2.StartMissionRequest(), metadata=self.route(route), timeout=RPC_TIMEOUT)

    def stop_mission(self, route: str) -> None:
        self.mission.StopMission(mission_pb2.StopMissionRequest(), metadata=self.route(route), timeout=RPC_TIMEOUT)

    def stop_service(self, name: str) -> None:
        """Stops a service, by its qualified name; the orchestrator passes it
        on to the remote daemon that runs it. It stays stopped, even across
        daemon restarts, until restart_service."""
        self.daemon.StopService(daemon_pb2.StopServiceRequest(service_name=name), timeout=RPC_TIMEOUT)

    def restart_service(self, name: str) -> None:
        self.daemon.RestartService(daemon_pb2.RestartServiceRequest(service_names=name), timeout=RPC_TIMEOUT)

    def hold(self, route: str) -> None:
        self.control.Hold(control_pb2.HoldRequest(), metadata=self.route(route), timeout=RPC_TIMEOUT)

    def set_dataflow(self, toml_bytes: bytes, route: str) -> None:
        self.data.SetDataflow(data_pb2.SetDataflowRequest(toml=toml_bytes), metadata=self.route(route), timeout=RPC_TIMEOUT)


def find_vehicle(resp: daemon_pb2.ListServicesResponse, vehicle: str):
    """Finds a vehicle by its qualified name ("{ip}.{name}") or plain name."""
    vehicles = [s for s in resp.services if s.category == daemon_pb2.SERVICE_CATEGORY_VEHICLE]
    for s in vehicles:
        if s.name == vehicle:
            return s
    matches = [s for s in vehicles if s.name.endswith("." + vehicle)]
    if len(matches) > 1:
        raise ValueError(f"vehicle {vehicle!r} is ambiguous: {', '.join(s.name for s in matches)}")
    return matches[0] if matches else None


def find_remote(resp: daemon_pb2.ListServicesResponse, d: DaemonCfg, include_failed: bool = False):
    """The registered remote d, matched by its hostname. A remote whose
    services can't be listed (it failed to start them, say) is left out,
    unless include_failed; it can still be configured."""
    for r in resp.remotes:
        if (include_failed or not r.error) and r.hostname in (d.key, d.hostname):
            return r
    return None


def is_start_failure(error: str) -> bool:
    """Whether a daemon's error says it failed to start its services, which
    lasts until it is reconfigured."""
    return "failed to start services" in error


def config_packages(config: Path) -> list[str]:
    """The package uris a daemon config's services (and simulator) use."""
    with open(config, "rb") as f:
        raw = tomllib.load(f)
    uris = [s.get("plugin", {}).get("package", "") for s in raw.get("services", [])]
    uris.append(raw.get("simulator", {}).get("package", ""))
    return sorted({u for u in uris if u})


# --update-packages (the default): reinstall each package a daemon's configs
# use whose code upstream differs from what was installed, once per run; with
# --no-update-packages, only missing packages are installed
UPDATE_PACKAGES = True
# (daemon key, package uri) installed, updated or checked in this run
_installed: set[tuple[str, str]] = set()
# What upstream tree each package was installed from, by "<daemon key> <uri>":
# the daemon keeps no git metadata, so the runner records it
PACKAGE_STATE = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state") / "mobisys2027" / "packages.json"
# Bare, blobless clones of the packages' repositories, to read their trees
GIT_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mobisys2027" / "git"
GIT_TIMEOUT = 300
# The commit each (repository, version) is at upstream, fetched once a run
_upstream: dict[tuple[str, str], str] = {}


def split_package_uri(uri: str) -> tuple[str, str, str]:
    """(repository url, subdir, version) of a package uri
    (https://host/repo//sub/dir@version), as the daemon reads it; subdir is
    "" for the root, version "" for the default branch."""
    scheme = uri.find("://")
    host_end = uri.find("/", scheme + 3) if scheme >= 0 else 0
    host_end = len(uri) if host_end < 0 else host_end
    at = uri.rfind("@", host_end)
    url, version = (uri, "") if at < 0 else (uri[:at], uri[at + 1:])
    start = scheme + 3 if scheme >= 0 else 0
    sep = url.find("//", start)
    repo, subdir = (url, "") if sep < 0 else (url[:sep], url[sep + 2:])
    return repo, subdir.strip("/"), version


def _git(repo_dir: Path, *args: str) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ALLOW_PROTOCOL": "https:file"}
    out = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True,
                         timeout=GIT_TIMEOUT, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


def upstream_package(uri: str) -> tuple[str, float]:
    """The id of the tree a package uri installs now (its subdir at its
    version's latest commit upstream), and the time of the last commit that
    changed it, from a cached blobless clone of its repository."""
    repo, subdir, version = split_package_uri(uri)
    ref = version or "HEAD"
    cache = GIT_CACHE / hashlib.sha256(repo.encode()).hexdigest()[:16]
    key = (repo, ref)
    if key not in _upstream:
        if not (cache / "HEAD").exists():
            cache.mkdir(parents=True, exist_ok=True)
            _git(cache, "init", "-q", "--bare")
            _git(cache, "remote", "add", "origin", repo)
        _git(cache, "fetch", "-q", "--filter=blob:none", "origin", ref)
        _upstream[key] = _git(cache, "rev-parse", "FETCH_HEAD")
    commit = _upstream[key]
    tree = _git(cache, "rev-parse", f"{commit}:{subdir}" if subdir else f"{commit}^{{tree}}")
    changed = _git(cache, "log", "-1", "--format=%ct", commit, *(["--", subdir] if subdir else []))
    return tree, float(changed or 0)


def _load_package_state() -> dict[str, str]:
    try:
        return json.loads(PACKAGE_STATE.read_text())
    except (OSError, ValueError):
        return {}


def _record_package(key: str, uri: str, tree: str) -> None:
    state = _load_package_state()
    state[f"{key} {uri}"] = tree
    PACKAGE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PACKAGE_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(PACKAGE_STATE)


def package_outdated(key: str, uri: str, installed_at: float) -> tuple[bool, str | None, str]:
    """Whether the package uri installed on daemon key at installed_at differs
    from upstream: (outdated, upstream tree or None if it couldn't be read,
    why). Known by the tree recorded when the runner installed it; for one it
    didn't, by whether upstream changed it after it was installed (and then
    the tree is recorded, so later runs compare trees)."""
    try:
        tree, changed_at = upstream_package(uri)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as e:
        return False, None, f"can't read upstream ({e}); keeping the installed one"
    recorded = _load_package_state().get(f"{key} {uri}")
    if recorded is not None:
        return recorded != tree, tree, "upstream changed" if recorded != tree else "same as upstream"
    if changed_at > installed_at:
        return True, tree, "upstream changed after it was installed"
    _record_package(key, uri, tree)
    return False, tree, "unchanged upstream since it was installed"


def install_packages(client: Daemon, key: str, config: Path, route: str | None = None) -> None:
    """Installs the packages config uses that daemon key doesn't have yet,
    and, with UPDATE_PACKAGES, reinstalls those whose code differs from
    upstream, the first time in this run. They take effect on the restart
    that configuring it causes."""
    needed = [u for u in config_packages(config) if (key, u) not in _installed]
    if not needed:
        return
    have = client.list_packages(route)
    for uri in needed:
        _installed.add((key, uri))
        tree = None
        if uri in have:
            if not UPDATE_PACKAGES:
                continue
            outdated, tree, why = package_outdated(key, uri, have[uri])
            if not outdated:
                if tree is None:
                    log(f"not updating {uri} on {key}: {why}")
                continue
            log(f"updating {uri} on {key} ({why})")
        else:
            log(f"installing {uri} on {key}")
            try:
                tree, _ = upstream_package(uri)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                tree = None  # installed anyway; the next run compares by time
        try:
            client.install_package(uri, route)
        except grpc.RpcError as e:
            _installed.discard((key, uri))
            raise RuntimeError(f"couldn't install {uri} on {key}: {e.details()}") from None
        if tree is not None:
            _record_package(key, uri, tree)


def config_services(config: Path) -> set[str]:
    """The names of the services a daemon config runs."""
    with open(config, "rb") as f:
        raw = tomllib.load(f)
    return {s["name"] for s in raw.get("services", []) if s.get("name")}


def start_stopped_services(client: Daemon, exp: Experiment, profile: str, timeout: float = READY_TIMEOUT) -> None:
    """Restarts the services of the experiment's daemon configs that are
    stopped by request (with StopService, e.g. `eagle stop-service`), which
    they stay across restarts and reconfigures, and waits for them to run.
    Other daemons registered with the orchestrator are left alone."""
    resp = client.list_services()
    remote_ips = {r.ip for r in resp.remotes if r.ip}
    # The qualifier of each of the experiment's daemons' services
    owners: dict[str, str] = {}
    for d in exp.remotes:
        listed = find_remote(resp, d, include_failed=True)
        if listed is not None:
            owners[listed.ip] = d.key
    wanted: list[str] = []
    for svc in resp.services:
        if not svc.stopped:
            continue
        ip = next((i for i in remote_ips if svc.name.startswith(i + ".")), None)
        if ip is None:
            key, base = exp.orchestrator.key, svc.name.split(".")[-1]
        elif ip in owners:
            key, base = owners[ip], svc.name[len(ip) + 1:]
        else:
            continue  # a daemon this experiment doesn't use
        if base in config_services(exp.configs[profile][key]):
            wanted.append(svc.name)
    if not wanted:
        return
    for name in wanted:
        log(f"starting {name}, which was stopped by request")
        client.restart_service(name)
    deadline = time.monotonic() + timeout
    next_progress = time.monotonic() + PROGRESS_INTERVAL
    while True:
        try:
            listed = {s.name: s for s in client.list_services().services}
        except grpc.RpcError:
            listed = {}
        waiting = [n for n in wanted if not (n in listed and listed[n].running)]
        if not waiting:
            return
        if time.monotonic() > deadline:
            errors = "; ".join(f"{n}: {listed[n].error or 'not running'}" if n in listed else f"{n}: not listed" for n in waiting)
            raise TimeoutError(f"service(s) not running after {timeout:.0f}s: {errors}")
        if time.monotonic() >= next_progress:
            log(f"still waiting for {', '.join(waiting)} to run")
            next_progress = time.monotonic() + PROGRESS_INTERVAL
        time.sleep(1)


def wait_registered(client: Daemon, d: DaemonCfg, timeout: float = READY_TIMEOUT):
    """Waits for remote d to be registered with the orchestrator, and returns
    its listing."""
    deadline = time.monotonic() + timeout
    logged = False
    last = "not listed"
    while True:
        try:
            remote, last = find_remote(client.list_services(), d, include_failed=True), "not listed"
        except grpc.RpcError as e:
            remote, last = None, f"ListServices: {e.code().name} {e.details()}"
        if remote is not None:
            return remote
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"remote {d.key} didn't register with the orchestrator within {timeout:.0f}s "
                f"({last}); it can only be configured through the orchestrator it is locked to"
            )
        if not logged:
            if last.startswith("ListServices"):
                log(f"waiting for remote {d.key}: can't reach the orchestrator's daemon at {client.socket} ({last})")
            else:
                log(f"waiting for remote {d.key} to register with the orchestrator")
            logged = True
        time.sleep(1)


def configure_daemons(client: Daemon, exp: Experiment, profile: str) -> None:
    """Restarts each remote, then the orchestrator, with its config for the
    profile, first installing any packages the config needs that the daemon
    doesn't have, and saving the daemon's own config for restore_configs. A
    remote is locked to the orchestrator, so it is configured through it, once
    it has registered."""
    configs = exp.configs[profile]
    for d in exp.remotes:
        config = configs[d.key]
        remote = wait_registered(client, d)
        save_original(client, d.key, route=remote.ip)
        install_packages(client, d.key, config, route=remote.ip)
        log(f"configuring remote {d.key} ({remote.ip}) through the orchestrator with {config.name}")
        client.configure(config.read_text(), route=remote.ip)
    o = exp.orchestrator
    save_original(client, o.key)
    install_packages(client, o.key, configs[o.key])
    log(f"configuring orchestrator {o.key} with {configs[o.key].name}")
    client.configure(configs[o.key].read_text())


def original_path(key: str) -> Path:
    return ORIGINALS_DIR / f"{key}.toml"


def save_original(client: Daemon, key: str, route: str | None = None) -> None:
    """Saves daemon key's config, before an experiment config replaces it,
    unless a run that didn't restore it already saved it."""
    path = original_path(key)
    if path.exists():
        log(f"{key}'s own config is still saved from a run that didn't restore it ({path}); keeping it")
        return
    text = client.get_configuration(route)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    log(f"saved {key}'s own config to {path}")


def restore_configs(client: Daemon, exp: Experiment) -> None:
    """Configures each daemon whose own config was saved with it again, the
    remotes first, through the orchestrator, then the orchestrator, and waits
    for them to come back. That also ends the trial's flow or mission, so no
    engine keeps working (or logging) for it. A config that can't be restored
    stays saved, for the next run to restore.

    The services stop_services stopped are restarted first, while the
    experiment configs (which have them all) still run: being stopped by
    request lasts across reconfigures, so one the restored config doesn't
    have would stay marked stopped on the daemon, and start stopped under a
    later config that has it. They have no flow, so they don't do anything
    before the restore restarts them again."""
    restart_stopped_by_runner(client)
    remotes = [d for d in exp.remotes if original_path(d.key).exists()]
    o = exp.orchestrator
    if not remotes and not original_path(o.key).exists():
        restart_stopped_by_runner(client)
        return
    restored: list[DaemonCfg] = []
    for d in remotes:
        path = original_path(d.key)
        try:
            remote = wait_registered(client, d, timeout=60)
            log(f"restoring remote {d.key}'s own config through the orchestrator")
            client.configure(path.read_text(), route=remote.ip)
            path.unlink()
            restored.append(d)
        except (grpc.RpcError, TimeoutError, OSError) as e:
            details = f"{e.code().name} {e.details()}" if isinstance(e, grpc.RpcError) else str(e)
            log(f"warning: couldn't restore {d.key}'s own config ({details}); it is still saved in {path}")
    path = original_path(o.key)
    if path.exists():
        try:
            log(f"restoring orchestrator {o.key}'s own config")
            client.configure(path.read_text())
            path.unlink()
            restored.append(o)
        except (grpc.RpcError, OSError) as e:
            details = f"{e.code().name} {e.details()}" if isinstance(e, grpc.RpcError) else str(e)
            log(f"warning: couldn't restore {o.key}'s own config ({details}); it is still saved in {path}")
    if restored:
        time.sleep(2)  # let a restarting daemon go down before polling it
        try:
            for d in exp.remotes:
                wait_registered(client, d)
            log("daemons are back on their own configs")
        except TimeoutError as e:
            log(f"warning: {e}")
    restart_stopped_by_runner(client)


def experiment_services(resp: daemon_pb2.ListServicesResponse, exp: Experiment) -> list:
    """The services of the experiment's daemons (the orchestrator and its
    remotes) in a ListServices response; other remotes' are left out."""
    remote_ips = {r.ip for r in resp.remotes if r.ip}
    ours = {listed.ip for d in exp.remotes if (listed := find_remote(resp, d, include_failed=True)) is not None}
    services = []
    for svc in resp.services:
        ip = next((i for i in remote_ips if svc.name.startswith(i + ".")), None)
        if ip is None or ip in ours:
            services.append(svc)
    return services


def stop_services(client: Daemon, exp: Experiment) -> None:
    """Stops every running service of the experiment's daemons, the vehicles
    first, so that nothing writes to the files that are collected next. They
    are recorded in STOPPED_FILE first, for restart_stopped_by_runner."""
    services = [s for s in experiment_services(client.list_services(), exp) if not s.stopped]
    order = {daemon_pb2.SERVICE_CATEGORY_VEHICLE: 0, daemon_pb2.SERVICE_CATEGORY_ENGINE: 1}
    services.sort(key=lambda s: order.get(s.category, 2))
    if not services:
        return
    names = [s.name for s in services]
    recorded = json.loads(STOPPED_FILE.read_text()) if STOPPED_FILE.exists() else []
    STOPPED_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOPPED_FILE.write_text(json.dumps(sorted(set(recorded) | set(names))))
    log(f"stopping {len(names)} service(s) before collecting: {', '.join(names)}")
    for name in names:
        try:
            client.stop_service(name)
        except grpc.RpcError as e:
            log(f"warning: couldn't stop {name}: {e.code().name} {e.details()}")


def restart_stopped_by_runner(client: Daemon) -> None:
    """Restarts the services stop_services stopped that are still stopped,
    which clears their stopped by request mark. A service that isn't listed
    (its daemon's config doesn't have it) is skipped."""
    if not STOPPED_FILE.exists():
        return
    try:
        names = json.loads(STOPPED_FILE.read_text())
        listed = {s.name: s for s in client.list_services().services}
        todo = [n for n in names if n in listed and listed[n].stopped]
        for name in todo:
            client.restart_service(name)
        if todo:
            log(f"restarted {len(todo)} service(s) the runner had stopped, so none stays marked stopped: {', '.join(todo)}")
        STOPPED_FILE.unlink()
    except grpc.RpcError as e:
        log(
            f"warning: couldn't restart the services the runner stopped ({e.code().name} {e.details()}); "
            f"they are listed in {STOPPED_FILE}, for the next run to restart"
        )


_sha256_cache: dict[tuple[str, int, int], str] = {}


def file_sha256(path: Path) -> str:
    """path's SHA-256, in hex, cached while its size and mtime stay the same."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _sha256_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        _sha256_cache[key] = h.hexdigest()
    return _sha256_cache[key]


def daemon_sha256(d: DaemonCfg, name: str) -> str | None:
    """The SHA-256 of the file `name` in daemon d's files directory, or None
    if it isn't there or can't be read (e.g. d has no prefix to ssh to): on
    this host for a daemon with no prefix, otherwise over ssh."""
    if not d.prefix:
        path = Path.home() / REMOTE_FILES_DIR / name
        return file_sha256(path) if path.is_file() else None
    if not d.ssh_host:
        return None
    files_dir = '"$HOME"/' + shlex.quote(REMOTE_FILES_DIR.rstrip("/"))
    r = subprocess.run(ssh(d.ssh_host, f"sha256sum {files_dir}/{shlex.quote(name)} 2>/dev/null"),
                       capture_output=True, text=True)
    out = r.stdout.split()
    return out[0] if r.returncode == 0 and out and len(out[0]) == 64 else None


def already_there(d: DaemonCfg, f: "Upload", what: str) -> bool:
    """Whether daemon d already has upload f, byte for byte (by SHA-256), so
    it needn't be pushed again; logs it if so."""
    if daemon_sha256(d, f.name) != file_sha256(f.path):
        return False
    log(f"{what} is already on {d.key} (same SHA-256); not pushing it")
    return True


def push_uploads(client: Daemon, exp: Experiment) -> None:
    """Pushes the experiment directory's uploads to every daemon's files
    directory (STEELEAGLE_FILES): the orchestrator's and each remote's,
    through the orchestrator, or with rsync if the remote isn't registered.
    Each is saved as its push_as, if set, so a driver's config can name one
    file for every video. A daemon that already has a file of that name with
    the same SHA-256 isn't pushed it again. Uploads marked no_upload are not
    pushed: they are assumed to already be there. Called once per directory,
    before its first experiment."""
    pushes = [u for u in exp.uploads if not u.no_upload]
    skipped = [u.name for u in exp.uploads if u.no_upload]
    if skipped:
        log(f"not pushing {', '.join(skipped)} (no_upload: assumed to be on the daemons already)")
    if not pushes:
        return
    try:
        listing = client.list_services()
    except grpc.RpcError:
        listing = None
    for f in pushes:
        what = f.path.name + (f" as {f.name}" if f.name != f.path.name else "")
        for d in exp.remotes:
            if already_there(d, f, what):
                continue
            remote = find_remote(listing, d) if listing else None
            if remote is not None:
                log(f"pushing {what} to remote {d.key} ({remote.ip}) through the orchestrator")
                client.transfer_file(f.path, route=remote.ip, name=f.name)
            elif d.ssh_host:
                log(f"remote {d.key} not registered; pushing {what} to it with rsync")
                files_dir = REMOTE_FILES_DIR.rstrip("/")
                dest = files_dir + "/" + f.name
                # The remote makes the files directory before its rsync starts
                # (rsync's --mkpath needs 3.2.3+ on both ends)
                cmd = [
                    *rsync_cmd(d.ssh_host), "-a",
                    f"--rsync-path=mkdir -p {shlex.quote(files_dir)} && rsync",
                    str(f.path), f"{d.ssh_host}:{dest}",
                ]
                subprocess.run(cmd, check=True)
            else:
                raise RuntimeError(f"remote {d.key} is not registered with the orchestrator and has no prefix to rsync to")
        if already_there(exp.orchestrator, f, what):
            continue
        log(f"pushing {what} to orchestrator {exp.orchestrator.key}")
        client.transfer_file(f.path, name=f.name)


# --------------------------------------------------------------------------
# File collection
# --------------------------------------------------------------------------


def source_files(exp: Experiment, src: str) -> tuple[DaemonCfg, list[str]]:
    """The daemon whose host holds result or catalog src, and the paths of the files
    (or directories) it names there: those its glob matches, or a
    directory's entries, as they are collected."""
    host, _, rest = src.partition("/")
    d = exp.daemons[host]
    path = "/" + rest
    if not d.prefix:
        if glob.has_magic(path):
            return d, sorted(glob.glob(path))
        if os.path.isdir(path):
            return d, sorted(os.path.join(path, n) for n in os.listdir(path) if not n.startswith("."))
        return d, [path] if os.path.exists(path) else []
    if glob.has_magic(path):
        expand = f"set -- {path}"  # the remote shell expands the glob
    else:
        q = shlex.quote(path)
        expand = f"if [ -d {q} ]; then set -- {q}/*; else set -- {q}; fi"
    script = f'{expand}; for f in "$@"; do [ -e "$f" ] && printf "%s\\0" "$f"; done; true'
    r = subprocess.run(ssh(d.ssh_host, script), capture_output=True, check=True)
    return d, sorted(p for p in r.stdout.decode().split("\0") if p)


def clear_results(exp: Experiment) -> None:
    """Deletes what each result source holds on its host before a trial, so
    files left there by an earlier run (or one that died before collecting
    them) aren't collected with the trial's. A directory the source names or
    matches is emptied, not removed; a file is removed."""
    for key, src in exp.results.items():
        host, _, rest = src.partition("/")
        d = exp.daemons[host]
        path = "/" + rest
        if path.rstrip("/") in ("", "/"):
            raise RuntimeError(f"results.{key} = {src!r}: refusing to clear the root directory")
        log(f"clearing {key} at {src}")
        if d.prefix:
            expand = path if glob.has_magic(path) else shlex.quote(path)
            script = (
                f'for f in {expand}; do if [ -d "$f" ]; then find "$f" -mindepth 1 -delete; '
                f'elif [ -e "$f" ] || [ -L "$f" ]; then rm -f -- "$f"; fi; done'
            )
            r = subprocess.run(ssh(d.ssh_host, script), capture_output=True)
            if r.returncode:
                log(f"warning: clearing {key} on {d.key} exited {r.returncode}: {r.stderr.decode().strip()}")
            continue
        for p in glob.glob(path) if glob.has_magic(path) else [path]:
            if os.path.isdir(p) and not os.path.islink(p):
                for entry in os.scandir(p):
                    if entry.is_dir(follow_symlinks=False):
                        shutil.rmtree(entry.path)
                    else:
                        os.remove(entry.path)
            elif os.path.lexists(p):
                os.remove(p)


def remove_result_dirs(exp: Experiment) -> None:
    """Removes each directory a result source names (or its glob matches) from
    its host once the experiment is done, like rm -rf: whatever is still in it
    is deleted too."""
    for key, src in exp.results.items():
        host, _, rest = src.partition("/")
        d = exp.daemons[host]
        path = "/" + rest
        if path.rstrip("/") in ("", "/"):
            raise RuntimeError(f"results.{key} = {src!r}: refusing to remove the root directory")
        if d.prefix:
            expand = path if glob.has_magic(path) else shlex.quote(path)
            script = (
                f'for f in {expand}; do if [ -d "$f" ] && [ ! -L "$f" ]; then '
                f'rm -rf -- "$f" && echo "removed $f"; fi; done; true'
            )
            r = subprocess.run(ssh(d.ssh_host, script), capture_output=True)
            for line in r.stdout.decode().splitlines():
                log(f"{key}: {line} on {d.key}")
            for line in r.stderr.decode().splitlines():
                log(f"warning: {key}: {line} on {d.key}")
            continue
        for p in glob.glob(path) if glob.has_magic(path) else [path]:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
                log(f"{key}: removed {p}")


def catalog_files(exp: Experiment, src: str) -> tuple[DaemonCfg, dict[str, tuple[str, str]]]:
    """The daemon whose host holds catalog src, and every file under what src
    names (its glob matches, or a directory's entries, as source_files),
    recursively: {path: (relative path, modification time)}. The relative path
    is below the match's parent directory, so a matched directory keeps its
    name and its subdirectories (e.g. 100.65.73.80.detection_log/raw/0.jpeg)."""
    host, _, rest = src.partition("/")
    d = exp.daemons[host]
    path = "/" + rest
    files: dict[str, tuple[str, str]] = {}
    if not d.prefix:
        if glob.has_magic(path):
            matches = sorted(glob.glob(path))
        elif os.path.isdir(path):
            matches = sorted(os.path.join(path, n) for n in os.listdir(path) if not n.startswith("."))
        else:
            matches = [path] if os.path.exists(path) else []
        for m in matches:
            parent = os.path.dirname(m.rstrip("/"))
            walk = [(os.path.dirname(m), [], [os.path.basename(m)])] if os.path.isfile(m) else os.walk(m)
            for root, _, names in walk:
                for n in names:
                    f = os.path.join(root, n)
                    if os.path.isfile(f):
                        files[f] = (os.path.relpath(f, parent), repr(os.stat(f).st_mtime_ns))
        return d, files
    if glob.has_magic(path):
        expand = f"set -- {path}"  # the remote shell expands the glob
    else:
        q = shlex.quote(path)
        expand = f"if [ -d {q} ]; then set -- {q}/*; else set -- {q}; fi"
    # Each match, then its files and their modification times, \0-separated
    script = (f'{expand}; for f in "$@"; do [ -e "$f" ] || continue; printf "M\\0%s\\0" "$f"; '
              f'find "$f" -type f -printf "%p\\0%T@\\0"; done; true')
    r = subprocess.run(ssh(d.ssh_host, script), capture_output=True, check=True)
    fields = r.stdout.decode().split("\0")
    parent, i = "", 0
    while i + 1 < len(fields):
        a, b = fields[i], fields[i + 1]
        i += 2
        if a == "M":
            parent = os.path.dirname(b.rstrip("/"))
        else:
            files[a] = (os.path.relpath(a, parent), b)
    return d, files


def deferred_files(exp: Experiment) -> dict[str, dict[str, str]]:
    """The files under each defer_pull catalog now, with their modification
    times, before a trial, so that only the ones the trial adds or rewrites
    are moved after it: an engine may reuse names from trial to trial
    (detection_log names its frames by number)."""
    return {key: {p: m for p, (_, m) in catalog_files(exp, exp.catalogs[key])[1].items()}
            for key in sorted(exp.defer_pull)}


def defer_pull(exp: Experiment, key: str, before: dict[str, str], profile: str, trial: int) -> str | None:
    """Moves the files catalog key gained or rewrote during the trial into
    <experiment>_<profile>_<trial>_<key>/ in the files directory
    (STEELEAGLE_FILES) of the host that holds them, to be transferred later,
    keeping their paths below the catalog's matches (see catalog_files). They
    are moved out of the engine's own directory, since an engine may empty
    that when it starts (detection_log does, with refresh), which it does on
    every daemon restart. If the directory exists already (from an earlier
    run), a new one with the time added is used. Returns "<daemon>:<dir>",
    or None if there was nothing to move."""
    d, files = catalog_files(exp, exp.catalogs[key])
    moves = [(p, rel) for p, (rel, mtime) in sorted(files.items()) if before.get(p) != mtime]
    if not moves:
        log(f"warning: {key} has no new files on {d.key} to move")
        return None
    name = f"{exp.name}_{profile}_{trial}_{key}"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if d.prefix:
        files_dir = '"$HOME"/' + shlex.quote(REMOTE_FILES_DIR.rstrip("/"))
        # In batches, a directory at a time (one mkdir, and xargs runs mv on
        # many files at once): a process per file moves only ~30 files/s on
        # a Tachyon. mv -n leaves a file whose name is taken where it is,
        # which the last line then reports
        groups: dict[str, list[str]] = {}
        for p, rel in moves:
            groups.setdefault(os.path.dirname(rel), []).append(shlex.quote(p))
        script = (
            f"set -e\nd={files_dir}/{shlex.quote(name)}\n"
            f'if [ -e "$d" ]; then d="$d-{stamp}"; fi\n'
            'mkdir -p "$d"\necho "$d"\nset +e\n'
            + "mkdir -p " + " ".join(f'"$d"/{shlex.quote(sub)}' for sub in groups) + "\n"
            + "".join(f'printf "%s\\0" {" ".join(ps)} | xargs -0 -r mv -n -t "$d"/{shlex.quote(sub)}\n'
                      for sub, ps in groups.items())
            + f'for f in {" ".join(q for ps in groups.values() for q in ps)}; do [ -e "$f" ] && echo "taken $f"; done; true\n'
        )
        # Sent on stdin, since a long list of files can be over the command
        # line's limit
        r = subprocess.run(ssh(d.ssh_host, "sh -s"), input=script.encode(), capture_output=True)
        out = r.stdout.decode().splitlines()
        if r.returncode != 0 or not out:
            raise RuntimeError(f"moving {key} on {d.key} failed: {r.stderr.decode().strip()}")
        dest, taken = out[0], [line[len("taken "):] for line in out[1:] if line.startswith("taken ")]
    else:
        target = Path.home() / REMOTE_FILES_DIR / name
        if target.exists():
            target = target.with_name(f"{name}-{stamp}")
        target.mkdir(parents=True)
        dest, taken = str(target), []
        for p, rel in moves:
            new = target / rel
            if new.exists():
                taken.append(str(new))
            else:
                new.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(p, new)
    log(f"deferred {key}: moved {len(moves) - len(taken)} file(s) on {d.key} to {dest}, to be transferred later")
    if taken:
        log(f"warning: {len(taken)} file(s) of {key} weren't moved, since their path is taken in {dest} (e.g. {taken[0]})")
    return f"{d.key}:{dest}"


def rsync(exp: Experiment, src: str, dest: Path, move: bool) -> bool:
    """Copies (or moves) src's files into dest. False if rsync failed; a
    local path that matches nothing is only warned about."""
    host, _, rest = src.partition("/")
    d = exp.daemons[host]
    path = "/" + rest
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [*rsync_cmd(d.ssh_host), "-a"] if d.prefix else ["rsync", "-a"]
    if move:
        cmd.append("--remove-source-files")
    # A directory's contents are collected, not the directory itself.
    # ssh and rsync are not captured, so their prompts (host key, password)
    # and errors reach the user's terminal.
    if d.prefix:
        if not glob.has_magic(path):
            test = ssh(d.ssh_host, f"test -d {shlex.quote(path)}")
            if subprocess.run(test).returncode == 0:
                path = path.rstrip("/") + "/"
        sources = [d.prefix + path]  # the remote shell expands globs
    else:
        sources = sorted(glob.glob(path))
        if not sources:
            log(f"warning: nothing matches {path}")
            return True
        if len(sources) == 1 and not glob.has_magic(path) and os.path.isdir(sources[0]):
            sources = [sources[0].rstrip("/") + "/"]
    r = subprocess.run(cmd + sources + [str(dest) + "/"])
    if r.returncode:
        log(f"warning: rsync of {src} exited {r.returncode}")
        return False
    return True


# --------------------------------------------------------------------------
# netsim
# --------------------------------------------------------------------------


def sudo_validate() -> None:
    """Caches sudo credentials for netsim, asking for a password if needed."""
    if subprocess.run(["sudo", "-v"]).returncode:
        raise SystemExit("netsim needs root: sudo failed")


class Netsim:
    """netsim, run for one trial. Under the none and ambient profiles it
    isn't run at all: the trial runs unattenuated. Under none, it has no
    timeline; under ambient, its timeline is the iperf3 measurement, ending
    at timeline_end."""

    def __init__(self, exp: Experiment, profile: str, trial_dir: Path):
        self.timeline = trial_dir / "timeline.csv"
        self.output = trial_dir / "netsim.log"
        self.enabled = profile not in NO_NETSIM
        self.profile = profile
        self.timeline_end: float | None = None
        cmd = [
            sys.executable, "-u", str(NETSIM),
            "-i", exp.interface, "-p", str(exp.port),
            profile, "--log", str(self.timeline),
            *exp.netsim_args,
        ]
        if profile in ("malicious", "outage") and exp.failpoints:
            cmd += ["--failpoints", exp.failpoints]
        if os.geteuid() != 0 and "--dry-run" not in exp.netsim_args:
            cmd = ["sudo", "-n"] + cmd
        self.needs_sudo = self.enabled and cmd[0] == "sudo"
        self.cmd = cmd
        self.proc: subprocess.Popen | None = None
        self.out = None

    def start(self) -> None:
        if not self.enabled:
            log(f"{self.profile} profile: running without netsim")
            return
        if self.needs_sudo:
            sudo_validate()  # refresh the cached credentials the -n above relies on
        self.out = open(self.output, "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.out, stderr=subprocess.STDOUT)
        log(f"netsim started (pid {self.proc.pid})")

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            # SIGINT makes netsim print its totals and tear down its tc state
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=10)
        else:
            log(f"warning: netsim exited early with code {self.proc.returncode}; see {self.output}")
        self.out.close()
        self.proc = None
        log("netsim stopped")


class PowerMonitor:
    """The power script (POWER_SCRIPT), run for one trial on the host of each
    remote daemon with a prefix to ssh to. Each writes its timeline to
    trial_dir/power_<daemon>.csv, and its messages to power_<daemon>.log. The
    script is sent with python3 -c, so nothing is installed on the host, and
    it stops when its ssh's stdin closes, so it never outlives the trial."""

    def __init__(self, exp: Experiment, trial_dir: Path):
        self.exp = exp
        self.trial_dir = trial_dir
        self.daemons = power_daemons(exp)
        self.procs: list[tuple[DaemonCfg, subprocess.Popen, object, object]] = []

    def timeline(self, d: DaemonCfg) -> Path:
        return self.trial_dir / f"power_{d.key}.csv"

    def output(self, d: DaemonCfg) -> Path:
        return self.trial_dir / f"power_{d.key}.log"

    def start(self) -> None:
        if not self.exp.power:
            return
        for d in self.exp.remotes:
            if not d.ssh_host:
                log(f"warning: remote {d.key} has no prefix to ssh to, so its power isn't measured")
        script = POWER_SCRIPT.read_text()
        for d in self.daemons:
            command = (
                f"{REMOTE_PATH} python3 -u -c {shlex.quote(script)} "
                f"--interval {self.exp.power_interval:g} --sample {min(0.1, self.exp.power_interval):g}"
            )
            out, err = open(self.timeline(d), "w"), open(self.output(d), "w")
            # In a session of its own, so a Ctrl-C reaches only the runner,
            # which then stops the script itself and gets its last row
            proc = subprocess.Popen(
                ssh(d.ssh_host, command), stdin=subprocess.PIPE, stdout=out, stderr=err,
                start_new_session=True,
            )
            self.procs.append((d, proc, out, err))
        if self.daemons:
            log(f"measuring the power of {', '.join(d.key for d in self.daemons)}")

    def stop(self) -> None:
        for d, proc, out, err in self.procs:
            try:
                proc.stdin.close()  # the script stops on EOF
            except OSError:
                pass
        for d, proc, out, err in self.procs:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=10)
            out.close()
            err.close()
            if proc.returncode:
                log(
                    f"warning: power measurement on {d.key} exited {proc.returncode}; "
                    f"see {self.output(d)}"
                )
        if self.procs:
            log("power measurement stopped")
        self.procs = []


def parse_netsim_summary(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    patterns = {
        "elapsed_sec": r"totals over ([\d.]+) s",
        "delivered_bytes": r"delivered through shaper\s*:.*?\((\d+) bytes",
        "dropped_off_bytes": r"dropped while port off\s*:.*?\((\d+) bytes",
        "dropped_queue_pkts": r"dropped by shaper queue\s*:\s*(\d+) pkts",
        "total_arrived_bytes": r"total arrived on port\s*:.*?\((\d+) bytes",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def read_timeline(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def weighted_percentile(samples: list[tuple[float, float]], q: float) -> float:
    """Percentile of (value, weight) samples, i.e. of the time-weighted rate."""
    samples = sorted(samples)
    total = sum(w for _, w in samples)
    acc = 0.0
    for v, w in samples:
        acc += w
        if acc >= q * total:
            return v
    return samples[-1][0]


def timed_samples(rows: list[dict[str, str]], column: str, end: float) -> list[tuple[float, float]]:
    """(value, seconds) samples of a timeline column: each row's value holds
    until the next row (or the end of the run). Rows that leave the column
    empty are skipped."""
    timeline = [(float(r["t_sec"]), float(r[column])) for r in rows if r.get(column)]
    samples = []
    for i, (t, v) in enumerate(timeline):
        t_next = timeline[i + 1][0] if i + 1 < len(timeline) else max(end, t)
        if t_next > t:
            samples.append((v, t_next - t))
    return samples or [(v, 1.0) for _, v in timeline]


def summarize(samples: list[tuple[float, float]], prefix: str, unit: str) -> dict[str, float]:
    if not samples:
        return {}
    total = sum(w for _, w in samples)
    return {
        f"{prefix}_mean_{unit}": sum(v * w for v, w in samples) / total,
        f"{prefix}_median_{unit}": weighted_percentile(samples, 0.5),
        f"{prefix}_p95_{unit}": weighted_percentile(samples, 0.95),
        f"{prefix}_min_{unit}": min(v for v, _ in samples),
        f"{prefix}_max_{unit}": max(v for v, _ in samples),
    }


def bandwidth_stats(rows: list[dict[str, str]], end: float) -> dict[str, float]:
    """Time-weighted statistics of the bandwidth allowed (by netsim, or, for
    ambient, measured by iperf3) and of the delay netsim added."""
    bw = timed_samples(rows, "applied_kbps", end)
    stats = summarize(bw, "bandwidth", "kbps")
    if bw:
        stats["time_off_sec"] = sum(w for v, w in bw if v == 0)
    stats.update(summarize(timed_samples(rows, "delay_ms", end), "delay", "ms"))
    return stats


def power_daemons(exp: Experiment) -> list[DaemonCfg]:
    """The remote daemons whose power is measured: those with a prefix to ssh to."""
    return [d for d in exp.remotes if d.ssh_host] if exp.power else []


def power_prefix(exp: Experiment, d: DaemonCfg) -> str:
    """The prefix of d's power columns in the stats CSV: power, or, if more
    than one daemon is measured, <daemon>_power."""
    return "power" if len(power_daemons(exp)) == 1 else f"{d.key}_power"


def power_stats(rows: list[dict[str, str]], prefix: str) -> dict[str, float]:
    """Mean, median and 99th percentile of a power timeline, each row weighted
    by its samples, so that a short last row counts for less."""
    samples = []
    for r in rows:
        try:
            samples.append((float(r["power_w"]), float(r.get("samples") or 1)))
        except (KeyError, TypeError, ValueError):
            continue  # e.g. a row cut short when the connection dropped
    samples = [(v, w) for v, w in samples if w > 0]
    if not samples:
        return {}
    total = sum(w for _, w in samples)
    return {
        f"{prefix}_mean_w": sum(v * w for v, w in samples) / total,
        f"{prefix}_median_w": weighted_percentile(samples, 0.5),
        f"{prefix}_p99_w": weighted_percentile(samples, 0.99),
    }


STATS_COLUMNS = [
    "trial", "duration_sec",
    "total_bytes_delivered", "total_bytes_dropped_while_off", "total_bytes_arrived",
    "pkts_dropped_by_queue",
    "bandwidth_mean_kbps", "bandwidth_median_kbps", "bandwidth_p95_kbps",
    "bandwidth_min_kbps", "bandwidth_max_kbps", "time_off_sec",
    "delay_mean_ms", "delay_median_ms", "delay_p95_ms", "delay_min_ms", "delay_max_ms",
]
TIMELINE_COLUMNS = ["trial", "t_sec", "drawn_kbps", "applied_kbps", "state", "delay_ms"]
POWER_COLUMNS = ["trial", "daemon", "t_sec", "power_w", "input_w", "battery_w", "battery_status", "samples"]


def stats_columns(exp: Experiment) -> list[str]:
    """STATS_COLUMNS, and the power columns of each daemon measured."""
    return STATS_COLUMNS + [
        f"{power_prefix(exp, d)}_{s}_w" for d in power_daemons(exp) for s in ("mean", "median", "p99")
    ]


def append_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Appends rows to a CSV, writing the header first if the file is new, so
    the trials finished so far are saved even if a later one fails."""
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def fmt(v):
    return int(v) if isinstance(v, float) and v.is_integer() else v


def write_stats(
    exp: Experiment, run_dir: Path, netsim: Netsim, power: PowerMonitor, label: str, trial: int,
    wall_sec: float,
) -> None:
    """Adds the trial's row to stats_<label>.csv, its timeline to
    timeline_<label>.csv and its power timelines to power_<label>.csv, where
    the label is <profile>_<experiment>."""
    summary = parse_netsim_summary(netsim.output.read_text() if netsim.output.is_file() else "")
    end = summary.get("elapsed_sec", wall_sec)
    timeline = read_timeline(netsim.timeline)
    stats = {
        "trial": trial,
        "duration_sec": round(end, 3),
        "total_bytes_delivered": summary.get("delivered_bytes", ""),
        "total_bytes_dropped_while_off": summary.get("dropped_off_bytes", ""),
        "total_bytes_arrived": summary.get("total_arrived_bytes", ""),
        "pkts_dropped_by_queue": summary.get("dropped_queue_pkts", ""),
        **{
            k: round(v, 3)
            for k, v in bandwidth_stats(
                timeline, netsim.timeline_end if netsim.timeline_end is not None else end,
            ).items()
        },
    }
    power_rows = []
    power_text = []
    for d in power.daemons:
        rows = read_timeline(power.timeline(d))
        power_rows += [{"trial": trial, "daemon": d.key, **r} for r in rows]
        prefix = power_prefix(exp, d)
        ps = {k: round(v, 3) for k, v in power_stats(rows, prefix).items()}
        stats.update(ps)
        if ps:
            power_text.append(
                f"{d.key} power mean {ps[prefix + '_mean_w']} / median {ps[prefix + '_median_w']} / "
                f"p99 {ps[prefix + '_p99_w']} W"
            )
        else:
            power_text.append(f"no power measured on {d.key}")
    append_csv(run_dir / f"stats_{label}.csv", stats_columns(exp), [{k: fmt(v) for k, v in stats.items()}])
    append_csv(run_dir / f"timeline_{label}.csv", TIMELINE_COLUMNS, [{"trial": trial, **r} for r in timeline])
    if power.daemons:
        append_csv(run_dir / f"power_{label}.csv", POWER_COLUMNS, power_rows)
    if "delay_mean_ms" in stats:
        summary_text = (
            f"delay mean {stats['delay_mean_ms']} / median {stats['delay_median_ms']} / "
            f"p95 {stats['delay_p95_ms']} ms"
        )
    elif not netsim.enabled and netsim.timeline_end is None:
        summary_text = "unattenuated"
    else:
        summary_text = (
            f"mean {stats.get('bandwidth_mean_kbps', '?')} / median {stats.get('bandwidth_median_kbps', '?')} / "
            f"p95 {stats.get('bandwidth_p95_kbps', '?')} Kbps"
        )
    log(
        f"stats ({label}, trial {trial}): {stats['total_bytes_delivered']} bytes delivered, {summary_text}"
        + "".join(f"; {t}" for t in power_text)
    )


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def wait_for(duration: float, what: str) -> float:
    """Waits for duration seconds, or for ENTER if it is 0. Returns seconds
    waited."""
    t0 = time.monotonic()
    if duration > 0:
        log(f"running {what} for {duration:g}s")
        PROGRESS.sleep(duration, "trial")
    else:
        input(f"Running {what}. Press ENTER to stop... ")
    return time.monotonic() - t0


def wait_grace(grace: float) -> float:
    """Waits out the grace period. Returns seconds waited."""
    if grace <= 0:
        return 0.0
    t0 = time.monotonic()
    log(f"grace period: waiting {grace:g}s more before stopping")
    PROGRESS.sleep(grace, "grace")
    return time.monotonic() - t0


def wait_until_done(limit: float, done, what: str) -> tuple[float, bool]:
    """Waits up to `limit` seconds, ending early once done() is true, which is
    checked every DONE_POLL_S. Returns (seconds waited, whether done)."""
    t0 = time.monotonic()
    while (left := limit - (time.monotonic() - t0)) > 0:
        if done():
            return time.monotonic() - t0, True
        PROGRESS.sleep(min(DONE_POLL_S, left), what)
    return time.monotonic() - t0, done()


def hook_env(exp: Experiment, hook: str, profile: str, trial: int, trial_dir: Path, vehicle: str) -> dict:
    """The environment a hook runs in: the runner's, and what it is told about the trial."""
    return {
        **os.environ,
        "EXPERIMENT_NAME": exp.name,
        "EXPERIMENT_PROFILE": profile,
        "EXPERIMENT_TRIAL": str(trial),
        "EXPERIMENT_TRIAL_DIR": str(trial_dir.resolve()),
        "EXPERIMENT_VEHICLE": vehicle,
        "EXPERIMENT_HOOK": hook,
    }


def run_hook(
    exp: Experiment, hook: str, profile: str, trial: int, trial_dir: Path, vehicle: str,
) -> None:
    """Runs the experiment's hook (setup, start, grace, grace_end or end), if
    it has one, in the experiment directory, and waits for it. Its output goes
    to the terminal. It is told about the trial through the environment
    (EXPERIMENT_NAME, EXPERIMENT_PROFILE, EXPERIMENT_TRIAL,
    EXPERIMENT_TRIAL_DIR, EXPERIMENT_VEHICLE, EXPERIMENT_HOOK). A hook that
    fails is only warned about, so the trial still stops and is collected."""
    cmd = exp.hooks.get(hook)
    if not cmd:
        return
    log(f"running {hook} hook: {shlex.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=exp.dir, env=hook_env(exp, hook, profile, trial, trial_dir, vehicle),
                           stdin=subprocess.DEVNULL)
    except OSError as e:
        log(f"warning: {hook} hook couldn't run: {e}")
        return
    if r.returncode:
        log(f"warning: {hook} hook exited {r.returncode}")


def hook_passes(
    exp: Experiment, hook: str, profile: str, trial: int, trial_dir: Path, vehicle: str,
) -> bool:
    """Whether the experiment's hook exits 0, run quietly as run_hook would; false
    without one, or if it can't run. For the done hook, which is polled."""
    cmd = exp.hooks.get(hook)
    if not cmd:
        return False
    try:
        return subprocess.run(cmd, cwd=exp.dir, env=hook_env(exp, hook, profile, trial, trial_dir, vehicle),
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return False


# A flow names its nodes relative to the vehicle, which qualifies them itself:
# vehicle.<stream> for the vehicle's telemetry or a camera,
# local.<engine>.<producer> for a producer of an engine on the vehicle's
# daemon, and local.<engine> or remote.<engine> for an engine on the vehicle's
# daemon or on the orchestrator. Producers of remote engines can't be used.
VEHICLE_PREFIX, LOCAL_PREFIX, REMOTE_PREFIX = "vehicle.", "local.", "remote."


def check_flow(path: Path) -> None:
    """Checks that the flow at path names its nodes as the vehicle expects,
    so an old flow with bare names fails before any daemon is restarted. The
    vehicle checks the rest (that the cameras and engines exist) when the
    flow is set."""
    try:
        flow = tomllib.loads(path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"flow {path}: {e}") from None
    for p in flow.get("producers", []):
        name = p.get("name", "")
        if name.startswith(REMOTE_PREFIX):
            raise ValueError(
                f"flow {path}: producer {name!r} is of a remote engine; only the vehicle's "
                f"streams and local engines' producers can be used"
            )
        if not name.startswith((VEHICLE_PREFIX, LOCAL_PREFIX)):
            raise ValueError(
                f"flow {path}: producer {name!r} must start with {VEHICLE_PREFIX!r} (a stream "
                f"of the vehicle) or {LOCAL_PREFIX!r} (a local engine's producer)"
            )
    for e in flow.get("engines", []):
        name, location = e.get("name", ""), e.get("location", "")
        if not name.startswith((LOCAL_PREFIX, REMOTE_PREFIX)):
            raise ValueError(
                f"flow {path}: engine {name!r} must start with {LOCAL_PREFIX!r} or {REMOTE_PREFIX!r}"
            )
        if name.split(".", 1)[0] != location:
            raise ValueError(f"flow {path}: engine {name!r} has location {location!r}")


TRIAL_DONE = "complete"  # written into a trial's directory once it has run to the end and been collected


def run_dir_of(exp: Experiment, profile: str) -> Path:
    return exp.dir / "runs" / exp.name / profile


def done_trials(run_dir: Path) -> set[int]:
    """The trials in run_dir that ran to the end (an interrupted or failed
    one has no TRIAL_DONE file)."""
    return {t for t in trial_numbers(run_dir) if (run_dir / f"trial_{t}" / TRIAL_DONE).is_file()}


def pending_trials(exp: Experiment, profile: str, only_trial: int | None, overwrite: bool) -> list[int]:
    """The trials of profile to run: trial only_trial (-t) or 1..trials,
    leaving out those already done unless overwrite."""
    wanted = [only_trial] if only_trial else list(range(1, exp.trials + 1))
    if overwrite:
        return wanted
    done = done_trials(run_dir_of(exp, profile))
    return [t for t in wanted if t not in done]


def prepare_run_dir(exp: Experiment, profile: str, trials: list[int], only_trial: int | None,
                    overwrite: bool) -> Path:
    """The profile's run directory, ready for `trials`. With overwrite, all
    its earlier results go (with -t, only that trial's); otherwise finished
    trials are kept and a trial left unfinished is cleared to run again."""
    run_dir = run_dir_of(exp, profile)
    label = f"{profile}_{exp.name}"
    if overwrite and not only_trial and run_dir.exists():
        log(f"--overwrite: deleting the earlier results in {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for t in trials:
        trial_dir = run_dir / f"trial_{t}"
        if trial_dir.exists():
            log(f"{'--overwrite: deleting' if overwrite else 'clearing the unfinished'} trial {trial_dir}")
            shutil.rmtree(trial_dir)
        drop_trial_rows(run_dir, label, t)
    return run_dir


def trial_numbers(run_dir: Path) -> list[int]:
    """The numbers of the trial_<n> directories in run_dir."""
    if not run_dir.is_dir():
        return []
    return sorted(
        int(m.group(1)) for p in run_dir.iterdir() if p.is_dir() and (m := re.fullmatch(r"trial_(\d+)", p.name))
    )


def drop_trial_rows(run_dir: Path, label: str, trial: int) -> None:
    """Removes trial's rows from the profile's stats, timeline and power CSVs."""
    for kind in ("stats", "timeline", "power"):
        path = run_dir / f"{kind}_{label}.csv"
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            columns = r.fieldnames or []
            rows = [row for row in r if row.get("trial") != str(trial)]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(rows)


def cleanup(
    exp: Experiment, run_dir: Path, trial_dir: Path, netsim: Netsim, power: PowerMonitor, label: str,
    profile: str, trial: int, wall_sec: float, deferred_before: dict[str, dict[str, str]], started: bool = True,
) -> bool:
    """Stops the trial's measurements and collects its files. False if its
    [results] could not all be collected, or nothing came back: neither a
    result nor a defer_pull catalog's files (saved on their host, which
    counts as collected)."""
    netsim.stop()
    power.stop()  # stopped already, unless stopping the mission failed
    collected = True
    for key, src in exp.results.items():
        log(f"moving {key} from {src}")
        collected &= rsync(exp, src, trial_dir / "result" / key, move=True)
    deferred = False
    for key, src in exp.catalogs.items():
        if key in exp.defer_pull:
            where = defer_pull(exp, key, deferred_before.get(key, {}), profile, trial)
            if where:
                # Where the trial's files wait on their host, to find them later
                (trial_dir / f"deferred_{key}").write_text(where + "\n")
                deferred = True
            continue
        log(f"copying {key} from {src}")
        rsync(exp, src, trial_dir / "catalog" / key, move=False)
    if exp.results and not any(p.is_file() for p in (trial_dir / "result").rglob("*")):
        if deferred:
            log("note: the trial's [results] came back empty, but its deferred catalogs were saved on their "
                "hosts, so it counts as collected")
        else:
            log("warning: the trial's [results] came back empty")
            collected = False
    if started:
        write_stats(exp, run_dir, netsim, power, label, trial, wall_sec)
    else:
        log("the trial didn't start, so no stats are written for it")
    log(f"trial saved to {trial_dir}")
    return collected


def describe(phase: Phase) -> str:
    if phase.video:
        return f"video {phase.video.name}, flow {phase.file.name}"
    return f"mission {phase.file.name}"


def netsim_direction(exp: Experiment) -> str:
    """The direction netsim shapes, from its --direction arg, if given."""
    args = exp.netsim_args
    for i, a in enumerate(args):
        if a == "--direction" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--direction="):
            return a.split("=", 1)[1]
    return "ingress"


@dataclass
class AmbientLink:
    """The link an ambient trial runs on, measured once (measure_ambient_once):
    iperf3's one-second throughput, as timeline rows, and its report."""
    rows: list[dict]
    end: float  # seconds the measurement lasted
    measured_at: str
    dir: Path  # iperf3.json and iperf3_server.log


# Per link (ambient_key), its measurement this run
AMBIENT_LINKS: dict[tuple, AmbientLink] = {}


def vehicle_host(client: Daemon, exp: Experiment) -> tuple[DaemonCfg, str]:
    """The remote daemon whose ambient config runs the experiment's vehicle,
    and its IP as registered with the orchestrator: found from the configs,
    so the vehicle needn't be running."""
    listing = client.list_services()
    for key, config in exp.configs[AMBIENT].items():
        d = exp.daemons[key]
        if d.orchestrator or exp.vehicle not in config_services(config):
            continue
        if not d.ssh_host:
            raise RuntimeError(
                f"ambient: {d.key}, which runs {exp.vehicle}, needs a prefix in [daemons], "
                f"to start iperf3 on it over ssh"
            )
        remote = find_remote(listing, d, include_failed=True)
        if remote is None or not remote.ip:
            raise RuntimeError(f"ambient: {d.key}, which runs {exp.vehicle}, isn't registered with the orchestrator")
        return d, remote.ip
    raise RuntimeError(
        f"ambient: no remote daemon's config runs the vehicle {exp.vehicle}, so there is no link to measure"
    )


def ambient_key(exp: Experiment, host: DaemonCfg, ip: str) -> tuple:
    """What a measurement of the link depends on: experiments that agree on
    it share one."""
    return host.ssh_host, ip, netsim_direction(exp), exp.iperf_port, exp.iperf_duration


def measure_ambient_once(client: Daemon, exps: list[Experiment], only_trial: int | None, overwrite: bool) -> None:
    """For the ambient profile: measures the link between the vehicle's host
    and this one with iperf3, in the direction netsim would shape (by
    default, from the vehicle's host to here), once for the whole run, before
    any daemon is configured for an experiment, rather than before each
    trial. The experiments' daemons' running services are stopped for it, so
    nothing else is sent, and restarted afterwards, even if it fails. Each
    ambient trial then gets a copy (use_ambient). An experiment on another
    link (host, direction, port or duration) gets its own measurement."""
    needed = [e for e in exps if AMBIENT in e.profiles and pending_trials(e, AMBIENT, only_trial, overwrite)]
    if not needed:
        return
    links = {}
    for e in needed:
        host, ip = vehicle_host(client, e)
        links.setdefault(ambient_key(e, host, ip), (e, host, ip))
    log(f"ambient: measuring {len(links)} link(s) once, before any trial, for "
        f"{', '.join(e.name for e in needed)}")
    stopped = set()
    try:
        for e in needed:
            if e.dir not in stopped:
                stop_services(client, e)
                stopped.add(e.dir)
        for key, (e, host, ip) in links.items():
            out = Path(tempfile.mkdtemp(prefix="ambient_"))
            report = run_iperf(e, host, ip, out)
            rows, end = [], 0.0
            for interval in report.get("intervals", []):
                sums = interval.get("sum", {})
                if "start" not in sums:
                    continue
                kbps = round(sums.get("bits_per_second", 0.0) / 1000, 1)
                rows.append({
                    "t_sec": f"{sums['start']:.3f}", "drawn_kbps": kbps, "applied_kbps": kbps,
                    "state": AMBIENT, "delay_ms": "",
                })
                end = max(end, float(sums.get("end", sums["start"])))
            if not rows:
                raise RuntimeError(f"ambient: iperf3 reported no intervals; see {out / 'iperf3.json'}")
            AMBIENT_LINKS[key] = AmbientLink(rows, end, datetime.now().isoformat(timespec="seconds"), out)
            mean = sum(float(r["applied_kbps"]) for r in rows) / len(rows)
            log(f"ambient: measured {mean:.1f} Kbps on average over {end:g}s, from {host.key}; "
                f"every ambient trial on this link uses it")
    finally:
        log("ambient: restarting the services stopped for the measurement")
        restart_stopped_by_runner(client)


def use_ambient(client: Daemon, exp: Experiment, trial_dir: Path) -> float:
    """Gives an ambient trial the run's measurement of its link
    (measure_ambient_once): trial_dir/timeline.csv, as netsim would write its
    timeline, iperf3's report (iperf3.json) and its server's log, and
    ambient_measured_at. Returns the end of the measurement, in seconds."""
    host, ip = vehicle_host(client, exp)
    link = AMBIENT_LINKS.get(ambient_key(exp, host, ip))
    if link is None:
        raise RuntimeError(f"ambient: the link from {host.key} wasn't measured at the start of the run")
    with open(trial_dir / "timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TIMELINE_COLUMNS[1:])
        w.writeheader()
        w.writerows(link.rows)
    for name in ("iperf3.json", "iperf3_server.log"):
        if (link.dir / name).exists():
            shutil.copy2(link.dir / name, trial_dir / name)
    (trial_dir / "ambient_measured_at").write_text(f"{link.measured_at}\n")
    log(f"ambient: using the link measured at {link.measured_at}")
    return link.end


def run_iperf(exp: Experiment, host: DaemonCfg, ip: str, trial_dir: Path) -> dict:
    """Starts a one-off iperf3 server on host over ssh, and runs the client
    here against it at ip. Returns the client's JSON report, also saved to
    trial_dir/iperf3.json."""
    port = exp.iperf_port
    # -R makes the server (the vehicle's host) send, as for ingress shaping
    reverse = ["-R"] if netsim_direction(exp) == "ingress" else []
    client_cmd = [
        "iperf3", "-c", ip, "-p", str(port), "-t", f"{exp.iperf_duration:g}", "-i", "1", "-J", *reverse,
    ]
    log(
        f"ambient: measuring {'from' if reverse else 'to'} {host.key} ({ip}) with iperf3 "
        f"on port {port} for {exp.iperf_duration:g}s"
    )
    with open(trial_dir / "iperf3_server.log", "w") as server_log:
        # -1: the server exits after one test
        server = subprocess.Popen(
            ssh(host.ssh_host, f"{REMOTE_PATH} iperf3 -s -1 -p {port}"),
            stdout=server_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                time.sleep(1)  # let the server start listening
                try:
                    r = subprocess.run(
                        client_cmd, capture_output=True, text=True, timeout=exp.iperf_duration + 60,
                    )
                except FileNotFoundError:
                    raise RuntimeError("ambient: iperf3 is not installed here") from None
                try:
                    report = json.loads(r.stdout)
                except ValueError:
                    report = {"error": (r.stderr or r.stdout).strip()}
                if r.returncode == 0 and "error" not in report:
                    break
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"ambient: iperf3 failed: {report.get('error')}"
                        + (
                            f" (the server on {host.key} exited {server.returncode}; "
                            f"see {trial_dir / 'iperf3_server.log'})"
                            if server.poll() is not None else ""
                        )
                    )
        finally:
            if server.poll() is None:
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.terminate()
                    server.wait(timeout=10)
    (trial_dir / "iperf3.json").write_text(json.dumps(report, indent=2))
    return report


def run_trial(
    client: Daemon, exp: Experiment, vehicle: str, profile: str, run_dir: Path, trial: int,
    deferred_before: dict[str, dict[str, str]],
) -> None:
    """Runs one trial into run_dir/trial_<n>, adding its stats to
    run_dir/stats_<profile>_<experiment>.csv. deferred_before is what each
    defer_pull result held before the trial (deferred_files)."""
    phase = exp.phase
    label = f"{profile}_{exp.name}"
    trial_dir = run_dir / f"trial_{trial}"
    trial_dir.mkdir(parents=True)
    netsim = Netsim(exp, profile, trial_dir)
    power = PowerMonitor(exp, trial_dir)
    what = f"profile {profile}, trial {trial}/{exp.trials}"
    log(f"=== {exp.name}: {what} ({describe(phase)}) ===")

    if profile == AMBIENT:
        netsim.timeline_end = use_ambient(client, exp, trial_dir)

    if phase.kind == "mission":
        if phase.no_upload:
            log(f"loading {phase.file.name} on {vehicle} (not uploaded)")
        elif phase.file.is_dir():
            log(f"compiling ({mission_source(phase.file).name}) and loading mission folder "
                f"{phase.file.name} on {vehicle}")
            client.compile_mission(phase.file, vehicle)
        else:
            log(f"uploading and loading {phase.file.name} on {vehicle}")
            client.upload_mission(phase.file, vehicle)
        client.load_mission(phase.file.name, vehicle)
    # The flow is sent as written: the vehicle qualifies its names
    flow = phase.file.read_bytes() if phase.kind == "video" else b""
    if not phase.auto:
        input(f"[{exp.name} {profile} {trial}/{exp.trials}] Press ENTER to start the {phase.kind}... ")
    run_hook(exp, "setup", profile, trial, trial_dir, vehicle)

    if phase.kind == "mission":
        # Before the trial's try, so a mission that never started isn't
        # stopped, nor its services stopped and collected: the daemons go
        # straight back to their own configs, and the run stops
        try:
            client.start_mission(vehicle)
        except grpc.RpcError as e:
            log("Failed to start mission, aborting!")
            log(f"StartMission: {e.code().name} {e.details()}")
            restore_configs(client, exp)
            raise SystemExit(1) from None

    wall = 0.0
    started = finished = False
    try:
        if phase.kind == "video":
            client.set_dataflow(flow, vehicle)
        started = True
        netsim.start()
        power.start()
        log(f"{phase.kind} started")
        run_hook(exp, "start", profile, trial, trial_dir, vehicle)
        polled = "done" in exp.hooks
        done = lambda: hook_passes(exp, "done", profile, trial, trial_dir, vehicle)
        if polled and phase.duration > 0:
            log(f"running {phase.kind} under {what} for at most {phase.duration:g}s, until done")
            wall, ended = wait_until_done(phase.duration, done, "trial")
        else:
            wall, ended = wait_for(phase.duration, f"{phase.kind} under {what}"), False
        if ended:
            log(f"done after {wall:.0f}s; skipping the grace period")
        else:
            # Everything keeps running through the grace period: the mission or
            # flow, netsim and the power measurement
            run_hook(exp, "grace", profile, trial, trial_dir, vehicle)
            if polled and phase.grace > 0:
                log(f"grace period: at most {phase.grace:g}s more, until done")
                waited, ended = wait_until_done(phase.grace, done, "grace")
                log(f"{'done' if ended else 'not done'} after {waited:.0f}s of grace")
            else:
                waited = wait_grace(phase.grace)
            wall += waited
        run_hook(exp, "grace_end", profile, trial, trial_dir, vehicle)
        finished = True
    finally:
        # Before the mission and services stop, so the power timeline covers
        # the trial's run only, as the bandwidth timeline does
        power.stop()
        if phase.kind == "mission":
            for name, call in (("Hold", client.hold), ("StopMission", client.stop_mission)):
                try:
                    call(vehicle)
                    log(f"{name} sent")
                except grpc.RpcError as e:
                    log(f"warning: {name} failed: {e.code().name} {e.details()}")
        # Nothing may write to the files while they are collected
        try:
            stop_services(client, exp)
        except grpc.RpcError as e:
            log(f"warning: couldn't stop the services: {e.code().name} {e.details()}")
        try:
            collected = cleanup(exp, run_dir, trial_dir, netsim, power, label, profile, trial, wall,
                                deferred_before, started)
            if finished and not collected:
                log(f"warning: not marking {trial_dir} complete, since its results weren't all collected; "
                    "the next run runs it again")
            elif finished:  # so a later run skips it
                (trial_dir / TRIAL_DONE).write_text(f"{datetime.now().isoformat(timespec='seconds')}\n")
        finally:
            # However the trial ended, once it has been stopped and collected
            run_hook(exp, "end", profile, trial, trial_dir, vehicle)


def run_experiment(
    client: Daemon, exp: Experiment, overwrite: bool = False, only_trial: int | None = None,
    written: list[Path] | None = None,
) -> None:
    """Runs the trials of each profile not done yet (pending_trials): every
    trial, or with only_trial, that one; with overwrite, done or not. Each
    profile's results directory is added to `written` as soon as it is made,
    so an interrupted run's are known too."""
    phase = exp.phase
    log(
        f"### experiment {exp.name} ({exp.dir}): {describe(phase)}"
        + (" (not uploaded)" if phase.no_upload else "")
        + f", profiles {', '.join(exp.profiles)}, "
        + (f"trial {only_trial} only" if only_trial else f"{exp.trials} trial(s) each")
    )
    if phase.duration_from_video:
        log(f"video {phase.video.name} is {phase.duration:g}s long; each trial runs that long")
    if "done" in exp.hooks:
        log(f"each trial ends once its done hook passes: within that, then up to {phase.grace:g}s of grace")
    elif phase.grace:
        log(f"each trial keeps running for a {phase.grace:g}s grace period after that")
    for profile in exp.profiles:
        trials = pending_trials(exp, profile, only_trial, overwrite)
        if not trials:
            log(f"profile {profile}: already done in {run_dir_of(exp, profile)} (--overwrite to run it again)")
            continue
        skipped = sorted(done_trials(run_dir_of(exp, profile)) - set(trials)) if not overwrite else []
        if skipped:
            log(f"profile {profile}: skipping trial(s) {', '.join(map(str, skipped))}, already done")
        run_dir = prepare_run_dir(exp, profile, trials, only_trial, overwrite)
        if written is not None and run_dir not in written:
            written.append(run_dir)
        for trial in trials:
            # Each trial starts from freshly restarted daemons, so no state
            # (like a running flow) carries over from the previous one.
            # The uploads were pushed once, before the directory's first
            # experiment, and stay in the files directory across restarts.
            # However the trial ends (its duration is up, ENTER, Ctrl-C or
            # an error), each daemon gets its own config back
            PROGRESS.start_trial()
            try:
                # Taken before the daemons restart, since an engine may
                # write its files as soon as it starts
                deferred_before = deferred_files(exp)
                clear_results(exp)
                configure_daemons(client, exp, profile)
                time.sleep(2)  # let the orchestrator go down before polling it
                vehicle = client.wait_ready(exp.vehicle, exp.remotes)
                start_stopped_services(client, exp, profile)
                log(f"daemons ready; targeting {vehicle}")
                if exp.phase.warmup > 0:
                    log(f"warmup: waiting {exp.phase.warmup:g}s for the engines to load before starting")
                    PROGRESS.sleep(exp.phase.warmup, "warmup")
                run_trial(client, exp, vehicle, profile, run_dir, trial, deferred_before)
            finally:
                restore_configs(client, exp)
                PROGRESS.end_trial()
        log(f"profile {profile} saved to {run_dir}")
    remove_result_dirs(exp)
    log(f"experiment {exp.name} saved to {exp.dir / 'runs' / exp.name}")


def zip_results(run_dirs: list[Path], out_dir: Path, partial: bool = False) -> Path | None:
    """Zips run_dirs (each a profile's results: its trials and CSVs) into
    out_dir/<date>-<time>[-partial].zip, to keep or copy off the machine. Paths
    in the zip are relative to the current directory when under it (e.g.
    lab/.../runs/<experiment>/<profile>/...), else to each experiment
    directory's parent."""
    dirs = [d for d in run_dirs if d.is_dir()]
    if not dirs:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}{'-partial' if partial else ''}.zip"
    cwd = Path.cwd().resolve()
    files = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for d in dirs:
            d = d.resolve()
            # <exp dir>/runs/<experiment>/<profile>: name it from the exp dir's parent
            base = cwd if d.is_relative_to(cwd) else d.parents[3]
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(base))
                    files += 1
    log(f"zipped {files} file(s) from {len(dirs)} results folder(s) into {path} "
        f"({path.stat().st_size / 1e6:.1f} MB)")
    return path


def find_experiment_dirs(root: Path) -> list[Path]:
    """Every experiment directory (one with an experiment.toml) under root, at
    any depth, in alphabetical order of path. An experiment directory's own
    subdirectories (e.g. runs/) are not searched."""

    def walk(d: Path):
        for p in sorted(d.iterdir()):
            if not p.is_dir() or p.name.startswith(".") or p.name in (VIDEOS_DIR, MEDIA_DIR, "runs"):
                continue
            if (p / EXPERIMENT_FILE).is_file():
                yield p
            else:
                yield from walk(p)

    return list(walk(root))


def list_experiments(exp_dir: Path, names: list[str] | None = None) -> None:
    """Prints the experiments in exp_dir's experiment.toml. Only the toml is
    read: the files it names are not checked."""
    tables = experiment_tables(exp_dir, read_toml(exp_dir), names)
    rows = [("NAME", "KIND", "RUNS", "VEHICLE", "PROFILES", "TRIALS")]
    for t in tables:
        mission, video, ns = (t.get(k) if isinstance(t.get(k), dict) else {} for k in ("mission", "video", "netsim"))
        if mission.get("file"):
            kind, runs = "mission", str(mission["file"])
        elif video.get("video") or video.get("flow"):
            kind, runs = "video", f"{video.get('video') or '?'} (flow {video.get('flow') or '?'})"
        else:
            kind, runs = "?", "(no mission or video set)"
        profiles = ns.get("profiles", ["*"])
        profiles = "all" if "*" in profiles else ",".join(map(str, profiles))
        rows.append((t["name"], kind, runs, str(t.get("vehicle", "")), profiles, str(ns.get("trials", 1))))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    print(f"{exp_dir}:")
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def select_profiles(exps: list[Experiment], picked: list[str]) -> list[Experiment]:
    """Narrows each experiment to the picked profiles it lists, keeping its
    own order (and the configs it resolved over all of its profiles). An
    experiment that lists none of them is skipped."""
    unknown = [p for p in picked if p not in NETSIM_PROFILES]
    if unknown:
        raise SystemExit(
            f"unknown netsim profile {', '.join(unknown)}; the profiles are {', '.join(NETSIM_PROFILES)}"
        )
    unused = [p for p in picked if not any(p in e.profiles for e in exps)]
    if unused:
        raise SystemExit(f"no selected experiment runs profile {', '.join(unused)}")
    kept = []
    for e in exps:
        e.profiles = [p for p in e.profiles if p in picked]
        if e.profiles:
            kept.append(e)
        else:
            log(f"skipping experiment {e.name} ({e.dir}): it runs none of profiles {', '.join(picked)}")
    return kept


def main():
    ap = argparse.ArgumentParser(
        description="Run SteelEagle experiments.", fromfile_prefix_chars="@",
        epilog="@FILE reads more paths (and options) from FILE, one per line.",
    )
    ap.add_argument(
        "paths", nargs="+", type=Path, metavar="path",
        help="experiment directories, run in the order given (with -r, directories of them)",
    )
    ap.add_argument(
        "-r", "--recursive", action="store_true",
        help="run every experiment directory inside each path, at any depth",
    )
    ap.add_argument(
        "-e", "--experiment", action="append", dest="experiments", metavar="NAME",
        help="run only the experiment with this name in path's experiment.toml (repeat for several)",
    )
    ap.add_argument(
        "-l", "--list", action="store_true",
        help="list the experiments in path's experiment.toml (with -r, in each directory) and exit",
    )
    ap.add_argument(
        "-p", "--profile", action="append", dest="profiles", metavar="PROFILE",
        help="run only this netsim profile, in each experiment that lists it "
        "(repeat, or separate with commas, for several)",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="run every trial again, deleting the earlier results of each profile run (with -t, only "
        "that trial's). Without it, trials already done are skipped and only the missing ones run",
    )
    ap.add_argument(
        "-t", "--trial", type=int, metavar="N",
        help="run only trial N of each profile, into its results folder (skipped if already done; "
        "with --overwrite, replaced)",
    )
    ap.add_argument(
        "--update-packages", action=argparse.BooleanOptionalAction, default=True,
        help="reinstall each package the daemons' configs use whose code differs from upstream "
        "(its subdirectory at its branch's latest commit), once per daemon per run; on by default. "
        "Missing packages are always installed",
    )
    ap.add_argument(
        "--zip-dir", type=Path, default=Path("archives"), metavar="DIR",
        help="when the run ends, zip the results folders it wrote into DIR/<date>-<time>.zip "
        "(-partial if it stopped early; default %(default)s)",
    )
    ap.add_argument("--no-zip", action="store_true", help="don't zip the results when the run ends")
    ap.add_argument("--socket", default=default_socket(), help="daemon control socket (default: %(default)s)")
    args = ap.parse_args()
    global UPDATE_PACKAGES
    UPDATE_PACKAGES = args.update_packages
    if args.trial is not None and args.trial < 1:
        ap.error("-t takes a trial number >= 1")
    if (args.recursive or len(args.paths) > 1) and args.experiments:
        ap.error("-e picks experiments in one directory's experiment.toml; it can't be used with -r "
                 "or several paths")

    exp_dirs: list[Path] = []
    for path in args.paths:
        root = path.resolve()
        if not root.is_dir():
            raise SystemExit(f"{path}: no such directory")
        if (root / EXPERIMENT_FILE).is_file():
            found = [root]
        elif args.recursive:
            found = find_experiment_dirs(root)
            if not found:
                log(f"skipping {path}: no experiment directories (with {EXPERIMENT_FILE}) in it")
        else:
            # e.g. a section directory caught by a shell glob: not an experiment
            log(f"skipping {path}: it has no {EXPERIMENT_FILE} (use -r for a directory of experiments)")
            found = []
        exp_dirs += [d for d in found if d not in exp_dirs]
    if not exp_dirs:
        raise SystemExit(f"no experiment directories (with {EXPERIMENT_FILE}) to run")

    try:
        if args.list:
            for i, d in enumerate(exp_dirs):
                if i:
                    print()
                list_experiments(d, args.experiments)
            return
        # Check every experiment up front, so a typo doesn't stop a batch midway
        exps = [e for d in exp_dirs for e in load_experiments(d, args.experiments)]
    except ValueError as e:
        raise SystemExit(f"invalid experiment: {e}")
    if args.profiles:
        picked = [p.strip() for arg in args.profiles for p in arg.split(",") if p.strip()]
        exps = select_profiles(exps, list(dict.fromkeys(picked)))

    # Ask for the sudo password now rather than in the middle of a run
    if os.geteuid() != 0 and any(
        "--dry-run" not in e.netsim_args and any(p not in NO_NETSIM for p in e.profiles) for e in exps
    ):
        log("netsim needs root; checking sudo")
        sudo_validate()

    # Ask for each remote's ssh password now too, once
    open_ssh(sorted({d.ssh_host for e in exps for d in e.remotes if d.ssh_host}))

    # A kill or a dropped ssh session exits through the finally blocks too, so
    # netsim is stopped and the daemons get their own configs back
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _: sys.exit(128 + signum))
    if not Path(args.socket).exists():
        found = sorted(glob.glob("/run/user/*/steeleagle/eagled.sock"))
        raise SystemExit(
            f"no daemon control socket at {args.socket}: the runner talks to the orchestrator's "
            f"eagled through the socket of the user it runs as (the invoking user's, under sudo). "
            + (f"Daemon sockets on this host: {', '.join(found)}; run as that user, or pass --socket."
               if found else "No eagled socket was found on this host; is eagled running here?")
        )
    client = Daemon(args.socket)
    todo = {id(e): sum(len(pending_trials(e, p, args.trial, args.overwrite)) for p in e.profiles) for e in exps}
    for e in exps:
        PROGRESS.add(e, todo[id(e)])
    if not PROGRESS.total:
        log("every trial is already done (--overwrite to run them again)")
        return
    done = sum(len(e.profiles) * (1 if args.trial else e.trials) for e in exps) - PROGRESS.total
    if done and not args.overwrite:
        log(f"{done} trial(s) already done are skipped (--overwrite to run them again)")
    PROGRESS.started = time.monotonic()
    log(f"{PROGRESS.total} trial(s) in {len(exps)} experiment(s); {PROGRESS.summary()}")
    pushed: set[Path] = set()
    written: list[Path] = []
    finished = False
    try:
        # Once, before any daemon is configured for an experiment
        measure_ambient_once(client, exps, args.trial, args.overwrite)
        for e in exps:
            if not todo[id(e)]:
                log(f"### experiment {e.name} ({e.dir}): already done")
                continue
            # Each directory's uploads are pushed once, before its first
            # experiment, so the drivers find them when the daemons restart
            if e.dir not in pushed:
                push_uploads(client, e)
                pushed.add(e.dir)
            run_experiment(client, e, args.overwrite, args.trial, written)
        finished = True
        log(f"done: {len(exps)} experiment(s), {PROGRESS.done} trial(s) in "
            f"{fmt_secs(time.monotonic() - PROGRESS.started)}")
    finally:
        # Even a stopped run's finished trials are worth keeping
        if not args.no_zip:
            try:
                zip_results(written, args.zip_dir, partial=not finished)
            except Exception as ex:  # never hide the run's own error
                log(f"could not zip the results: {ex}")


if __name__ == "__main__":
    main()
