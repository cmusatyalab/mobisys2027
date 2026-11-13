# steeleagle-experiment

An automatic experiment runner for SteelEagle. You give it an experiment
directory, whose `experiment.toml` holds a list of experiments. It first
pushes the files and videos the directory declares to the daemons, once. Then,
for each experiment, and for each of its network profiles, it:

- restarts the daemons with that profile's configs,
- starts a mission, or a dataflow over a video, on a vehicle,
- limits the network with [netsim](src/mobisys2027/netsim/README.md),
- stops everything when the run ends,
- measures the power draw of the remote daemons' hosts over the run,
- collects the result files, catalog files, and bandwidth and power statistics
  into the experiment directory.

## Running it

Run everything from the root of this repository. [uv](https://docs.astral.sh/uv/)
sets up the environment on first use. The experiment directories are in `lab/`
and `live/`:

```bash
uv run experiment lab/yolo_ed          # one experiment directory
uv run experiment -r lab               # every experiment directory in lab/
```

To call `experiment` and `netsim` without `uv run`, activate the environment
with `source .venv/bin/activate` after the first `uv sync`.

### Picking and listing experiments

By default every experiment in the directory's `experiment.toml` runs, in the
order they appear. `-e` runs only the named experiment, and can be repeated:

```bash
uv run experiment path/to/my_experiment -e onboard_track -e parking
```

`-l` lists the experiments and exits, without running anything or checking
the files they name:

```
$ uv run experiment path/to/my_experiment -l
path/to/my_experiment:
  NAME           KIND     RUNS                             VEHICLE  PROFILES          TRIALS
  onboard_track  video    traintrack.mp4 (flow flow.toml)  drone    all               3
  parking        video    parking.mp4 (flow flow.toml)     drone    normal,malicious  1
  survey         mission  survey.zip                       drone    all               3
```

### Running a batch of experiment directories

`-r` runs every experiment in every subdirectory that contains an
`experiment.toml`, in alphabetical order of directory:

```bash
uv run experiment -r path/to/experiments/
```

Every experiment in the batch is checked before the first one starts.
`-l` works with `-r` too, listing each directory's experiments; `-e` doesn't. A typo
in the last experiment stops the batch before anything runs, not halfway
through.

### Options

| Option            | Meaning                                                                     |
|-------------------|-----------------------------------------------------------------------------|
| `path`            | An experiment directory, or with `-r`, a directory of experiment directories |
| `-r, --recursive` | Run every experiment directory inside `path`                                |
| `-e, --experiment NAME` | Run only the experiment named `NAME` (repeat for several). Not with `-r` |
| `-l, --list`      | List the experiments in `path` (with `-r`, in each directory) and exit      |
| `--update-packages` | Reinstall every package the daemons' configs use, once per daemon per run, to get a branch's latest code (see below) |
| `--socket PATH`   | eagled's control socket (default: `$XDG_RUNTIME_DIR/steeleagle/eagled.sock`) |

Before configuring a daemon, the runner installs any package its config uses
that the daemon doesn't have, through the orchestrator. A package that is
already installed is left alone, even if its branch has moved on since; with
`--update-packages`, each one is reinstalled the first time a daemon is
configured in the run, so it picks up the branch's latest code. Updated
plugins rebuild their environments when the daemon restarts, which makes the
first trial slower.

## sudo: run it as yourself, not as root

netsim changes kernel network settings, so it needs root. Run the runner as
your own user anyway:

- eagled is a systemd **user** service. Its control socket is
  `/run/user/<your uid>/steeleagle/eagled.sock`.
- The runner runs only netsim through `sudo`. It asks for your sudo password
  once at startup, and checks again before each trial, so an expired sudo
  timestamp asks again rather than failing.
- rsync and ssh to the remote daemons use **your** ssh keys and `known_hosts`.
  They run attached to your terminal, so if a host asks for a password or to
  confirm its host key, answer the prompt there.

If you do run it as root (for example `sudo "$(which uv)" run experiment ...`), the default
socket is that of the user who ran `sudo` (`SUDO_UID`). In that case rsync and
ssh use root's ssh setup.

## Running netsim by hand

The project also installs a `netsim` command, to shape the link outside of
an experiment:

```bash
sudo .venv/bin/netsim [-i INTERFACE] [-p PORT] [-f FAILPOINTS] PROFILE
```

| Option              | Default      | Meaning                                                     |
|---------------------|--------------|-------------------------------------------------------------|
| `PROFILE`           | required     | `normal`, `limited`, `degraded`, `intermittent`, `denied`, `malicious`, `outage` or `delay` |
| `-i, --interface`   | `tailscale0` | Interface to shape                                          |
| `-p, --port`        | `9001`       | Port to shape                                               |
| `-f, --failpoints`  | none         | `malicious` and `outage` only: forced outages, e.g. `30:10,90:5` |

For example, `sudo .venv/bin/netsim -f 30:10,90:5 malicious`. Giving
`malicious` or `outage` without `-f` prints a warning, since it then behaves
like `intermittent` or `normal`. Giving `-f` with any other profile is an
error. It runs until Ctrl-C; see the
[netsim README](src/mobisys2027/netsim/README.md) for the other options.
(`sudo` needs the full path, as it resets `PATH`.)

## Requirements

- On the machine you run it on:
  - Python 3.12+ (uv installs it if needed).
  - A running eagled, which acts as the orchestrator.
  - `rsync` and `ssh`.
  - `tc` and `ip` (iproute2) and the `ifb` kernel module, for netsim.
- On each remote daemon's host, `python3` (3.8+) for the power measurement,
  which is written for the Particle Tachyon (see
  [`[experiments.power]`](#experimentspower)).
- Remote daemons locked to the orchestrator. They are configured through it,
  so the runner waits (up to 5 minutes) for each one to register before
  configuring it.

## The experiment directory

Every file the experiment uses lives next to its `experiment.toml`:

```
my_experiment/
├── experiment.toml
├── traintrack.mp4, flow.toml        # the videos (or in a videos/ folder above), flows and missions the experiments name
├── labels.json                      # (optional) any other files listed in [[uploads]]
├── lambda2.toml                     # config for daemon "lambda2", for every experiment
├── fermi.toml                       # config for daemon "fermi", for every experiment
├── fermi_onboard_track.toml         # (optional) fermi's config for experiment onboard_track
├── fermi_onboard_track_malicious.toml  # (optional) ...and for its malicious profile
└── runs/                            # written by the runner
```

See [`src/mobisys2027/example/experiment.toml`](src/mobisys2027/example/experiment.toml)
for a commented template.

## experiment.toml

`experiment.toml` holds a list of experiments, each an `[[experiments]]`
table with its own `[experiments.mission]` or `[experiments.video]` and
`[experiments.netsim]` sections. They run in the order they appear. The
`[[uploads]]`, `[daemons]`, `[results]` and `[catalogs]` tables are shared by
all of them.

```toml
[[experiments]]
name = "traintrack"
vehicle = "drone"
[experiments.video]
video = "traintrack.mp4"
flow = "flow.toml"
[experiments.netsim]
interface = "tailscale0"
port = 9001
trials = 3

[[experiments]]
name = "parking"
vehicle = "drone"
[experiments.video]
video = "parking.mp4"
flow = "flow.toml"
[experiments.netsim]
profiles = ["normal", "malicious"]
interface = "tailscale0"
port = 9001
failpoints = [[5, 5], [60, 20]]

[[uploads]]
file = "traintrack.mp4"
[[uploads]]
file = "parking.mp4"
no_upload = true   # already on every daemon

[daemons.lambda2]
orchestrator = true
```

Each section header must start with `experiments.` (plural). After an
`[[experiments]]` header, `[experiment.video]` would make a separate table, so
the runner rejects it.

### `[[experiments]]`

| Key       | Meaning                                                                       |
|-----------|-------------------------------------------------------------------------------|
| `name`    | Unique name. Used by `-e`, in the daemon config file names, and for the `runs/` folder |
| `vehicle` | Vehicle to target: its name (`drone`) or full name (`100.64.0.2.drone`)  |

Then set **exactly one** of `[experiments.mission]` or `[experiments.video]`.
The other may be left out, or kept with its files set to `""`.

| Key                  | Meaning                                                                                  |
|----------------------|------------------------------------------------------------------------------------------|
| `file` (mission)     | Mission file in the experiment directory                                                 |
| `no_upload` (mission)| `true`: don't upload the mission; it's assumed to already be on the vehicle under its own name, and is just loaded |
| `video` (video)      | A single video, which must be one of the [`[[uploads]]`](#uploads), by its file name or the name it is pushed as. For several videos, add an experiment for each |
| `flow` (video)       | Flow file in the experiment directory, sent with `SetDataflow` to start each trial       |
| `auto`               | `false`: press ENTER to start each trial. `true`: start right away                       |
| `duration`           | Seconds to run each trial. `0`: run until you press ENTER. For a mission, defaults to `0`; for a video, defaults to the video's length (see below) |

The video is pushed (or not) as its `[[uploads]]` entry says. A driver such
as `fake_drone` then finds it by name in `STEELEAGLE_FILES`, so set the
driver's `video` arg to the name the video is pushed as. If its entry has
`no_upload = true` and the file isn't here, its length can't be found, so
`duration` must be set.

If a video experiment leaves `duration` out, each trial runs for the length of
its video. The length comes from `ffprobe` if it is installed, or else from the
header of an MP4/MOV file; if neither works, the experiment is rejected before
it starts and you need to set `duration`. Setting `duration` (including `0`)
overrides the video's length.

A flow can name its engines by their bare service names (`rescale`,
`verifier`), as in the daemon configs. uriel only finds an engine by the name
its daemon serves it under, `<daemon ip>.<name>` (e.g.
`100.101.84.40.rescale`), so before each trial the runner qualifies each bare
name, and the edges with it, from the orchestrator's `ListServices`:

- a `local` engine is the one of that name on the vehicle's daemon;
- a `remote` engine is the only one of that name on any other daemon.

A bare name with no such engine, or (remote) more than one, stops the trial
before it starts, naming the engines it found. Names that are already
qualified are sent as they are. The runner logs what it mapped each name to.

### `[[uploads]]`

Every file and video the experiments need on the daemons (a video for a
driver, a model, a label map) is declared once, for the whole directory, as an
`[[uploads]]` table. Before the directory's first experiment runs, each is
pushed **once** to `STEELEAGLE_FILES` (`~/.local/share/steeleagle/files`) on
the orchestrator and every remote: through the orchestrator's `TransferFile`,
or with rsync if a remote isn't registered. They stay there across the
daemon restarts of every trial, profile and experiment in the directory.

```toml
[[uploads]]
file = "labels.json"          # pushed as labels.json
[[uploads]]
file = "model_v2.pt"
push_as = "model.pt"          # pushed as model.pt
[[uploads]]
file = "parkinglot.mp4"
no_upload = true              # already on every daemon: not pushed
```

| Key         | Meaning                                                                                     |
|-------------|---------------------------------------------------------------------------------------------|
| `file`      | File name, looked for first in the nearest `videos/` folder at or above the experiment directory, then in the experiment directory |
| `push_as`   | (optional) Name it is saved as on the daemons; defaults to its own name. Every upload is pushed at once, so two videos can't share one name; give each driver the name of its own video |
| `no_upload` | (optional) `true`: don't push it; it's assumed to already be in every daemon's `STEELEAGLE_FILES` under its own name. The file then need not be here, and `push_as` can't be set |

Two uploads can't be pushed under the same name. With `-e`, every upload is
still pushed. Since uploads are pushed only once, a `[results]` path that would
move one off its daemon (e.g. `fermi/root/.local/share/steeleagle/files/*`) is
rejected; collect it under `[catalogs]` instead.

### `[experiments.netsim]`

Each experiment has its own netsim settings.

| Key          | Meaning                                                                                       |
|--------------|-----------------------------------------------------------------------------------------------|
| `profiles`   | Profiles to run, in order. `["*"]` runs all ten (see below)                                   |
| `interface`  | Interface netsim shapes, e.g. `tailscale0`                                                    |
| `port`       | Port netsim shapes                                                                            |
| `failpoints` | Outages for the `malicious` and `outage` profiles, as `[event_time, dead_time]` pairs in seconds|
| `trials`     | (optional) Times to run each profile, default `1`. Every trial runs before the next profile   |
| `args`       | (optional) Extra netsim flags, passed through as is, e.g. `["--direction", "egress", "--seed", "1"]` |
| `iperf_port` | (optional) `ambient` only: port iperf3 measures the link on, default `5201`                    |
| `iperf_duration` | (optional) `ambient` only: seconds iperf3 measures the link for, default `30`              |

The profiles, in the order `["*"]` runs them:

| Profile        | Bandwidth (Kbps)                         |
|----------------|------------------------------------------|
| `normal`       | 2000 – 3000                              |
| `limited`      | 500 – 1000                               |
| `degraded`     | 50 – 100                                 |
| `intermittent` | 0 – 100, often dropping to 0             |
| `denied`       | always 0                                 |
| `malicious`    | like intermittent, plus `failpoints`     |
| `outage`       | like normal, plus `failpoints`           |
| `delay`        | unshaped, plus 0 – 300 ms of variable latency |
| `none`         | unshaped: netsim isn't run               |
| `ambient`      | like `none`, but measured with iperf3 first |

`delay` leaves the bandwidth alone. Every second it picks a new one-way delay
uniformly between 0 and 300 ms and applies it to the port's traffic with a
netem qdisc. Change the range with netsim's `--min-delay-ms` and
`--max-delay-ms` in `args`. When the delay drops, packets sent under the new
delay can overtake ones still held under the old one, so TCP sees some
reordering.

`none` runs the experiment on the network as it is, without netsim. Its
trials have no timeline, and their stats CSV has only the trial and duration
filled in.

`ambient` also runs the experiment on the network as it is. Before each trial,
once the daemons are up, the runner:

1. stops the vehicle (`StopService`), so it isn't sending anything,
2. starts a one-off `iperf3` server on the vehicle's host over ssh, and runs
   the client here against it for `iperf_duration` seconds on `iperf_port`,
   in the direction netsim would shape (by default the vehicle's host sends,
   with `-R`; with `--direction egress` in `args`, this host sends),
3. restarts the vehicle (`RestartService`, even if iperf3 failed) and waits for
   it to be running again,

and then runs the trial without netsim. The per-second throughput iperf3
measured is the trial's bandwidth timeline (state `ambient`), and its full
report is saved as `trial_<n>/iperf3.json`. This needs `iperf3` installed on
both hosts, the vehicle running on a remote daemon (not the orchestrator),
and that daemon in `[daemons]` with a `prefix` to ssh to. A stopped service
stays stopped across daemon restarts, so if the runner is killed hard while
measuring, restart the vehicle with `eagle restart-service <vehicle>`.

Each failpoint `[30, 10]` cuts the port off from 30 s to 40 s after the run
starts, whatever the bandwidth draw says. Failpoints may be given in any order,
and overlapping ones are merged into one outage. Only the `malicious` and
`outage` profiles use them: `malicious` adds them to intermittent's 0 – 100
Kbps, and `outage` adds them to normal's 2000 – 3000 Kbps, for measuring an
outage on an otherwise healthy link. With no failpoints, `malicious` behaves
like `intermittent` and `outage` like `normal` (netsim prints a warning).

```toml
failpoints = [[30, 10], [90, 5], [150, 20]]
```

Since each experiment has its own `failpoints`, each video gets its own
outages by being its own experiment.

See the [netsim README](src/mobisys2027/netsim/README.md) for how
bandwidth is drawn and what the other flags do.

### `[experiments.power]`

While each trial runs, the runner measures the power draw of the host of every
remote daemon that has a `prefix` (not the orchestrator). The section is
optional; leave it out to measure with the defaults.

| Key        | Meaning                                                              |
|------------|----------------------------------------------------------------------|
| `enabled`  | (optional) Whether to measure power, default `true`                  |
| `interval` | (optional) Seconds per row of the power timeline, at least `0.1`, default `1` |

```toml
[experiments.power]
interval = 0.5
```

The measuring script,
[`src/mobisys2027/power/tachyon_power.py`](src/mobisys2027/power/tachyon_power.py),
is written for the Particle Tachyon. The runner sends it over ssh with
`python3 -c`, so nothing is installed on the host. It starts with netsim, once
the vehicle has started the mission or flow, and stops when the duration is up
(before `Hold` and the services stop), so the power timeline covers the same
span as the bandwidth timeline. It stops when its ssh connection closes, so it
never outlives the trial, even if the runner is killed.

The Tachyon's battery gauge reports `power_now`, but that value doesn't follow
the load. The script works the board's power out from the voltage and current
of its supplies instead: the external input that is online (USB or wireless),
minus what goes into the battery while it charges, plus what comes out of it
while it discharges. So it measures the same thing whether the Tachyon runs on
USB power or on its battery. The input side is measured before the charger, so
it includes the charger's losses. Single readings are noisy (the input current
reads in 5 mA steps), so the script samples every 0.1 s and each row is the
mean of its interval. On the Tachyon, it reads about 1.1 W idle and 6.7 W with
all six cores busy.

If a host has none of these supplies (it isn't a Tachyon), the script exits,
the runner prints a warning naming its `power_<daemon>.log`, and the trial
goes on without power statistics. You can also run the script by hand on a
Tachyon: `python3 tachyon_power.py --no-stdin > power.csv` (stop it with
Ctrl-C).

### `[daemons]`

One table per daemon, named after the daemon:

```toml
[daemons.lambda2]
orchestrator = true      # exactly one daemon is the orchestrator: the one this runs on
[daemons.fermi]
prefix = "root@fermi:"   # rsync/ssh prefix for this host
```

| Key            | Meaning                                                                              |
|----------------|--------------------------------------------------------------------------------------|
| `orchestrator` | `true` for the local daemon the runner talks to. Exactly one must be set            |
| `prefix`       | `user@host:` prefix for rsync and ssh. Empty for the orchestrator (local files)     |

Remote daemons are matched to the orchestrator's registered remotes by
hostname: either the table name (`fermi`) or the host in `prefix`.

#### Daemon configs

There is no `config` key. Each daemon's config file is named after the daemon
and the experiment, and the runner picks one for each of the experiment's
profiles, in profile order:

1. `<daemon>_<experiment>_<profile>.toml` if it exists, e.g.
   `fermi_onboard_track_normal.toml`;
2. otherwise the config the previous profile used;
3. otherwise, before any profile-specific file has been found,
   `<daemon>_<experiment>.toml`, e.g. `fermi_onboard_track.toml`;
4. otherwise `<daemon>.toml`, shared by every experiment.

For example, for experiment `track` with
`profiles = ["normal", "limited", "degraded", "intermittent"]`:

| Files present                                         | normal               | limited                    | degraded                    | intermittent                |
|-------------------------------------------------------|----------------------|----------------------------|-----------------------------|-----------------------------|
| `fermi.toml`                                          | `fermi.toml`         | `fermi.toml`               | `fermi.toml`                | `fermi.toml`                |
| `fermi.toml`, `fermi_track.toml`                      | `fermi_track.toml`   | `fermi_track.toml`         | `fermi_track.toml`          | `fermi_track.toml`          |
| `fermi.toml`, `fermi_track_limited.toml`              | `fermi.toml`         | `fermi_track_limited.toml` | `fermi_track_limited.toml`  | `fermi_track_limited.toml`  |
| `fermi_track_normal.toml`, `fermi_track_degraded.toml`| `fermi_track_normal.toml` | `fermi_track_normal.toml` | `fermi_track_degraded.toml` | `fermi_track_degraded.toml` |

If a profile has no config to use (no specific file, no earlier one, and
neither `<daemon>_<experiment>.toml` nor `<daemon>.toml`), the experiment is
rejected before it starts. An `experiment.toml` that still has the old
`config` key is rejected too.

### `[results]` and `[catalogs]`

Files to collect after each trial, as `name = "<daemon>/<absolute path>"`.
The first part of the path names the daemon whose host holds the files. The
rest is an absolute path on that host, without its leading `/`:

```toml
[results]   # moved: the sources are deleted (rsync --remove-source-files)
frames_seen = "fermi/home/me/.local/share/steeleagle/files/seen"
frames_sent = "lambda2/home/me/.local/share/steeleagle/files/sent"
[catalogs]  # copied: the sources are kept
mcap = "lambda2/home/me/.local/share/steeleagle/files/mcap*"   # globs work
```

A directory's contents are collected into `result/<name>/` (or
`catalog/<name>/`). The runner doesn't put the directory itself inside. Moving
leaves the emptied source directories in place. A source that matches nothing
gives a warning, not an error.

A catalog can instead be left on its host, to be transferred later, with
`defer_pull`:

```toml
[catalogs]
dets_onboard = { path = "heisenberg/root/.local/share/steeleagle/files/detection_log/*", defer_pull = true }
```

It isn't copied; the files the trial added are moved, on their host, into
`<experiment>_<profile>_<trial>_<catalog>/` in the files directory
(`~/.local/share/steeleagle/files`), keeping their names, e.g.
`case_study_2_cop_delay_2_dets_onboard/10002.jpg`. They are moved out of the
engine's own directory because an engine may empty that whenever it starts
(detection_log does with `refresh`), and the daemons restart twice in every
trial. Before each trial (before the daemons restart) the runner lists what
the path holds, so only files written during the trial are moved. If the
trial's directory already exists, e.g. from an earlier run, a new one with
the time added (`...-20261111-134131`) is used. A file whose name is taken in
the directory is left where it is, with a warning. `defer_pull` is only for
catalogs: a result is
removed from its host once it is pulled, so it can't be left there.

## What a run does

Before the first experiment of each directory, its `[[uploads]]` (except
those with `no_upload`) are pushed to every daemon's files directory. Then, for
each experiment, in order, for each profile in its `profiles`, and for each of
its `trials`:

1. **Restart the daemons.** Each remote daemon gets its config for the experiment and
   profile, sent through the orchestrator once the remote has registered, and
   then the orchestrator gets its own config. Applying a config restarts the
   daemon. Before its config is replaced, each daemon's own config is saved
   (see step 10).
2. **Wait until ready.** The runner polls `ListServices` until every remote has
   re-registered and the vehicle is listed and running (5-minute timeout).
3. **Measure** (`ambient` only). Stop the vehicle, measure the link with
   iperf3, and restart the vehicle (see [`[experiments.netsim]`](#experimentsnetsim)).
4. **Prepare.** For a mission: upload it (unless `no_upload` is set) and load
   it on the vehicle.
5. **Wait for you.** If `auto = false`: press ENTER to start the trial.
6. **Start.** `StartMission`, or `SetDataflow` with the video's flow (its bare
   engine names qualified; see [`[[experiments]]`](#experiments)), then start netsim
   as soon as the vehicle answers (except under `none` and `ambient`), and the
   power measurement on each remote's host (see
   [`[experiments.power]`](#experimentspower)).
7. **Run** for `duration` seconds (for a video with no `duration`, the video's
   length), or until you press ENTER if it is `0`.
8. **Stop.** Stop the power measurement. For a mission: `Hold`, then `StopMission`. These are sent even if
   the run is interrupted. Then every service of the experiment's daemons is
   stopped (`StopService`, the vehicles first), so nothing is still writing to
   the files collected next, and netsim is stopped with SIGINT, so it prints
   its byte totals and removes its `tc` rules.
9. **Collect** results and catalogs, and write the statistics. A
   `defer_pull` catalog is moved aside on its host instead (see
   [`[results]` and `[catalogs]`](#results-and-catalogs)).
10. **Restore the daemons.** The services stopped in step 8 are restarted
    first: a service stopped with `StopService` stays marked stopped across
    reconfigures, so one the daemon's own config doesn't have would otherwise
    start stopped under a later config that has it. With no flow set, they
    don't do anything. Then each daemon is configured with its own config
    again: the remotes through the orchestrator, then the orchestrator. That
    restarts them, which also ends the trial's flow, so no engine (e.g. a
    detection log) keeps working for it afterwards. This happens however the
    trial ends: its duration is up, you press ENTER, Ctrl-C, an error, or the
    runner is killed (SIGTERM, or SIGHUP when its ssh session drops).

Because the daemons restart for every trial, nothing (a running flow, a
loaded mission) carries over from one trial, profile or experiment to the next.

Each daemon's own config is saved in
`~/.local/state/steeleagle-experiment/original-configs/<daemon>.toml` from
just before it is configured until it is restored. If a run dies before it
can restore one (e.g. `kill -9`), the file stays, and the next run restores
that config, rather than the experiment config it finds on the daemon, so it
can also be restored by running the runner again.

## Output

Each profile's trials go to `runs/<experiment>/<profile>/` in the experiment
directory. If that folder already exists (from an earlier run), a timestamp is
added (`runs/onboard_track/normal-20261110-031500/`), so nothing is
overwritten.

```
runs/onboard_track/normal/
├── trial_1/
│   ├── result/<name>/    files from [results], moved off their hosts
│   ├── catalog/<name>/   files from [catalogs], copied
│   ├── timeline.csv      the bandwidth netsim allowed over time in this trial
│   ├── netsim.log        netsim's full output
│   ├── power_fermi.csv   remote fermi's power draw over time in this trial
│   └── power_fermi.log   the power script's messages
├── trial_2/ ...
├── stats_normal_onboard_track.csv      summary statistics, one row per trial
├── timeline_normal_onboard_track.csv   every trial's timeline, with a trial column
└── power_normal_onboard_track.csv      every trial's power timelines, with trial and daemon columns
```

The CSVs are named `stats_<profile>_<experiment>.csv`,
`timeline_<profile>_<experiment>.csv` and `power_<profile>_<experiment>.csv`,
so they stay distinct when gathered together. They are appended to as each trial finishes, so the trials done so
far are kept if a later one fails.

### stats_\<profile\>_\<experiment\>.csv

One row per trial, with these columns:

| Column                          | Meaning                                                                                 |
|---------------------------------|-----------------------------------------------------------------------------------------|
| `trial`                         | The trial number, from 1                                                                |
| `duration_sec`                  | How long netsim ran                                                                     |
| `total_bytes_delivered`         | Bytes that got through netsim on the shaped port (IP layer, link headers removed)       |
| `total_bytes_dropped_while_off` | Bytes discarded while the port was off (rate 0 or a failpoint)                          |
| `total_bytes_arrived`           | Delivered + dropped while off (not counting packets dropped by the shaper queue)        |
| `pkts_dropped_by_queue`         | Packets that overflowed the shaper queue (tc doesn't record their size)                 |
| `bandwidth_mean_kbps`           | Mean bandwidth allowed                                                                  |
| `bandwidth_median_kbps`         | Median bandwidth allowed                                                                |
| `bandwidth_p95_kbps`            | 95th percentile of bandwidth allowed                                                    |
| `bandwidth_min_kbps`, `bandwidth_max_kbps` | Range of bandwidth allowed                                                   |
| `time_off_sec`                  | Time the port was fully off                                                             |
| `delay_mean_ms`, `delay_median_ms`, `delay_p95_ms`, `delay_min_ms`, `delay_max_ms` | Latency netsim added (`delay` only) |
| `power_mean_w`                  | Mean power drawn by the remote daemon's host, in watts                                  |
| `power_median_w`                | Median power drawn                                                                      |
| `power_p99_w`                   | 99th percentile of power drawn                                                          |

The power columns are there when power is measured (see
[`[experiments.power]`](#experimentspower)). When more than one remote is
measured, each gets its own, named after it: `fermi_power_mean_w`, and so on.
They are computed over the rows of the power timeline, each weighted by its
number of samples, so a short last row counts for less. The power is measured
under every profile, including `none` and `ambient`.

The byte totals come from netsim's own counters. See the netsim README for
exactly what they include.

The bandwidth statistics describe the **allowed** rate, not the measured
throughput. They are weighted by time: each rate counts for as long as it was
in effect, and periods with the port off count as 0. Under `delay` the
bandwidth isn't limited, so they're empty and the delay columns are filled in
instead. Under `ambient`, they describe the throughput iperf3 measured before
the trial, and the byte totals are empty since netsim doesn't run.

### timeline_\<profile\>_\<experiment\>.csv

netsim's trace for each trial, with one row each time it drew a new rate or
crossed a failpoint boundary. `t_sec` restarts at 0 for each trial:

```
trial,t_sec,drawn_kbps,applied_kbps,state,delay_ms
1,0.000,2389,2389,on,
1,1.000,0,0,off,
1,2.000,48,0,failpoint,
2,0.000,2710,2710,on,
```

Each `trial_<n>/timeline.csv` holds the same rows for one trial, without the
`trial` column.

- `drawn_kbps` is what the distribution gave.
- `applied_kbps` is what was in effect.
- `state` is `on`, `off` (the draw was 0), `failpoint`, `delay` or `ambient`.
- `delay_ms` is the latency added, under `delay` (whose kbps columns are empty).
- Under `ambient`, the rows are iperf3's one-second intervals, and both kbps
  columns hold the measured throughput.

### power_\<profile\>_\<experiment\>.csv

The power timeline of each measured remote, for each trial, one row per
`interval` seconds. `t_sec` restarts at 0 for each trial:

```
trial,daemon,t_sec,power_w,input_w,battery_w,battery_status,samples
1,fermi,0.000,1.161,1.161,0.000,Full,11
1,fermi,1.000,1.407,1.407,0.000,Full,10
```

Each `trial_<n>/power_<daemon>.csv` holds the same rows for one trial and
daemon, without the `trial` and `daemon` columns.

- `power_w` is the board's power draw: `input_w + battery_w`.
- `input_w` is the power from the external supplies (USB, wireless).
- `battery_w` is positive while the battery discharges and negative while it
  charges. It is 0 when the battery is full.
- `battery_status` is the battery's status at the end of the row, e.g. `Full`,
  `Charging` or `Discharging`.
- `samples` is the number of readings the row averages.

## Trying it without the hardware

Add `"--dry-run"` to an experiment's `netsim.args` to have netsim print its `tc` commands
instead of running them. It then needs no root, and the runner won't ask for
sudo. The byte totals in the stats CSV are empty in this mode, but the timeline
and bandwidth statistics are filled in. `none` and `ambient` don't use netsim,
so `--dry-run` doesn't affect them: `ambient` still stops the vehicle and runs
iperf3.

## Troubleshooting

- **`daemon not ready after 300s: ...`**: the message says what was missing.
  - *remote not registered*: the remote didn't reconnect to the orchestrator.
  - *vehicle not listed*: `vehicle` names nothing the orchestrator lists.
  - *not running*: the vehicle is listed but stopped, and its error is shown.

  Check with `eagle list-services`.
- **Network still limited after a crash**: if the runner is killed hard
  (`kill -9`, power loss), netsim's `tc` rules stay in place. The next run
  removes them by itself before it starts. To remove them right away:

  ```bash
  sudo .venv/bin/netsim -i tailscale0 -p 9001 --cleanup
  ```

  Don't run this if something else on the machine uses a clsact qdisc on that
  interface.
- **`vehicle 'drone' is ambiguous`**: more than one daemon has a vehicle with
  that name. Use its full name (`<ip>.drone`), as shown by
  `eagle list-services`.
