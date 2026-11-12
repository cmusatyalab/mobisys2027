# DDIL 2026 experiments

Experiment directories for `steeleagle/tools/experiment`. lambda2 is the
orchestrator (run the runner there); fermi runs the fake drone and the edge
engines. All are `auto = true`, 3 trials per profile, and each trial runs for
the length of its video.

```bash
cd steeleagle/tools/experiment
uv run experiment -r ../../../ddil2026          # everything
uv run experiment ../../../ddil2026/yolo        # one experiment
```

| Experiment | Video(s) | Profiles | Edge (fermi) → network → lambda2 | Results |
|---|---|---|---|---|
| `yolo` | parkinglot, street, traintrack | normal, limited, degraded, intermittent | detection-qnn (person, early discard) → detection-log; → JPEG → decoder. Degraded/intermittent send only the best person crop (detection-crop) | detection-log frames (`raw/`, `out/`) |
| `cliped_text` / `cliped_image` | parkinglot, street, traintrack | normal, limited, degraded, intermittent, malicious | rescale 720p → cliped → WebP → VLM verifier, `log_tiles` on both | cliped and verifier tiles |

The yolo flow can't change per profile, so it names a generic `reduce` stage,
and the profile's daemon config picks the plugin behind it
(`fermi_degraded.toml` applies to degraded and intermittent).

cliped's malicious failpoints (`netsim.failpoint_map`) cover the
person-visible frames in `steeleagle_plugins/engines/cliped/tools/labels/`,
taken at 30 fps and padded 1 s on each side:

| Video | TP frames | Failpoints `[start, length]` (s) |
|---|---|---|
| parkinglot | 1489–1930 | `[48.6, 16.8]` |
| street | 1203–1520, 2098–2421 | `[39.1, 12.6], [68.9, 12.8]` |
| traintrack | 666–858, 1863–2443 | `[21.2, 8.4], [61.1, 21.4]` |

## Files to add

- `yolo/`, `cliped_text/`, `cliped_image/`: `parkinglot.mp4`, `street.mp4`, `traintrack.mp4`
- `cliped_image/`: `prompt.jpeg`

## Before running

- lambda2 needs Ollama running for cliped's verifier (`qwen2.5vl:3b`, pulled at startup if missing).
- Results are collected from `/root/.local/share/steeleagle/files` on both hosts.
