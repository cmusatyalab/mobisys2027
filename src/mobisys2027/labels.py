"""Ground-truth boxes for the test videos and image datasets, and the
helpers that read, write and use them.

A label file, `labels/<name>.json` at the root of the repository, holds the
boxes of one source, which is a video (`parkinglot.mp4`) or a directory of
images (`sard`), frame by frame:

    {
      "source": "parkinglot.mp4",
      "kind": "video",                 # or "images"
      "fps": 30.0,                     # playback rate (the fake drone's fps for images)
      "frames": 2589,                  # number of frames
      "width": 1280, "height": 720,    # size of the frames (informational)
      "target": "mannequin",           # the class searched for
      "names": ["a.jpg", ...],         # images only: file of each frame, in playback order
      "boxes": {"123": [["mannequin", x0, y0, x1, y1], ["person", ...]], ...}
    }

Frame numbers are the fake drone's frame ids: decode order for a video, and
the natural order of the file names for images (see fake_drone's images.py),
which `natural_key` here mirrors. Box corners are normalized to [0, 1] of the
frame's width and height, so they hold at any resolution the frames are
played or labeled at. A frame missing from `boxes` has no boxes.

Image datasets come with labels in YOLO format (one `<class> <cx> <cy> <w>
<h>` line per box, normalized), which `import_yolo` converts:

- SARD: `images/*.jpg` with `labels/*.txt` beside the images directory;
- WiSARD: `*.jpeg` and `*.txt` side by side;
- MMLA (The Wilds): `*.jpg` and `*.txt` side by side, classes in classes.txt.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "labels"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})

# The class searched for in each source, and what else is labeled.
MANNEQUIN = "mannequin"
PERSON = "person"


def natural_key(name: str):
    """Sort key ordering the digit runs of `name` by value, as the fake
    drone orders the images it plays."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def list_images(directory: Path) -> list[Path]:
    """The images directly in `directory`, in the fake drone's playback order."""
    files = [p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: natural_key(p.name))


Box = tuple[str, float, float, float, float]  # label, x0, y0, x1, y1 (normalized)


@dataclass
class Labels:
    source: str
    kind: str
    fps: float
    frames: int
    target: str
    width: int = 0
    height: int = 0
    names: list[str] = field(default_factory=list)
    boxes: dict[int, list[Box]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The name the label file is saved under: the source without its
        extension."""
        return Path(self.source).stem

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def frame_boxes(self, frame: int, label: str | None = None) -> list[Box]:
        boxes = self.boxes.get(frame, [])
        return [b for b in boxes if label is None or b[0] == label]

    def has(self, frame: int, label: str | None = None) -> bool:
        """Whether frame `frame` has a box (of `label`, by default the target)."""
        return bool(self.frame_boxes(frame, label or self.target))

    def target_frames(self) -> list[int]:
        return sorted(f for f in self.boxes if self.has(f))

    def target_runs(self) -> list[tuple[int, int]]:
        """The runs of consecutive frames with the target, as (first, last)."""
        runs: list[tuple[int, int]] = []
        for f in self.target_frames():
            if runs and f == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], f)
            else:
                runs.append((f, f))
        return runs

    def to_json(self) -> dict:
        out = {
            "source": self.source,
            "kind": self.kind,
            "fps": self.fps,
            "frames": self.frames,
            "width": self.width,
            "height": self.height,
            "target": self.target,
        }
        if self.names:
            out["names"] = self.names
        out["boxes"] = {
            str(f): [[b[0], *(round(v, 5) for v in b[1:])] for b in boxes]
            for f, boxes in sorted(self.boxes.items())
            if boxes
        }
        return out

    @classmethod
    def from_json(cls, data: dict) -> "Labels":
        return cls(
            source=data["source"],
            kind=data["kind"],
            fps=float(data["fps"]),
            frames=int(data["frames"]),
            target=data.get("target", MANNEQUIN),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            names=list(data.get("names", [])),
            boxes={int(f): [(b[0], *map(float, b[1:5])) for b in boxes] for f, boxes in data.get("boxes", {}).items()},
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or label_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=None, separators=(",", ":")) + "\n")
        tmp.replace(path)
        return path


def label_path(name: str) -> Path:
    return LABELS_DIR / f"{Path(name).stem}.json"


def load(name: str) -> Labels:
    """The labels of the source `name` (e.g. "parkinglot", "parkinglot.mp4" or
    "sard"), from labels/."""
    path = Path(name) if str(name).endswith(".json") else label_path(name)
    return Labels.from_json(json.loads(path.read_text()))


def try_load(name: str) -> Labels | None:
    try:
        return load(name)
    except FileNotFoundError:
        return None


# ---- image datasets ---------------------------------------------------------


def yolo_label_file(image: Path) -> Path | None:
    """The YOLO label file of `image`: beside it (WiSARD, MMLA), or in a
    `labels` directory beside its `images` directory (SARD, Roboflow)."""
    for candidate in (image.with_suffix(".txt"), image.parent.parent / "labels" / f"{image.stem}.txt"):
        if candidate.is_file():
            return candidate
    return None


def read_yolo(path: Path | None, class_names: dict[int, str]) -> list[Box]:
    if path is None:
        return []
    boxes: list[Box] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:5])
        label = class_names.get(cls, str(cls))
        boxes.append(
            (label, max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2))
        )
    return boxes


def image_dir_of(dataset: Path) -> Path:
    """Where a dataset's images are: its `images` directory if it has one."""
    return dataset / "images" if (dataset / "images").is_dir() else dataset


def import_yolo(
    dataset: Path,
    *,
    name: str,
    target: str,
    class_names: dict[int, str] | None = None,
    fps: float,
) -> Labels:
    """The Labels of the YOLO-labeled image dataset `dataset` (see the module
    docstring for the layouts read). `class_names` maps class ids to names;
    by default, those of a classes.txt in or above the dataset, or else
    every class is `target`."""
    images = list_images(image_dir_of(dataset))
    if not images:
        raise SystemExit(f"no images in {image_dir_of(dataset)}")
    if class_names is None:
        class_names = read_classes(dataset) or {}
        if not class_names:
            class_names = _AllAre(target)
    boxes = {}
    for i, image in enumerate(images):
        found = read_yolo(yolo_label_file(image), class_names)
        if found:
            boxes[i] = found
    width, height = image_size(images[0])
    return Labels(
        source=name,
        kind="images",
        fps=fps,
        frames=len(images),
        target=target,
        width=width,
        height=height,
        names=[p.name for p in images],
        boxes=boxes,
    )


class _AllAre(dict):
    """Names every class id `label`."""

    def __init__(self, label: str):
        super().__init__()
        self.label = label

    def get(self, key, default=None):
        return self.label


def read_classes(dataset: Path) -> dict[int, str] | None:
    for d in (dataset, *dataset.parents):
        path = d / "classes.txt"
        if path.is_file():
            names = [n.strip() for n in path.read_text().splitlines() if n.strip()]
            return dict(enumerate(names))
    return None


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return 0, 0
    return img.shape[1], img.shape[0]


# ---- uses of the labels ---------------------------------------------------------


def failpoints(labels: Labels, pad: float = 1.0, offset: float = 0.0) -> list[list[float]]:
    """netsim failpoints, [start, length] in seconds, that cut the link while
    the target is in view: each run of target frames, padded by `pad` seconds
    each side, with overlapping ones merged. `offset` is added to each start,
    for a playback that starts late."""
    spans: list[list[float]] = []
    for first, last in labels.target_runs():
        start = max(0.0, first / labels.fps - pad) + offset
        end = (last + 1) / labels.fps + pad + offset
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return [[round(s, 1), round(e - s, 1)] for s, e in spans]


def overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection of the boxes `a` and `b` (x0, y0, x1, y1) over the smaller
    one's area: 1 when one holds the other."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0
