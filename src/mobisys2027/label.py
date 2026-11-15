"""Draw and fix the ground-truth boxes of a test video or image dataset.

    uv run label parkinglot                  # videos/parkinglot.mp4, labels/parkinglot.json
    uv run label path/to/street.mp4
    uv run label sard --images ~/Downloads/sard

    # without the video here: make a small proxy of it on lambda2 and copy it over
    uv run label traintrack --fetch root@lambda2:/root/mobisys2027/videos/traintrack.mp4

    # turn a dataset's YOLO labels into labels/<name>.json
    uv run label sard --import-yolo ~/Downloads/sard --target person --fps 10

The boxes are saved in labels/<name>.json (see mobisys2027/labels.py),
normalized to the frame, so a proxy at a lower resolution labels the full
video. Label the mannequin as "mannequin" and anyone else in view as
"person".

Mouse: drag to draw a box of the current class; click a box to select it.

Keys:
    d / right   next frame             a / left   previous frame
    l / j       forward / back 30      L / J      forward / back 300
    n / b       next / previous frame whose boxes differ from this one's
    1 / 2       class mannequin / person (and set the selected box to it)
    x / del     delete the selected box    X   delete every box on this frame
    backspace   delete the box drawn last on this frame (or the selected one)
    c           copy the previous frame's boxes onto this one
    v           copy this frame's boxes onto the next 30 frames
    i           interpolate every frame between the last frame you edited and this one
    u / ctrl+z  undo (and go to the frame it changes)
    U / ctrl+y  redo                    p    play / pause
    s           save                    q    save and quit      esc  unselect
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

from mobisys2027 import labels as L

WINDOW = "label"
MAX_VIEW_W, MAX_VIEW_H = 1280, 720
CACHE_WIDTH = 960
COLORS = {L.MANNEQUIN: (60, 220, 60), L.PERSON: (0, 160, 255)}
OTHER_COLOR = (200, 200, 200)
KEY_LEFT, KEY_RIGHT = (0xFF51, 65361, 2424832), (0xFF53, 65363, 2555904)
KEY_ESC = 27
KEY_BACKSPACE = (8, 0xFF08)
KEY_DELETE = (0xFFFF, 0xFF9F, 3014656)
KEY_UNDO = (ord("u"), 26)  # u, ctrl+z
KEY_REDO = (ord("U"), 25)  # U, ctrl+y
CLASSES = {ord("1"): L.MANNEQUIN, ord("2"): L.PERSON}
PROXY_WIDTH = 960


# ---- frame sources -------------------------------------------------------------


class VideoFrames:
    """Random access to a video's frames, decoded strictly in order (seeking
    can land on the wrong frame) and cached as downscaled JPEGs."""

    def __init__(self, path: Path):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise SystemExit(f"could not open video {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.estimate = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cache: list[bytes] = []
        self.exhausted = False

    def _decode_next(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.exhausted = True
            self.cap.release()
            return
        h, w = frame.shape[:2]
        if w > CACHE_WIDTH:
            frame = self.cv2.resize(frame, (CACHE_WIDTH, round(h * CACHE_WIDTH / w)), interpolation=self.cv2.INTER_AREA)
        ok, buf = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 85])
        self.cache.append(buf.tobytes())

    def get(self, i: int):
        while i >= len(self.cache) and not self.exhausted:
            self._decode_next()
        if i >= len(self.cache):
            return None
        import numpy as np

        return self.cv2.imdecode(np.frombuffer(self.cache[i], np.uint8), self.cv2.IMREAD_COLOR)

    def count(self) -> int:
        """The number of frames, decoding to the end if needed."""
        while not self.exhausted:
            self._decode_next()
        return len(self.cache)

    def known(self) -> int:
        return len(self.cache) if self.exhausted else max(self.estimate, len(self.cache))


class ImageFrames:
    def __init__(self, files: list[Path]):
        import cv2

        self.cv2 = cv2
        self.files = files

    def get(self, i: int):
        if not 0 <= i < len(self.files):
            return None
        img = self.cv2.imread(str(self.files[i]))
        if img is not None and img.shape[1] > CACHE_WIDTH:
            h, w = img.shape[:2]
            img = self.cv2.resize(img, (CACHE_WIDTH, round(h * CACHE_WIDTH / w)), interpolation=self.cv2.INTER_AREA)
        return img

    def count(self) -> int:
        return len(self.files)

    def known(self) -> int:
        return len(self.files)


def find_video(name: str) -> Path | None:
    """A video named `name` (with or without .mp4): as a path, or in videos/,
    videos/proxy/ or media/."""
    p = Path(name).expanduser()
    if p.is_file():
        return p
    stem = Path(name).stem
    for d in (L.REPO_ROOT / "videos", L.REPO_ROOT / "videos" / "proxy", L.REPO_ROOT / "media", Path.cwd()):
        for candidate in (d / name, d / f"{stem}.mp4"):
            if candidate.is_file():
                return candidate
    return None


def fetch_proxy(remote: str, name: str) -> Path:
    """Makes a small proxy of the video at `remote` (user@host:path) on its
    host with ffmpeg, keeping every frame, and copies it to videos/proxy/."""
    host, path = remote.split(":", 1)
    dest = L.REPO_ROOT / "videos" / "proxy" / f"{Path(name).stem}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"/tmp/label-proxy-{Path(path).stem}.mp4"
    cmd = (
        f"ffmpeg -y -loglevel error -i {path} -an -vf scale={PROXY_WIDTH}:-2 -vsync passthrough "
        f"-c:v libx264 -preset veryfast -crf 30 -bf 0 {tmp}"
    )
    print(f"making a {PROXY_WIDTH}px proxy of {remote} ...", file=sys.stderr)
    subprocess.run(["ssh", host, cmd], check=True)
    subprocess.run(["rsync", "-a", f"{host}:{tmp}", str(dest)], check=True)
    subprocess.run(["ssh", host, f"rm -f {tmp}"], check=False)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
    return dest


# ---- editing --------------------------------------------------------------------


def box_at(boxes, x, y):
    """Index of the smallest box holding the normalized point (x, y), or None."""
    hits = [(abs((b[3] - b[1]) * (b[4] - b[2])), i) for i, b in enumerate(boxes) if b[1] <= x <= b[3] and b[2] <= y <= b[4]]
    return min(hits)[1] if hits else None


def center(b):
    return ((b[1] + b[3]) / 2, (b[2] + b[4]) / 2)


def match_boxes(a, b):
    """Pairs the boxes of `a` and `b` of the same label, nearest centers
    first. Returns [(box of a, box of b)]."""
    pairs, used_a, used_b = [], set(), set()
    candidates = sorted(
        (((center(x)[0] - center(y)[0]) ** 2 + (center(x)[1] - center(y)[1]) ** 2, i, j)
         for i, x in enumerate(a) for j, y in enumerate(b) if x[0] == y[0]),
    )
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((a[i], b[j]))
    return pairs


def interpolate(labels: L.Labels, f0: int, f1: int) -> int:
    """Fills the frames strictly between f0 and f1 with the boxes of f0 and
    f1, matched by label and position, moved linearly. Returns how many
    frames it filled."""
    if f0 > f1:
        f0, f1 = f1, f0
    pairs = match_boxes(labels.frame_boxes(f0), labels.frame_boxes(f1))
    for f in range(f0 + 1, f1):
        t = (f - f0) / (f1 - f0)
        labels.boxes[f] = [(a[0], *(a[k] + (b[k] - a[k]) * t for k in range(1, 5))) for a, b in pairs]
    return max(0, f1 - f0 - 1)


class Editor:
    def __init__(self, labels: L.Labels, frames, label_file: Path):
        self.labels = labels
        self.frames = frames
        self.label_file = label_file
        self.i = 0
        self.cls = labels.target if labels.target in (L.MANNEQUIN, L.PERSON) else L.MANNEQUIN
        self.selected: int | None = None
        self.drag_start = None
        self.drag_now = None
        self.last_edited: int | None = None
        self.undo: list[list[tuple[int, list]]] = []
        self.redo: list[list[tuple[int, list]]] = []
        self.dirty = False
        self.playing = False
        self.view_scale = 1.0
        self.message = ""

    # --- state changes
    def remember(self, frames):
        self.undo.append(self.snapshot(frames))
        del self.undo[:-200]
        self.redo.clear()
        self.dirty = True

    def snapshot(self, frames):
        return [(f, copy.deepcopy(self.labels.boxes.get(f, []))) for f in frames]

    def restore(self, src, dst, verb):
        """Pops a change off `src`, saving the boxes it replaces onto `dst`."""
        if not src:
            self.message = f"nothing to {verb}"
            return
        change = src.pop()
        dst.append(self.snapshot([f for f, _ in change]))
        for f, old in change:
            self.set_boxes(f, old)
        frames = [f for f, _ in change]
        if frames and self.i not in frames:
            self.goto(frames[0])
        self.selected = None
        self.dirty = True
        span = f"frame {frames[0]}" if len(frames) == 1 else f"frames {frames[0]}-{frames[-1]}" if frames else "nothing"
        self.message = f"{verb}: {span}  ({len(self.undo)} undo, {len(self.redo)} redo left)"

    def set_boxes(self, f, boxes):
        if boxes:
            self.labels.boxes[f] = boxes
        else:
            self.labels.boxes.pop(f, None)

    def edited(self):
        self.last_edited = self.i

    # --- mouse
    def on_mouse(self, event, x, y, flags, _):
        import cv2

        nx, ny = x / self.view_w, y / self.view_h
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (nx, ny)
            self.drag_now = (nx, ny)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_now = (nx, ny)
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            (x0, y0), (x1, y1) = self.drag_start, (nx, ny)
            self.drag_start = self.drag_now = None
            boxes = list(self.labels.frame_boxes(self.i))
            if abs(x1 - x0) * self.view_w < 4 or abs(y1 - y0) * self.view_h < 4:
                self.selected = box_at(boxes, nx, ny)
                return
            box = (self.cls, max(0, min(x0, x1)), max(0, min(y0, y1)), min(1, max(x0, x1)), min(1, max(y0, y1)))
            self.remember([self.i])
            boxes.append(box)
            self.set_boxes(self.i, boxes)
            self.selected = len(boxes) - 1
            self.edited()

    # --- keys
    def on_key(self, key) -> bool:
        """Handles `key`; False to quit."""
        n = self.frames.known()
        ch = key & 0xFFFF if key not in (*KEY_LEFT, *KEY_RIGHT, *KEY_DELETE) else key
        boxes = list(self.labels.frame_boxes(self.i))
        if ch in (ord("d"),) or key in KEY_RIGHT:
            self.goto(self.i + 1)
        elif ch in (ord("a"),) or key in KEY_LEFT:
            self.goto(self.i - 1)
        elif ch == ord("l"):
            self.goto(self.i + 30)
        elif ch == ord("j"):
            self.goto(self.i - 30)
        elif ch == ord("L"):
            self.goto(self.i + 300)
        elif ch == ord("J"):
            self.goto(self.i - 300)
        elif ch in (ord("n"), ord("b")):
            step = 1 if ch == ord("n") else -1
            here = self.signature(self.i)
            f = self.i + step
            while 0 <= f < self.frames.known() and self.frames.get(f) is not None and self.signature(f) == here:
                f += step
            self.goto(f)
        elif ch in CLASSES:
            self.cls = CLASSES[ch]
            if self.selected is not None and self.selected < len(boxes):
                self.remember([self.i])
                b = boxes[self.selected]
                boxes[self.selected] = (self.cls, *b[1:])
                self.set_boxes(self.i, boxes)
                self.edited()
        elif ch in KEY_BACKSPACE:
            if boxes:
                k = self.selected if self.selected is not None and self.selected < len(boxes) else len(boxes) - 1
                self.remember([self.i])
                del boxes[k]
                self.set_boxes(self.i, boxes)
                self.selected = None
                self.message = "deleted a box"
                self.edited()
        elif ch == ord("x") or key in KEY_DELETE:
            if self.selected is not None and self.selected < len(boxes):
                self.remember([self.i])
                del boxes[self.selected]
                self.set_boxes(self.i, boxes)
                self.selected = None
                self.edited()
        elif ch == ord("X"):
            self.remember([self.i])
            self.set_boxes(self.i, [])
            self.selected = None
            self.edited()
        elif ch == ord("c"):
            if self.i > 0:
                self.remember([self.i])
                self.set_boxes(self.i, list(self.labels.frame_boxes(self.i - 1)))
                self.edited()
        elif ch == ord("v"):
            end = min(self.i + 31, n)
            self.remember(range(self.i + 1, end))
            for f in range(self.i + 1, end):
                self.set_boxes(f, list(boxes))
            self.message = f"copied to frames {self.i + 1}-{end - 1}"
        elif ch == ord("i"):
            if self.last_edited is None or self.last_edited == self.i:
                self.message = "edit a frame, then edit another and press i"
            else:
                lo, hi = sorted((self.last_edited, self.i))
                self.remember(range(lo + 1, hi))
                filled = interpolate(self.labels, lo, hi)
                self.message = f"interpolated {filled} frames between {lo} and {hi}"
                self.edited()
        elif ch in KEY_UNDO:
            self.restore(self.undo, self.redo, "undo")
        elif ch in KEY_REDO:
            self.restore(self.redo, self.undo, "redo")
        elif ch == ord("p"):
            self.playing = not self.playing
        elif ch == ord("s"):
            self.save()
        elif ch == ord("q"):
            self.save()
            return False
        elif ch == KEY_ESC:
            self.selected = None
        return True

    def signature(self, f):
        return sorted((b[0] for b in self.labels.frame_boxes(f)))

    def goto(self, f):
        total = self.frames.known()
        f = max(0, min(f, total - 1))
        if self.frames.get(f) is None:  # past the end of the video
            f = self.frames.count() - 1
        if f != self.i:
            self.selected = None
        self.i = f

    def save(self):
        # Saving finds the exact frame count, decoding the rest of a video
        self.labels.frames = self.frames.count()
        self.labels.save(self.label_file)
        self.dirty = False
        self.message = f"saved {self.label_file.relative_to(L.REPO_ROOT) if self.label_file.is_relative_to(L.REPO_ROOT) else self.label_file}"

    # --- drawing
    def render(self):
        import cv2

        img = self.frames.get(self.i)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(MAX_VIEW_W / w, MAX_VIEW_H / h, 2.0)
        img = cv2.resize(img, (round(w * scale), round(h * scale)))
        self.view_h, self.view_w = img.shape[:2]
        for k, b in enumerate(self.labels.frame_boxes(self.i)):
            color = COLORS.get(b[0], OTHER_COLOR)
            p0 = (round(b[1] * self.view_w), round(b[2] * self.view_h))
            p1 = (round(b[3] * self.view_w), round(b[4] * self.view_h))
            cv2.rectangle(img, p0, p1, color, 3 if k == self.selected else 1)
            cv2.putText(img, b[0], (p0[0], max(12, p0[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if self.drag_start and self.drag_now:
            p0 = (round(self.drag_start[0] * self.view_w), round(self.drag_start[1] * self.view_h))
            p1 = (round(self.drag_now[0] * self.view_w), round(self.drag_now[1] * self.view_h))
            cv2.rectangle(img, p0, p1, COLORS.get(self.cls, OTHER_COLOR), 1)
        total = self.frames.known()
        status = (
            f"{self.i}/{total - 1}  t={self.i / self.labels.fps:6.2f}s  class={self.cls}"
            f"{'  [unsaved]' if self.dirty else ''}{'  edited@' + str(self.last_edited) if self.last_edited is not None else ''}"
        )
        cv2.rectangle(img, (0, 0), (self.view_w, 44), (0, 0, 0), -1)
        cv2.putText(img, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, self.message, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
        # Timeline of the target along the bottom
        y = self.view_h - 6
        cv2.line(img, (0, y), (self.view_w, y), (60, 60, 60), 6)
        for first, last in self.labels.target_runs():
            x0, x1 = round(first / total * self.view_w), max(round((last + 1) / total * self.view_w), round(first / total * self.view_w) + 1)
            cv2.line(img, (x0, y), (x1, y), COLORS[L.MANNEQUIN] if self.labels.target == L.MANNEQUIN else (255, 200, 0), 6)
        xi = round(self.i / total * self.view_w)
        cv2.line(img, (xi, y - 8), (xi, y + 6), (255, 255, 255), 2)
        return img

    def run(self):
        import cv2

        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            img = self.render()
            if img is not None:
                cv2.imshow(WINDOW, img)
            key = cv2.waitKeyEx(int(1000 / self.labels.fps) if self.playing else 30)
            if self.playing:
                self.goto(self.i + 1)
            if key != -1 and not self.on_key(key):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                if self.dirty:
                    self.save()
                break
        cv2.destroyAllWindows()


# ---- entry point -----------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="video name or path (parkinglot, videos/street.mp4), or dataset name with --images")
    parser.add_argument("--images", type=Path, help="label the images in this directory (a dataset's root or images/ dir)")
    parser.add_argument("--labels", type=Path, help="label file (default labels/<name>.json)")
    parser.add_argument("--target", help="class searched for (default mannequin, or person for --import-yolo)")
    parser.add_argument("--fps", type=float, default=None, help="playback rate of an image dataset (default 10)")
    parser.add_argument("--fetch", metavar="HOST:PATH", help="make a small proxy of this remote video and label it")
    parser.add_argument("--import-yolo", type=Path, metavar="DIR",
                        help="write labels/<source>.json from this dataset's YOLO labels and exit")
    parser.add_argument("--class-name", action="append", default=[], metavar="ID=NAME",
                        help="with --import-yolo, name a class id (repeatable); default classes.txt, or the target")
    args = parser.parse_args(argv)

    name = Path(args.source).stem
    label_file = args.labels or L.label_path(name)

    if args.import_yolo:
        names = None
        if args.class_name:
            names = {int(k): v for k, v in (c.split("=", 1) for c in args.class_name)}
        labels = L.import_yolo(args.import_yolo.expanduser(), name=name, target=args.target or L.PERSON,
                               class_names=names, fps=args.fps or 10.0)
        path = labels.save(label_file)
        print(f"{path}: {labels.frames} images, {len(labels.target_frames())} with a {labels.target}")
        return

    existing = L.try_load(str(label_file)) if label_file.is_file() else None
    if args.images or (existing and existing.kind == "images"):
        directory = (args.images or Path.home() / "Downloads" / name).expanduser()
        files = L.list_images(L.image_dir_of(directory))
        if not files:
            raise SystemExit(f"no images in {L.image_dir_of(directory)}; give --images")
        frames = ImageFrames(files)
        labels = existing or L.Labels(source=name, kind="images", fps=args.fps or 10.0, frames=len(files),
                                      target=args.target or L.PERSON, names=[f.name for f in files])
        if labels.names and labels.names != [f.name for f in files]:
            raise SystemExit(f"{directory} doesn't hold the images {label_file} was made for")
    else:
        video = fetch_proxy(args.fetch, name) if args.fetch else find_video(args.source)
        if video is None:
            raise SystemExit(f"no video {args.source!r} in videos/, videos/proxy/ or media/; "
                             f"give its path, or --fetch user@host:/path/to/{name}.mp4")
        frames = VideoFrames(video)
        source = existing.source if existing else f"{name}.mp4"
        labels = existing or L.Labels(source=source, kind="video", fps=frames.fps, frames=frames.estimate,
                                      target=args.target or L.MANNEQUIN, width=frames.width, height=frames.height)
        print(f"labeling {video} -> {label_file}", file=sys.stderr)
    Editor(labels, frames, label_file).run()


if __name__ == "__main__":
    main()
