"""Prepares the image datasets for the experiments, and pushes them to fermi.

Run on lambda2 (it has the room; this downloads ~15 GB of MMLA), from the
root of the repository:

    uv run --with huggingface_hub python scripts/prepare_datasets.py \\
        --sard /path/to/sard --wisard /path/to/wisard --push root@fermi

For each dataset it writes data/<name>/: its images, flattened into one
directory and scaled down to at most --max-width pixels wide (JPEG), in the
order the fake drone plays them, with each image's YOLO label file beside it
(the fake drone skips them). Then labels/<name>.json, the ground truth the
analysis scores against (`uv run label <name>` to look at it). With --push,
data/<name>/ is copied to fermi's files directory, where the experiments'
fake drone plays it from (`images = "<name>"`).

- sard: a Roboflow export, images/ and labels/ (class 0: person);
- wisard: images and labels side by side (class 0: person);
- mmla: clips of MMLA The Wilds (imageomics/mmla_wilds on Hugging Face),
  downloaded here, spliced as MMLA_CLIPS: four 50-frame African painted
  dog clips (class 3, the target) spread through zebra and giraffe footage,
  so 5% of the frames hold a dog and the rest are negatives. Each
  image is prefixed with its place in the splice, so the fake drone plays
  them in that order.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2

from mobisys2027 import labels as L

MMLA_REPO = "imageomics/mmla_wilds"
# The dog video's frames are in zips (32 GB); partition_3 (frames 11000-13500,
# 5.9 GB) has every dog clip below
MMLA_ZIPS = {"session_1/DJI_0035/partition_3": "session_1/DJI_0035/partition_3.zip"}
# (video, first frame, frames or None for the rest), played in this order:
# 200 dog frames (5%) among 3800 of zebra and giraffe. The pack thins out over
# the dog video: the clips go from ~8 dogs a frame to ~2.
MMLA_CLIPS = [
    ("session_2/P0140018", 0, 1000),  # giraffe
    ("session_1/DJI_0035/partition_3", 100, 50),  # dogs (~8)
    ("session_4/P0070010", 0, None),  # zebra (506)
    ("session_1/DJI_0035/partition_3", 800, 50),  # dogs (~6)
    ("session_2/P0140018", 1000, 1000),  # giraffe
    ("session_1/DJI_0035/partition_3", 1500, 50),  # dogs (~3)
    ("session_2/P0150019", 0, None),  # giraffe (327)
    ("session_1/DJI_0035/partition_3", 2300, 50),  # dogs (~2)
    ("session_2/P0140018", 2000, 967),  # giraffe
]
MMLA_CLASSES = {0: "zebra", 1: "giraffe", 2: "onager", 3: "dog"}


def flatten(images: list[Path], out: Path, max_width: int, numbered: bool = False) -> None:
    """Copies `images`, scaled down to `max_width`, with their YOLO labels,
    into `out`; `numbered`, each prefixed with its index, to keep their order."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for k, image in enumerate(images):
        img = cv2.imread(str(image))
        if img is None:
            print(f"  skipping unreadable {image}")
            continue
        h, w = img.shape[:2]
        if w > max_width:
            img = cv2.resize(img, (max_width, round(h * max_width / w / 2) * 2), interpolation=cv2.INTER_AREA)
        stem = f"{k:05d}_{image.stem}" if numbered else image.stem
        cv2.imwrite(str(out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        label = L.yolo_label_file(image)
        if label is not None:
            shutil.copy(label, out / f"{stem}.txt")
        if k % 500 == 0:
            print(f"  {k}/{len(images)}")


def download_mmla(cache: Path) -> list[Path]:
    import zipfile

    from huggingface_hub import hf_hub_download, snapshot_download

    videos = dict.fromkeys(v for v, _, _ in MMLA_CLIPS)
    root = Path(snapshot_download(
        MMLA_REPO, repo_type="dataset", local_dir=cache,
        allow_patterns=[f"{v}/*" for v in videos if v not in MMLA_ZIPS] + ["classes.txt"],
    ))
    for v in videos:
        if v in MMLA_ZIPS and not ((root / v).is_dir() and L.list_images(root / v)):
            print(f"  downloading {MMLA_ZIPS[v]}")
            z = Path(hf_hub_download(MMLA_REPO, MMLA_ZIPS[v], repo_type="dataset", local_dir=cache))
            with zipfile.ZipFile(z) as f:
                f.extractall((root / v).parent)  # the zip holds <partition>/
            z.unlink()
    images = []
    for v, first, n in MMLA_CLIPS:
        frames = L.list_images(root / v)
        clip = frames[first:None if n is None else first + n]
        if n is not None and len(clip) != n:
            raise SystemExit(f"{v} has {len(frames)} frames, not the {first + n} MMLA_CLIPS needs")
        images += clip
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sard", type=Path, help="the SARD sample (images/, labels/)")
    parser.add_argument("--wisard", type=Path, help="the WiSARD sample (images and labels together)")
    parser.add_argument("--no-mmla", action="store_true", help="skip MMLA")
    parser.add_argument("--mmla-cache", type=Path, default=Path("data/mmla_wilds"), help="where MMLA is downloaded")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--fps", type=float, default=10.0, help="playback rate recorded in the labels (lab.IMAGE_FPS)")
    parser.add_argument("--push", metavar="USER@HOST", help="rsync each dataset to this host's steeleagle files directory")
    args = parser.parse_args()

    jobs = []
    if args.sard:
        jobs.append(("sard", L.list_images(L.image_dir_of(args.sard.expanduser())), "person", {0: "person"}))
    if args.wisard:
        jobs.append(("wisard", L.list_images(args.wisard.expanduser()), "person", {0: "person"}))
    if not args.no_mmla:
        jobs.append(("mmla", download_mmla(args.mmla_cache), "dog", MMLA_CLASSES))

    for name, images, target, classes in jobs:
        out = args.out / name
        print(f"{name}: {len(images)} images -> {out}")
        flatten(images, out, args.max_width, numbered=name == "mmla")
        labels = L.import_yolo(out, name=name, target=target, class_names=classes, fps=args.fps)
        path = labels.save()
        print(f"{name}: {path} ({labels.frames} frames, {len(labels.target_frames())} with a {target})")
        if args.push:
            dest = f"{args.push}:.local/share/steeleagle/files/"
            subprocess.run(["rsync", "-a", "--delete", str(out), dest], check=True)
            print(f"{name}: pushed to {dest}{name}")


if __name__ == "__main__":
    main()
