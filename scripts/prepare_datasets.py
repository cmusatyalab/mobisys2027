"""Prepares the image datasets for the experiments, and pushes them to fermi.

Run on lambda2 (it has the room; this downloads ~13 GB of MMLA), from the
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
- mmla: the onager videos of MMLA The Wilds (imageomics/mmla_wilds on
  Hugging Face, session_3, class 2: onager), downloaded here, back to back.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2

from mobisys2027 import labels as L

MMLA_REPO = "imageomics/mmla_wilds"
MMLA_ONAGER_VIDEOS = ["P0070007", "P0080008", "P0090009", "P0100010", "P0110011", "P0120012", "P0160016"]
MMLA_CLASSES = {0: "zebra", 1: "giraffe", 2: "onager", 3: "dog"}


def flatten(images: list[Path], out: Path, max_width: int) -> None:
    """Copies `images`, scaled down to `max_width`, with their YOLO labels,
    into `out`."""
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
        cv2.imwrite(str(out / f"{image.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        label = L.yolo_label_file(image)
        if label is not None:
            shutil.copy(label, out / f"{image.stem}.txt")
        if k % 500 == 0:
            print(f"  {k}/{len(images)}")


def download_mmla(cache: Path) -> list[Path]:
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(
        MMLA_REPO, repo_type="dataset", local_dir=cache,
        allow_patterns=[f"session_3/{v}/*" for v in MMLA_ONAGER_VIDEOS] + ["classes.txt"],
    ))
    images = []
    for v in MMLA_ONAGER_VIDEOS:
        images += L.list_images(root / "session_3" / v)
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
        jobs.append(("mmla", download_mmla(args.mmla_cache), "onager", MMLA_CLASSES))

    for name, images, target, classes in jobs:
        out = args.out / name
        print(f"{name}: {len(images)} images -> {out}")
        flatten(images, out, args.max_width)
        labels = L.import_yolo(out, name=name, target=target, class_names=classes, fps=args.fps)
        path = labels.save()
        print(f"{name}: {path} ({labels.frames} frames, {len(labels.target_frames())} with a {target})")
        if args.push:
            dest = f"{args.push}:.local/share/steeleagle/files/"
            subprocess.run(["rsync", "-a", "--delete", str(out), dest], check=True)
            print(f"{name}: pushed to {dest}{name}")


if __name__ == "__main__":
    main()
