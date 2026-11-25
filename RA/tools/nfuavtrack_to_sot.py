"""Convert NF-UAVTrack (Hugging Face layout: images/<video>/frame-XXXXXX.jpg, labels/<video>/frame-XXXXXX.txt with
normalised YOLO boxes; an empty label file = target absent) into the single-object-tracking view used by run_ra.py:

  <out>/NF_UAVTrack/<video>/img/000001.jpg ...   (symlinks by default, --copy to copy)
  <out>/NF_UAVTrack/<video>/groundtruth.txt      (x,y,w,h in pixels per frame; nan,nan,nan,nan = absent)

The conversion (clip to the image, 6-decimal formatting) is the one used for the paper.

  python tools/nfuavtrack_to_sot.py --src /path/to/NF-UAVTrack --out views --split /path/to/NF-UAVTrack-repo/splits/test.txt
"""
import argparse, math, os, shutil
from pathlib import Path
import cv2

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_yolo_label(path, width, height):
    lines = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not lines:
        return [math.nan, math.nan, math.nan, math.nan]
    if len(lines) > 1:
        raise ValueError(f'{path} contains {len(lines)} labels; expected at most one')
    parts = lines[0].split()
    if len(parts) != 5:
        raise ValueError(f'{path} must contain YOLO cls xc yc w h')
    _, x_center, y_center, box_w, box_h = [float(part) for part in parts]
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
        raise ValueError(f'{path} has out-of-range normalized box')
    abs_w = box_w * width
    abs_h = box_h * height
    abs_x = (x_center - box_w / 2.0) * width
    abs_y = (y_center - box_h / 2.0) * height
    x1 = max(0.0, min(float(width), abs_x))
    y1 = max(0.0, min(float(height), abs_y))
    x2 = max(0.0, min(float(width), abs_x + abs_w))
    y2 = max(0.0, min(float(height), abs_y + abs_h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return [math.nan, math.nan, math.nan, math.nan]
    return [x1, y1, clipped_w, clipped_h]


def valid_box(row):
    return all(math.isfinite(value) for value in row) and row[2] > 0 and row[3] > 0


def format_box(row):
    if not valid_box(row):
        return 'nan,nan,nan,nan'
    return ','.join(f'{value:.6f}'.rstrip('0').rstrip('.') for value in row)


def convert_sequence(src, out, seq, copy=False):
    image_dir, label_dir = src / 'images' / seq, src / 'labels' / seq
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = sorted(label_dir.glob('*.txt'))
    if len(image_paths) != len(label_paths):
        raise ValueError(f'{seq}: image/label count mismatch: {len(image_paths)} vs {len(label_paths)}')
    out_img = out / seq / 'img'; out_img.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        label_path = label_dir / f'{image_path.stem}.txt'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'could not read image: {image_path}')
        height, width = image.shape[:2]
        rows.append(read_yolo_label(label_path, width, height))
        dst = out_img / f'{index:06d}{image_path.suffix.lower()}'
        if dst.exists() or dst.is_symlink(): dst.unlink()
        if copy: shutil.copy2(image_path, dst)
        else: os.symlink(image_path.resolve(), dst)
    (out / seq / 'groundtruth.txt').write_text('\n'.join(format_box(r) for r in rows) + '\n', encoding='utf-8')
    return len(rows), sum(valid_box(r) for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, required=True, help='NF-UAVTrack root containing images/ and labels/')
    p.add_argument('--out', type=Path, required=True, help='views root; sequences go to <out>/NF_UAVTrack/<video>')
    p.add_argument('--split', type=Path, help='text file with one video name per line (default: all videos)')
    p.add_argument('--copy', action='store_true', help='copy frames instead of symlinking them')
    a = p.parse_args()
    seqs = [l.split()[-1] for l in a.split.read_text().splitlines() if l.strip()] if a.split else \
        sorted(d.name for d in (a.src / 'images').iterdir() if d.is_dir())
    out = a.out / 'NF_UAVTrack'
    for s in seqs:
        n, v = convert_sequence(a.src, out, s, a.copy); print(s, n, 'frames', v, 'with box', flush=True)


if __name__ == '__main__':
    main()
