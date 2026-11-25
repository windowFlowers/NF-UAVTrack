"""Anti-UAV (Jiang et al., T-MM 2021; https://github.com/ucas-vg/Anti-UAV) visible stream -> SOT views used for the
external-benchmark experiment: <out>/AntiUAV_RGB/<seq>/{img/000001.jpg.., groundtruth.txt}. Absent frames (exist=0 or
empty rect) -> nan,nan,nan,nan. Known visible-start corrections from the Anti-UAV repository are applied.

  python tools/prep_antiuav_rgb.py --src /path/to/Anti-UAV-RGBT --split test --out views_antiuav_test [--shard 0 --nshard 4]
"""
import argparse, json, cv2
from pathlib import Path

FIX = {'20190926_134054_1_1': 150, '20190925_194211_1_3': 150, '20190925_194211_1_4': 1}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, required=True, help='Anti-UAV root with <split>/<seq>/visible.mp4 and visible.json')
    p.add_argument('--split', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    a = p.parse_args()
    seqs = sorted(q for q in (a.src / a.split).iterdir() if (q / 'visible.mp4').exists())
    for i, q in enumerate(seqs):
        if i % a.nshard != a.shard: continue
        o = a.out / 'AntiUAV_RGB' / q.name
        if (o / 'groundtruth.txt').exists(): continue
        (o / 'img').mkdir(parents=True, exist_ok=True)
        js = json.load(open(q / 'visible.json')); ex, rc = js['exist'], js['gt_rect']
        cap = cv2.VideoCapture(str(q / 'visible.mp4')); n = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            n += 1; cv2.imwrite(str(o / 'img' / f'{n:06d}.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        m = min(n, len(ex)); lines = []
        for k in range(m):
            r = rc[k] if k < len(rc) else []
            ok = ex[k] == 1 and isinstance(r, list) and len(r) == 4 and r[2] > 0 and r[3] > 0 and k >= FIX.get(q.name, 0)
            lines.append(','.join(f'{v:g}' for v in r) if ok else 'nan,nan,nan,nan')
        for f in sorted((o / 'img').iterdir())[m:]: f.unlink()
        (o / 'groundtruth.txt').write_text('\n'.join(lines) + '\n')
        print(a.split, q.name, n, len(ex), sum(l[0] != 'n' for l in lines), flush=True)
    print('DONE', a.split, a.shard, flush=True)


if __name__ == '__main__':
    main()
