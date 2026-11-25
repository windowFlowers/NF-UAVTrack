"""Cache full-frame YOLO detections for every frame of every sequence in a manifest.

Output: <out>/<domain>__<sequence>.json = {"weights_sha256", "dir", "dets": {image_name: [[x, y, w, h, conf], ...]}}
with conf >= 0.05 (RA itself uses conf >= kappa0 = 0.25). Running the detector once and caching is equivalent to
running it on demand at the trigger frames; it only lets tracker variants share the detections.
The paper used YOLOv12-M (https://github.com/sunsmarterjie/yolov12, Ultralytics API), imgsz 640, FP16, batch 32.

  python tools/build_yolo_cache.py --weights best.pt --manifest manifests/test.json --out det_cache [--shard 0 --nshard 2]
"""
import argparse, hashlib, json, os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', type=Path, required=True); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--device', default='0')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    a = p.parse_args()
    from ultralytics import YOLO
    a.out.mkdir(parents=True, exist_ok=True)
    recs = json.loads(a.manifest.read_text(encoding='utf-8-sig'))['records']
    sha = hashlib.sha256(open(a.weights, 'rb').read()).hexdigest()
    model = YOLO(str(a.weights))
    for i, r in enumerate(recs):
        if i % a.nshard != a.shard: continue
        f = a.out / f"{r['domain']}__{r['sequence']}.json"
        if f.exists(): continue
        img = Path(r['root']) / 'img'
        names = sorted(x for x in os.listdir(img) if x.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        res = {}
        for k in range(0, len(names), 32):
            chunk = names[k:k + 32]
            for n, o in zip(chunk, model.predict([str(img / n) for n in chunk], imgsz=640, conf=0.05, device=a.device, verbose=False, half=True)):
                xyxy = o.boxes.xyxy.cpu().numpy().tolist(); cf = o.boxes.conf.cpu().numpy().tolist()
                res[n] = [[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2), round(s, 4)] for (x1, y1, x2, y2), s in zip(xyxy, cf)]
        tmp = f.with_suffix('.tmp'); json.dump(dict(weights_sha256=sha, dir=str(Path(r['root'])), dets=res), open(tmp, 'w')); tmp.rename(f)
        print(r['domain'], r['sequence'], len(names), flush=True)
    print('SHARD DONE', flush=True)


if __name__ == '__main__':
    main()
