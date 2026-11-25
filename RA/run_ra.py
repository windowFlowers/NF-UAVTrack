"""Run a host tracker with or without Recovery Augmentation (RA) on a list of sequences (one-pass evaluation).

  python run_ra.py --tracker hiptrack --repo /path/to/HIPTrack --checkpoint /path/to/hiptrack.pth.tar \
      --manifest manifests/test.json --variant soft --tau 0.5 --output runs/hiptrack_soft_t0.5

``--variant none`` runs the unchanged tracker (baseline); ``soft`` is RA as in the paper. The tracker is initialised
with the ground-truth box on the first frame that has one; no ground truth is used afterwards. Per sequence the run
writes predictions.txt (x,y,w,h,present), ra_log.npy (p1, score, triggered, accepted, re-searches, candidates,
detections) and receipt.json (metrics); summary.json holds per-dataset and dataset-equal (macro) means.
"""
import argparse, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np, cv2, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ra import RA
from ra.recovery import VARIANTS
from adapters import ADAPTERS, build_adapter
from eval.eval_uav_sot import compute_tracking_metrics, load_groundtruth_file
from eval.label_normalization import frame_path_sort_key

METRICS = ('success_auc', 'precision_20', 'normalized_precision', 'recall_iou_0.3', 'recall_iou_0.5', 'drift_rate')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest(path, det_cache_dir=None):
    """Manifest: {"records": [{"domain", "sequence", "root", "cache"}]}; root holds img/ and groundtruth.txt,
    cache is the detection JSON for that sequence (relative paths are relative to the manifest file)."""
    path = Path(path).resolve(); m = json.loads(path.read_text(encoding='utf-8-sig'))
    out = []
    for r in m['records']:
        root = Path(r['root']); root = root if root.is_absolute() else path.parent / root
        if det_cache_dir is not None:
            cache = Path(det_cache_dir) / f"{r['domain']}__{r['sequence']}.json"
        else:
            cache = Path(r['cache']); cache = cache if cache.is_absolute() else path.parent / cache
        out.append(dict(domain=r['domain'], sequence=r['sequence'], root=root.resolve(), cache=Path(cache).resolve()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tracker', choices=tuple(ADAPTERS), required=True)
    p.add_argument('--repo', type=Path, required=True, help='clone of the official tracker repository')
    p.add_argument('--checkpoint', type=Path, required=True, help='tracker weights (absolute, or relative to --repo)')
    p.add_argument('--tracker-config', help='tracker YAML (default: the one used in the paper, relative to --repo)')
    p.add_argument('--manifest', type=Path, required=True, help='sequence manifest, see tools/make_manifest.py')
    p.add_argument('--det-cache-dir', type=Path, help='override: read <dir>/<domain>__<sequence>.json for every sequence')
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--trigger', choices=('conf', 'periodic'), default='conf'); p.add_argument('--period', type=int, default=15)
    p.add_argument('--kappa1', type=float, default=0.5)
    p.add_argument('--tau', type=float, default=0.5); p.add_argument('--accept', type=float)
    p.add_argument('--conf-min', type=float, default=0.25); p.add_argument('--topk', type=int, default=3)
    p.add_argument('--gate-frames', type=int, default=30)
    p.add_argument('--max-frames', type=int); p.add_argument('--limit-seqs', type=int); p.add_argument('--seqs')
    p.add_argument('--cudnn-tf32', action='store_true', default=os.environ.get('RA_TF32') == '1',
                   help='allow cuDNN TF32 (PyTorch default); off by default. RA_TF32=1 sets it too.')
    p.add_argument('--torch-hub-dir', type=Path, help='torch.hub cache directory (some trackers fetch backbone weights)')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.accept is None: a.accept = a.tau
    a.output = a.output.resolve(); a.repo = a.repo.resolve(); a.manifest = a.manifest.resolve()
    if not a.checkpoint.is_absolute() and (Path.cwd() / a.checkpoint).exists(): a.checkpoint = (Path.cwd() / a.checkpoint).resolve()
    assert not a.output.exists(), f'{a.output} exists'; a.output.mkdir(parents=True)
    recs = load_manifest(a.manifest, a.det_cache_dir.resolve() if a.det_cache_dir else None)[:a.limit_seqs]
    if a.seqs: recs = [r for r in recs if r['sequence'] in a.seqs.split(',')]
    if a.torch_hub_dir: torch.hub.set_dir(str(a.torch_hub_dir.resolve()))
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = a.cudnn_tf32; torch.backends.cudnn.benchmark = False
    ad = build_adapter(a.tracker, a.repo, a.checkpoint, a.tracker_config)
    json.dump(dict({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, checkpoint=ad.checkpoint,
                   checkpoint_sha256=sha(ad.checkpoint), started=time.time(),
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), open(a.output / 'run.json', 'w'), indent=1)
    ra = RA(ad, a.variant, a.tau, a.accept, a.conf_min, a.topk, a.gate_frames, trigger=a.trigger, period=a.period, kappa1=a.kappa1)
    results = []
    for rec in recs:
        paths = sorted((x for x in (rec['root'] / 'img').iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')), key=frame_path_sort_key)
        gt = load_groundtruth_file(rec['root'] / 'groundtruth.txt'); assert len(gt) == len(paths), rec['root']
        dets = json.load(open(rec['cache']))['dets']
        missing = sum(x.name not in dets for x in paths)
        valid = np.isfinite(gt).all(1) & (gt[:, 2:] > 0).all(1); first = int(np.flatnonzero(valid)[0])
        count = min(len(paths), first + 1 + a.max_frames) if a.max_frames else len(paths)
        pred = np.full((count, 5), np.nan, dtype=np.float32); pred[:, 4] = 0
        log = np.full((count, 7), np.nan, dtype=np.float32)   # p1, score, trig, used, tried, ncand, ndet
        t0 = time.perf_counter()
        for i, path in enumerate(paths[:count]):
            if i < first: continue
            image = cv2.imread(str(path))
            if ad.color == 'RGB': image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            with torch.inference_mode():
                if i == first:
                    ra.init(image, gt[i].astype(float).tolist(), rec['sequence']); pred[i] = [*gt[i], 1.]
                else:
                    b, info = ra.track(image, dets.get(path.name, []))
                    b = np.asarray(b, dtype=np.float32); assert b.shape == (4,) and np.isfinite(b).all()
                    pred[i] = [*b, 1.]; log[i] = [info[k] for k in ('p1', 'score', 'trig', 'used', 'tried', 'ncand', 'ndet')]
        torch.cuda.synchronize(); wall = time.perf_counter() - t0
        seqout = a.output / rec['domain'] / rec['sequence']; seqout.mkdir(parents=True)
        np.savetxt(seqout / 'predictions.txt', pred, delimiter=',', fmt='%.9g'); np.save(seqout / 'ra_log.npy', log)
        met = compute_tracking_metrics(pred, gt[:count])
        r = dict(domain=rec['domain'], sequence=rec['sequence'], frames=count, init_sequence_index=first,
                 det_frames_missing=missing, metrics={k: float(met[k]) for k in METRICS}, wall_seconds=wall,
                 triggered=int(np.nansum(log[:, 2])), accepted=int(np.nansum(log[:, 3])), researches=int(np.nansum(log[:, 4])))
        json.dump(r, open(seqout / 'receipt.json', 'w'), indent=1); results.append(r)
        print(json.dumps(dict(seq=rec['domain'] + '/' + rec['sequence'], auc=r['metrics']['success_auc'], trig=r['triggered'],
                              acc=r['accepted'], fps=round(count / wall, 2))), flush=True)
    doms = {d: {k: float(np.mean([r['metrics'][k] for r in results if r['domain'] == d])) for k in METRICS}
            for d in dict.fromkeys(r['domain'] for r in results)}
    macro = {k: float(np.mean([m[k] for m in doms.values()])) for k in METRICS}
    json.dump(dict(status='completed', args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                   results=results, domain_metrics=doms, domain_equal_macro=macro), open(a.output / 'summary.json', 'w'), indent=1)
    print(json.dumps(dict(status='completed', macro_auc=macro['success_auc'], domains={d: m['success_auc'] for d, m in doms.items()})), flush=True)


if __name__ == '__main__':
    main()
