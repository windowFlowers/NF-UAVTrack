"""Write the sequence manifest read by run_ra.py.

Views layout: <views>/<domain>/<sequence>/{img/, groundtruth.txt}. Detection caches (tools/build_yolo_cache.py):
<cache_dir>/<domain>__<sequence>.json. A split file lists "<domain> <sequence>" per line (configs/splits/*.txt);
without one every sequence found under --views is listed.

  python tools/make_manifest.py --views views --cache-dir det_cache --split configs/splits/test.txt --out manifests/test.json
"""
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--views', type=Path, required=True); p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--split', type=Path); p.add_argument('--domains', help='comma list, keep only these datasets')
    p.add_argument('--allow-missing', action='store_true', help='skip sequences whose view is missing instead of failing')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.split:
        pairs = [tuple(l.split()[:2]) for l in a.split.read_text().splitlines() if l.strip() and not l.startswith('#')]
    else:
        pairs = [(d.name, q.name) for d in sorted(a.views.iterdir()) if d.is_dir() for q in sorted(d.iterdir())
                 if (q / 'groundtruth.txt').exists()]
    if a.domains: pairs = [x for x in pairs if x[0] in a.domains.split(',')]
    recs = []
    for d, q in pairs:
        root = (a.views / d / q).resolve(); cache = (a.cache_dir / f'{d}__{q}.json').resolve()
        if not (root / 'groundtruth.txt').exists():
            if a.allow_missing: print('missing view, skipped:', d, q); continue
            raise FileNotFoundError(root)
        recs.append(dict(domain=d, sequence=q, root=str(root), cache=str(cache)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(records=recs), open(a.out, 'w'), indent=1); print(len(recs), 'sequences ->', a.out)


if __name__ == '__main__':
    main()
