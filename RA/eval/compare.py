"""Paired per-sequence comparison of two runs (e.g. a tracker without RA vs with RA), tau selection on validation,
and the RA plug-in table of the paper.

Statistics (as in the paper): datasets weighted equally (mean of per-dataset means of per-sequence AUC, in points);
95% CI from 5000 per-sequence bootstrap resamples drawn within each dataset; two-sided p from 20000 paired sign-flips;
up/down = sequences whose AUC rises/falls by more than 0.01 points. Random draws use numpy default_rng(0).

  python -m eval.compare pair --base runs/hiptrack_none --ra runs/hiptrack_soft_t0.5
  python -m eval.compare select-tau --pattern 'runs_val/ostrack_soft_t{tau}'
  python -m eval.compare table --runs-root runs --spec configs/paper_plugin_table.json
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

DOMS = ('DDT', 'MAVVID', 'NF_UAVTrack', 'DUT_AntiUAV')
TAUS = ('0.3', '0.4', '0.5', '0.6', '0.7')


def load(run):
    """{(domain, sequence): success AUC in points} from the receipts of one run directory."""
    out = {}
    for f in glob.glob(str(Path(run) / '*/*/receipt.json')):
        x = json.load(open(f)); out[(x['domain'], x['sequence'])] = 100 * x['metrics']['success_auc']
    return out


def domain_order(keys):
    present = {k[0] for k in keys}
    return [d for d in DOMS if d in present] + sorted(present - set(DOMS))


def stats(A, B, keys, rng, nb=5000, nf=20000):
    doms = domain_order(keys)
    diff = {k: B[k] - A[k] for k in keys}; byd = {d: np.array([diff[k] for k in keys if k[0] == d]) for d in doms}
    dm = lambda v: np.mean([np.mean([v[k] for k in keys if k[0] == d]) for d in doms])
    g = np.mean([byd[d].mean() for d in doms])
    boots = [np.mean([rng.choice(byd[d], len(byd[d])).mean() for d in doms]) for _ in range(nb)]
    flips = [np.mean([(byd[d] * rng.choice([-1, 1], len(byd[d]))).mean() for d in doms]) for _ in range(nf)]
    return dict(n=len(keys), base=float(dm(A)), ra=float(dm(B)), gain=float(g), lo=float(np.percentile(boots, 2.5)),
                hi=float(np.percentile(boots, 97.5)), p=float(np.mean(np.abs(flips) >= abs(g) - 1e-12)),
                up=int(sum(v > 0.01 for v in diff.values())), down=int(sum(v < -0.01 for v in diff.values())),
                dom={d: float(byd[d].mean()) for d in doms})


def select_tau(macro_by_tau):
    """tau with the best validation dataset-equal AUC; ties within 0.01 AUC points go to the tau closest to 0.5."""
    best = max(macro_by_tau.values())
    return min((t for t in macro_by_tau if macro_by_tau[t] >= best - 1e-4), key=lambda t: abs(float(t) - 0.5))


def macro(run):
    return json.load(open(Path(run) / 'summary.json'))['domain_equal_macro']['success_auc']


def fmt_p(p): return '<0.001' if p < 0.001 else f'{p:.3f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('pair'); a1.add_argument('--base', required=True); a1.add_argument('--ra', required=True)
    a1.add_argument('--exclude-domains', default=''); a1.add_argument('--json')
    a2 = sub.add_parser('select-tau'); a2.add_argument('--pattern', required=True, help="run dir with '{tau}' placeholder")
    a2.add_argument('--taus', default=','.join(TAUS))
    a3 = sub.add_parser('table'); a3.add_argument('--runs-root', required=True); a3.add_argument('--spec', required=True)
    a = ap.parse_args()
    if a.cmd == 'pair':
        A, B = load(a.base), load(a.ra)
        assert set(A) == set(B) and A, f'runs cover different sequences ({len(A)} vs {len(B)})'
        keys = sorted(k for k in A if k[0] not in a.exclude_domains.split(','))
        st = stats(A, B, keys, np.random.default_rng(0))
        print(f"n={st['n']} base {st['base']:.2f} RA {st['ra']:.2f} gain {st['gain']:+.2f} 95% CI [{st['lo']:+.2f}, {st['hi']:+.2f}] "
              f"p={fmt_p(st['p'])} up/down {st['up']}/{st['down']} | " + ' '.join(f'{d} {g:+.2f}' for d, g in st['dom'].items()))
        if a.json: json.dump(st, open(a.json, 'w'), indent=1)
    elif a.cmd == 'select-tau':
        v = {t: macro(a.pattern.format(tau=t)) for t in a.taus.split(',')}
        print(json.dumps(v, indent=1)); print('selected tau', select_tau(v))
    else:
        # one generator shared by all rows, in spec order (as for the paper table)
        spec = json.load(open(a.spec)); rng = np.random.default_rng(0); R = Path(a.runs_root)
        for row in spec['rows']:
            A, B = load(R / row['base']), load(R / row['ra'].format(tau=row['tau']))
            if not A or set(A) != set(B):
                print(f"{row['label']}: missing runs"); continue
            st = stats(A, B, sorted(A), rng)
            print(f"{row['label']} & {row['tau']} & {st['base']:.2f} & {st['ra']:.2f} & {st['gain']:+.2f} & [{st['lo']:.2f}, {st['hi']:.2f}] & "
                  f"${fmt_p(st['p'])}$ & {st['up']}/{st['down']} & " + ' & '.join(f'{x:+.2f}' for x in st['dom'].values()) + ' \\\\')


if __name__ == '__main__':
    main()
