"""Check compact intervals and directional margins against direct formulas."""
from pathlib import Path
import json
import numpy as np
from compact_certificates import compact_intervals
from tail_certificates import intervals, js, entropy, top_l, certificate


def main() -> None:
    rng = np.random.default_rng(20270912)
    tolerance = 1e-12
    max_interval_error = 0.0
    max_margin_error = 0.0
    certified = 0
    pools = 500
    for idx in range(pools):
        cnum = [2, 3, 5, 10, 77][idx % 5]
        heads = rng.dirichlet(np.ones(cnum) * .6, size=3)
        if idx % 17 == 0:
            heads[0] = 1 / cnum  # tied minima
        if idx % 19 == 0:
            heads[1] = np.eye(cnum)[0]  # zero-query-tail boundary
        folds = rng.integers(0, 3, size=30)
        labels = rng.integers(0, cnum, size=30)
        raw = rng.dirichlet(np.ones(cnum)*.7, size=30)
        rho = raw[np.arange(30), labels]
        anchored = rho[:, None] * raw
        anchored[np.arange(30), labels] += 1-rho
        masses = anchored[np.arange(30), labels]
        rel = rng.uniform(-.2, 1., size=30)
        qb = heads.mean(axis=0)
        concentration = 1-entropy(qb)/np.log2(cnum)
        lo, hi, bounds = intervals(heads[folds], labels, masses, rel, qb, concentration)
        clo, chi = compact_intervals(heads, folds, labels, masses, rel, qb, concentration)
        max_interval_error = max(max_interval_error, float(np.max(np.abs(clo-lo))), float(np.max(np.abs(chi-hi))))
        assert np.allclose(clo,lo,atol=tolerance,rtol=0) and np.allclose(chi,hi,atol=tolerance,rtol=0)
        nominal = .5*rel + .5*(concentration*qb[labels]+(1-concentration)*np.array([1-js(q,p) for q,p in zip(heads[folds], anchored)]))
        residual = np.array([js(q,p)-v.coarse for q,p,v in zip(heads[folds], anchored,bounds)])
        kappa=.5*(1-concentration)
        down=kappa*(np.array([v.tail_max for v in bounds])-residual)
        up=kappa*residual
        observed_pair=nominal[:,None]-nominal[None,:]-(down[:,None]+up[None,:])
        max_margin_error=max(max_margin_error,float(np.max(np.abs(observed_pair-(lo[:,None]-hi[None,:])))))
        assert max_margin_error < tolerance
        a_minus=top_l(clo,8)
        if certificate(clo,chi,a_minus)['certified']:
            certified += 1
            assert set(a_minus)==set(top_l(nominal,8))
    report={'kind':'compact_interval_checks',
            'seed':20270912,'pools':pools,'candidates_per_pool':30,
            'interval_comparisons':pools*30,'pair_margin_comparisons':pools*30*30,
            'compact_max_interval_error':max_interval_error,
            'directional_margin_max_error':max_margin_error,
            'certified_exact_set_shortcuts':certified,'all_checks_passed':True,
            'target_llm_calls':0,'tolerance':tolerance,
            'note':'Synthetic checks; neither coverage estimates nor wall-clock benchmark results.'}
    path=Path(__file__).resolve().parent/'runs/theory/compact_validation.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
