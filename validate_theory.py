"""Check the sharp bounds, selection certificates, and constructive witnesses."""
from pathlib import Path
from itertools import product, combinations
import csv, json, platform, time
import numpy as np
from tail_certificates import entropy,js,tail_bound,intervals,certificate,top_l,rank_ranges
ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20270911)
start = time.perf_counter()
TOL = 1e-12
report = dict(kind='synthetic_theorem_checks',seed=20270911,
              target_llm_calls=0,numerics='numpy float64',tolerance=TOL,
              python=platform.python_version(),numpy=np.__version__)
max_decomp=max_vertex=max_min=0.
violations=0; vertices=0
for c in (2,3,5,10,77):
    for t in range(500):
        q=rng.dirichlet(np.full(c,.3 if t%2 else 2.))
        y=int(rng.integers(c)); b=float(rng.uniform()); bound=tail_bound(q,y,b)
        a=q[y]; idx=np.flatnonzero(np.arange(c)!=y); r=q[idx]/(1-a);r=r/r.sum()
        vals=[]
        for j in idx:
            p=np.zeros(c);p[y]=b;p[j]=1-b
            vals.append(js(q,p)-bound.coarse)
        vertices+=len(idx)
        max_vertex=max(max_vertex,abs(max(vals)-bound.tail_max))
        max_min=max(max_min,abs(js(q,bound.upper_posterior)-bound.coarse))
        w=(2-a-b)/2; alpha=(1-a)/(2-a-b)
        for _ in range(5):
            s=rng.dirichlet(np.full(c-1,.6));p=np.zeros(c);p[y]=b;p[idx]=(1-b)*s
            residual=js(q,p)-bound.coarse
            max_decomp=max(max_decomp,abs(residual-w*js(r,s,alpha)))
            violations += int(residual < -TOL or residual > bound.tail_max+TOL)
report['pair_checks']=dict(pairs=2500,random_tails=12500,vertices=vertices,
    max_decomposition_error=max_decomp,max_upper_endpoint_error=max_vertex,
    max_lower_endpoint_error=max_min,interval_violations=violations)

certified=uncertified=witness_failures=cert_violations=tail_worlds=0
for c in (3,5,10,77):
    for t in range(250):
        k,L=30,8
        folds=rng.dirichlet(np.full(c,.1 if t%2 else 1.),size=3)
        qb=folds.mean(0);con=1-entropy(qb)/np.log2(c)
        if t%4==0: con=.99
        elif t%4==1: con=0.
        # Fixed concentration overrides are deliberate general-score probes.
        q=folds[np.arange(k)%3]; y=rng.integers(c,size=k)
        pp=rng.dirichlet(np.full(c,.7),size=k);rho=pp[np.arange(k),y]
        p=rho[:,None]*pp;p[np.arange(k),y]+=1-rho
        b=p[np.arange(k),y];rel=rng.uniform(-.2,1,size=k)
        lo,hi,bds=intervals(q,y,b,rel,qb,con)
        def score(posteriors):
            return .5*rel+.5*(con*qb[y]+(1-con)*np.array([1-js(qi,pi) for qi,pi in zip(q,posteriors)]))
        nominal=score(p); selected=top_l(nominal,L); original=set(selected.tolist())
        cert=certificate(lo,hi,selected)
        if cert['certified']:
            certified+=1
            for _ in range(20):
                pert=np.zeros_like(p)
                for i in range(k):
                    idx=np.flatnonzero(np.arange(c)!=y[i]);pert[i,y[i]]=b[i]
                    pert[i,idx]=(1-b[i])*rng.dirichlet(np.full(c-1,.5))
                cert_violations+=int(set(top_l(score(pert),L))!=original);tail_worlds+=1
        else:
            uncertified+=1;i,j=cert['pair'];pert=p.copy()
            pert[i]=bds[i].lower_posterior;pert[j]=bds[j].upper_posterior
            witness_failures+=int(set(top_l(score(pert),L))==original)
report['posterior_pool_checks']=dict(pools=1000,K=30,L=8,certified=certified,
    uncertified=uncertified,certified_random_worlds=tail_worlds,
    certified_violations=cert_violations,constructive_witness_failures=witness_failures)

rank_errors=certificate_errors=0;max_minimax_error=0.
for _ in range(200):
    k,L=6,2;lo=rng.uniform(-1,1,k);hi=lo+rng.uniform(0,.5,k);pi=rng.permutation(k)
    worlds=np.array([np.where(bits,hi,lo) for bits in product((0,1),repeat=k)])
    order=np.array([top_l(w,k,pi) for w in worlds]);ranks=np.argsort(order,axis=1)+1
    best,worst=rank_ranges(lo,hi,pi)
    rank_errors += int(not(np.array_equal(best,ranks.min(0)) and np.array_equal(worst,ranks.max(0))))
    allsets=list(combinations(range(k),L))
    candidate=top_l(lo,L,pi)
    brute=max(min(float(w[list(a)].sum()) for w in worlds) for a in allsets)
    max_minimax_error=max(max_minimax_error,abs(float(lo[candidate].sum())-brute))
    for a in allsets:
        observed=all(set(o[:L])==set(a) for o in order)
        certificate_errors+=int(certificate(lo,hi,a,pi)['certified']!=observed)
report['exhaustive_interval_checks']=dict(pools=200,K=6,L=2,endpoint_worlds=12800,
    set_certificates=3000,rank_range_mismatches=rank_errors,
    certificate_mismatches=certificate_errors,max_minimax_error=max_minimax_error)

ties=[([1,1,0],[1,1,0],[0],True),([1,1,0],[1,1,0],[1],False),
      ([.5,.2],[.5,.5],[0],True),([.5,.2],[.5,.5],[1],False)]
for lo,hi,a,expected in ties:
    assert certificate(lo,hi,a)['certified']==expected
for q,y,b in [([1,0,0],0,.5),([.2,.3,.5],1,1),([1,0],0,1),([0,1],0,0),([.3,.7],0,.4)]:
    assert tail_bound(q,y,b).tail_max==0.
report['boundary_checks']=dict(tie_cases=4,zero_tail_or_binary_cases=5,failures=0)
assert max(max_decomp,max_vertex,max_min,max_minimax_error)<TOL
assert violations+cert_violations+witness_failures+rank_errors+certificate_errors==0
report['all_checks_passed']=True
report['elapsed_seconds']=time.perf_counter()-start
(ROOT/'runs/theory').mkdir(parents=True,exist_ok=True)
(ROOT/'runs/theory/theory_validation.json').write_text(json.dumps(report,indent=2)+'\n')
q=np.array([.2,.56,.20,.04]); b=.8;r=q[1:]/.8;extreme=np.array([0.,0.,1.]);bb=tail_bound(q,0,b)
with (ROOT/'runs/theory/sharpness_curve.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['t','residual','sharp_bound','generic_bound'])
    for t in np.linspace(0,1,41):
        s=(1-t)*r+t*extreme;p=np.r_[b,(1-b)*s]
        writer.writerow([f'{t:.6f}',f'{js(q,p)-bb.coarse:.12f}',f'{bb.tail_max:.12f}',f'{.5*entropy([.8,.2]):.12f}'])
print(json.dumps(report,indent=2))
