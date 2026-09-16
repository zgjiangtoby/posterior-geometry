"""Exact-real tail-certificate formulas evaluated in NumPy float64.

This module implements the mathematics in the manuscript. Numerical tests use
an explicit tolerance; these are not interval-arithmetic machine certificates.
Scores are additive selector scores, never target-model accuracy guarantees.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def probability_vector(value) -> np.ndarray:
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 1 or len(p) < 1 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected a finite, nonnegative probability vector.')
    if not np.isclose(p.sum(), 1., rtol=0., atol=1e-10):
        raise ValueError('Probabilities must sum to one.')
    return p


def entropy(value) -> float:
    p = probability_vector(value)
    positive = p > 0
    return float(-np.sum(p[positive] * np.log2(p[positive])))


def js(first, second, weight: float = .5) -> float:
    p, q = probability_vector(first), probability_vector(second)
    if p.shape != q.shape or not 0 <= weight <= 1:
        raise ValueError('Shapes must match and mixture weight must lie in [0,1].')
    if weight in (0., 1.):
        return 0.
    m = weight * p + (1 - weight) * q
    ip, iq = p > 0, q > 0
    return float(weight * np.sum(p[ip] * np.log2(p[ip] / m[ip]))
                 + (1 - weight) * np.sum(q[iq] * np.log2(q[iq] / m[iq])))


def phi(weight: float, z: float) -> float:
    """Weighted JS between a fixed tail and a vertex at tail coordinate z."""
    if not 0 <= weight <= 1 or not 0 <= z <= 1:
        raise ValueError('Arguments must lie in [0,1].')
    if weight in (0., 1.) or z == 1.:
        return 0.
    m = 1 - weight + weight * z
    azlogz = 0. if z == 0. else weight * z * np.log2(z)
    return float(-m * np.log2(m) - weight * (1 - z) * np.log2(weight) + azlogz)


@dataclass(frozen=True)
class TailBound:
    coarse: float
    tail_max: float
    # 'lower'/'upper' refer to the selector score, not to JS divergence.
    lower_posterior: np.ndarray
    upper_posterior: np.ndarray


def tail_bound(query, label: int, mass: float) -> TailBound:
    q = probability_vector(query)
    if len(q) < 2 or not 0 <= label < len(q) or not 0 <= mass <= 1:
        raise ValueError('At least two classes and a valid label/mass are required.')
    a = float(q[label]); tq = 1 - a; tp = 1 - mass
    idx = np.flatnonzero(np.arange(len(q)) != label)
    r = q[idx] / tq if tq > 0 else np.full(len(idx), 1 / len(idx))
    r = r / r.sum()
    upper = np.zeros_like(q); upper[label] = mass; upper[idx] = tp * r
    lower = np.zeros_like(q); lower[label] = mass
    lower[idx[int(np.argmin(r))]] = tp
    coarse = js([a, tq], [mass, tp])
    if tq == 0. or tp == 0. or len(q) == 2:
        return TailBound(coarse, 0., lower, upper)
    w = (tq + tp) / 2
    alpha = tq / (tq + tp)
    return TailBound(coarse, max(0., w * phi(alpha, float(r.min()))), lower, upper)


def intervals(queries, labels, masses, semantic, query_mean,
              concentration: float, semantic_weight: float = .5):
    q = np.asarray(queries, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    b = np.asarray(masses, dtype=np.float64)
    rel = np.asarray(semantic, dtype=np.float64)
    qb = probability_vector(query_mean)
    if q.ndim != 2 or q.shape[1] != len(qb) or any(len(v) != len(q) for v in (y,b,rel)):
        raise ValueError('Candidate array shapes do not match.')
    if not 0 <= concentration <= 1 or not 0 <= semantic_weight <= 1:
        raise ValueError('Weights must be in [0,1].')
    bounds = [tail_bound(qi, int(yi), float(bi)) for qi,yi,bi in zip(q,y,b)]
    coarse = np.array([v.coarse for v in bounds])
    uncertainty = np.array([v.tail_max for v in bounds])
    upper = semantic_weight * rel + (1-semantic_weight) * (
        concentration * qb[y] + (1-concentration) * (1-coarse))
    lower = upper - (1-semantic_weight) * (1-concentration) * uncertainty
    return lower, upper, bounds


def _priority(size: int, priority=None) -> np.ndarray:
    pi = np.arange(size) if priority is None else np.asarray(priority)
    if pi.shape != (size,) or not np.issubdtype(pi.dtype, np.number) or not np.isfinite(pi).all() or len(np.unique(pi)) != size:
        raise ValueError('Priorities must be a distinct scalar for every candidate.')
    return pi


def top_l(scores, size: int, priority=None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 <= size <= len(values):
        raise ValueError('Invalid score vector or cardinality.')
    return np.lexsort((_priority(len(values), priority), -values))[:size]


def certificate(lower, upper, selected, priority=None) -> dict:
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.ndim != 1 or lo.shape != hi.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo>hi).any():
        raise ValueError('Invalid candidate intervals.')
    pi = _priority(len(lo), priority)
    sel = np.asarray(selected, dtype=int)
    if sel.ndim != 1 or len(np.unique(sel)) != len(sel) or (sel<0).any() or (sel>=len(lo)).any():
        raise ValueError('Invalid candidate set.')
    excluded = sorted(set(range(len(lo))) - set(sel.tolist()))
    if len(sel) == 0 or len(excluded) == 0:
        return dict(certified=True, margin=float('inf'), pair=None)
    i = min(sel, key=lambda t:(lo[t], -pi[t]))
    j = max(excluded, key=lambda t:(hi[t], -pi[t]))
    ok = (lo[i], -pi[i]) > (hi[j], -pi[j])
    return dict(certified=bool(ok), margin=float(lo[i]-hi[j]),
                pair=None if ok else (int(i),int(j)))


def rank_ranges(lower, upper, priority=None):
    lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lo.ndim != 1 or lo.shape != hi.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo>hi).any():
        raise ValueError('Invalid intervals.')
    pi = _priority(len(lo), priority)
    lowkeys = [(lo[i],-pi[i]) for i in range(len(lo))]
    highkeys = [(hi[i],-pi[i]) for i in range(len(lo))]
    best = [1 + sum(lowkeys[j]>highkeys[i] for j in range(len(lo)) if j!=i) for i in range(len(lo))]
    worst = [1 + sum(highkeys[j]>lowkeys[i] for j in range(len(lo)) if j!=i) for i in range(len(lo))]
    return np.array(best), np.array(worst)
