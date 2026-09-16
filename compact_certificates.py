"""Query-factored evaluation of the same intervals as tail_certificates.py.

This is an algebraic implementation check, not a runtime benchmark. All formulas
use float64; the exact-real guarantees in the paper do not certify roundoff.
"""
from __future__ import annotations
import numpy as np
from tail_certificates import probability_vector, js, phi


def compact_intervals(query_heads, fold_ids, labels, masses, semantic,
                      query_mean, concentration: float,
                      semantic_weight: float = 0.5):
    """Return lower/upper score arrays in O(F*C + K) arithmetic.

    Candidate tails are deliberately not an argument. The observed labels and
    their anchored masses are assumed to have been computed offline.
    """
    heads = np.asarray(query_heads, dtype=np.float64)
    if heads.ndim != 2 or heads.shape[0] < 1 or heads.shape[1] < 2:
        raise ValueError('query_heads must have shape (F, C), F >= 1, C >= 2.')
    for head in heads:
        probability_vector(head)
    f = np.asarray(fold_ids)
    y = np.asarray(labels)
    b = np.asarray(masses, dtype=np.float64)
    rel = np.asarray(semantic, dtype=np.float64)
    if f.ndim != 1 or y.ndim != 1 or not np.issubdtype(f.dtype, np.integer) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('fold_ids and labels must be one-dimensional integer arrays.')
    n = len(f)
    if any(v.shape != (n,) for v in (y, b, rel)):
        raise ValueError('Candidate arrays must have the same length.')
    if (f < 0).any() or (f >= len(heads)).any() or (y < 0).any() or (y >= heads.shape[1]).any():
        raise ValueError('Invalid fold or class index.')
    if not np.isfinite(b).all() or not np.isfinite(rel).all() or (b < 0).any() or (b > 1).any():
        raise ValueError('Masses must lie in [0, 1]; inputs must be finite.')
    if not 0 <= concentration <= 1 or not 0 <= semantic_weight <= 1:
        raise ValueError('Mixture coefficients must lie in [0, 1].')
    qb = probability_vector(query_mean)
    if qb.shape != (heads.shape[1],):
        raise ValueError('query_mean has the wrong number of classes.')

    # Keep the first and second *entries*, including ties, not distinct values.
    idx_min = np.argmin(heads, axis=1)
    smallest_two = np.partition(heads, kth=1, axis=1)[:, :2]
    first = np.min(smallest_two, axis=1)
    second = np.max(smallest_two, axis=1)
    minimum_excluding = np.where(y == idx_min[f], second[f], first[f])

    # Prefix/suffix sums avoid cancellation in a nearly one-hot head. These
    # have the same O(F*C) preprocessing cost as scanning the minimum entries.
    prefix = np.pad(np.cumsum(heads, axis=1), ((0, 0), (1, 0)))
    suffix = np.pad(np.cumsum(heads[:, ::-1], axis=1)[:, ::-1], ((0, 0), (0, 1)))
    actual_tail_sum = prefix[f, y] + suffix[f, y + 1]
    a = heads[f, y]
    coarse = np.empty(n, dtype=np.float64)
    upper_tail = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tq, tp = 1 - a[i], 1 - b[i]
        coarse[i] = js([a[i], tq], [b[i], tp])
        if tq > 0 and tp > 0 and heads.shape[1] > 2:
            z = np.clip(minimum_excluding[i] / actual_tail_sum[i], 0.0, 1.0)
            upper_tail[i] = max(0.0, (tq + tp) * 0.5 * phi(tq / (tq + tp), float(z)))
    upper = semantic_weight * rel + (1-semantic_weight) * (
        concentration * qb[y] + (1-concentration) * (1-coarse))
    lower = upper - (1-semantic_weight) * (1-concentration) * upper_tail
    return lower, upper
