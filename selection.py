"""Full posterior alignment, its two main controls, and the tail certificate."""
from __future__ import annotations
import numpy as np
from compact_certificates import compact_intervals
from tail_certificates import certificate, entropy, intervals, js, top_l
from common import (METHODS, aligned, check_output, load_arrays, read_json,
                    save_arrays, save_json, sha256)


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim not in (1, 2) or p.shape[-1] < 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite nonnegative posterior vectors with at least two labels")
    sums = p.sum(axis=-1, keepdims=True)
    if (sums <= 0).any() or not np.allclose(sums, 1, atol=1e-5, rtol=0):
        raise ValueError("Posterior rows must sum to one")
    return p / sums


def anchored_posteriors(raw, labels):
    p = probabilities(raw)
    labels = np.asarray(labels)
    if p.ndim != 2 or labels.shape != (len(p),) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Candidate labels must be an aligned integer vector")
    if (labels < 0).any() or (labels >= p.shape[1]).any():
        raise ValueError("Candidate labels are outside the label space")
    rho = p[np.arange(len(p)), labels]
    anchored = rho[:, None] * p
    anchored[np.arange(len(p)), labels] += 1 - rho
    return anchored


def query_concentration(query):
    q = probabilities(query)
    return float(np.clip(1 - entropy(q) / np.log2(len(q)), 0, 1))


def fpa_scores(semantic, query_mean, matched_queries, anchored, labels, weight=0.5):
    q, p = probabilities(matched_queries), probabilities(anchored)
    qb = probabilities(query_mean)
    rel, y = np.asarray(semantic, dtype=np.float64), np.asarray(labels)
    if q.shape != p.shape or rel.shape != (len(p),) or y.shape != rel.shape:
        raise ValueError("Candidate scoring arrays do not align")
    if qb.shape != (p.shape[1],) or not np.isfinite(rel).all() or not 0 <= weight <= 1:
        raise ValueError("Invalid query, relevance, or scoring weight")
    midpoint = .5 * (q + p)
    def kl(a):
        ratio = np.ones_like(a)
        np.divide(a, midpoint, out=ratio, where=a > 0)
        return np.sum(a * np.log2(ratio), axis=1)
    divergence = np.clip(.5 * (kl(q) + kl(p)), 0, 1)
    c = query_concentration(qb)
    return weight * rel + (1 - weight) * (c * qb[y] + (1 - c) * (1 - divergence))


def select_query(semantic, heads, folds, raw_candidates, labels, shots=8, weight=.5):
    """Return candidate-pool positions; query gold labels are not an input."""
    raw_heads = np.asarray(heads, dtype=np.float64)
    qb = probabilities(raw_heads.mean(axis=0))
    heads = probabilities(raw_heads)
    p = anchored_posteriors(raw_candidates, labels)
    scores = fpa_scores(semantic, qb, heads[folds], p, labels, weight)
    lm = weight * np.asarray(semantic) + (1 - weight) * qb[labels]
    return np.stack([top_l(semantic, shots), top_l(lm, shots), top_l(scores, shots)])


def load_inputs(args):
    ret_path = args.run_dir / "retrieval" / args.task / "retrieval.npz"
    apm_path = args.run_dir / "apm" / args.task / f"seed_{args.seed}" / "posteriors.npz"
    retrieval, apm = load_arrays(ret_path), load_arrays(apm_path)
    rm, am = read_json(ret_path.with_name("metadata.json")), read_json(apm_path.with_name("metadata.json"))
    if rm["source_data_metadata"] != am["source_data_metadata"]:
        raise ValueError("Retrieval and APM data versions differ")
    if am["settings"]["seed"] != args.seed:
        raise ValueError("APM seed differs from the requested seed")
    aligned(retrieval, apm, ("candidate_ids", "test_ids", "candidate_labels", "test_labels"))
    n, c = apm["candidate_probs_oof"].shape
    if n != len(apm["candidate_ids"]) or apm["test_probs_folds"].shape[1:] != (len(apm["test_ids"]), c):
        raise ValueError("Posterior shapes do not match the data")
    folds = apm["candidate_fold"]
    if folds.shape != (n,) or not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any() or (folds >= len(apm["test_probs_folds"])).any():
        raise ValueError("Invalid candidate fold assignments")
    pools = retrieval["topk_indices"]
    if pools.ndim != 2 or pools.shape != retrieval["topk_semantic_scores"].shape or len(pools) != len(apm["test_ids"]):
        raise ValueError("Invalid retrieval shape")
    if not np.issubdtype(pools.dtype, np.integer) or (pools < 0).any() or (pools >= n).any():
        raise ValueError("Invalid candidate pool indices")
    if any(len(set(row)) != len(row) for row in pools) or args.shots > pools.shape[1]:
        raise ValueError("Candidate pools must contain enough distinct examples")
    meta = dict(task=args.task, seed=args.seed, shots=args.shots, weight=args.weight,
                source_data_metadata=rm["source_data_metadata"],
                retrieval_sha256=sha256(ret_path), apm_sha256=sha256(apm_path),
                limits={"train": rm["settings"]["limit_train"], "test": rm["settings"]["limit_test"]})
    return retrieval, apm, meta


def select(args):
    ret, apm, meta = load_inputs(args)
    out = args.run_dir / "selections" / args.task / f"seed_{args.seed}"
    check_output(out, ("selections.npz", "metadata.json"), args.overwrite)
    chosen = []
    for index, pool in enumerate(ret["topk_indices"]):
        positions = select_query(ret["topk_semantic_scores"][index], apm["test_probs_folds"][:, index],
                                 apm["candidate_fold"][pool], apm["candidate_probs_oof"][pool],
                                 apm["candidate_labels"][pool], args.shots, args.weight)
        chosen.append(pool[positions])
    save_arrays(out / "selections.npz", methods=np.asarray(METHODS),
                candidate_indices=np.stack(chosen, axis=1),
                **{k: ret[k] for k in ("candidate_ids", "test_ids", "test_labels")})
    save_json(out / "metadata.json", dict(meta, methods=METHODS, prompt_order="descending selector score",
                                         query_gold_used_for_selection=False))
    return out


def audit(args):
    ret, apm, meta = load_inputs(args)
    out = args.run_dir / "audit" / args.task / f"seed_{args.seed}"
    check_output(out, ("facts.npz", "summary.json"), args.overwrite)
    indices = np.arange(len(ret["test_ids"]))
    if args.query_registry:
        wanted = read_json(args.query_registry)[args.task]
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError("Registry must contain unique query IDs")
        lookup = {str(q): i for i, q in enumerate(ret["test_ids"])}
        indices = np.asarray([lookup[str(q)] for q in wanted], dtype=np.int64)
    certified, margins, boundary_pairs, endpoint_changes = [], [], [], []
    for index in indices:
        pool = ret["topk_indices"][index]
        rel = ret["topk_semantic_scores"][index]
        raw_heads = np.asarray(apm["test_probs_folds"][:, index], dtype=np.float64)
        qb = probabilities(raw_heads.mean(0))
        heads = probabilities(raw_heads)
        y, folds = apm["candidate_labels"][pool], apm["candidate_fold"][pool]
        p = anchored_posteriors(apm["candidate_probs_oof"][pool], y)
        lo, hi = compact_intervals(heads, folds, y, p[np.arange(len(y)), y], rel, qb,
                                   query_concentration(qb), args.weight)
        full = fpa_scores(rel, qb, heads[folds], p, y, args.weight)
        if (full < lo - 1e-12).any() or (full > hi + 1e-12).any():
            raise ArithmeticError("Observed scores fall outside the derived intervals")
        chosen = top_l(full, args.shots)
        result = certificate(lo, hi, chosen)
        changed = set(top_l(lo, args.shots)) != set(chosen)
        if result["certified"] and changed:
            raise ArithmeticError("Certified set changed at lower endpoints")
        certified.append(result["certified"])
        margins.append(result["margin"])
        boundary_pairs.append(result["pair"] or (-1, -1))
        endpoint_changes.append(changed)
    save_arrays(out / "facts.npz", query_ids=ret["test_ids"][indices], query_indices=indices,
                certified=np.asarray(certified), margin=np.asarray(margins),
                boundary_pair=np.asarray(boundary_pairs), endpoint_changed=np.asarray(endpoint_changes))
    summary = dict(meta, rows=len(indices), certified=int(sum(certified)),
                   coverage_percent=100 * float(np.mean(certified)),
                   lower_endpoint_changes=int(sum(endpoint_changes)),
                   population="query_registry" if args.query_registry else "all_available_evaluation_queries",
                   query_registry_sha256=sha256(args.query_registry) if args.query_registry else None)
    save_json(out / "summary.json", summary)
    return out
