"""Paired seed/example bootstrap for the main classification comparison."""
from __future__ import annotations
import hashlib
import math
import csv
from types import SimpleNamespace
from typing import Any, Sequence
import numpy as np
from common import METHODS, load_arrays, read_json, save_json, atomic_path

def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _bootstrap_summary(
    draws: np.ndarray,
    *,
    observed_difference: float,
    confidence_level: float,
    p_value: float,
) -> dict[str, float]:
    tail = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(draws, [tail, 1.0 - tail])
    return {
        "difference": float(observed_difference),
        "difference_pp": float(100.0 * observed_difference),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci_low_pp": float(100.0 * ci_low),
        "ci_high_pp": float(100.0 * ci_high),
        "p_value": float(p_value),
    }


def _difference_distribution(
    truth: np.ndarray, predictions_a: np.ndarray, predictions_b: np.ndarray
) -> tuple[np.ndarray, float, float]:
    correct_a = np.asarray(predictions_a) == truth
    correct_b = np.asarray(predictions_b) == truth
    difference = correct_a.astype(np.int8) - correct_b.astype(np.int8)
    return difference, float(np.mean(correct_a)), float(np.mean(correct_b))


def _crossed_randomization_draws(
    differences: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomize one shared sign per test ID on its seed-mean difference."""

    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("Crossed randomization requires a seed-by-example matrix")
    mean_by_example = matrix.mean(axis=0)
    draws = np.zeros(n_resamples, dtype=np.float64)
    magnitudes, counts = np.unique(
        np.abs(mean_by_example[mean_by_example != 0.0]), return_counts=True
    )
    for magnitude, count in zip(magnitudes, counts):
        positive = rng.binomial(int(count), 0.5, size=n_resamples)
        draws += (2.0 * positive - int(count)) * float(magnitude)
    return draws / matrix.shape[1]


def _randomization_p_value(draws: np.ndarray, observed: float) -> float:
    """Return a conservative two-sided Monte Carlo randomization p-value."""

    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8.0
    extreme = np.count_nonzero(np.abs(draws) + tolerance >= abs(observed))
    return float((extreme + 1.0) / (len(draws) + 1.0))


def _randomization_details(draws: np.ndarray, observed: float) -> dict[str, Any]:
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8.0
    extreme = int(np.count_nonzero(np.abs(draws) + tolerance >= abs(observed)))
    p_value = float((extreme + 1.0) / (len(draws) + 1.0))
    return {
        "p_value": p_value,
        "p_value_extreme_count": extreme,
        "p_value_mc_standard_error": float(
            math.sqrt(p_value * (1.0 - p_value) / (len(draws) + 1.0))
        ),
    }


def _crossed_seed_example_draws(
    artifacts: Sequence[Any],
    method_a: str,
    method_b: str,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    seed_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Crossed bootstrap: resample seeds, then one shared test-ID sample."""

    if not artifacts:
        raise ValueError("Crossed bootstrap requires at least one seed")
    differences: list[np.ndarray] = []
    accuracies_a: list[float] = []
    accuracies_b: list[float] = []
    for artifact in artifacts:
        difference, accuracy_a, accuracy_b = _difference_distribution(
            artifact.gold_labels,
            artifact.predictions_for(method_a),
            artifact.predictions_for(method_b),
        )
        differences.append(difference)
        accuracies_a.append(accuracy_a)
        accuracies_b.append(accuracy_b)

    difference_matrix = np.stack(differences)
    n_seeds, n_examples = difference_matrix.shape
    if seed_counts is None:
        seed_counts = rng.multinomial(
            n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=n_resamples
        )
    else:
        seed_counts = np.asarray(seed_counts, dtype=np.int64)
        if seed_counts.shape != (n_resamples, n_seeds):
            raise ValueError("Shared seed-count bootstrap has incompatible shape")
        if np.any(seed_counts < 0) or np.any(seed_counts.sum(axis=1) != n_seeds):
            raise ValueError("Every shared seed-count row must sum to n_seeds")
    draws = np.empty(n_resamples, dtype=np.float64)
    unique_counts, inverse = np.unique(seed_counts, axis=0, return_inverse=True)
    for pattern_index, counts_by_seed in enumerate(unique_counts):
        positions = np.flatnonzero(inverse == pattern_index)
        per_example = counts_by_seed @ difference_matrix / n_seeds
        values, value_counts = np.unique(per_example, return_counts=True)
        sampled_value_counts = rng.multinomial(
            n_examples,
            value_counts.astype(np.float64) / n_examples,
            size=len(positions),
        )
        draws[positions] = sampled_value_counts @ values / n_examples
    observed = float(np.mean([np.mean(difference) for difference in differences]))
    return draws, observed, float(np.mean(accuracies_a)), float(np.mean(accuracies_b))


def paired_macro_crossed_bootstrap(
    artifacts_by_task: dict[str, Sequence[Any]],
    method_a: str,
    method_b: str,
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Fixed-task macro crossed bootstrap with shared seed resampling."""

    tasks = list(artifacts_by_task)
    if not tasks:
        raise ValueError("Macro crossed bootstrap requires at least one task")
    models = {
        artifact.model
        for task_artifacts in artifacts_by_task.values()
        for artifact in task_artifacts
    }
    if len(models) != 1:
        raise ValueError("Macro crossed bootstrap must contain exactly one model")
    rng = np.random.default_rng(seed)
    ordered_by_task = {
        task: sorted(artifacts_by_task[task], key=lambda artifact: artifact.seed)
        for task in tasks
    }
    seed_lists = [
        [artifact.seed for artifact in ordered_by_task[task]] for task in tasks
    ]
    if any(seeds != seed_lists[0] for seeds in seed_lists[1:]):
        raise ValueError("Macro crossed bootstrap requires identical seed IDs by task")
    n_seeds = len(seed_lists[0])
    shared_seed_counts = rng.multinomial(
        n_seeds, np.full(n_seeds, 1.0 / n_seeds), size=n_resamples
    )
    task_draws: list[np.ndarray] = []
    task_permutation_draws: list[np.ndarray] = []
    observed_differences: list[float] = []
    accuracies_a: list[float] = []
    accuracies_b: list[float] = []
    for task in tasks:
        ordered = ordered_by_task[task]
        draws, observed, accuracy_a, accuracy_b = _crossed_seed_example_draws(
            ordered,
            method_a,
            method_b,
            n_resamples=n_resamples,
            rng=rng,
            seed_counts=shared_seed_counts,
        )
        task_draws.append(draws)
        differences = [
            _difference_distribution(
                artifact.gold_labels,
                artifact.predictions_for(method_a),
                artifact.predictions_for(method_b),
            )[0]
            for artifact in ordered
        ]
        task_permutation_draws.append(
            _crossed_randomization_draws(
                np.stack(differences), n_resamples=n_resamples, rng=rng
            )
        )
        observed_differences.append(observed)
        accuracies_a.append(accuracy_a)
        accuracies_b.append(accuracy_b)
    macro_draws = np.mean(np.stack(task_draws), axis=0)
    macro_permutation_draws = np.mean(np.stack(task_permutation_draws), axis=0)
    macro_observed = float(np.mean(observed_differences))
    randomization = _randomization_details(
        macro_permutation_draws, macro_observed
    )
    return {
        "scope": "macro_task_fixed_seed_example_crossed",
        "model": next(iter(models)),
        "task": None,
        "seed": None,
        "method_a": method_a,
        "method_b": method_b,
        "accuracy_a": float(np.mean(accuracies_a)),
        "accuracy_b": float(np.mean(accuracies_b)),
        **_bootstrap_summary(
            macro_draws,
            observed_difference=macro_observed,
            confidence_level=confidence_level,
            p_value=float(randomization["p_value"]),
        ),
        "p_value_extreme_count": randomization["p_value_extreme_count"],
        "p_value_mc_standard_error": randomization[
            "p_value_mc_standard_error"
        ],
        "p_value_method": (
            "two_sided_test_id_randomization_monte_carlo_conditional_on_seeds"
        ),
        "confidence_level": confidence_level,
        "n_examples": int(
            sum(len(artifacts_by_task[task][0].gold_labels) for task in tasks)
        ),
        "n_seeds": min(len(artifacts_by_task[task]) for task in tasks),
        "seeds_by_task": {
            task: sorted(artifact.seed for artifact in artifacts_by_task[task])
            for task in tasks
        },
        "n_tasks": len(tasks),
        "tasks": tasks,
        "n_resamples": n_resamples,
        "bootstrap_seed": seed,
    }


def summarize(args):
    rows, comparisons = [], []
    for model in args.models:
        by_task = {}
        for task in args.tasks:
            cells = []
            for seed in args.seeds:
                root = args.run_dir / "predictions" / model / task / f"seed_{seed}"
                z = load_arrays(root / "predictions.npz")
                if tuple(z["methods"]) != METHODS or z["predictions"].shape != (len(METHODS), len(z["test_ids"])):
                    raise ValueError(f"Unexpected prediction arrays: {root}")
                if len(set(z["test_ids"])) != len(z["test_ids"]) or not len(z["test_ids"]):
                    raise ValueError(f"Invalid query IDs: {root}")
                if not np.array_equal(z["valid"], z["predictions"] >= 0) or (z["gold_labels"] < 0).any():
                    raise ValueError(f"Invalid predictions or gold labels: {root}")
                if cells and (not np.array_equal(cells[0].test_ids, z["test_ids"]) or
                              not np.array_equal(cells[0].gold_labels, z["gold_labels"])):
                    raise ValueError("Seed runs must evaluate identical query IDs and labels")
                cells.append(SimpleNamespace(model=model, task=task, seed=seed,
                    test_ids=z["test_ids"], gold_labels=z["gold_labels"],
                    predictions_for=lambda method, z=z: z["predictions"][METHODS.index(method)]))
            by_task[task] = cells
        for method in METHODS:
            per_task = []
            for task, cells in by_task.items():
                accuracy = [100 * np.mean(c.predictions_for(method) == c.gold_labels) for c in cells]
                per_task.append(accuracy)
                rows.append(dict(model=model, task=task, method=method, mean=float(np.mean(accuracy)),
                                 sd=float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else None))
            macro = np.mean(per_task, axis=0)
            rows.append(dict(model=model, task="Macro", method=method, mean=float(np.mean(macro)),
                             sd=float(np.std(macro, ddof=1)) if len(macro) > 1 else None))
        for comparator in ("topk", "label_mass"):
            # Keep the original bootstrap stream despite clearer public method names.
            legacy_model = {"qwen": "qwen2.5-7b-instruct", "mistral": "mistral-7b-instruct-v0.3"}[model]
            legacy_control = "gold_soft" if comparator == "label_mass" else comparator
            seed = _stable_seed(2027, "macro-crossed", legacy_model, "confidence_gated_gold_anchor", legacy_control)
            comparisons.append(paired_macro_crossed_bootstrap(by_task, "fpa", comparator,
                                n_resamples=args.resamples, confidence_level=.95, seed=seed))
    out = args.run_dir / "results"
    with atomic_path(out / "accuracy.csv") as temporary, temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "task", "method", "mean", "sd"))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / "comparisons.json", dict(numpy_version=np.__version__, tasks=args.tasks,
              seeds=args.seeds, models=args.models, resamples=args.resamples, comparisons=comparisons))
    return out


def coverage(args):
    cells = []
    for task in args.tasks:
        reference = None
        for seed in args.seeds:
            root = args.run_dir / "audit" / task / f"seed_{seed}"
            summary, facts = read_json(root / "summary.json"), load_arrays(root / "facts.npz")
            if reference is not None and not np.array_equal(reference, facts["query_ids"]):
                raise ValueError("Certificate seeds must audit identical query IDs")
            reference = facts["query_ids"]
            if len(reference) != summary["rows"] or int(facts["certified"].sum()) != summary["certified"]:
                raise ValueError("Certificate facts differ from their summary")
            cells.append(summary)
    out = args.run_dir / "results" / "coverage.json"
    rows = sum(c["rows"] for c in cells)
    save_json(out, dict(tasks=args.tasks, seeds=args.seeds, query_seed_rows=rows,
        equal_task_seed_percent=float(np.mean([c["coverage_percent"] for c in cells])),
        query_weighted_percent=100 * sum(c["certified"] for c in cells) / rows,
        by_task={task: float(np.mean([c["coverage_percent"] for c in cells if c["task"] == task]))
                 for task in args.tasks}, cells=cells))
    return out
