#!/usr/bin/env python3
"""Run the main posterior-geometry experiments, one stage at a time."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import TARGETS, load_data
from data import TASK_NAMES, load_task, materialize_task

AUDIT_TASKS = ("agnews", "mnli", "massive", "sst5", "banking77")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("data", "retrieve", "apm", "select", "infer", "audit", "summarize", "coverage"):
        q = sub.add_parser(stage)
        q.add_argument("--run-dir", type=Path, default=Path("runs"))
        if stage in ("summarize", "coverage"):
            q.add_argument("--tasks", nargs="+", choices=TASK_NAMES,
                           default=list(TASK_NAMES if stage == "summarize" else AUDIT_TASKS))
            q.add_argument("--seeds", nargs="+", type=int,
                           default=[521, 997, 2027] if stage == "summarize" else [4099, 8191, 16381])
            if stage == "summarize":
                q.add_argument("--models", nargs="+", choices=TARGETS, default=list(TARGETS))
                q.add_argument("--resamples", type=int, default=100000)
            continue
        q.add_argument("--task", required=True, choices=TASK_NAMES)
        q.add_argument("--overwrite", action="store_true")
        if stage in ("data", "retrieve", "apm", "infer"):
            q.add_argument("--data-dir", type=Path, default=Path("data"))
        if stage in ("apm", "select", "infer", "audit"):
            q.add_argument("--seed", type=int, default=521)
        if stage in ("retrieve", "apm", "infer"):
            q.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
            q.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
        if stage in ("retrieve", "apm"):
            q.add_argument("--limit-train", type=int)
            q.add_argument("--limit-test", type=int)
        if stage == "retrieve":
            q.add_argument("--retriever", default="Alibaba-NLP/gte-base-en-v1.5")
            q.add_argument("--retriever-revision")
            q.add_argument("--pool-size", type=int, default=30)
            q.add_argument("--encode-batch-size", type=int, default=128)
            q.add_argument("--query-batch-size", type=int, default=256)
        elif stage == "apm":
            q.add_argument("--apm", default="FacebookAI/roberta-base")
            q.add_argument("--apm-revision")
            q.add_argument("--folds", type=int, default=3)
            q.add_argument("--epochs", type=float, default=3)
            q.add_argument("--learning-rate", type=float, default=2e-5)
            q.add_argument("--train-batch-size", type=int, default=16)
            q.add_argument("--eval-batch-size", type=int, default=64)
            q.add_argument("--gradient-accumulation-steps", type=int, default=2)
            q.add_argument("--max-length", type=int, default=256)
            q.add_argument("--workers", type=int, default=4)
            q.add_argument("--max-steps", type=int, default=-1, help="Optional small training check; paper runs use epochs")
        elif stage in ("select", "audit"):
            q.add_argument("--shots", type=int, default=8)
            q.add_argument("--weight", type=float, default=.5, help="Semantic score weight")
            if stage == "audit":
                q.add_argument("--query-registry", type=Path, help="JSON mapping task names to ordered query IDs")
        elif stage == "infer":
            q.add_argument("--model", choices=TARGETS, default="qwen")
            q.add_argument("--target-model", help="Optional local checkpoint or Hub ID")
            q.add_argument("--target-revision")
            q.add_argument("--batch-size", type=int, default=2)
            q.add_argument("--max-input-tokens", type=int, default=16384)
            q.add_argument("--max-new-tokens", type=int, default=16)
    return p


def main():
    p = parser()
    args = p.parse_args()
    positive = ("limit_train", "limit_test", "pool_size", "encode_batch_size", "query_batch_size",
                "epochs", "learning_rate", "train_batch_size", "eval_batch_size",
                "gradient_accumulation_steps", "max_length", "batch_size", "max_input_tokens",
                "max_new_tokens", "resamples")
    for name in positive:
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "folds", 3) < 2 or getattr(args, "workers", 0) < 0:
        p.error("--folds must be at least 2 and --workers must be nonnegative")
    if getattr(args, "shots", 8) < 0 or not 0 <= getattr(args, "weight", .5) <= 1:
        p.error("--shots must be nonnegative and --weight must lie in [0, 1]")
    if getattr(args, "max_steps", -1) == 0 or getattr(args, "max_steps", -1) < -1:
        p.error("--max-steps must be -1 or positive")
    for name in ("tasks", "models", "seeds"):
        values = getattr(args, name, [])
        if len(set(values)) != len(values):
            p.error(f"--{name} must contain unique values")
    if args.stage == "data":
        metadata = args.data_dir / args.task / "metadata.json"
        if metadata.exists() and not args.overwrite:
            load_data(args.data_dir, args.task)
            print(f"Using prepared data: {metadata.parent}")
        else:
            print(materialize_task(load_task(args.task), args.data_dir))
    elif args.stage in ("retrieve", "apm", "infer"):
        import models
        print(getattr(models, args.stage)(args))
    elif args.stage in ("select", "audit"):
        import selection
        print(getattr(selection, args.stage)(args))
    else:
        import results
        print(getattr(results, args.stage)(args))


if __name__ == "__main__":
    main()
