"""Semantic retrieval, cross-fitted APM training, and target-LLM inference."""
from __future__ import annotations
import gc
import gzip
import json
import shutil
import time
import numpy as np
from common import (METHODS, TARGETS, aligned, atomic_path, check_output, load_arrays,
                    load_data, model_kwargs, read_json, row_arrays, run_metadata,
                    save_arrays, save_json, sha256)
from prompts import build_prompt, label_verbalizers, parse_prediction


def device_settings(args):
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU or use --device cpu --dtype float32 for a small check")
    if args.device == "cpu" and args.dtype != "float32":
        raise ValueError("Use --dtype float32 with --device cpu")
    if args.device.startswith("cuda") and torch.cuda.device_count() != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to a single GPU (run.sh uses GPU=0 by default)")
    return getattr(torch, args.dtype)


def semantic_topk(query, corpus, k, batch_size):
    """Descending cosine, with original memory row as the exact-tie priority."""
    import torch
    if not 1 <= k <= len(corpus):
        raise ValueError("Candidate pool size exceeds the memory")
    all_indices, all_scores = [], []
    for start in range(0, len(query), batch_size):
        similarities = query[start:start + batch_size] @ corpus.T
        values, indices = torch.topk(similarities, k, dim=1, sorted=True)
        # Resolve boundary ties against the complete memory, not only the K returned rows.
        for row in torch.nonzero((similarities == values[:, -1:]).sum(1) > 1).flatten().tolist():
            above = torch.nonzero(similarities[row] > values[row, -1]).flatten()
            tied = torch.nonzero(similarities[row] == values[row, -1]).flatten()
            selected = torch.cat([above, tied[:k - len(above)]])
            indices[row], values[row] = selected, similarities[row, selected]
        idx, val = indices.cpu().numpy(), values.cpu().float().numpy()
        for i in range(len(idx)):
            order = np.lexsort((idx[i], -val[i]))
            all_indices.append(idx[i, order])
            all_scores.append(val[i, order])
    return np.asarray(all_indices, dtype=np.int32), np.asarray(all_scores, dtype=np.float32)


def retrieve(args):
    import torch
    from sentence_transformers import SentenceTransformer
    device_settings(args)
    train, test, metadata = load_data(args.data_dir, args.task, args.limit_train, args.limit_test)
    out = args.run_dir / "retrieval" / args.task
    check_output(out, ("retrieval.npz", "metadata.json"), args.overwrite)
    options = model_kwargs(args.retriever, args.retriever_revision)
    # GTE loads custom model code from a separate repository; pin both revisions.
    if args.retriever == "Alibaba-NLP/gte-base-en-v1.5":
        options["model_kwargs"] = {"code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"}
        options["config_kwargs"] = {"code_revision": "40ced75c3017eb27626c9d4ea981bde21a2662f4"}
    model = SentenceTransformer(args.retriever, device=args.device, trust_remote_code=True, **options)
    separator = model.tokenizer.sep_token or "[SEP]"
    def texts(rows):
        return [separator.join(r["text"]) if isinstance(r["text"], list) else r["text"] for r in rows]
    started = time.perf_counter()
    corpus, query = [model.encode(texts(rows), batch_size=args.encode_batch_size,
                                  normalize_embeddings=True, convert_to_tensor=True,
                                  show_progress_bar=True).to(dtype=torch.float32)
                     for rows in (train, test)]
    with torch.inference_mode():
        indices, scores = semantic_topk(query, corpus, args.pool_size, args.query_batch_size)
    save_arrays(out / "retrieval.npz", topk_indices=indices, topk_semantic_scores=scores,
                **row_arrays(train, test))
    save_json(out / "metadata.json", run_metadata(args, metadata, model_options=options,
                                                  elapsed_seconds=time.perf_counter() - started))
    return out


def apm(args):
    import torch
    from datasets import Dataset
    from sklearn.model_selection import StratifiedKFold
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                              TrainingArguments, DataCollatorWithPadding, set_seed)
    device_settings(args)
    train, test, metadata = load_data(args.data_dir, args.task, args.limit_train, args.limit_test)
    arrays = row_arrays(train, test)
    labels, classes = arrays["candidate_labels"], len(metadata["label_names"])
    if np.bincount(labels, minlength=classes).min() < args.folds:
        raise ValueError("Each class needs at least --folds training examples; increase --limit-train")
    out = args.run_dir / "apm" / args.task / f"seed_{args.seed}"
    check_output(out, ("posteriors.npz", "metadata.json"), args.overwrite)
    options = model_kwargs(args.apm, args.apm_revision)
    tokenizer = AutoTokenizer.from_pretrained(args.apm, use_fast=True, **options)
    def tokenize(rows):
        paired = isinstance(rows[0]["text"], list)
        data = Dataset.from_dict({"text": [r["text"] for r in rows], "label": [r["label"] for r in rows]})
        def encode(batch):
            parts = ([t[0] for t in batch["text"]], [t[1] for t in batch["text"]]) if paired else (batch["text"],)
            return tokenizer(*parts, truncation=True, max_length=args.max_length)
        return data.map(encode, batched=True, remove_columns=["text"])
    # Fold models train on memory data only; evaluation labels are used after inference.
    train_data, test_data = tokenize(train), tokenize(test)
    posterior = np.full((len(train), classes), np.nan, dtype=np.float32)
    query_posterior = np.empty((args.folds, len(test), classes), dtype=np.float32)
    fold_ids = np.full(len(train), -1, dtype=np.int16)
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    started = time.perf_counter()
    summaries = []
    for fold, (training, holdout) in enumerate(splitter.split(np.zeros(len(train)), labels)):
        seed = int(np.random.SeedSequence([args.seed, fold]).generate_state(1)[0])
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.apm, num_labels=classes,
                   problem_type="single_label_classification", **options)
        settings = TrainingArguments(
            output_dir=str(out / "trainer_tmp" / str(fold)), do_train=True, do_eval=False,
            eval_strategy="no", save_strategy="no", report_to=[], logging_steps=100,
            learning_rate=args.learning_rate, weight_decay=.01, warmup_ratio=.06,
            num_train_epochs=args.epochs, per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps, seed=seed, data_seed=seed,
            bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16",
            tf32=args.device.startswith("cuda"), use_cpu=args.device == "cpu",
            dataloader_num_workers=args.workers, dataloader_pin_memory=args.device.startswith("cuda"))
        trainer = Trainer(model=model, args=settings, train_dataset=train_data.select(training),
                          data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
        result = trainer.train()
        def predict(dataset):
            # No gold labels are provided to the posterior prediction call.
            logits = np.asarray(trainer.predict(dataset.remove_columns("label")).predictions, dtype=np.float64)
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
        posterior[holdout] = predict(train_data.select(holdout))
        query_posterior[fold] = predict(test_data)
        fold_ids[holdout] = fold
        summaries.append(dict(fold=fold, seed=seed, training_rows=len(training), holdout_rows=len(holdout),
                              training_loss=float(result.training_loss)))
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not np.isfinite(posterior).all() or (fold_ids < 0).any():
        raise RuntimeError("Incomplete out-of-fold posteriors")
    save_arrays(out / "posteriors.npz", candidate_probs_oof=posterior,
                test_probs_folds=query_posterior, candidate_fold=fold_ids, **arrays)
    save_json(out / "metadata.json", run_metadata(args, metadata, model_options=options,
                folds=summaries, torch_version=torch.__version__,
                transformers_version=__import__("transformers").__version__,
                elapsed_seconds=time.perf_counter() - started))
    shutil.rmtree(out / "trainer_tmp", ignore_errors=True)
    return out


def chat_ids(tokenizer, task, prompt, max_input_tokens):
    system = ("You are a text classifier. Return exactly one label and no explanation. "
              f"Valid labels, in order, are: {', '.join(label_verbalizers(task))}.")
    ids = tokenizer.apply_chat_template([{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}],
                                       tokenize=True, add_generation_prompt=True)
    if len(ids) > max_input_tokens:
        raise ValueError(f"Prompt has {len(ids)} tokens, exceeding --max-input-tokens={max_input_tokens}; increase the limit")
    return ids


def infer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    dtype = device_settings(args)
    source = args.run_dir / "selections" / args.task / f"seed_{args.seed}"
    meta, selection = read_json(source / "metadata.json"), load_arrays(source / "selections.npz")
    train, test, metadata = load_data(args.data_dir, args.task, meta["limits"]["train"], meta["limits"]["test"])
    if meta["source_data_metadata"] != metadata:
        raise ValueError("Selections and prompt data differ")
    aligned(selection, row_arrays(train, test), ("candidate_ids", "test_ids", "test_labels"))
    if tuple(selection["methods"]) != METHODS:
        raise ValueError("Unexpected selector names")
    selected = selection["candidate_indices"]
    if selected.shape != (len(METHODS), len(test), meta["shots"]) or (selected < 0).any() or (selected >= len(train)).any():
        raise ValueError("Invalid demonstration indices")
    out = args.run_dir / "predictions" / args.model / args.task / f"seed_{args.seed}"
    check_output(out, ("predictions.npz", "responses.jsonl.gz", "metadata.json"), args.overwrite)
    name = args.target_model or TARGETS[args.model]
    options = model_kwargs(name, args.target_revision)
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, padding_side="left", **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype,
                    attn_implementation="sdpa", low_cpu_mem_usage=True, **options).to(args.device)
    model.eval()
    set_seed(args.seed)
    predictions = np.full((len(METHODS), len(test)), -1, dtype=np.int16)
    started = time.perf_counter()
    with atomic_path(out / "responses.jsonl.gz") as temporary, gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for m, method in enumerate(METHODS):
            for start in range(0, len(test), args.batch_size):
                positions = range(start, min(start + args.batch_size, len(test)))
                ids = []
                for i in positions:
                    prompt = build_prompt(args.task, [train[j] for j in selected[m, i]], {"text": test[i]["text"]})
                    ids.append(chat_ids(tokenizer, args.task, prompt, args.max_input_tokens))
                encoded = tokenizer.pad([{"input_ids": row} for row in ids], padding=True,
                                        pad_to_multiple_of=8, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
                responses = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, response in zip(positions, responses):
                    label = parse_prediction(args.task, response)
                    if label is not None:
                        predictions[m, i] = label
                    stream.write(json.dumps(dict(method=method, query_id=test[i]["id"],
                                prediction=label, response=response, candidate_indices=selected[m, i].tolist()), ensure_ascii=False) + "\n")
            print(f"{args.task}/{method}: {np.mean(predictions[m] == selection['test_labels']):.4f}", flush=True)
    save_arrays(out / "predictions.npz", methods=np.asarray(METHODS), predictions=predictions,
                valid=predictions >= 0, gold_labels=selection["test_labels"], test_ids=selection["test_ids"])
    save_json(out / "metadata.json", run_metadata(args, metadata, target_model=name, model_options=options,
                selection_sha256=sha256(source / "selections.npz"), prompt_order=meta["prompt_order"],
                invalid_answers_counted_incorrect=True, elapsed_seconds=time.perf_counter() - started))
    return out
