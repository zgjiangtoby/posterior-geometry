# Posterior Geometry for Demonstration Selection

This code is for the paper **When Does Posterior Geometry Matter for In-Context Demonstration Selection?**

The implementation includes full posterior alignment (FPA), the Semantic TopK and Label-Mass controls, candidate-tail certificates, and numerical verification.

## Installation

Use Python 3.11 or 3.12. Model experiments use a single CUDA GPU; numerical verification runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example usage

1. Set the model names or local checkpoint paths at the top of `run.sh`. The defaults use GTE-base-en-v1.5, RoBERTa-base, Qwen2.5-7B-Instruct, and Mistral-7B-Instruct-v0.3. Public model revisions are fixed in `common.py`; downloads use the Hugging Face cache.
2. Run one task and one seed first:

```bash
GPU=0 TASKS="sst5" SEEDS="521" MODELS="qwen" bash run.sh classification
```

Data are downloaded automatically and prepared as:

```text
data/
  sst2/       # train.jsonl, validation.jsonl, test.jsonl, metadata.json
  sst5/
  cr/
  subj/
  agnews/
  mnli/
  qnli/
  banking77/
  massive/
```

SST-2 uses the SetFit sentence-level split; MNLI and QNLI evaluate the labeled validation splits; MASSIVE uses en-US. The loader preserves source row order and label mappings. Existing prepared data can be used with `DATA_DIR=/path/to/data`.

## Main classification experiment

```bash
GPU=0 bash run.sh classification
```

This runs all nine tasks, seeds **521, 997, 2027**, and both target models. Each query uses eight demonstrations from the same semantic Top-30 pool. The APM uses three stratified folds, three training epochs per fold, and candidate/query posteriors from the same fold model. FPA uses observed-label anchoring, an entropy-based query mixture, and semantic weight 0.5.

Selections retain descending selector-score order in the prompt, as in the original classification implementation. Generation is greedy with at most 16 new tokens; invalid answers count as errors. Prompts exceeding the input limit raise an error instead of being truncated.

Outputs are written under `runs/`: retrieval arrays, out-of-fold posteriors, selected indices, generated responses, and predictions. `runs/results/accuracy.csv` reports task and macro accuracy with sample standard deviation across seeds; `comparisons.json` reports the paired seed/example bootstrap using 100,000 draws. Macro averages weight tasks equally.

To run or resume an individual stage:

```bash
python main.py data --task sst5
CUDA_VISIBLE_DEVICES=0 python main.py retrieve --task sst5
CUDA_VISIBLE_DEVICES=0 python main.py apm --task sst5 --seed 521
python main.py select --task sst5 --seed 521
CUDA_VISIBLE_DEVICES=0 python main.py infer --task sst5 --seed 521 --model qwen
python main.py summarize --tasks sst5 --seeds 521 --models qwen
```

Use `--help` after any stage for its options. `--run-dir` selects the output directory. Existing experiment outputs require `--overwrite` to replace; use a separate run directory for a new configuration. `BATCH_SIZE=1` reduces inference memory use. Local model paths can be set through `RETRIEVER`, `APM`, `QWEN`, and `MISTRAL` in `run.sh`.

## Certificate coverage

```bash
GPU=0 RUN_DIR=runs_certificate bash run.sh certificate
```

This runs AG News, MNLI, MASSIVE, SST-5, and Banking77 with APM seeds **4099, 8191, 16381**. It computes attainable score intervals, tie-aware Top-8 invariance, and the lower-endpoint membership check. No target-LLM inference is needed. Results include per-query facts and equal-task/seed and query-weighted coverage in `runs_certificate/results/coverage.json`.

The default population is all evaluation queries. The original source package does not contain the historical query registry for the paper's 100,455 query-seed rows. To use a retained registry, set `QUERY_REGISTRY=/path/to/registry.json`; its format is `{"sst5": ["sst5:test:0", "sst5:test:1"], ...}` with the desired IDs for each task. Each report records its actual population and row count.

## Numerical verification

Only NumPy is needed for these checks:

```bash
pip install numpy==1.26.4
bash run.sh theory
```

The checks cover the JS decomposition and sharp endpoints, invariant sets and constructive counterexamples, rank ranges, exact ties, and compact interval computation. Reports and the Figure 2 curve are generated under `runs/theory/`. Computation uses float64 with an explicit numerical tolerance.

## Code layout

```text
main.py                    Experiment entry point
run.sh                     Main experiment commands and model paths
data.py, prompts.py        Dataset preparation, prompts, and label parsing
models.py                  Retrieval, cross-fitted APMs, and LLM inference
selection.py               FPA, TopK, Label-Mass, and certificate audit
tail_certificates.py       Sharp bounds, invariance, and rank formulas
compact_certificates.py    Intervals without candidate-tail access
results.py                 Accuracy, paired statistics, and coverage summaries
common.py                  Input validation, model revisions, and file I/O
validate_*.py              Synthetic numerical verification
```
