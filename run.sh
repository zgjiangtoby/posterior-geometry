#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Model names can also be local checkpoint directories.
PYTHON="${PYTHON:-python}"
RETRIEVER="${RETRIEVER:-Alibaba-NLP/gte-base-en-v1.5}"
APM="${APM:-FacebookAI/roberta-base}"
QWEN="${QWEN:-Qwen/Qwen2.5-7B-Instruct}"
MISTRAL="${MISTRAL:-mistralai/Mistral-7B-Instruct-v0.3}"
DATA_DIR="${DATA_DIR:-data}"
RUN_DIR="${RUN_DIR:-runs}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export TOKENIZERS_PARALLELISM=false

mode="${1:-classification}"
if [[ "$mode" == "theory" ]]; then
    "$PYTHON" validate_theory.py
    "$PYTHON" validate_compact.py
    exit 0
elif [[ "$mode" == "classification" ]]; then
    read -r -a tasks <<< "${TASKS:-sst2 sst5 cr subj agnews mnli qnli banking77 massive}"
    read -r -a seeds <<< "${SEEDS:-521 997 2027}"
elif [[ "$mode" == "certificate" ]]; then
    read -r -a tasks <<< "${TASKS:-agnews mnli massive sst5 banking77}"
    read -r -a seeds <<< "${SEEDS:-4099 8191 16381}"
else
    echo "Usage: bash run.sh [classification|certificate|theory]" >&2
    exit 2
fi
read -r -a models <<< "${MODELS:-qwen mistral}"
for model in "${models[@]}"; do
    [[ "$model" == qwen || "$model" == mistral ]] || { echo "Unknown model: $model" >&2; exit 2; }
done
device=(--device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}")
common=(--data-dir "$DATA_DIR" --run-dir "$RUN_DIR")

for task in "${tasks[@]}"; do
    "$PYTHON" main.py data --task "$task" "${common[@]}"
    "$PYTHON" main.py retrieve --task "$task" "${common[@]}" "${device[@]}" --retriever "$RETRIEVER"
    for seed in "${seeds[@]}"; do
        "$PYTHON" main.py apm --task "$task" --seed "$seed" "${common[@]}" "${device[@]}" --apm "$APM" --workers "${WORKERS:-4}"
        if [[ "$mode" == "certificate" ]]; then
            audit_args=(--task "$task" --seed "$seed" --run-dir "$RUN_DIR")
            if [[ -n "${QUERY_REGISTRY:-}" ]]; then audit_args+=(--query-registry "$QUERY_REGISTRY"); fi
            "$PYTHON" main.py audit "${audit_args[@]}"
        else
            "$PYTHON" main.py select --task "$task" --seed "$seed" --run-dir "$RUN_DIR"
            for model in "${models[@]}"; do
                target="$QWEN"; if [[ "$model" == mistral ]]; then target="$MISTRAL"; fi
                "$PYTHON" main.py infer --task "$task" --seed "$seed" "${common[@]}" "${device[@]}" \
                    --model "$model" --target-model "$target" --batch-size "${BATCH_SIZE:-2}"
            done
        fi
    done
done
if [[ "$mode" == "certificate" ]]; then
    "$PYTHON" main.py coverage --run-dir "$RUN_DIR" --tasks "${tasks[@]}" --seeds "${seeds[@]}"
else
    "$PYTHON" main.py summarize --run-dir "$RUN_DIR" --tasks "${tasks[@]}" --seeds "${seeds[@]}" --models "${models[@]}"
fi
