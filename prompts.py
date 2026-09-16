"""Task prompt templates and deterministic prediction parsing."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from data import Record, _LABEL_NAMES

_TASK_ALIASES = {
    "sst-2": "sst2", "sst-5": "sst5", "ag_news": "agnews",
    "banking-77": "banking77", "massive_en_us": "massive", "massive-en-us": "massive",
}

# Task-specific prediction strings. Dataset label names remain
# canonical in data.py; this table is the explicit output verbalizer mapping.
_VERBALIZERS: dict[str, tuple[str, ...]] = {
    "sst2": ("negative", "positive"),
    "sst5": ("terrible", "bad", "okay", "good", "great"),
    "cr": ("negative", "positive"),
    "subj": ("objective", "subjective"),
    "agnews": ("world", "sports", "business", "science and technology"),
    "mnli": ("Yes", "Maybe", "No"),
    "qnli": ("Yes", "No"),
    "banking77": tuple(name.replace("_", " ").rstrip("?") for name in _LABEL_NAMES["banking77"]),
    "massive": tuple(name.replace("_", " ") for name in _LABEL_NAMES["massive"]),
}


def _task_name(task_name: str) -> str:
    name = task_name.strip().lower()
    return _TASK_ALIASES.get(name, name)


def label_verbalizers(
    task_name: str,
    *,
    label_mapping: Mapping[int, str] | None = None,
) -> tuple[str, ...]:
    """Return the ordered output strings used by prompts and decoding."""
    task = _task_name(task_name)
    if task not in _VERBALIZERS:
        raise ValueError(f"Unknown task {task_name!r}")
    return _label_words(task, label_mapping)


def _text_parts(task: str, value: Record | str | Sequence[str]) -> tuple[str, ...]:
    text: Any = value.get("text") if isinstance(value, Mapping) else value
    if task in {"mnli", "qnli"}:
        if not isinstance(text, Sequence) or isinstance(text, str) or len(text) != 2:
            raise ValueError(f"{task} text must be a two-element sequence")
        return str(text[0]), str(text[1])
    if not isinstance(text, str):
        raise ValueError(f"{task} text must be a string")
    return (text,)


def _label_words(task: str, label_mapping: Mapping[int, str] | None) -> tuple[str, ...]:
    if label_mapping is None:
        return _VERBALIZERS[task]
    expected = set(range(len(_LABEL_NAMES[task])))
    if set(label_mapping) != expected:
        raise ValueError(f"label_mapping keys must be exactly {sorted(expected)}")
    return tuple(str(label_mapping[idx]) for idx in range(len(expected)))


def prompt_template(task_name: str) -> str:
    """Return the literal placeholder template used by the classification experiments."""

    task = _task_name(task_name)
    templates = {
        "sst2": 'Review: "<X>" Sentiment: <label>',
        "sst5": 'Review: "<X>" Sentiment: <label>',
        "cr": 'Review: "<X>" Sentiment: <label>',
        "subj": 'Input: "<X>" Type: <label>',
        "agnews": '"<X>" It is about <label>.',
        "mnli": '"<C>" Can we know "<X>"? <label>',
        "qnli": '"<C>" Can we know "<X>"? <label>',
        "banking77": (
            '"<X>" Which banking customer-service intent is this query about? '
            'Choose one of: <label list>. <label>'
        ),
        "massive": (
            '"<X>" Which virtual-assistant intent is this utterance about? '
            'Choose one of: <label list>. <label>'
        ),
    }
    try:
        return templates[task]
    except KeyError as exc:
        raise ValueError(f"Unknown task {task_name!r}") from exc


def _prefix(task: str, value: Record | str | Sequence[str], labels: tuple[str, ...]) -> str:
    parts = _text_parts(task, value)
    if task in {"sst2", "sst5", "cr"}:
        return f'Review: "{parts[0]}" Sentiment:'
    if task == "subj":
        return f'Input: "{parts[0]}" Type:'
    if task == "agnews":
        return f'"{parts[0]}" It is about'
    if task in {"mnli", "qnli"}:
        return f'"{parts[0]}" Can we know "{parts[1]}"?'
    label_list = ", ".join(labels)
    if task == "banking77":
        return (
            f'"{parts[0]}" Which banking customer-service intent is this query about? '
            f"Choose one of: {label_list}."
        )
    if task == "massive":
        return (
            f'"{parts[0]}" Which virtual-assistant intent is this utterance about? '
            f"Choose one of: {label_list}."
        )
    raise ValueError(f"Unknown task {task!r}")


def format_example(
    task_name: str,
    example: Record,
    *,
    include_answer: bool = True,
    label_mapping: Mapping[int, str] | None = None,
) -> str:
    """Format one demonstration or answer-free query using the appendix template."""

    task = _task_name(task_name)
    labels = _label_words(task, label_mapping)
    prefix = _prefix(task, example, labels)
    if not include_answer:
        return prefix + (" " if task == "agnews" else "")
    label = int(example["label"])
    if not 0 <= label < len(labels):
        raise ValueError("Cannot format an unlabeled record as a demonstration")
    suffix = labels[label]
    if task == "agnews":
        return f"{prefix} {suffix}."
    return f"{prefix} {suffix}"


def build_prompt(
    task_name: str,
    demonstrations: Sequence[Record],
    query: Record | str | Sequence[str],
    *,
    label_mapping: Mapping[int, str] | None = None,
    separator: str = "\n\n",
) -> str:
    """Build an ICL prompt, preserving the selector-provided demonstration order."""

    blocks = [
        format_example(task_name, demo, include_answer=True, label_mapping=label_mapping)
        for demo in demonstrations
    ]
    query_record: Record = query if isinstance(query, Mapping) else {"text": query}  # type: ignore[assignment]
    blocks.append(
        format_example(task_name, query_record, include_answer=False, label_mapping=label_mapping)
    )
    return separator.join(blocks)


def _normalized(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower().strip()
    value = value.replace("_", " ").replace("/", " ")
    return re.sub(r"\s+", " ", value)


def _aliases(
    task: str,
    labels: tuple[str, ...],
    *,
    include_canonical: bool,
) -> dict[int, set[str]]:
    result = {idx: {_normalized(verbalizer)} for idx, verbalizer in enumerate(labels)}
    if include_canonical:
        for idx, canonical in enumerate(_LABEL_NAMES[task]):
            result[idx].add(_normalized(canonical))
    if include_canonical and task == "sst5":
        extras = ("very negative", "negative", "neutral", "positive", "very positive")
        for idx, alias in enumerate(extras):
            result[idx].add(alias)
    elif include_canonical and task == "agnews":
        result[0].update({"world news", "international"})
        result[3].update({"sci tech", "science technology", "technology"})
    elif include_canonical and task == "mnli":
        result[0].update({"yes", "entails", "entailed"})
        result[1].update({"maybe", "unknown"})
        result[2].update({"no", "contradicts", "contradicted"})
    elif include_canonical and task == "qnli":
        result[0].update({"yes", "entails", "entailed"})
        result[1].update({"no", "not entailment", "does not entail"})
    return result


def parse_prediction(
    task_name: str,
    prediction: str,
    *,
    label_mapping: Mapping[int, str] | None = None,
) -> int | None:
    """Parse a generated answer into a label ID without substring collisions.

    The parser accepts appendix verbalizers, canonical dataset label names, and
    common MNLI/QNLI/SST-5 aliases.  It returns ``None`` instead of guessing
    when no label is present or distinct labels first occur at the same offset.
    """

    task = _task_name(task_name)
    labels = _label_words(task, label_mapping)
    text = _normalized(prediction)[:4096]
    # Decoder outputs occasionally include an answer cue or an echoed template.
    cue_matches = list(re.finditer(r"(?:sentiment|type|answer|intent)\s*:\s*", text))
    if cue_matches:
        text = text[cue_matches[-1].end():]
    exact = text.strip(" \t\r\n.!,;:'\"`()[]{}")
    hits: list[tuple[int, int, int]] = []
    for label_id, variants in _aliases(
        task, labels, include_canonical=label_mapping is None
    ).items():
        for variant in variants:
            if not variant:
                continue
            match = re.search(rf"(?<![\w]){re.escape(variant)}(?![\w])", text)
            if match:
                hits.append((match.start(), -len(variant), label_id))
    if not hits:
        return None
    hits.sort()
    best_position = hits[0][0]
    labels_at_best = {label for position, _, label in hits if position == best_position}
    if len(labels_at_best) != 1:
        return None
    # Longest match wins for nested labels such as entailment/not entailment.
    return min((hit for hit in hits if hit[0] == best_position), key=lambda hit: hit[1])[2]
