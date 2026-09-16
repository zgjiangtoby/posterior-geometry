"""Pinned, normalized data loaders for the nine classification tasks.

Every task has the same three splits and every record has exactly the common
fields ``id``, ``text``, ``label``, and ``label_name``.  Pair-classification
tasks use a two-element JSON-compatible list for ``text``:

* MNLI: ``[premise, hypothesis]``
* QNLI: ``[sentence, question]``

Official GLUE test labels are unavailable.  Consequently MNLI/QNLI test
records use ``label=-1`` and ``label_name="__unlabeled__"``; the paper's
evaluation split for both tasks is validation.  MNLI matched and mismatched
partitions are deterministically concatenated (matched first).
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, TypeAlias

Record: TypeAlias = dict[str, Any]

TASK_NAMES: tuple[str, ...] = (
    "sst2",
    "sst5",
    "cr",
    "subj",
    "agnews",
    "mnli",
    "qnli",
    "banking77",
    "massive",
)

# Expected counts for the classification data protocol.
DATASET_STATS: dict[str, dict[str, int]] = {
    "sst2": {"train": 6_920, "validation": 872, "test": 1_821, "labels": 2},
    "sst5": {"train": 8_544, "validation": 1_101, "test": 2_210, "labels": 5},
    "cr": {"train": 3_394, "validation": 0, "test": 376, "labels": 2},
    "subj": {"train": 8_000, "validation": 0, "test": 2_000, "labels": 2},
    "agnews": {"train": 120_000, "validation": 0, "test": 7_600, "labels": 4},
    "mnli": {
        "train": 392_702,
        "validation": 19_647,
        "test": 19_643,
        "labels": 3,
    },
    "qnli": {"train": 104_743, "validation": 5_463, "test": 5_463, "labels": 2},
    "banking77": {"train": 10_003, "validation": 0, "test": 3_080, "labels": 77},
    "massive": {
        "train": 11_514,
        "validation": 2_033,
        "test": 2_974,
        "labels": 60,
    },
}

_HF_REVISIONS = {
    # Immutable Hugging Face dataset revisions used to reconstruct Table 1.
    "sst2": ("SetFit/sst2", None, "00ea8ccb7a54b4e3780a3e51aa3f80361ff849c0"),
    "sst5": ("SetFit/sst5", None, "e51bdcd8cd3a30da231967c1a249ba59361279a3"),
    "cr": ("SetFit/CR", None, "01e427e689e9d3a9097f85eab7a91ce937cf5f98"),
    "subj": ("SetFit/subj", None, "f3c1162e678417f664d76b21864fdb87b0615fcf"),
    "agnews": ("ag_news", None, "eb185aade064a813bc0b7f42de02595523103ca4"),
    "mnli": ("glue", "mnli", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
    "qnli": ("glue", "qnli", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
}

_LABEL_NAMES: dict[str, tuple[str, ...]] = {
    "sst2": ("negative", "positive"),
    # Appendix verbalizers; SetFit calls these very negative ... very positive.
    "sst5": ("terrible", "bad", "okay", "good", "great"),
    "cr": ("negative", "positive"),
    "subj": ("objective", "subjective"),
    "agnews": ("World", "Sports", "Business", "Sci/Tech"),
    "mnli": ("entailment", "neutral", "contradiction"),
    "qnli": ("entailment", "not_entailment"),
    "banking77": (
        "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
        "automatic_top_up", "balance_not_updated_after_bank_transfer",
        "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
        "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
        "card_delivery_estimate", "card_linking", "card_not_working",
        "card_payment_fee_charged", "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
        "contactless_not_working", "country_support", "declined_card_payment",
        "declined_cash_withdrawal", "declined_transfer",
        "direct_debit_payment_not_recognised", "disposable_card_limits",
        "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
        "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
        "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
        "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
        "order_physical_card", "passcode_forgotten", "pending_card_payment",
        "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
        "receiving_money", "Refund_not_showing_up", "request_refund",
        "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
        "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
        "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
        "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
        "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
        "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
        "why_verify_identity", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ),
    "massive": (
        "datetime_query", "iot_hue_lightchange", "transport_ticket", "takeaway_query",
        "qa_stock", "general_greet", "recommendation_events", "music_dislikeness",
        "iot_wemo_off", "cooking_recipe", "qa_currency", "transport_traffic",
        "general_quirky", "weather_query", "audio_volume_up", "email_addcontact",
        "takeaway_order", "email_querycontact", "iot_hue_lightup",
        "recommendation_locations", "play_audiobook", "lists_createoradd", "news_query",
        "alarm_query", "iot_wemo_on", "general_joke", "qa_definition", "social_query",
        "music_settings", "audio_volume_other", "calendar_remove", "iot_hue_lightdim",
        "calendar_query", "email_sendemail", "iot_cleaning", "audio_volume_down",
        "play_radio", "cooking_query", "datetime_convert", "qa_maths",
        "iot_hue_lightoff", "iot_hue_lighton", "transport_query", "music_likeness",
        "email_query", "play_music", "audio_volume_mute", "social_post", "alarm_set",
        "qa_factoid", "calendar_set", "play_game", "alarm_remove", "lists_remove",
        "transport_taxi", "recommendation_movies", "iot_coffee", "music_query",
        "play_podcasts", "lists_query",
    ),
}


@dataclass(slots=True)
class TaskData:
    """Normalized in-memory representation consumed by experiment code."""

    name: str
    label_names: tuple[str, ...]
    splits: dict[str, list[Record]]
    source: dict[str, Any]
    evaluation_split: str

    def __getitem__(self, split: str) -> list[Record]:
        return self.splits[split]

    @property
    def train(self) -> list[Record]:
        return self.splits["train"]

    @property
    def validation(self) -> list[Record]:
        return self.splits["validation"]

    @property
    def test(self) -> list[Record]:
        return self.splits["test"]


def _hf_dataset(task_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on minimal systems
        raise RuntimeError(
            'Hugging Face datasets is required. Install the datasets package in the Python environment used to run this command.'
        ) from exc
    repo, config, revision = _HF_REVISIONS[task_name]
    kwargs: dict[str, Any] = {"revision": revision}
    return load_dataset(repo, config, **kwargs) if config else load_dataset(repo, **kwargs)


def _record(task: str, split: str, source_id: str, text: str | list[str], label: int) -> Record:
    names = _LABEL_NAMES[task]
    label_name = "__unlabeled__" if label == -1 else names[label]
    return {
        "id": f"{task}:{split}:{source_id}",
        "text": text,
        "label": label,
        "label_name": label_name,
    }


def _convert_rows(
    task: str,
    split: str,
    rows: Iterable[Mapping[str, Any]],
    text_fields: tuple[str, ...],
    id_prefix: str = "",
) -> list[Record]:
    result: list[Record] = []
    for position, row in enumerate(rows):
        source_id = str(row.get("idx", position))
        if id_prefix:
            source_id = f"{id_prefix}:{source_id}"
        values = [str(row[field]) for field in text_fields]
        text: str | list[str] = values[0] if len(values) == 1 else values
        result.append(_record(task, split, source_id, text, int(row["label"])))
    return result


def _load_hf_task(task_name: str) -> TaskData:
    dataset = _hf_dataset(task_name)
    if task_name in {"sst2", "sst5", "cr", "subj", "agnews"}:
        split_map = {
            "train": "train",
            "validation": "validation" if "validation" in dataset else None,
            "test": "test",
        }
        splits = {
            target: (
                _convert_rows(task_name, target, dataset[source], ("text",))
                if source is not None
                else []
            )
            for target, source in split_map.items()
        }
    elif task_name == "mnli":
        splits = {
            "train": _convert_rows("mnli", "train", dataset["train"], ("premise", "hypothesis")),
            "validation": (
                _convert_rows(
                    "mnli", "validation", dataset["validation_matched"],
                    ("premise", "hypothesis"), "matched",
                )
                + _convert_rows(
                    "mnli", "validation", dataset["validation_mismatched"],
                    ("premise", "hypothesis"), "mismatched",
                )
            ),
            "test": (
                _convert_rows(
                    "mnli", "test", dataset["test_matched"],
                    ("premise", "hypothesis"), "matched",
                )
                + _convert_rows(
                    "mnli", "test", dataset["test_mismatched"],
                    ("premise", "hypothesis"), "mismatched",
                )
            ),
        }
    else:  # QNLI
        splits = {
            split: _convert_rows(
                "qnli", split, dataset[split], ("sentence", "question")
            )
            for split in ("train", "validation", "test")
        }
    repo, config, revision = _HF_REVISIONS[task_name]
    return TaskData(
        name=task_name,
        label_names=_LABEL_NAMES[task_name],
        splits=splits,
        source={"kind": "huggingface", "dataset": repo, "config": config, "revision": revision},
        evaluation_split="validation" if task_name in {"mnli", "qnli"} else "test",
    )


_BANKING_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
_BANKING_BASE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{_BANKING_COMMIT}/banking_data"
)
_MASSIVE_PARQUET = {
    # HF dataset-server's script conversion of AmazonScience/massive en-US.
    # Individual files avoid downloading the entire 52-language v1.1 archive.
    "train": (
        "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
        "refs%2Fconvert%2Fparquet/en-US/train/0000.parquet",
        "9c8b3ce8a96ec3b0d4aee104d8778d9f6bacdbd18f2cea232f6d8333d24401b1",
    ),
    "validation": (
        "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
        "refs%2Fconvert%2Fparquet/en-US/validation/0000.parquet",
        "cf22bf142963c86ee164c5bb1fe22c67c1261df5c52dfd7217ea87b977c7bb43",
    ),
    "test": (
        "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
        "refs%2Fconvert%2Fparquet/en-US/test/0000.parquet",
        "c418da9a5f3a7425b5cf44941bcea032a31040dbda2da85e0dd9323785c7731e",
    ),
}


def _raw_cache_dir() -> Path:
    root = os.environ.get("PG_DATA_CACHE")
    path = Path(root) if root else Path.home() / ".cache" / "posterior_geometry" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, filename: str, expected_sha256: str | None = None) -> Path:
    destination = _raw_cache_dir() / filename
    # Multiple seed jobs commonly start together.  Serialize cache fills and
    # re-check after acquiring the lock so they never write the same partial.
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size:
            if expected_sha256 is None or _sha256(destination) == expected_sha256:
                return destination
            destination.unlink()
        temporary = destination.with_suffix(destination.suffix + ".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "posterior-geometry/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        if expected_sha256 is not None:
            observed = _sha256(destination)
            if observed != expected_sha256:
                destination.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Checksum mismatch for {url}: expected {expected_sha256}, got {observed}"
                )
        return destination


def _load_banking77() -> TaskData:
    label_to_id = {name: idx for idx, name in enumerate(_LABEL_NAMES["banking77"])}
    splits: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    for split in ("train", "test"):
        path = _download(f"{_BANKING_BASE}/{split}.csv", f"banking77-{_BANKING_COMMIT}-{split}.csv")
        with path.open(encoding="utf-8", newline="") as stream:
            for position, row in enumerate(csv.DictReader(stream)):
                label_name = row["category"]
                # The historical HF builder used two spelling/case quirks.
                if label_name == "refund_not_showing_up":
                    label_name = "Refund_not_showing_up"
                elif label_name == "reverted_card_payment?":
                    label_name = "reverted_card_payment?"
                label = label_to_id[label_name]
                splits[split].append(_record("banking77", split, str(position), row["text"], label))
    return TaskData(
        name="banking77",
        label_names=_LABEL_NAMES["banking77"],
        splits=splits,
        source={
            "kind": "canonical_raw_fallback",
            "huggingface_dataset": "PolyAI/banking77",
            "upstream_commit": _BANKING_COMMIT,
            "urls": [f"{_BANKING_BASE}/train.csv", f"{_BANKING_BASE}/test.csv"],
            "note": "datasets>=4 cannot execute the repository's legacy dataset script",
        },
        evaluation_split="test",
    )


def _load_massive() -> TaskData:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - datasets installs pyarrow
        raise RuntimeError("pyarrow is required to read MASSIVE parquet files") from exc
    splits: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    urls: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for split, (url, checksum) in _MASSIVE_PARQUET.items():
        path = _download(url, f"massive-en-US-{split}.parquet", checksum)
        urls[split] = url
        checksums[split] = checksum
        table = parquet.read_table(path, columns=["id", "intent", "utt"])
        for batch in table.to_batches(max_chunksize=8192):
            for row in batch.to_pylist():
                splits[split].append(
                    _record("massive", split, str(row["id"]), row["utt"], int(row["intent"]))
                )
    return TaskData(
        name="massive",
        label_names=_LABEL_NAMES["massive"],
        splits=splits,
        source={
            "kind": "canonical_raw_fallback",
            "huggingface_dataset": "AmazonScience/massive",
            "huggingface_config": "en-US",
            "dataset_version": "1.1",
            "parquet_urls": urls,
            "sha256": checksums,
            "note": "datasets>=4 cannot execute the repository's legacy dataset script",
        },
        evaluation_split="test",
    )


def load_task(task_name: str, *, verify: bool = True) -> TaskData:
    """Load one normalized classification task.

    Args:
        task_name: One of :data:`TASK_NAMES` (case-insensitive; ``sst-2``,
            ``sst-5``, ``ag_news``, and ``massive_en_us`` aliases are accepted).
        verify: Enforce the expected split sizes and record schema.
    """

    aliases = {
        "sst-2": "sst2", "sst-5": "sst5", "ag_news": "agnews",
        "banking-77": "banking77", "massive_en_us": "massive", "massive-en-us": "massive",
    }
    normalized = task_name.strip().lower()
    normalized = aliases.get(normalized, normalized)
    if normalized not in TASK_NAMES:
        raise ValueError(f"Unknown task {task_name!r}; choose one of {', '.join(TASK_NAMES)}")
    if normalized == "banking77":
        task = _load_banking77()
    elif normalized == "massive":
        task = _load_massive()
    else:
        task = _load_hf_task(normalized)
    if verify:
        verify_task(task)
    return task


def verify_task(task: TaskData) -> dict[str, int]:
    """Raise ``AssertionError`` on any schema/count/label mismatch."""

    expected = DATASET_STATS[task.name]
    assert set(task.splits) == {"train", "validation", "test"}
    assert len(task.label_names) == expected["labels"]
    observed: dict[str, int] = {}
    all_ids: set[str] = set()
    for split in ("train", "validation", "test"):
        records = task.splits[split]
        observed[split] = len(records)
        assert len(records) == expected[split], (
            f"{task.name}/{split}: expected {expected[split]:,}, got {len(records):,}"
        )
        for row in records:
            assert set(row) == {"id", "text", "label", "label_name"}
            assert isinstance(row["id"], str) and row["id"]
            assert row["id"] not in all_ids, f"Duplicate id: {row['id']}"
            all_ids.add(row["id"])
            text = row["text"]
            assert (
                isinstance(text, str)
                or (isinstance(text, list) and len(text) == 2 and all(isinstance(x, str) for x in text))
            )
            assert text and (not isinstance(text, list) or all(text))
            label = row["label"]
            assert isinstance(label, int)
            if label == -1:
                assert task.name in {"mnli", "qnli"} and split == "test"
                assert row["label_name"] == "__unlabeled__"
            else:
                assert 0 <= label < len(task.label_names)
                assert row["label_name"] == task.label_names[label]
    return observed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Record]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return count, _sha256(path)


def materialize_task(task: TaskData, output_dir: str | Path) -> Path:
    """Write ``train/validation/test.jsonl`` and a checksum metadata file."""

    verify_task(task)
    task_dir = Path(output_dir) / task.name
    task_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        path = task_dir / f"{split}.jsonl"
        count, checksum = _atomic_jsonl(path, task.splits[split])
        files[split] = {"file": path.name, "rows": count, "sha256": checksum}
    metadata = {
        "schema_version": 1,
        "task": task.name,
        "record_fields": ["id", "text", "label", "label_name"],
        "pair_text_order": (
            ["premise", "hypothesis"] if task.name == "mnli"
            else ["sentence", "question"] if task.name == "qnli" else None
        ),
        "label_names": list(task.label_names),
        "evaluation_split": task.evaluation_split,
        "source": task.source,
        "files": files,
    }
    metadata_path = task_dir / "metadata.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".metadata.", dir=task_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary_name, metadata_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return metadata_path
