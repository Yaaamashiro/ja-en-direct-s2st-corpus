from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tarfile
import time
import urllib.request
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from .config import AppConfig, CorpusSourceConfig
from .io import atomic_write_jsonl
from .normalization import minimal_normalize


_URL_OR_EMAIL = re.compile(
    r"(?:https?://|www\.|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b)",
    re.IGNORECASE,
)
_MARKUP = re.compile(r"<[^>]+>|&(?:nbsp|lt|gt|amp|quot|apos);", re.IGNORECASE)
_CUE_ONLY = re.compile(
    r"^\s*[\[(（【]?\s*"
    r"(?:music|applause|laughter|laughs|sighs?|gasps?|noise|silence|"
    r"音楽|拍手|笑い|ため息|歓声|雑音)"
    r"\s*[\])）】]?\s*[.!。…-]*\s*$",
    re.IGNORECASE,
)
_JA_CHARACTER = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_EN_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_CONTENT = re.compile(r"[A-Za-z\u3040-\u30ff\u3400-\u9fff]")
_PUNCTUATION = re.compile(r"[^\w\u3040-\u30ff\u3400-\u9fff]+")
_PREPARATION_ALGORITHM_VERSION = "prepare-v1"

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    pair_id TEXT PRIMARY KEY,
    corpus TEXT NOT NULL,
    original_split TEXT NOT NULL,
    source_id TEXT NOT NULL,
    license_id TEXT NOT NULL,
    ja_text_raw TEXT NOT NULL,
    en_text_raw TEXT NOT NULL,
    ja_text_norm TEXT NOT NULL,
    en_text_norm TEXT NOT NULL,
    fingerprint TEXT NOT NULL UNIQUE,
    priority TEXT NOT NULL,
    ja_chars INTEGER NOT NULL,
    en_words INTEGER NOT NULL,
    length_bin TEXT NOT NULL,
    estimated_ja_seconds REAL NOT NULL,
    selected INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS candidate_selection
ON candidates(corpus, original_split, length_bin, selected, priority);
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _download(
    source_name: str,
    source: CorpusSourceConfig,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        digest = sha256_file(destination)
        if source.sha256 is not None and digest != source.sha256:
            raise RuntimeError(
                f"{destination} exists but SHA-256 is {digest}; "
                f"expected {source.sha256}. Move the file aside and retry."
            )
        return {
            "source": source_name,
            "url": source.url,
            "archive": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": digest,
            "reused": True,
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        if not temporary.is_file():
            raise RuntimeError(
                f"incomplete download path is not a file: {temporary}"
            )
        print(f"[download] discarding incomplete file: {temporary}")
        temporary.unlink()
    print(f"[download] {source_name}: {source.url}")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "ja-en-direct-s2st-corpus/0.1"},
    )
    started = time.time()
    with closing(urllib.request.urlopen(request, timeout=60)) as response:
        with temporary.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    digest = sha256_file(temporary)
    if source.sha256 is not None and digest != source.sha256:
        raise RuntimeError(
            f"downloaded {source_name} SHA-256 is {digest}; "
            f"expected {source.sha256}"
        )
    temporary.replace(destination)
    return {
        "source": source_name,
        "url": source.url,
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": digest,
        "reused": False,
        "download_seconds": round(time.time() - started, 3),
    }


def _safe_tar_members(archive: tarfile.TarFile, destination: Path) -> list[tarfile.TarInfo]:
    destination_resolved = destination.resolve()
    safe: list[tarfile.TarInfo] = []
    for member in archive.getmembers():
        pure = PurePosixPath(member.name)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise RuntimeError(f"unsafe archive member: {member.name!r}")
        target = (destination / Path(*pure.parts)).resolve()
        try:
            target.relative_to(destination_resolved)
        except ValueError as error:
            raise RuntimeError(
                f"archive member escapes destination: {member.name!r}"
            ) from error
        safe.append(member)
    return safe


def _extract(
    source_name: str,
    archive_path: Path,
    work_dir: Path,
    archive_sha256: str,
) -> Path:
    destination = work_dir / "extracted" / f"{source_name}-{archive_sha256[:16]}"
    marker = destination / ".complete.json"
    if marker.is_file():
        return destination
    if destination.exists():
        raise RuntimeError(
            f"incomplete extraction exists at {destination}; move it aside and retry"
        )
    temporary = destination.with_name(destination.name + ".extracting")
    if temporary.exists():
        raise RuntimeError(
            f"incomplete extraction exists at {temporary}; move it aside and retry"
        )
    temporary.mkdir(parents=True)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = _safe_tar_members(archive, temporary)
            archive.extractall(temporary, members=members)
        _atomic_json(
            temporary / ".complete.json",
            {
                "source": source_name,
                "archive_sha256": archive_sha256,
                "members": len(members),
            },
        )
        temporary.replace(destination)
    except BaseException:
        # Do not delete a potentially useful forensic artifact automatically.
        raise
    return destination


def _find_one(root: Path, relative_candidates: Iterable[str]) -> Path:
    for relative in relative_candidates:
        candidate = root / relative
        if candidate.is_file():
            return candidate
    names = ", ".join(relative_candidates)
    raise FileNotFoundError(f"none of [{names}] found under {root}")


def iter_jesc(
    root: Path,
    splits: Iterable[str] = ("dev", "test", "train"),
) -> Iterator[tuple[str, int, str, str]]:
    for split in splits:
        path = _find_one(root, (f"split/{split}", split))
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = line.rstrip("\r\n").split("\t")
                if len(row) != 2:
                    raise ValueError(
                        f"{path}:{line_number}: expected 2 TSV columns, got {len(row)}"
                    )
                # The official JESC split is English in column 1 and Japanese in 2.
                yield split, line_number, row[1], row[0]


def iter_kftt(
    root: Path,
    splits: Iterable[str] = ("dev", "test", "train"),
) -> Iterator[tuple[str, int, str, str]]:
    base_candidates = (
        "kftt-data-1.0/data/orig",
        "data/orig",
    )
    base: Path | None = None
    for relative in base_candidates:
        candidate = root / relative
        if candidate.is_dir():
            base = candidate
            break
    if base is None:
        raise FileNotFoundError(f"KFTT data/orig directory not found under {root}")
    for split in splits:
        ja_path = base / f"kyoto-{split}.ja"
        en_path = base / f"kyoto-{split}.en"
        if not ja_path.is_file() or not en_path.is_file():
            raise FileNotFoundError(f"missing KFTT {split} language files in {base}")
        with ja_path.open("r", encoding="utf-8-sig") as ja_handle:
            with en_path.open("r", encoding="utf-8-sig") as en_handle:
                line_number = 0
                while True:
                    ja_line = ja_handle.readline()
                    en_line = en_handle.readline()
                    if not ja_line and not en_line:
                        break
                    line_number += 1
                    if not ja_line or not en_line:
                        raise ValueError(
                            f"KFTT {split} files have different line counts"
                        )
                    yield (
                        split,
                        line_number,
                        ja_line.rstrip("\r\n"),
                        en_line.rstrip("\r\n"),
                    )


def _text_fingerprint(ja_text: str, en_text: str) -> str:
    payload = f"{ja_text}\0{en_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pair_id(
    corpus: str,
    split: str,
    source_index: int,
    fingerprint: str,
) -> str:
    return f"{corpus}_{split}_{source_index:08d}_{fingerprint[:12]}"


def _selection_priority(seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{pair_id}".encode("utf-8")).hexdigest()


def _length_bin(
    ja_chars: int,
    short_max: int,
    medium_max: int,
) -> str:
    if ja_chars <= short_max:
        return "short"
    if ja_chars <= medium_max:
        return "medium"
    return "long"


def filter_pair(
    corpus: str,
    ja_raw: str,
    en_raw: str,
) -> tuple[str | None, str, str, int, int]:
    ja = minimal_normalize(ja_raw)
    en = minimal_normalize(en_raw)
    if not ja or not en:
        return "empty", ja, en, 0, 0
    if _URL_OR_EMAIL.search(ja) or _URL_OR_EMAIL.search(en):
        return "url_or_email", ja, en, 0, 0
    if _MARKUP.search(ja) or _MARKUP.search(en):
        return "markup", ja, en, 0, 0
    if _CUE_ONLY.fullmatch(ja) or _CUE_ONLY.fullmatch(en):
        return "subtitle_cue", ja, en, 0, 0
    if not _CONTENT.search(ja) or not _CONTENT.search(en):
        return "no_content", ja, en, 0, 0

    ja_chars = len(_PUNCTUATION.sub("", ja))
    en_words = len(_EN_WORD.findall(en))
    if ja_chars < 8 or en_words < 4:
        return "too_short", ja, en, ja_chars, en_words
    max_ja = 160 if corpus == "jesc" else 180
    max_en = 50 if corpus == "jesc" else 60
    if ja_chars > max_ja or en_words > max_en:
        return "too_long", ja, en, ja_chars, en_words

    ja_script_chars = len(_JA_CHARACTER.findall(ja))
    if ja_script_chars / max(ja_chars, 1) < 0.45:
        return "not_japanese", ja, en, ja_chars, en_words
    latin_chars = sum(character.isascii() and character.isalpha() for character in en)
    en_letters = sum(character.isalpha() for character in en)
    if latin_chars / max(en_letters, 1) < 0.80:
        return "not_english", ja, en, ja_chars, en_words

    ratio = ja_chars / max(en_words, 1)
    if not 0.8 <= ratio <= 8.0:
        return "length_ratio", ja, en, ja_chars, en_words
    return None, ja, en, ja_chars, en_words


def _create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(_CREATE_SCHEMA)
    return connection


def _reset_database(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS candidates")
    connection.executescript(_CREATE_SCHEMA)


def _ingest(
    connection: sqlite3.Connection,
    config: AppConfig,
    corpus: str,
    rows: Iterable[tuple[str, int, str, str]],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    inserted = 0
    duplicates = 0
    source = config.sources[corpus]
    batch: list[tuple[Any, ...]] = []

    def flush() -> None:
        nonlocal inserted, duplicates
        if not batch:
            return
        before = connection.total_changes
        connection.executemany(
            """
            INSERT OR IGNORE INTO candidates (
                pair_id, corpus, original_split, source_id, license_id,
                ja_text_raw, en_text_raw, ja_text_norm, en_text_norm,
                fingerprint, priority, ja_chars, en_words, length_bin,
                estimated_ja_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        added = connection.total_changes - before
        inserted += added
        duplicates += len(batch) - added
        batch.clear()
        connection.commit()

    for processed, (split, source_index, ja_raw, en_raw) in enumerate(rows, start=1):
        if split == "dev" and not config.prepare.include_dev:
            continue
        if split == "test" and not config.prepare.include_test:
            continue
        reason, ja, en, ja_chars, en_words = filter_pair(
            corpus, ja_raw, en_raw
        )
        if reason is not None:
            counts[reason] = counts.get(reason, 0) + 1
            continue
        fingerprint = _text_fingerprint(ja, en)
        pair_id = _pair_id(corpus, split, source_index, fingerprint)
        estimated = (
            ja_chars / config.prepare.ja_chars_per_second
            + config.prepare.utterance_padding_seconds
        )
        length_bin = _length_bin(
            ja_chars,
            config.prepare.short_max_ja_chars,
            config.prepare.medium_max_ja_chars,
        )
        batch.append(
            (
                pair_id,
                corpus,
                split,
                f"{corpus}:{split}:{source_index}",
                source.license_id,
                ja_raw,
                en_raw,
                ja,
                en,
                fingerprint,
                _selection_priority(config.prepare.selection_seed, pair_id),
                ja_chars,
                en_words,
                length_bin,
                estimated,
            )
        )
        if len(batch) >= 20_000:
            flush()
        if processed % 100_000 == 0:
            print(
                f"[prepare] {corpus}: processed={processed:,} "
                f"eligible={inserted:,} duplicates={duplicates:,}"
            )
    flush()
    return {
        "eligible": inserted,
        "duplicates": duplicates,
        "filtered": dict(sorted(counts.items())),
    }


def _select_until_hours(
    connection: sqlite3.Connection,
    corpus: str,
    target_hours: float,
    length_bin: str | None,
) -> tuple[int, float]:
    clauses = ["corpus = ?", "original_split = 'train'", "selected = 0"]
    parameters: list[Any] = [corpus]
    if length_bin is not None:
        clauses.append("length_bin = ?")
        parameters.append(length_bin)
    query = (
        "SELECT pair_id, estimated_ja_seconds FROM candidates WHERE "
        + " AND ".join(clauses)
        + " ORDER BY priority"
    )
    selected_ids: list[tuple[str]] = []
    seconds = 0.0
    target_seconds = target_hours * 3600
    for pair_id, estimated in connection.execute(query, parameters):
        if seconds >= target_seconds:
            break
        selected_ids.append((pair_id,))
        seconds += float(estimated)
    if selected_ids:
        connection.executemany(
            "UPDATE candidates SET selected = 1 WHERE pair_id = ?",
            selected_ids,
        )
    connection.commit()
    return (
        int(
            connection.execute(
                "SELECT COUNT(*) FROM candidates "
                "WHERE corpus = ? AND original_split = 'train' AND selected = 1",
                (corpus,),
            ).fetchone()[0]
        ),
        seconds,
    )


def _select_training(
    connection: sqlite3.Connection,
    config: AppConfig,
) -> dict[str, Any]:
    fractions = {
        "short": config.prepare.short_time_fraction,
        "medium": config.prepare.medium_time_fraction,
        "long": config.prepare.long_time_fraction,
    }
    corpus_targets = {
        "jesc": config.prepare.target_train_hours * config.prepare.jesc_fraction,
        "kftt": config.prepare.target_train_hours
        * (1 - config.prepare.jesc_fraction),
    }
    summary: dict[str, Any] = {}
    for corpus, target_hours in corpus_targets.items():
        for bin_name, fraction in fractions.items():
            _select_until_hours(
                connection,
                corpus,
                target_hours * fraction,
                bin_name,
            )
        selected_seconds = float(
            connection.execute(
                "SELECT COALESCE(SUM(estimated_ja_seconds), 0) "
                "FROM candidates WHERE corpus = ? AND original_split = 'train' "
                "AND selected = 1",
                (corpus,),
            ).fetchone()[0]
        )
        if selected_seconds < target_hours * 3600:
            _select_until_hours(
                connection,
                corpus,
                target_hours - selected_seconds / 3600,
                None,
            )
            selected_seconds = float(
                connection.execute(
                    "SELECT COALESCE(SUM(estimated_ja_seconds), 0) "
                    "FROM candidates WHERE corpus = ? AND original_split = 'train' "
                    "AND selected = 1",
                    (corpus,),
                ).fetchone()[0]
            )
        if selected_seconds < target_hours * 3600:
            available = float(
                connection.execute(
                    "SELECT COALESCE(SUM(estimated_ja_seconds), 0) "
                    "FROM candidates WHERE corpus = ? AND original_split = 'train'",
                    (corpus,),
                ).fetchone()[0]
            )
            raise RuntimeError(
                f"{corpus} has only {available / 3600:.2f} estimated train hours; "
                f"{target_hours:.2f} are required"
            )
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM candidates "
                "WHERE corpus = ? AND original_split = 'train' AND selected = 1",
                (corpus,),
            ).fetchone()[0]
        )
        selected_by_bin = {
            bin_name: seconds / 3600
            for bin_name, seconds in connection.execute(
                "SELECT length_bin, SUM(estimated_ja_seconds) "
                "FROM candidates WHERE corpus = ? "
                "AND original_split = 'train' AND selected = 1 "
                "GROUP BY length_bin",
                (corpus,),
            )
        }
        summary[corpus] = {
            "target_hours": target_hours,
            "estimated_selected_hours": selected_seconds / 3600,
            "selected_pairs": count,
            "estimated_hours_by_length_bin": selected_by_bin,
        }
    return summary


def _manifest_rows(
    connection: sqlite3.Connection,
    config: AppConfig,
) -> Iterator[dict[str, Any]]:
    query = """
    SELECT pair_id, corpus, original_split, source_id, license_id,
           ja_text_raw, en_text_raw, ja_text_norm, en_text_norm,
           priority, ja_chars, en_words, length_bin, estimated_ja_seconds
    FROM candidates
    WHERE selected = 1 OR original_split != 'train'
    ORDER BY
        CASE original_split WHEN 'train' THEN 0 WHEN 'dev' THEN 1 ELSE 2 END,
        corpus,
        priority
    """
    columns = [
        "pair_id",
        "corpus",
        "split",
        "source_id",
        "license_id",
        "ja_text_raw",
        "en_text_raw",
        "ja_text",
        "en_text",
        "selection_priority",
        "ja_chars",
        "en_words",
        "length_bin",
        "estimated_ja_seconds",
    ]
    for values in connection.execute(query):
        row = dict(zip(columns, values, strict=True))
        row["corpus_version"] = config.sources[row["corpus"]].version
        row["normalizer_version"] = "minimal-nfkc-v1"
        yield row


def _config_fingerprint(config: AppConfig) -> str:
    payload = {
        "algorithm_version": _PREPARATION_ALGORITHM_VERSION,
        "prepare": {
            **asdict(config.prepare),
            "sources_dir": str(config.prepare.sources_dir),
            "work_dir": str(config.prepare.work_dir),
            "reports_dir": str(config.prepare.reports_dir),
        },
        "sources": {
            name: asdict(source) for name, source in sorted(config.sources.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reuse_summary(config: AppConfig) -> dict[str, Any] | None:
    path = config.prepare.reports_dir / "prepare-summary.json"
    if not path.is_file() or not config.run.input_jsonl.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("config_fingerprint") != _config_fingerprint(config):
        raise RuntimeError(
            "an existing prepared manifest uses different preparation settings; "
            "use a new CORPUS_DATA_ROOT"
        )
    current_sha = sha256_file(config.run.input_jsonl)
    if summary.get("input_manifest_sha256") != current_sha:
        raise RuntimeError(
            "the prepared manifest was modified after preparation; "
            "use a new CORPUS_DATA_ROOT"
        )
    summary["reused"] = True
    return summary


def prepare_corpora(config: AppConfig) -> dict[str, Any]:
    reused = _reuse_summary(config)
    if reused is not None:
        print(f"[resume] prepared manifest found: {config.run.input_jsonl}")
        return reused

    downloads: dict[str, dict[str, Any]] = {}
    roots: dict[str, Path] = {}
    for name in ("jesc", "kftt"):
        source = config.sources[name]
        archive = config.prepare.sources_dir / source.archive_name
        downloads[name] = _download(name, source, archive)
        roots[name] = _extract(
            name,
            archive,
            config.prepare.work_dir,
            downloads[name]["sha256"],
        )

    database_path = config.prepare.work_dir / "pairs.sqlite3"
    with _create_database(database_path) as connection:
        _reset_database(connection)
        # Both corpora's evaluation rows go first so exact duplicates in
        # either training source cannot leak into an evaluation split.
        ingestion = {
            "jesc": {
                "evaluation": _ingest(
                    connection,
                    config,
                    "jesc",
                    iter_jesc(roots["jesc"], ("dev", "test")),
                ),
            },
            "kftt": {
                "evaluation": _ingest(
                    connection,
                    config,
                    "kftt",
                    iter_kftt(roots["kftt"], ("dev", "test")),
                ),
            },
        }
        ingestion["jesc"]["train"] = _ingest(
            connection,
            config,
            "jesc",
            iter_jesc(roots["jesc"], ("train",)),
        )
        ingestion["kftt"]["train"] = _ingest(
            connection,
            config,
            "kftt",
            iter_kftt(roots["kftt"], ("train",)),
        )
        selection = _select_training(connection, config)
        atomic_write_jsonl(
            config.run.input_jsonl,
            _manifest_rows(connection, config),
        )
        split_counts = dict(
            connection.execute(
                "SELECT original_split || ':' || corpus, COUNT(*) "
                "FROM candidates "
                "WHERE selected = 1 OR original_split != 'train' "
                "GROUP BY original_split, corpus"
            ).fetchall()
        )

    summary = {
        "reused": False,
        "config_fingerprint": _config_fingerprint(config),
        "input_manifest": str(config.run.input_jsonl),
        "input_manifest_sha256": sha256_file(config.run.input_jsonl),
        "downloads": downloads,
        "ingestion": ingestion,
        "selection": selection,
        "manifest_counts": split_counts,
    }
    _atomic_json(config.prepare.reports_dir / "prepare-summary.json", summary)
    return summary
