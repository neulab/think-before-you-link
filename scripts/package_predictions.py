#!/usr/bin/env python3
"""Package per-example predictions and lossless compressed traces."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_run(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("run must be FAMILY:LABEL:PATH")
    family, label, path = parts
    return family, label, Path(path)


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith(("/home/", "/data/")):
        return f"<external>/{Path(value).name}"
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_lossless(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as raw_dst:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_dst,
            compresslevel=9,
            mtime=0,
        ) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def compact_record(record, index):
    score_detail = record.get("score_detail") or {}
    layer1_meta = record.get("layer1_meta") or {}
    layer2_meta = record.get("layer2_meta") or {}
    return {
        "index": index,
        "article_title": record.get("title"),
        "entity_name": record.get("entity"),
        "wikidata_id": record.get("wikidata_id"),
        "image_filename": record.get("image"),
        "gold_title": record.get("target"),
        "predicted_title": record.get("layer2_prediction"),
        "is_correct": record.get("is_correct"),
        "match_type": score_detail.get("match_type"),
        "seen_title_count": len(record.get("seen_titles") or []),
        "layer1_iterations": layer1_meta.get("iterations"),
        "layer1_completion_tokens": (
            (layer1_meta.get("token_usage") or {}).get("completion_tokens")
        ),
        "layer2_completion_tokens": (
            (layer2_meta.get("token_usage") or {}).get("completion_tokens")
        ),
        "duration_seconds": record.get("total_duration_s"),
    }


def language_from_path(path):
    stem = path.stem
    marker = "__"
    if marker not in stem:
        raise ValueError(f"Cannot identify language from {path}")
    return stem.rsplit(marker, 1)[-1]


def package_run(family, label, source_dir, output_root):
    files = sorted(source_dir.glob("english_wiki_results__*.json"))
    if len(files) != 5:
        raise ValueError(f"{source_dir} has {len(files)} result files, expected 5")

    manifest_rows = []
    for source in files:
        language = language_from_path(source)
        with source.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        compact = [compact_record(record, index) for index, record in enumerate(records)]

        prediction_path = (
            output_root / "predictions" / family / label / f"{language}.parquet"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(compact),
            prediction_path,
            compression="zstd",
        )

        trace_path = output_root / "traces" / family / label / f"{language}.json.gz"
        gzip_lossless(source, trace_path)
        manifest_rows.append(
            {
                "family": family,
                "configuration": label,
                "language": language,
                "rows": len(records),
                "source_sha256": sha256(source),
                "trace_sha256": sha256(trace_path),
                "trace_bytes": trace_path.stat().st_size,
            }
        )
        print(f"{family}/{label}/{language}: {len(records):,} predictions")

    config_path = source_dir / "run_config.json"
    if config_path.is_file():
        config = sanitize(json.loads(config_path.read_text(encoding="utf-8")))
        destination = output_root / "metadata" / "run_configs" / family / f"{label}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return manifest_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        required=True,
        help="Repeatable FAMILY:LABEL:PATH specification",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.output_root / "metadata" / "artifact_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = []
    existing = {
        (row["family"], row["configuration"], row["language"]) for row in manifest
    }

    for family, label, source_dir in args.run:
        rows = package_run(family, label, source_dir, args.output_root)
        for row in rows:
            key = (row["family"], row["configuration"], row["language"])
            if key in existing:
                raise ValueError(f"Duplicate packaged artifact: {key}")
            manifest.append(row)
            existing.add(key)

    manifest.sort(
        key=lambda row: (row["family"], row["configuration"], row["language"])
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
