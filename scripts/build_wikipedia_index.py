#!/usr/bin/env python3
"""Build the title-description Wikipedia index used by the retrievers."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SCHEMA = pa.schema(
    [
        ("wiki_name", pa.string()),
        ("wiki_wikidata_id", pa.string()),
        ("wiki_url", pa.string()),
        ("wiki_description", pa.string()),
    ]
)


def iter_records(paths, description_chars):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                main_entity = item.get("main_entity") or {}
                description = item.get("description") or ""
                yield {
                    "wiki_name": item.get("name") or "",
                    "wiki_wikidata_id": main_entity.get("identifier"),
                    "wiki_url": item.get("url") or "",
                    "wiki_description": description[:description_chars] or None,
                }


def flush(writer, records):
    if not records:
        return writer
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    if writer is None:
        writer = pq.ParquetWriter(str(flush.output_path), SCHEMA, compression="zstd")
    writer.write_table(table)
    return writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the English structured-wikipedia JSONL shards",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/wikipedia_index.parquet"),
    )
    parser.add_argument(
        "--pattern",
        default="enwiki_namespace_0_*.jsonl",
        help="Shard filename glob relative to --source-dir",
    )
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--description-chars", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = sorted(args.source_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(
            f"No shards matching {args.pattern!r} under {args.source_dir}"
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"{args.output} already exists; pass --overwrite to replace it")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    flush.output_path = args.output
    writer = None
    batch = []
    count = 0
    try:
        for record in tqdm(iter_records(paths, args.description_chars), unit="article"):
            batch.append(record)
            if len(batch) >= args.chunk_size:
                writer = flush(writer, batch)
                count += len(batch)
                batch.clear()
        writer = flush(writer, batch)
        count += len(batch)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise SystemExit("The input shards contained no valid records")
    print(f"Wrote {count:,} articles to {args.output}")


if __name__ == "__main__":
    main()
