#!/usr/bin/env python3
"""Build the five MERLIN-Rare Hugging Face configurations."""

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, Image


LANGUAGES = {
    "hi": "hindi",
    "id": "indonesian",
    "ja": "japanese",
    "ta": "tamil",
    "vi": "vietnamese",
}
METRICS = [
    "wikipedia_pageviews_90d",
    "wikipedia_backlinks",
    "article_size_bytes",
    "reference_count",
    "revision_count",
    "unique_editors",
    "category_count",
    "image_count",
    "external_links",
    "wikidata_incoming_links",
    "wikidata_outgoing_links",
    "language_editions",
    "statement_count",
    "qualifier_count",
    "entity_age_days",
]
BASE_FIELDS = [
    "Article Title",
    "Entity Name",
    "Wikidata ID",
    "English Wikipedia Title",
]


def identity(row):
    return tuple(row.get(field) for field in BASE_FIELDS)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def has_valid_wikipedia_title(row):
    title = str(row.get("English Wikipedia Title") or "").strip()
    return bool(title) and title.upper() != "NIL"


def recalculate_percentiles(rows):
    """Match the percentile calculation used to construct the rare slices."""
    ranked_rows = [row for row in rows if has_valid_wikipedia_title(row)]
    for metric in METRICS:
        values = sorted(row.get(metric, 0) for row in ranked_rows if metric in row)
        if not values:
            continue

        for row in rows:
            percentile_field = f"{metric}_percentile"
            eligible = has_valid_wikipedia_title(row)
            if not eligible or metric not in row:
                row[percentile_field] = None
                continue
            rank = bisect.bisect_right(values, row.get(metric, 0))
            row[percentile_field] = round(rank / len(values) * 100, 1)
    return rows


def build_language(config, language, enriched_root, slices_root, images_root):
    enriched_rows = load_json(enriched_root / config / "enriched_data.json")
    recalculate_percentiles(enriched_rows)
    enriched_by_id = {identity(row): row for row in enriched_rows}

    union = {}
    memberships = {}
    slice_counts = Counter()
    for metric in METRICS:
        path = slices_root / f"bottom_5_{metric}" / f"{language}.json"
        rows = load_json(path)
        slice_counts[metric] = len(rows)
        for row in rows:
            filename = row["Image Name"]
            key = identity(row) + (filename,)
            union[key] = row
            memberships.setdefault(key, set()).add(metric)

    output_rows = []
    for key in sorted(union, key=lambda item: (str(item[0]), str(item[1]), str(item[2]))):
        source = union[key]
        enriched = enriched_by_id.get(identity(source))
        if enriched is None:
            raise KeyError(f"No enriched metrics for {identity(source)!r}")

        filename = source["Image Name"]
        image_path = images_root / language / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        row = {field: source[field] for field in BASE_FIELDS}
        row["Image Filename"] = filename
        row["Image Name"] = {
            "bytes": image_path.read_bytes(),
            "path": filename,
        }
        for metric in METRICS:
            row[metric] = enriched.get(metric)
            row[f"{metric}_percentile"] = enriched.get(f"{metric}_percentile")
            row[f"rare_{metric}"] = metric in memberships[key]
        output_rows.append(row)

    return output_rows, dict(slice_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched-root", type=Path, required=True)
    parser.add_argument("--slices-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    all_counts = {}
    total_rows = 0
    unique_images = set()
    for config, language in LANGUAGES.items():
        rows, slice_counts = build_language(
            config=config,
            language=language,
            enriched_root=args.enriched_root,
            slices_root=args.slices_root,
            images_root=args.images_root,
        )
        dataset = Dataset.from_list(rows).cast_column("Image Name", Image())
        destination = (
            args.output_root / "data" / config / "test-00000-of-00001.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(destination))

        all_counts[config] = {
            "rows": len(rows),
            "unique_images": len({row["Image Filename"] for row in rows}),
            "slices": slice_counts,
        }
        total_rows += len(rows)
        unique_images.update((config, row["Image Filename"]) for row in rows)
        print(f"{config}: {len(rows):,} rows -> {destination}")

    metadata = {
        "metric_snapshot": {
            "collection_started": "2025-10-07",
            "collection_completed": "2025-10-08",
            "pageview_window_days": 90,
        },
        "exclude_nil_entities": True,
        "metrics": METRICS,
        "languages": all_counts,
        "total_rows": total_rows,
        "unique_images": len(unique_images),
    }
    metadata_path = args.output_root / "metadata" / "slice_counts.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Total: {total_rows:,} rows, {len(unique_images):,} unique images")


if __name__ == "__main__":
    main()
