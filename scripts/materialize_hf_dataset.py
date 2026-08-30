#!/usr/bin/env python3
"""Materialize a MERLIN Hugging Face dataset for the evaluation CLI."""

import argparse
import io
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


LANGUAGES = {
    "hi": "hindi",
    "id": "indonesian",
    "ja": "japanese",
    "ta": "tamil",
    "vi": "vietnamese",
}


def save_image(value, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Image.Image):
        value.save(destination)
        return
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(io.BytesIO(value["bytes"])) as image:
                image.save(destination)
            return
        if value.get("path"):
            with Image.open(value["path"]) as image:
                image.save(destination)
            return
    if isinstance(value, (str, os.PathLike)):
        with Image.open(value) as image:
            image.save(destination)
        return
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def materialize_config(repo_id, config, split, output_dir, token, overwrite):
    language = LANGUAGES[config]
    dataset_dir = output_dir / "dataset"
    images_dir = output_dir / "images" / language
    json_path = dataset_dir / f"{language}.json"

    if json_path.exists() and not overwrite:
        raise FileExistsError(f"{json_path} exists; pass --overwrite to replace it")
    if images_dir.exists() and any(images_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{images_dir} is not empty; pass --overwrite to replace its files"
        )

    dataset = load_dataset(
        repo_id,
        config,
        split=split,
        token=token,
    )
    records = []
    written_images = set()
    for index, row in enumerate(dataset):
        image = row["Image Name"]
        filename = row.get("Image Filename")
        if not filename:
            source_name = Path(getattr(image, "filename", "") or "").name
            filename = source_name or f"{language}_{index}.jpg"

        image_path = images_dir / filename
        if filename not in written_images:
            save_image(image, image_path)
            written_images.add(filename)

        record = {key: value for key, value in row.items() if key != "Image Name"}
        record["Image Name"] = filename
        record.pop("Image Filename", None)
        records.append(record)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"{config}: wrote {len(records):,} rows and images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="neulab/merlin-rare")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(LANGUAGES),
        default=sorted(LANGUAGES),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    for config in args.configs:
        materialize_config(
            repo_id=args.repo_id,
            config=config,
            split=args.split,
            output_dir=args.output_dir,
            token=token,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
