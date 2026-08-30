import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from merlin_eval.config import EvalConfig
from merlin_eval.pipeline import _layer1_agentic_once
from merlin_eval.scoring import (
    get_score_detailed,
    load_aliases_cache,
    load_entity_links,
)
from scripts.build_merlin_rare_dataset import (
    METRICS,
    build_language,
    recalculate_percentiles,
)
from scripts import materialize_hf_dataset
from scripts.build_wikipedia_index import iter_records
from scripts.materialize_hf_dataset import save_image


def test_scoring_and_optional_metadata():
    assert get_score_detailed("Afghanistan", "Afghanistan")[0]
    assert get_score_detailed("afghanistan", "Afghanistan")[0]
    assert get_score_detailed("Mercury", "Mercury (planet)")[0]
    assert not get_score_detailed("", "Afghanistan")[0]
    assert load_entity_links("") == {}
    assert load_aliases_cache("") == {}


def test_wikipedia_index_extraction(tmp_path):
    shard = tmp_path / "enwiki_namespace_0_0.jsonl"
    shard.write_text(
        json.dumps(
            {
                "name": "Afghanistan",
                "main_entity": {"identifier": "Q889"},
                "url": "https://en.wikipedia.org/wiki/Afghanistan",
                "description": "A country in Central and South Asia.",
            }
        )
        + "\nnot-json\n",
        encoding="utf-8",
    )
    rows = list(iter_records([shard], description_chars=10))
    assert rows == [
        {
            "wiki_name": "Afghanistan",
            "wiki_wikidata_id": "Q889",
            "wiki_url": "https://en.wikipedia.org/wiki/Afghanistan",
            "wiki_description": "A country ",
        }
    ]


def test_merlin_rare_membership_and_image_materialization(tmp_path):
    enriched_root = tmp_path / "enriched"
    slices_root = tmp_path / "slices"
    images_root = tmp_path / "images"
    enriched_root.joinpath("hi").mkdir(parents=True)
    images_root.joinpath("hindi").mkdir(parents=True)

    source = {
        "Article Title": "शीर्षक",
        "Entity Name": "इकाई",
        "Wikidata ID": "Q1",
        "English Wikipedia Title": "Universe",
        "Image Name": "hindi_1.jpg",
    }
    enriched = {key: value for key, value in source.items() if key != "Image Name"}
    for index, metric in enumerate(METRICS):
        enriched[metric] = index
        enriched[f"{metric}_percentile"] = float(index)

        metric_dir = slices_root / f"bottom_5_{metric}"
        metric_dir.mkdir(parents=True)
        metric_dir.joinpath("hindi.json").write_text(
            json.dumps([source], ensure_ascii=False),
            encoding="utf-8",
        )

    enriched_root.joinpath("hi", "enriched_data.json").write_text(
        json.dumps([enriched], ensure_ascii=False),
        encoding="utf-8",
    )
    Image.new("RGB", (4, 4), "blue").save(
        images_root / "hindi" / "hindi_1.jpg"
    )

    rows, counts = build_language(
        "hi", "hindi", enriched_root, slices_root, images_root
    )
    assert len(rows) == 1
    assert all(rows[0][f"rare_{metric}"] for metric in METRICS)
    assert set(counts) == set(METRICS)

    image_value = rows[0]["Image Name"]
    assert image_value["path"] == "hindi_1.jpg"
    assert image_value["bytes"].startswith(b"\xff\xd8")
    destination = tmp_path / "copy.jpg"
    save_image(Image.open(io.BytesIO(image_value["bytes"])), destination)
    assert destination.is_file()


def test_slice_percentiles_exclude_nil_entities():
    rows = [
        {"English Wikipedia Title": "Sparse", **{metric: 0 for metric in METRICS}},
        {"English Wikipedia Title": "Dense", **{metric: 10 for metric in METRICS}},
        {"English Wikipedia Title": "NIL", **{metric: 0 for metric in METRICS}},
    ]

    recalculate_percentiles(rows)

    for metric in METRICS:
        percentile = f"{metric}_percentile"
        assert rows[0][percentile] == 50.0
        assert rows[1][percentile] == 100.0
        assert rows[2][percentile] is None


def test_materializer_writes_a_shared_image_once(tmp_path, monkeypatch):
    rows = [
        {
            "Article Title": f"article {index}",
            "Entity Name": f"entity {index}",
            "Image Filename": "shared.jpg",
            "Image Name": Image.new("RGB", (4, 4), "blue"),
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        materialize_hf_dataset,
        "load_dataset",
        lambda *args, **kwargs: rows,
    )

    materialize_hf_dataset.materialize_config(
        "unused", "hi", "test", tmp_path, token=None, overwrite=False
    )

    records = json.loads((tmp_path / "dataset/hindi.json").read_text())
    assert len(records) == 2
    assert len(list((tmp_path / "images/hindi").glob("*"))) == 1
    with pytest.raises(FileExistsError):
        materialize_hf_dataset.materialize_config(
            "unused", "hi", "test", tmp_path, token=None, overwrite=False
        )


class FakeRetriever:
    def search(self, query, top_k):
        assert query == "Afghanistan"
        assert top_k == 1
        return [
            {
                "title": "Afghanistan",
                "description": "Country in Asia",
                "wikidata_id": "Q889",
                "url": "https://en.wikipedia.org/wiki/Afghanistan",
                "score": 1.0,
            }
        ]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            tool_call = SimpleNamespace(
                id="call-1",
                type="function",
                function=SimpleNamespace(
                    name="search_wikipedia",
                    arguments='{"query": "Afghanistan", "limit": 1}',
                ),
            )
            message = SimpleNamespace(
                content=None,
                reasoning_content="searching",
                tool_calls=[tool_call],
            )
            choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        else:
            message = SimpleNamespace(
                content="The entity is Afghanistan.",
                reasoning_content=None,
                tool_calls=[],
            )
            choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])


def test_first_tool_call_is_forced():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = EvalConfig(
        rag=True,
        no_image=True,
        layer1_force_first_tool=True,
        max_searches=20,
    )

    explanation, _, seen_titles, meta = _layer1_agentic_once(
        client=client,
        entity_name="अफ़ग़ानिस्तान",
        article_title="समाचार",
        image_path="unused.jpg",
        language="hindi",
        retriever=FakeRetriever(),
        config=config,
        max_iterations=2,
    )

    assert explanation == "The entity is Afghanistan."
    assert seen_titles == ["Afghanistan"]
    assert meta["iterations"] == 2
    assert completions.calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_wikipedia"},
    }
