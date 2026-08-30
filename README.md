# Think Before You Link

Code for **Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking**.

The paper studies multilingual multimodal entity linking on [MERLIN](https://huggingface.co/datasets/neulab/merlin). It characterizes entity rarity with 15 Wikipedia and Wikidata metrics and evaluates a vision-language model that iteratively reasons and searches English Wikipedia.

- MERLIN-Rare dataset and per-example artifacts: [neulab/merlin-rare](https://huggingface.co/datasets/neulab/merlin-rare)
- Code license: MIT
- MERLIN-Rare license: CC BY-SA 4.0

## Repository contents

- `merlin_eval/` contains the evaluation framework, prompts, retrievers, and scoring.
- `scripts/build_wikipedia_index.py` creates the title-description index used by BM25 and embedding retrieval.
- `scripts/materialize_hf_dataset.py` exports MERLIN or MERLIN-Rare into the local JSON and image layout expected by the evaluator.
- `scripts/build_merlin_rare_dataset.py` reconstructs the five Hugging Face MERLIN-Rare configurations from the collected metrics and slice definitions.
- `scripts/package_predictions.py` creates compact prediction tables and lossless compressed traces.

Analysis notebooks, paper figures, model weights, retrieval indexes, and cluster-specific job scripts are intentionally not included.

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Serve the chosen Qwen3-VL checkpoint with an OpenAI-compatible server. The experiments used SGLang. For a Thinking checkpoint:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Thinking \
  --port 30000 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen
```

For an Instruct checkpoint, omit `--reasoning-parser qwen3`.

## Prepare MERLIN

The Hugging Face release stores images inside Parquet files. Materialize MERLIN-Rare for the evaluation CLI:

```bash
python scripts/materialize_hf_dataset.py \
  --repo-id neulab/merlin-rare \
  --split test \
  --output-dir data/merlin_rare
```

This creates:

```text
data/merlin_rare/
  dataset/{hindi,indonesian,japanese,tamil,vietnamese}.json
  images/{hindi,indonesian,japanese,tamil,vietnamese}/*.jpg
```

To materialize the original MERLIN Hugging Face release, use `--repo-id neulab/merlin --split train`. The upstream repository uses the split name `train` even though MERLIN is an evaluation benchmark.

## Prepare the Wikipedia index

Download the English namespace-0 shards from [wikimedia/structured-wikipedia](https://huggingface.co/datasets/wikimedia/structured-wikipedia), then run:

```bash
python scripts/build_wikipedia_index.py \
  --source-dir /path/to/structured-wikipedia/en \
  --output data/wikipedia_index.parquet
```

The index contains each English Wikipedia title, its short description, URL, and Wikidata identifier.

BM25 builds `data/wikipedia_bm25s_index/` on first use. Embedding retrieval builds an exact FAISS index using `intfloat/multilingual-e5-large-instruct`. The full embedding index requires about 30 GB of memory and storage at runtime.

## Run the evaluation

No retrieval:

```bash
python -m merlin_eval \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --server_url http://localhost:30000/v1 \
  --merlin_dataset_dir data/merlin_rare/dataset \
  --merlin_images_dir data/merlin_rare/images \
  --output_dir output/no_rag \
  --layer1_max_tokens 32000 \
  --max_workers 16
```

BM25 retrieval:

```bash
python -m merlin_eval \
  --rag --retriever bm25 \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --server_url http://localhost:30000/v1 \
  --merlin_dataset_dir data/merlin_rare/dataset \
  --merlin_images_dir data/merlin_rare/images \
  --wiki_index_path data/wikipedia_index.parquet \
  --output_dir output/bm25 \
  --layer1_max_tokens 32000 \
  --max_workers 16
```

Embedding retrieval:

```bash
python -m merlin_eval \
  --rag --retriever embedding \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --server_url http://localhost:30000/v1 \
  --merlin_dataset_dir data/merlin_rare/dataset \
  --merlin_images_dir data/merlin_rare/images \
  --wiki_index_path data/wikipedia_index.parquet \
  --embedding_cache_path data/wikipedia_embeddings_cache \
  --embedding_device cpu \
  --output_dir output/embedding \
  --layer1_max_tokens 32000 \
  --max_workers 10
```

The paper evaluates the Cartesian product of:

- Qwen3-VL 2B, 4B, and 8B
- Thinking and Instruct checkpoints
- no retrieval, BM25 retrieval, and embedding retrieval

The 2B and 4B experiments use 16,000 Layer-1 tokens. The 8B experiments use 32,000. Retrieval runs force the first Wikipedia tool call because smaller models often do not call the tool when it is only offered as optional. Later iterations let the model decide whether to search again, up to 20 iterations.

Each language produces a JSON file containing the prediction, target, score, reasoning, retrieved titles, tool conversation, token metadata, and duration. `run_config.json` records the full run configuration.

## MERLIN-Rare

MERLIN-Rare contains 1,105 entity mentions over 790 unique images. It is the union of the bottom-5% sets for 15 metrics. Each of the five language configurations includes:

- the original MERLIN fields and image
- all 15 metric values
- the within-language percentile among examples with a valid English Wikipedia title
- a `rare_<metric>` boolean for every slice

For example:

```python
from datasets import load_dataset

dataset = load_dataset("neulab/merlin-rare", "hi", split="test")
language_edition_slice = dataset.filter(
    lambda row: row["rare_language_editions"]
)
```

Metric collection ran from October 7 through October 8, 2025. Pageviews cover a 90-day window. See the dataset card and `metadata/slice_counts.json` in the Hugging Face repository for the complete definitions and counts.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Citation

```bibtex
@article{pengpun2026think,
  title={Think Before You Link: Rarity, Reasoning, and Retrieval in Multilingual Entity Linking},
  author={Pengpun, Parinthapat and Khanuja, Simran and Neubig, Graham},
  year={2026}
}
```

Please also cite the original MERLIN paper when using MERLIN-Rare.
