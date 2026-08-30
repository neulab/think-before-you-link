"""
Unified MERLIN Entity Linking Evaluation.

Usage examples:

  # Reasoning + RAG (BM25)  [was: merlin_agentic_final_optimized.py]
  python -m merlin_eval.run --rag --retriever bm25 \
    --server_url http://localhost:30000/v1 ...

  # Reasoning + RAG + Constrained  [was: _constrained.py]
  python -m merlin_eval.run --rag --retriever bm25 --constrained \
    --server_url http://localhost:30000/v1 ...

  # Reasoning + RAG (Embedding) + Constrained  [was: _constrained_embedding.py]
  python -m merlin_eval.run --rag --retriever embedding --constrained \
    --server_url http://localhost:30000/v1 ...

  # Reasoning only (No RAG)  [was: _constrained_no_rag.py]
  python -m merlin_eval.run \
    --server_url http://localhost:30000/v1 ...

  # No image ablation  [was: _constrained_no_image.py]
  python -m merlin_eval.run --rag --retriever bm25 --constrained --no_image \
    --server_url http://localhost:30000/v1 ...
"""

import argparse
import datetime
import json
import os
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from merlin_eval.config import EvalConfig
from merlin_eval.pipeline import process_single_sample
from merlin_eval.scoring import (
    filter_nil_samples,
    load_aliases_cache,
    load_entity_links,
)
from merlin_eval.utils import validate_languages

warnings.filterwarnings("ignore")


# ============================================================================
# Evaluate a single language
# ============================================================================

def evaluate_language(
    language: str,
    client: OpenAI,
    retriever,
    aliases_cache: dict,
    entity_links: dict,
    config: EvalConfig,
) -> dict:
    """Evaluate a single language.

    Returns a dict with keys: accuracy, correct, total, total_raw,
    nil_filtered, duration_s, error (if any).
    """
    lang_start = time.time()

    json_path = os.path.join(config.merlin_dataset_dir, f"{language}.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {language} dataset: {e}")
        return {"accuracy": 0.0, "correct": 0, "total": 0, "total_raw": 0,
                "nil_filtered": 0, "duration_s": 0, "error": str(e)}

    total_raw = len(data)
    print(f"\nOriginal dataset size: {total_raw}")
    data = filter_nil_samples(data)
    nil_filtered = total_raw - len(data)
    print(f"After filtering NIL: {len(data)}")
    if not data:
        print("No valid samples after filtering NIL")
        return {"accuracy": 0.0, "correct": 0, "total": 0,
                "total_raw": total_raw, "nil_filtered": nil_filtered,
                "duration_s": round(time.time() - lang_start, 2)}

    mode_parts = []
    if config.rag:
        mode_parts.append(f"RAG({config.retriever})")
    if config.constrained:
        mode_parts.append(f"constrained({config.constrained_mode})")
    if config.no_image:
        mode_parts.append("no-image")
    mode_str = " + ".join(mode_parts) if mode_parts else "reasoning-only"

    print(f"\n{'='*60}")
    print(f"Evaluating {language.upper()} [{mode_str}]")
    print(f"Samples: {len(data)}  |  Workers: {config.max_workers}")
    print(f"{'='*60}\n")

    results = []
    num_correct = 0

    def _process(sample):
        return process_single_sample(
            sample=sample,
            client=client,
            language=language,
            images_dir=config.merlin_images_dir,
            aliases_cache=aliases_cache,
            entity_links=entity_links,
            retriever=retriever,
            config=config,
        )

    if config.max_workers == 1:
        for sample in tqdm(data, desc=f"{language}", unit="ex"):
            result = _process(sample)
            results.append(result)
            if result["is_correct"]:
                num_correct += 1
            print(f"  >> Running: {num_correct}/{len(results)} correct "
                  f"({num_correct/len(results):.1%})", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(_process, s): s for s in data}
            for future in tqdm(as_completed(futures), total=len(data),
                               desc=f"{language}", unit="ex"):
                result = future.result()
                results.append(result)
                if result["is_correct"]:
                    num_correct += 1
                print(f"  >> Running: {num_correct}/{len(results)} correct "
                      f"({num_correct/len(results):.1%})", flush=True)

    accuracy = num_correct / len(data) if data else 0.0
    duration_s = round(time.time() - lang_start, 2)

    print(f"\n{'='*60}")
    print(f"Results for {language.upper()} [{mode_str}]")
    print(f"Correct: {num_correct}/{len(data)}  |  Accuracy: {accuracy:.2%}")
    print(f"Wall time: {duration_s:.1f}s")
    print(f"{'='*60}\n")

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    dataset_name = Path(config.merlin_dataset_dir).name
    output_file = os.path.join(
        config.output_dir,
        f"english_wiki_results__{dataset_name}__{language.lower()}.json"
    )
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

    return {
        "accuracy": accuracy,
        "correct": num_correct,
        "total": len(data),
        "total_raw": total_raw,
        "nil_filtered": nil_filtered,
        "duration_s": duration_s,
    }


# ============================================================================
# CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified MERLIN Entity Linking Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Feature toggles
    g = p.add_argument_group("Feature toggles")
    g.add_argument("--rag", action="store_true",
                   help="Enable RAG (agentic Wikipedia search in Layer 1)")
    g.add_argument("--retriever", choices=["bm25", "embedding"], default="bm25",
                   help="Retriever backend when --rag is set (default: bm25)")
    g.add_argument("--constrained", action="store_true",
                   help="Constrained decoding in Layer 2 (requires SGLang)")
    g.add_argument("--constrained_mode", choices=["regex", "trie"], default="trie",
                   help="Constrained decoding mode: 'regex' (seen titles only, "
                        "requires --rag) or 'trie' (all Wikipedia titles). "
                        "Default: trie")
    g.add_argument("--no_image", action="store_true",
                   help="Disable image input to the model")

    # Model / server
    g = p.add_argument_group("Model & server")
    g.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")
    g.add_argument("--server_url", default="http://localhost:30000/v1",
                   help="Model server URL (default: http://localhost:30000/v1)")

    # Data paths
    g = p.add_argument_group("Data paths")
    g.add_argument("--merlin_dataset_dir", required=True,
                   help="Directory containing Merlin language JSON files")
    g.add_argument("--merlin_images_dir", required=True,
                   help="Directory containing Merlin images")
    g.add_argument("--entity_links_path", default="",
                   help="Optional JSON containing Wikidata incoming-link metadata")
    g.add_argument("--aliases_path", default="",
                   help="Optional alias cache JSON (used only with --use_alias)")

    # Trie constrained decoding
    g = p.add_argument_group("Trie constrained decoding")
    g.add_argument("--trie_server_url", default="http://127.0.0.1:8099/lookup",
                   help="Trie server URL for trie-mode constrained decoding "
                        "(default: http://127.0.0.1:8099/lookup)")
    g.add_argument("--trie_processor_path", default="",
                   help="Path to serialized processor .json file "
                        "(default: auto-detect from data/trie_processors/)")
    g.add_argument("--trie_path", default="./data/wikipedia_trie.pkl",
                   help="Path to trie pickle (for reference/validation only)")

    # Retriever paths
    g = p.add_argument_group("Retriever paths")
    g.add_argument("--wiki_index_path", default="./data/wikipedia_index.parquet")
    g.add_argument("--bm25_cache_path", default="./data/wikipedia_bm25s_index")
    g.add_argument("--embedding_cache_path",
                   default="./data/wikipedia_embeddings_cache")
    g.add_argument("--embedding_device", default="cuda",
                   help="Device for embedding model (default: cuda)")
    g.add_argument("--force_rebuild_index", action="store_true",
                   help="Force rebuild retriever index even if cache exists")

    # Evaluation
    g = p.add_argument_group("Evaluation")
    g.add_argument("--languages", nargs="+", default=["all"],
                   help='Languages to evaluate (e.g. hindi tamil) or "all"')
    g.add_argument("--max_workers", type=int, default=1,
                   help="Parallel workers (default: 1)")
    g.add_argument("--target_in_pred", action="store_true",
                   help="Accept if target appears within prediction")
    g.add_argument("--use_alias", action="store_true",
                   help="Use alias information for evaluation")
    g.add_argument("--output_dir", default="output",
                   help="Output directory for results")

    # Robustness
    g = p.add_argument_group("Robustness")
    g.add_argument("--layer1_max_tokens", type=int, default=32000,
                   help="Max tokens for Layer 1 generation (default: 32000)")
    g.add_argument("--layer1_retries", type=int, default=2,
                   help="Number of retries if Layer 1 fails (default: 2)")
    g.add_argument("--no_force_first_tool", action="store_true",
                   help="Disable forcing tool_choice=required on iteration 0")
    g.add_argument("--api_timeout", type=float, default=3600.0,
                   help="Per-request API timeout in seconds (default: 3600 = 60 min)")
    g.add_argument("--max_searches", type=int, default=0,
                   help="Max search calls per sample (0 = unlimited, 1 = single-shot)")

    # Sampling controls
    g = p.add_argument_group("Sampling")
    g.add_argument("--temperature", type=float, default=0.0,
                   help="Layer 1 sampling temperature (default: 0 = greedy; "
                        ">0 used for self-consistency)")
    g.add_argument("--top_p", type=float, default=1.0,
                   help="Layer 1 nucleus sampling top_p (default: 1.0)")
    g.add_argument("--top_k", type=int, default=-1,
                   help="Layer 1 top_k sampling (default: -1 = unset; Gemma 4 uses 64). "
                        "Passed via SGLang extra_body.")
    g.add_argument("--seed", type=int, default=-1,
                   help="Layer 1 sampling seed (default: -1 = unset; Gemma 4 uses 42). "
                        "Passed via SGLang extra_body.")
    g.add_argument("--n_samples", type=int, default=1,
                   help="Number of independent Layer 1 samples (>1 → self-consistency mode)")
    g.add_argument("--enable_thinking", choices=["unset", "true", "false"], default="unset",
                   help="Gemma 4 reasoning toggle (default: unset = don't pass; "
                        "true/false = pass via chat_template_kwargs)")
    g.add_argument("--model_family", choices=["qwen", "gemma", "internvl"], default="qwen",
                   help="Model family for prompt/post-processing routing (default: qwen)")
    g.add_argument("--system_prompt_prefix", default="",
                   help="Optional string prepended to all Layer 1 system prompts. "
                        "Used by InternVL3_5 to inject R1_SYSTEM_PROMPT for thinking mode.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Build config
    config = EvalConfig(
        rag=args.rag,
        retriever=args.retriever,
        constrained=args.constrained,
        constrained_mode=args.constrained_mode,
        no_image=args.no_image,
        model=args.model,
        server_url=args.server_url,
        merlin_dataset_dir=args.merlin_dataset_dir,
        merlin_images_dir=args.merlin_images_dir,
        entity_links_path=args.entity_links_path,
        aliases_path=args.aliases_path,
        trie_server_url=args.trie_server_url,
        trie_processor_path=args.trie_processor_path,
        trie_path=args.trie_path,
        wiki_index_path=args.wiki_index_path,
        bm25_cache_path=args.bm25_cache_path,
        embedding_cache_path=args.embedding_cache_path,
        embedding_device=args.embedding_device,
        languages=args.languages,
        max_workers=args.max_workers,
        target_in_pred=args.target_in_pred,
        use_alias=args.use_alias,
        output_dir=args.output_dir,
        force_rebuild_index=args.force_rebuild_index,
        layer1_max_tokens=args.layer1_max_tokens,
        layer1_retries=args.layer1_retries,
        layer1_force_first_tool=not args.no_force_first_tool,
        api_timeout=args.api_timeout,
        max_searches=args.max_searches,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        n_samples=args.n_samples,
        enable_thinking={"unset": -1, "true": 1, "false": 0}[args.enable_thinking],
        model_family=args.model_family,
        system_prompt_prefix=args.system_prompt_prefix,
    )

    # --- Validation ---
    if config.constrained and config.constrained_mode == "regex" and not config.rag:
        print("WARNING: --constrained --constrained_mode regex requires --rag "
              "(need seen titles to constrain to). Ignoring --constrained.")
        config.constrained = False

    if config.retriever == "embedding" and not config.rag:
        print("WARNING: --retriever embedding has no effect without --rag.")

    # --- Load trie processor (if trie mode + constrained) ---
    if config.constrained and config.constrained_mode == "trie":
        processor_path = config.trie_processor_path
        if not processor_path:
            # Auto-detect based on model name
            model_lower = config.model.lower()
            if "qwen3" in model_lower:
                proc_name = "qwen3"
            elif "deepseek" in model_lower and "r1" in model_lower:
                proc_name = "deepseek_r1"
            elif "llama" in model_lower:
                proc_name = "llama3"
            else:
                proc_name = "qwen3"  # default
            processor_path = f"./data/trie_processors/{proc_name}.json"
            print(f"Auto-detected trie processor: {processor_path}")

        if os.path.exists(processor_path):
            with open(processor_path, "r") as f:
                config.trie_processor_str = f.read().strip()
            print(f"Loaded trie processor from {processor_path} "
                  f"({len(config.trie_processor_str)} bytes)")
        else:
            print(f"ERROR: Trie processor not found at {processor_path}")
            print("Run merlin_eval/trie/build_wikipedia_trie.py first, or "
                  "specify --trie_processor_path.")
            return

    # --- Banner ---
    mode_parts = []
    if config.rag:
        mode_parts.append(f"RAG({config.retriever})")
    else:
        mode_parts.append("reasoning-only")
    if config.constrained:
        mode_parts.append(f"constrained({config.constrained_mode})")
    if config.no_image:
        mode_parts.append("no-image")
    mode_str = " + ".join(mode_parts)

    print("\n" + "=" * 60)
    print(f"MERLIN Entity Linking Evaluation")
    print("=" * 60)
    print(f"Mode:       {mode_str}")
    print(f"Model:      {config.model}")
    print(f"Server:     {config.server_url}")
    if config.constrained and config.constrained_mode == "trie":
        print(f"Trie:       {config.trie_server_url}")
    print(f"Dataset:    {config.merlin_dataset_dir}")
    print(f"Images:     {config.merlin_images_dir}"
          f"{' (disabled)' if config.no_image else ''}")
    print(f"Workers:    {config.max_workers}")
    print(f"L1 tokens:  {config.layer1_max_tokens}")
    print(f"L1 retries: {config.layer1_retries}")
    print(f"Force tool: {config.layer1_force_first_tool}")
    print(f"Timeout:    {config.api_timeout}s")
    if config.rag and config.retriever == "embedding":
        print(f"Emb device: {config.embedding_device}")
    print("=" * 60 + "\n")

    # --- OpenAI client ---
    client = OpenAI(base_url=config.server_url, api_key="EMPTY")

    # --- Auxiliary data ---
    print("Loading auxiliary data...")
    entity_links = load_entity_links(config.entity_links_path)
    aliases_cache = load_aliases_cache(config.aliases_path)
    print()

    # --- Languages ---
    try:
        languages = validate_languages(config.languages)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # --- Retriever (only if RAG) ---
    retriever = None
    if config.rag:
        print(f"{'='*60}")
        print(f"Setting up retriever: {config.retriever}")
        print(f"{'='*60}")

        if config.retriever == "bm25":
            from merlin_eval.retriever import BM25Retriever
            retriever = BM25Retriever(
                parquet_path=config.wiki_index_path,
                bm25_cache_path=config.bm25_cache_path,
                force_rebuild=config.force_rebuild_index,
            )
        elif config.retriever == "embedding":
            from merlin_eval.retriever import EmbeddingRetriever
            retriever = EmbeddingRetriever(
                parquet_path=config.wiki_index_path,
                embedding_cache_path=config.embedding_cache_path,
                device=config.embedding_device,
                force_rebuild=config.force_rebuild_index,
            )
        print()

    # --- Evaluate ---
    run_start = time.time()
    lang_results = {}
    for language in languages:
        lang_info = evaluate_language(
            language=language,
            client=client,
            retriever=retriever,
            aliases_cache=aliases_cache,
            entity_links=entity_links,
            config=config,
        )
        lang_results[language] = lang_info
    run_duration_s = round(time.time() - run_start, 2)

    # --- Summary ---
    if len(lang_results) > 1:
        print("\n" + "=" * 60)
        print(f"OVERALL RESULTS [{mode_str}]")
        print("=" * 60)
        for lang, info in lang_results.items():
            print(f"  {lang.capitalize()}: {info['accuracy']:.2%}")
        avg = sum(r["accuracy"] for r in lang_results.values()) / len(lang_results)
        print(f"\n  Average: {avg:.2%}")
        print(f"  Total wall time: {run_duration_s:.1f}s")
        print("=" * 60 + "\n")

    # --- Save run metadata ---
    # Serialize config (exclude large trie_processor_str)
    config_dict = {}
    for k, v in config.__dict__.items():
        if k.startswith("_"):
            continue
        if k == "trie_processor_str" and v:
            config_dict[k] = f"<{len(v)} bytes>"
        else:
            config_dict[k] = v

    # Try to capture git commit
    git_commit = None
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        pass

    run_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": git_commit,
        "config": config_dict,
        "languages": {
            lang: info for lang, info in lang_results.items()
        },
        "run_duration_s": run_duration_s,
    }

    os.makedirs(config.output_dir, exist_ok=True)
    meta_path = os.path.join(config.output_dir, "run_config.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)
    print(f"Run metadata saved to {meta_path}")

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
