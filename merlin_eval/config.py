from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """All evaluation configuration in one place."""

    # Feature toggles
    rag: bool = False
    retriever: str = "bm25"  # "bm25" or "embedding"
    constrained: bool = False
    constrained_mode: str = "trie"  # "regex" (seen_titles only) or "trie" (all Wikipedia)
    no_image: bool = False

    # Model / server
    model: str = "Qwen/Qwen3-VL-8B-Thinking"
    server_url: str = "http://localhost:30000/v1"

    # Data paths
    merlin_dataset_dir: str = ""
    merlin_images_dir: str = ""
    entity_links_path: str = ""
    aliases_path: str = ""

    # Retriever paths
    wiki_index_path: str = "./data/wikipedia_index.parquet"
    bm25_cache_path: str = "./data/wikipedia_bm25s_index"
    embedding_cache_path: str = "./data/wikipedia_embeddings_cache"
    embedding_device: str = "cuda"

    # Trie constrained decoding
    trie_server_url: str = "http://127.0.0.1:8099/lookup"
    trie_processor_path: str = ""  # path to serialized processor .json
    trie_path: str = "./data/wikipedia_trie.pkl"
    trie_processor_str: str = ""  # loaded at runtime, not from CLI

    # Evaluation
    languages: list = field(default_factory=lambda: ["all"])
    max_workers: int = 1
    target_in_pred: bool = False
    use_alias: bool = False
    output_dir: str = "output"

    # Rebuild flags
    force_rebuild_index: bool = False

    # Robustness settings
    layer1_max_tokens: int = 32000          # max tokens for Layer 1 (was 40000)
    layer1_retries: int = 2                 # retry on failure (total attempts = 1 + retries)
    layer1_force_first_tool: bool = True    # force tool_choice="required" on iteration 0
    api_timeout: float = 3600.0             # per-request timeout in seconds (60 min)
    max_searches: int = 0                   # max search calls per sample (0 = unlimited)

    # Sampling controls — added for self-consistency (B3) and Gemma 4 (Task E)
    temperature: float = 0.0                # Layer 1 sampling temperature; 0 = greedy; >0 used for SC
    top_p: float = 1.0                      # Layer 1 nucleus sampling
    top_k: int = -1                         # Layer 1 top_k; -1 = unset (don't pass); Gemma 4 uses 64
    seed: int = -1                          # Layer 1 sampling seed; -1 = unset (don't pass); Gemma 4 uses 42
    n_samples: int = 1                      # number of independent Layer 1 samples (>1 for self-consistency)
    # enable_thinking: -1 = unset (don't pass); 0 = False; 1 = True. Used for Gemma 4 reasoning toggle.
    enable_thinking: int = -1
    # model_family: "qwen" (default), "gemma", or "internvl" — routes prompt/post-processing tweaks
    model_family: str = "qwen"
    # system_prompt_prefix: prepended to all Layer 1 system prompts. Used by InternVL3_5
    # to inject R1_SYSTEM_PROMPT when enable_thinking=True (mode-toggle without separate weights).
    system_prompt_prefix: str = ""

    @property
    def use_image(self) -> bool:
        return not self.no_image

    def get_extra_body(self) -> dict:
        """Build the extra_body kwarg for client.chat.completions.create() based on
        family-specific knobs. Currently used by Gemma 4 to set enable_thinking via
        chat_template_kwargs and to pass top_k / seed (non-OpenAI-standard) through
        SGLang's OpenAI-compat layer.
        """
        eb = {}
        if self.enable_thinking != -1:
            eb["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        if self.top_k != -1:
            eb["top_k"] = self.top_k
        if self.seed != -1:
            eb["seed"] = self.seed
        return eb
