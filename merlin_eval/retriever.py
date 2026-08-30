import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
from tqdm import tqdm


class BaseRetriever(ABC):
    """Common interface for Wikipedia retrievers."""

    def __init__(self, parquet_path: str):
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Wikipedia index not found: {self.parquet_path}")

        print(f"Loading Wikipedia index from parquet...")
        self.df = pl.read_parquet(self.parquet_path)
        print(f"Loaded {len(self.df):,} articles")

        self.titles = self.df["wiki_name"].to_list()
        self.descriptions = self.df["wiki_description"].fill_null("").to_list()
        self.wikidata_ids = self.df["wiki_wikidata_id"].fill_null("").to_list()
        self.urls = self.df["wiki_url"].fill_null("").to_list()
        self.title_to_idx = {title: idx for idx, title in enumerate(self.titles)}

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        ...

    def get_article_info(self, title: str) -> Optional[Dict]:
        """Get article info by exact title match."""
        idx = self.title_to_idx.get(title)
        if idx is not None:
            return {
                'title': self.titles[idx],
                'description': self.descriptions[idx],
                'wikidata_id': self.wikidata_ids[idx],
                'url': self.urls[idx]
            }
        return None


class BM25Retriever(BaseRetriever):
    """BM25-based Wikipedia retriever using bm25s library."""

    def __init__(self, parquet_path: str, bm25_cache_path: str,
                 force_rebuild: bool = False):
        super().__init__(parquet_path)

        import bm25s
        import Stemmer

        self.bm25_cache_path = Path(bm25_cache_path)
        self.retriever = bm25s.BM25(backend='numba')
        self.stemmer = Stemmer.Stemmer("english")
        self._bm25s = bm25s

        if force_rebuild and self.bm25_cache_path.exists():
            import shutil
            print(f"Removing existing BM25 cache: {self.bm25_cache_path}")
            shutil.rmtree(self.bm25_cache_path, ignore_errors=True)

        self._build_or_load()

    def _build_or_load(self):
        if self.bm25_cache_path.exists():
            print(f"Loading cached BM25 index...")
            try:
                self.retriever = self._bm25s.BM25.load(
                    self.bm25_cache_path, load_corpus=False, mmap=True)
                self.retriever.backend = "numba"
                print(f"Loaded BM25 cache")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}, rebuilding...")

        print(f"Building BM25 index (this may take a few minutes)...")
        pre_tokenized_docs = []
        for title, desc in tqdm(zip(self.titles, self.descriptions),
                                total=len(self.titles), desc="Tokenizing"):
            combined = f"{title} {desc}" if desc else title
            pre_tokenized_docs.append(combined)
        corpus_tokens = self._bm25s.tokenize(pre_tokenized_docs, stemmer=self.stemmer)
        self.retriever.index(corpus_tokens)

        print(f"Caching BM25 index...")
        self.bm25_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.retriever.save(self.bm25_cache_path)
        print(f"Cached to {self.bm25_cache_path}")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        if not query or not query.strip():
            return []
        query_tokens = self._bm25s.tokenize(query, stemmer=self.stemmer)
        if not query_tokens:
            return []
        bm25_results = self.retriever.retrieve(query_tokens, k=top_k)

        results = []
        for idx, score in zip(bm25_results[0][0], bm25_results[1][0]):
            if score > 0:
                results.append({
                    'title': self.titles[idx],
                    'description': self.descriptions[idx] or "No description available",
                    'wikidata_id': self.wikidata_ids[idx],
                    'url': self.urls[idx],
                    'score': float(score)
                })
        return results


class EmbeddingRetriever(BaseRetriever):
    """FAISS + E5 embedding-based Wikipedia retriever.

    Uses IndexFlatIP for exact search (~30 GB at runtime for 7.4M x 1024-dim).
    The one-time build streams embeddings through a disk-backed mmap file and
    adds to FAISS in chunks, so peak build RAM ≈ runtime RAM (never 3x).

    All 40 worker threads share the single FAISS index object — read-only
    search is thread-safe with zero copies.  A lock protects the encoding
    path so workers don't stomp on each other's GPU/model state.
    """

    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_BATCH_SIZE = 64
    EMBEDDING_MAX_LENGTH = 512
    SEARCH_TASK_INSTRUCTION = (
        "Given a web search query, retrieve relevant Wikipedia articles "
        "that match the query"
    )
    # How many docs to encode before flushing to the mmap file
    ENCODE_CHUNK_SIZE = 10_000

    def __init__(self, parquet_path: str, embedding_cache_path: str,
                 device: str = "cuda", force_rebuild: bool = False):
        super().__init__(parquet_path)

        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
        import faiss

        self._torch = torch
        self._F = F
        self._faiss = faiss

        self.embedding_cache_path = Path(embedding_cache_path)
        self.device = torch.device(device)
        self._encode_lock = threading.Lock()

        print(f"Loading embedding model: {self.EMBEDDING_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.EMBEDDING_MODEL_NAME)
        self.model.eval()
        self.model.to(self.device)
        print(f"Embedding model loaded on {self.device}")

        self.embedding_dim = self.model.config.hidden_size
        self.faiss_index = None

        if force_rebuild and self.embedding_cache_path.exists():
            import shutil
            print(f"Removing existing embedding cache: {self.embedding_cache_path}")
            shutil.rmtree(self.embedding_cache_path, ignore_errors=True)

        self._build_or_load()

    def _average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @property
    def _instruct_prefix(self):
        return f'Instruct: {self.SEARCH_TASK_INSTRUCTION}\nQuery: '

    def _encode_batch(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode a small batch of texts. Caller must hold _encode_lock."""
        torch = self._torch
        F = self._F

        if is_query:
            texts = [f'{self._instruct_prefix}{t}' for t in texts]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.EMBEDDING_BATCH_SIZE):
                batch_texts = texts[i:i + self.EMBEDDING_BATCH_SIZE]
                batch_dict = self.tokenizer(
                    batch_texts,
                    max_length=self.EMBEDDING_MAX_LENGTH,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                outputs = self.model(**batch_dict)
                embeddings = self._average_pool(
                    outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def _build_or_load(self):
        faiss = self._faiss
        faiss_file = self.embedding_cache_path / "faiss_flat_index.bin"

        if faiss_file.exists():
            print(f"Loading cached FAISS index...")
            try:
                self.faiss_index = faiss.read_index(str(faiss_file))
                print(f"FAISS IndexFlatIP loaded "
                      f"({self.faiss_index.ntotal:,} vectors, "
                      f"~{self.faiss_index.ntotal * self.embedding_dim * 4 / 1e9:.1f} GB)")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}, rebuilding...")

        n = len(self.titles)
        index_gb = n * self.embedding_dim * 4 / 1e9
        print(f"Building exact embedding index for {n:,} documents "
              f"(~{index_gb:.1f} GB)...")

        self.embedding_cache_path.mkdir(parents=True, exist_ok=True)

        # Create the FAISS index up front — vectors are added in chunks
        # so we never hold a separate full-size numpy array alongside it.
        index = faiss.IndexFlatIP(self.embedding_dim)

        combined_texts = [
            f"{t}. {d}" if d else t
            for t, d in zip(self.titles, self.descriptions)
        ]

        chunk = self.ENCODE_CHUNK_SIZE
        with self._encode_lock:
            for start in tqdm(range(0, n, chunk), desc="Encoding & adding"):
                end = min(start + chunk, n)
                # Encode chunk → add to FAISS → free chunk
                chunk_embs = self._encode_batch(
                    combined_texts[start:end], is_query=False)
                index.add(chunk_embs)
                del chunk_embs

        del combined_texts

        print(f"Saving FAISS index to {faiss_file}...")
        faiss.write_index(index, str(faiss_file))
        print(f"Cached ({index.ntotal:,} vectors)")

        self.faiss_index = index

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built")
        if not query or not query.strip():
            return []

        with self._encode_lock:
            query_embedding = self._encode_batch([query], is_query=True)

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and score > 0:
                results.append({
                    'title': self.titles[idx],
                    'description': self.descriptions[idx] or "No description available",
                    'wikidata_id': self.wikidata_ids[idx],
                    'url': self.urls[idx],
                    'score': float(score)
                })
        del query_embedding
        return results
