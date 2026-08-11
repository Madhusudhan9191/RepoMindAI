import math
import re
from collections import defaultdict, Counter


class BM25Retriever:
    """
    In-memory BM25 (Okapi BM25) Lexical Retriever for codebase chunks.
    Complements dense vector embeddings with exact code symbol matching.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = {}
        self.avg_doc_len = 0.0
        self.doc_count = 0
        self.term_df = defaultdict(int)
        self.doc_term_freqs = {}
        self.chunks_map = {}

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on non-alphanumeric, camelCase, snake_case
        raw_tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        tokens = []
        for tok in raw_tokens:
            tokens.append(tok)
            # CamelCase splitting
            sub_words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[0-9]+", tok)
            if len(sub_words) > 1:
                for sub in sub_words:
                    tokens.append(sub.lower())
        return tokens

    def index_chunks(self, chunks: list[dict]):
        """
        Build or update BM25 inverted index from a list of chunk dictionaries or Qdrant points.
        """
        self.doc_len.clear()
        self.term_df.clear()
        self.doc_term_freqs.clear()
        self.chunks_map.clear()

        if not chunks:
            self.doc_count = 0
            self.avg_doc_len = 0.0
            return

        total_length = 0

        for chunk in chunks:
            chunk_id = chunk.get("id") or str(chunk.get("payload", {}).get("id"))
            path = chunk.get("path") or chunk.get("payload", {}).get("path", "")
            name = chunk.get("name") or chunk.get("payload", {}).get("name", "")
            content = chunk.get("content") or chunk.get("payload", {}).get("content", "")

            # Combine path, symbol name, and content for lexical index
            full_text = f"{path} {name} {content}"
            tokens = self._tokenize(full_text)

            self.chunks_map[chunk_id] = chunk
            length = len(tokens)
            self.doc_len[chunk_id] = length
            total_length += length

            tf = Counter(tokens)
            self.doc_term_freqs[chunk_id] = tf

            for term in tf.keys():
                self.term_df[term] += 1

        self.doc_count = len(self.chunks_map)
        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 1.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Compute Okapi BM25 score for all indexed chunks against query tokens.
        Returns list of (chunk_id, bm25_score) tuples sorted by score descending.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return []

        scores = defaultdict(float)

        for token in set(query_tokens):
            df = self.term_df.get(token, 0)
            if df == 0:
                continue

            # IDF calculation with smoothing
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

            for chunk_id, tf_map in self.doc_term_freqs.items():
                freq = tf_map.get(token, 0)
                if freq > 0:
                    doc_l = self.doc_len[chunk_id]
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_l / self.avg_doc_len))
                    term_score = idf * (freq * (self.k1 + 1.0)) / denom
                    scores[chunk_id] += term_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
