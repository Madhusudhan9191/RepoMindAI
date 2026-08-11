import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

class RerankerService:
    """
    Reranks candidate code chunks using explicit hybrid scoring:
    hybrid_score = alpha * dense_similarity + beta * normalized_bm25 + gamma * symbol_overlap
    with optional Cross-Encoder transformer model execution.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", self.MODEL_NAME)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._model = None

    @property
    def model(self):
        """
        Lazy-loads the CrossEncoder model on the first inference call
        to keep application startup fast and offline-friendly.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidate_points: List, alpha: float = None, beta: float = None, gamma: float = None) -> List:
        """
        Computes explicit hybrid scores for candidate points against the query:
        hybrid_score = alpha * dense_score + beta * bm25_score + gamma * symbol_overlap

        Annotates each point with `rerank_score` and `hybrid_score` in its payload.
        """
        if not candidate_points:
            return []

        a = alpha if alpha is not None else self.alpha
        b = beta if beta is not None else self.beta
        g = gamma if gamma is not None else self.gamma

        # Find max BM25 score among candidates for normalization
        max_bm25 = max([getattr(p, "bm25_score", p.payload.get("bm25_score", 0.0) if p.payload else 0.0) for p in candidate_points], default=1.0)
        if max_bm25 <= 0:
            max_bm25 = 1.0

        for point in candidate_points:
            if point.payload is None:
                point.payload = {}

            dense_score = float(getattr(point, "score", point.payload.get("score", 0.0)))
            bm25_raw = float(getattr(point, "bm25_score", point.payload.get("bm25_score", 0.0)))
            norm_bm25 = bm25_raw / max_bm25

            content = point.payload.get("content", "")
            symbol_name = point.payload.get("name", "")
            path_name = point.payload.get("path", "")
            full_text = f"{path_name} {symbol_name} {content}"

            overlap = self._calculate_lexical_overlap(query, full_text)

            # Compute explicit hybrid score
            hybrid_score = (a * dense_score) + (b * norm_bm25) + (g * overlap)

            point.payload["dense_score"] = dense_score
            point.payload["bm25_score"] = bm25_raw
            point.payload["norm_bm25"] = norm_bm25
            point.payload["symbol_overlap"] = overlap
            point.payload["hybrid_score"] = float(hybrid_score)
            point.payload["rerank_score"] = float(hybrid_score)

        # If production CrossEncoder is explicitly loaded and not mock
        if "mock" not in self.model_name.lower() and os.getenv("USE_CROSS_ENCODER", "false").lower() in ("true", "1"):
            try:
                pairs = [[query, p.payload.get("content", "")] for p in candidate_points]
                scores = self.model.predict(pairs)
                for point, score in zip(candidate_points, scores):
                    point.payload["cross_encoder_score"] = float(score)
                    point.payload["rerank_score"] = float(0.5 * point.payload["hybrid_score"] + 0.5 * score)
            except Exception as e:
                logger.warning(f"CrossEncoder inference skipped: {e}")

        return sorted(candidate_points, key=lambda p: p.payload.get("rerank_score", p.score), reverse=True)

    def _calculate_lexical_overlap(self, query: str, document: str) -> float:
        """
        Calculates lexical word overlap ratio between query and document text.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        doc_words = set(re.findall(r"\w+", document.lower()))
        if not query_words:
            return 0.0
        intersection = query_words.intersection(doc_words)
        return len(intersection) / len(query_words)

