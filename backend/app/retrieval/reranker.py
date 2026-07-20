import logging
import os
import re
from typing import List

logger = logging.getLogger(__name__)

class RerankerService:
    """
    Reranks candidate code chunks using a Cross-Encoder transformer model.
    Optimizes for precision by processing the query and document content
    jointly through self-attention layers.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("RERANKER_MODEL", self.MODEL_NAME)
        self._model = None

    @property
    def model(self):
        """
        Lazy-loads the CrossEncoder model on the first inference call
        to keep application startup fast and offline-friendly.
        """
        if self._model is None:
            # We import only when the model is loaded to avoid loading sentence_transformers on startup
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidate_points: List) -> List:
        """
        Computes joint attention scores for candidate points against the query
        in a single batch call. Annotates each point with a `rerank_score`
        key in its payload dictionary and returns the sorted candidates (highest score first).

        If the model name contains 'mock', offline mock scoring is executed
        (using lexical overlap + length coefficient + original score).
        """
        if not candidate_points:
            return []

        # 1. Offline Mock Reranking logic
        if "mock" in self.model_name.lower():
            for point in candidate_points:
                if point.payload is None:
                    point.payload = {}
                content = point.payload.get("content", "")
                overlap = self._calculate_lexical_overlap(query, content)
                # Score combines original similarity, lexical overlap boost, and length coefficient
                point.payload["rerank_score"] = float(point.score + 0.1 * overlap + 0.0001 * len(content))
            
            return sorted(candidate_points, key=lambda p: p.payload.get("rerank_score", p.score), reverse=True)

        # 2. Production Transformer Batch Reranking logic
        try:
            # Construct (query, document) pairs for all candidates
            pairs = [[query, p.payload.get("content", "")] for p in candidate_points]
            
            # Predict relevance scores in a single batch
            scores = self.model.predict(pairs)
            
            # Annotate points with their cross-encoder rerank score
            for point, score in zip(candidate_points, scores):
                if point.payload is None:
                    point.payload = {}
                point.payload["rerank_score"] = float(score)

            # Return candidates sorted in descending order of rerank score
            return sorted(candidate_points, key=lambda p: p.payload.get("rerank_score", p.score), reverse=True)

        except Exception as e:
            logger.warning(f"Reranking failed (falling back to vector score): {str(e)}")
            # If inference fails, fall back to sorting by original vector score
            for point in candidate_points:
                if point.payload is None:
                    point.payload = {}
                point.payload["rerank_score"] = float(point.score)
            return sorted(candidate_points, key=lambda p: p.score, reverse=True)

    def _calculate_lexical_overlap(self, query: str, document: str) -> float:
        """
        Calculates lexical word overlap ratio between query and document text.
        Helps simulate semantic reranking offline without downloading model weights.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        doc_words = set(re.findall(r"\w+", document.lower()))
        if not query_words:
            return 0.0
        intersection = query_words.intersection(doc_words)
        return len(intersection) / len(query_words)
