import time
import uuid

class TelemetryCollector:
    """
    Utility helper to track and measure latency stages, counts, and
    metadata throughout a single RAG request pipeline.
    """

    def __init__(self, request_id: str = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.start_time = time.perf_counter()
        
        # Latency trackers
        self.stages = {}
        self._current_stage = None
        self._stage_start = None
        
        # Metrics trackers
        self.counts = {
            "candidates_retrieved": 0,
            "after_mmr": 0,
            "after_reranker": 0,
            "original_context_lines": 0,
            "compressed_context_lines": 0,
            "original_context_tokens": 0,
            "compressed_context_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        
        # Pipeline configuration flags
        self.config = {
            "rewrite_query": False,
            "use_mmr": False,
            "use_reranker": False,
            "use_compressor": False,
        }
        
        # Average scores
        self.scores = {
            "avg_vector_similarity": 0.0,
            "avg_mmr_score": 0.0,
            "avg_rerank_score": 0.0,
        }

    def start_stage(self, stage_name: str):
        """
        Starts timing a specific execution stage. Ends any previously active stage.
        """
        if self._current_stage:
            self.end_stage()
        self._current_stage = stage_name
        self._stage_start = time.perf_counter()

    def end_stage(self):
        """
        Ends timing the currently active execution stage.
        """
        if self._current_stage and self._stage_start:
            elapsed = int((time.perf_counter() - self._stage_start) * 1000)
            self.stages[f"{self._current_stage}_ms"] = elapsed
            self._current_stage = None
            self._stage_start = None

    def set_config(self, **kwargs):
        """
        Sets active retrieval configuration flags.
        """
        self.config.update(kwargs)

    def set_count(self, key: str, value: int):
        """
        Sets a metrics count parameter.
        """
        if key in self.counts:
            self.counts[key] = value

    def set_scores(self, **kwargs):
        """
        Sets average score values.
        """
        self.scores.update(kwargs)

    def export(self) -> dict:
        """
        Stops all active timers and packages the full telemetry dataset.
        Calculates derived ratios (e.g. compression percentage) and returns the dictionary.
        """
        if self._current_stage:
            self.end_stage()
            
        total_ms = int((time.perf_counter() - self.start_time) * 1000)
        self.stages["total_ms"] = total_ms
        
        # Calculate derived compression ratio
        orig_lines = self.counts["original_context_lines"]
        comp_lines = self.counts["compressed_context_lines"]
        compression_ratio = 0.0
        if orig_lines > 0:
            compression_ratio = round((orig_lines - comp_lines) / orig_lines, 4)
            
        return {
            "request_id": self.request_id,
            "latencies": self.stages,
            "counts": self.counts,
            "config": self.config,
            "scores": self.scores,
            "compression_ratio": compression_ratio,
        }
