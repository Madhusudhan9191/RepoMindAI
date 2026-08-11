from evaluation.eval_suite import run_benchmark


def test_eval_suite_execution():
    """
    Verifies that the RAG benchmark evaluation suite executes cleanly and returns valid metric dictionaries.
    """
    metrics_table = run_benchmark()
    assert len(metrics_table) == 4, f"Expected 4 evaluation stages, got {len(metrics_table)}"
    
    for stage_result in metrics_table:
        assert "precision_at_5" in stage_result
        assert "recall_at_5" in stage_result
        assert "mrr" in stage_result
        assert "compression_pct" in stage_result
        assert "avg_latency_ms" in stage_result
        assert stage_result["precision_at_5"] >= 0.0
        assert stage_result["recall_at_5"] >= 0.0
        assert stage_result["mrr"] >= 0.0


if __name__ == "__main__":
    test_eval_suite_execution()
    print("\nAll evaluation suite tests passed successfully!")
