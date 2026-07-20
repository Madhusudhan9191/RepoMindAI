import sys
from backend.app.services.query_rewriter import QueryRewriter
from backend.app.services.llm_service import MockLLMProvider
from backend.app.prompts.prompt_builder import PromptBuilder

def run_tests():
    print("Running QueryRewriter tests...")
    
    # 1. Initialize mock dependencies
    mock_llm = MockLLMProvider(model_name="mock-llama")
    rewriter = QueryRewriter(llm_service=mock_llm)
    
    # Test 2: Stateless / empty history should skip rewriting immediately
    q1 = "Where is scanner.py?"
    res1 = rewriter.rewrite(q1, [])
    assert res1 == q1, f"Expected original query, got: {res1}"
    print("Test 1 Passed: Stateless query skipped rewriting successfully.")

    # Test 3: Very long question should skip rewriting immediately
    long_q = "Scanner " * 100
    res2 = rewriter.rewrite(long_q, [{"role": "user", "content": "hello"}])
    assert res2 == long_q, "Expected original long query to be returned as-is."
    print("Test 2 Passed: Excessively long query skipped rewriting successfully.")

    # Test 4: Follow-up question with pronoun is rewritten using mock logic
    history = [
        {"role": "user", "content": "Where is IngestionPipeline defined?"},
        {"role": "assistant", "content": "In ingest_pipeline.py, class IngestionPipeline."}
    ]
    res3 = rewriter.rewrite("What about its imports?", history)
    assert res3 == "What about Scanner's imports?", f"Expected mock query rewritten, got: {res3}"
    print("Test 3 Passed: Pronoun resolved successfully using Mock provider rewrite rule.")

    # Test 5: Fallback logic when LLM returns empty string
    class EmptyLLM:
        def generate(self, messages, **kwargs):
            return ""
            
    rewriter_empty = QueryRewriter(llm_service=EmptyLLM())
    res4 = rewriter_empty.rewrite("What about its imports?", history)
    assert res4 == "What about its imports?", f"Expected fallback to original question, got: {res4}"
    print("Test 4 Passed: Graceful fallback on empty LLM response verified.")

    # Test 6: Fallback logic when LLM returns excessively long rewritten response
    class LongLLM:
        def generate(self, messages, **kwargs):
            return "Standalone Query " * 100
            
    rewriter_long = QueryRewriter(llm_service=LongLLM())
    res5 = rewriter_long.rewrite("What about its imports?", history)
    assert res5 == "What about its imports?", f"Expected fallback to original question, got: {res5}"
    print("Test 5 Passed: Graceful fallback on overly long LLM output verified.")

    # Test 7: Fallback logic when LLM raises exception
    class ExplodingLLM:
        def generate(self, messages, **kwargs):
            raise RuntimeError("API limit exceeded")
            
    rewriter_explode = QueryRewriter(llm_service=ExplodingLLM())
    res6 = rewriter_explode.rewrite("What about its imports?", history)
    assert res6 == "What about its imports?", f"Expected fallback to original question on error, got: {res6}"
    print("Test 6 Passed: Graceful fallback on LLM exception verified.")

    print("\nAll QueryRewriter tests passed successfully!")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError as e:
        print(f"\nTEST FAILED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during tests: {str(e)}")
        sys.exit(1)
