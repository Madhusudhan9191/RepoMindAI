from backend.app.chunking.code_chunker import CodeChunker


def main():
    print("Testing Hybrid Code Chunking Engine...")
    
    # 1. Test AST function chunks on Python
    chunker_py = CodeChunker("backend/app/main.py")
    py_ast_chunks = chunker_py.get_function_chunks()
    assert len(py_ast_chunks) > 0, "Expected AST function chunks for backend/app/main.py"
    print(f"Test 1 Passed: Extracted {len(py_ast_chunks)} Python AST function chunks.")

    # 2. Test sliding window chunking on Markdown/Non-Python
    chunker_md = CodeChunker("README.md")
    md_chunks = chunker_md.get_chunks()
    assert len(md_chunks) > 0, "Expected sliding window chunks for README.md"
    assert md_chunks[0]["language"] == "Markdown"
    print(f"Test 2 Passed: Extracted {len(md_chunks)} Markdown sliding window chunks.")

    # 3. Test get_chunks unified hybrid API
    all_chunks = chunker_py.get_chunks()
    assert len(all_chunks) > 0, "Expected hybrid chunks from get_chunks()"
    print(f"Test 3 Passed: Hybrid API get_chunks() verified ({len(all_chunks)} chunks).")

    print("\nAll CodeChunker tests passed successfully!")


if __name__ == "__main__":
    main()