from backend.app.chunking.code_chunker import CodeChunker


def main():

    chunker = CodeChunker(
        "backend/app/main.py"
    )

    chunks = chunker.get_function_chunks()

    for chunk in chunks:

        print("=" * 60)

        print(chunk)


if __name__ == "__main__":
    main()