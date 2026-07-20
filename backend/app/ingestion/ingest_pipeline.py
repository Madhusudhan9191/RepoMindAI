import time
from pathlib import Path

from backend.app.ingestion.scanner import RepositoryScanner
from backend.app.parsers.python_parser import PythonParser
from backend.app.chunking.code_chunker import CodeChunker
from backend.app.core.container import get_container
from backend.app.core.exceptions import RepositoryNotFoundException


class IngestionPipeline:
    """
    Repository Ingestion Pipeline

    Repository
        ↓
    Scanner
        ↓
    Parser
        ↓
    Chunker
        ↓
    Embedding
        ↓
    Qdrant
    """

    def __init__(self, repo_path: str):

        self.repo_path = Path(repo_path)

        self.scanner = RepositoryScanner(self.repo_path)

        container = get_container()

        self.embedding_service = container.embedding_service
        self.qdrant = container.qdrant_service

    def ingest(self) -> dict:
        start_time = time.perf_counter()

        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise RepositoryNotFoundException(
                f"Repository path '{self.repo_path}' is not a valid directory."
            )

        print("=" * 60)
        print("Starting Repository Ingestion...")
        print("=" * 60)

        inventory = self.scanner.scan()

        print(f"Files discovered : {len(inventory)}")

        total_chunks = 0
        indexed_files = 0

        for file in inventory:

            relative_path = file["path"]

            full_path = self.repo_path / relative_path

            if full_path.suffix != ".py":
                continue

            try:

                print(f"\nProcessing : {relative_path}")

                parser = PythonParser(full_path)

                metadata = parser.extract_metadata()

                print(
                    f"Functions : {len(metadata['functions'])}"
                )

                print(
                    f"Classes   : {len(metadata['classes'])}"
                )

                chunker = CodeChunker(full_path)

                chunks = chunker.get_function_chunks()

                if not chunks:

                    print("No chunks found.")

                    continue

                chunks = self.embedding_service.embed_chunks(
                    chunks
                )

                self.qdrant.insert_chunks(chunks)

                indexed_files += 1

                total_chunks += len(chunks)

                print(
                    f"Chunks indexed : {len(chunks)}"
                )

            except Exception as e:

                print(
                    f"Failed : {relative_path}"
                )

                print(e)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        print("\n" + "=" * 60)
        print("Repository Indexed Successfully")
        print("=" * 60)

        print(f"Python Files Indexed : {indexed_files}")

        print(f"Total Chunks         : {total_chunks}")

        print(
            f"Vectors in Qdrant    : {self.qdrant.count()}"
        )

        print(f"Duration             : {duration_ms} ms")

        print("=" * 60)

        return {
            "files_scanned": len(inventory),
            "python_files": indexed_files,
            "chunks": total_chunks,
            "vectors": self.qdrant.count(),
            "duration_ms": duration_ms
        }