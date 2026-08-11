import time
import hashlib
import json
from pathlib import Path

from backend.app.ingestion.scanner import RepositoryScanner
from backend.app.chunking.code_chunker import CodeChunker
from backend.app.core.container import get_container
from backend.app.core.exceptions import RepositoryNotFoundException


class IngestionPipeline:
    """
    Production Repository Ingestion Pipeline

    Repository
        ↓
    Scanner & SHA-256 Incremental Delta Detector
        ↓
    Hybrid Multi-Language Chunker
        ↓
    Batch Inference Embedding
        ↓
    Qdrant Batch Upsert
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.scanner = RepositoryScanner(self.repo_path)
        self.hash_manifest_file = self.repo_path / ".file_hashes.json"

        container = get_container()
        self.embedding_service = container.embedding_service
        self.qdrant = container.qdrant_service

    def _compute_hash(self, file_path: Path) -> str:
        try:
            return hashlib.sha256(file_path.read_bytes()).hexdigest()
        except Exception:
            return ""

    def _load_manifest(self) -> dict:
        if self.hash_manifest_file.exists():
            try:
                return json.loads(self.hash_manifest_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_manifest(self, manifest: dict):
        try:
            self.hash_manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

    def ingest(self, force_reindex: bool = False) -> dict:
        start_time = time.perf_counter()

        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise RepositoryNotFoundException(
                f"Repository path '{self.repo_path}' is not a valid directory."
            )

        print("=" * 60)
        print("Starting Incremental Repository Ingestion...")
        print("=" * 60)

        inventory = self.scanner.scan()
        print(f"Files discovered : {len(inventory)}")

        previous_manifest = {} if force_reindex else self._load_manifest()
        current_manifest = {}

        current_paths = set()
        files_to_process = []
        files_deleted = []

        # 1. Identify new, modified, and unchanged files
        for file_info in inventory:
            rel_path = file_info["path"]
            full_path = self.repo_path / rel_path
            current_paths.add(rel_path)

            file_hash = self._compute_hash(full_path)
            current_manifest[rel_path] = file_hash

            prev_hash = previous_manifest.get(rel_path)
            if force_reindex or prev_hash != file_hash:
                files_to_process.append((rel_path, full_path, file_info["language"], prev_hash is not None))

        # 2. Identify deleted files
        for old_path in previous_manifest.keys():
            if old_path not in current_paths and old_path != ".file_hashes.json":
                files_deleted.append(old_path)

        # Sync deletion in Qdrant for modified & deleted files
        for deleted_path in files_deleted:
            print(f"Syncing deletion : {deleted_path}")
            self.qdrant.delete_chunks_by_path(deleted_path)

        for rel_path, _, _, is_modified in files_to_process:
            if is_modified:
                print(f"Re-indexing modified file : {rel_path}")
                self.qdrant.delete_chunks_by_path(rel_path)

        # 3. Chunk all modified/new files
        all_chunks = []
        processed_files_count = 0

        for rel_path, full_path, language, _ in files_to_process:
            try:
                chunker = CodeChunker(full_path, language=language)
                chunks = chunker.get_chunks()

                if chunks:
                    all_chunks.extend(chunks)
                    processed_files_count += 1
            except Exception as e:
                print(f"Failed chunking {rel_path}: {e}")

        # 4. Batch Embed & Batch Upsert
        if all_chunks:
            print(f"\nBatch Embedding {len(all_chunks)} chunks across {processed_files_count} files...")
            all_chunks = self.embedding_service.embed_chunks(all_chunks)

            print(f"Batch Upserting vectors into Qdrant...")
            self.qdrant.insert_chunks(all_chunks)

        # 5. Save updated manifest
        self._save_manifest(current_manifest)

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        print("\n" + "=" * 60)
        print("Repository Ingestion Completed")
        print("=" * 60)
        print(f"Files Scanned      : {len(inventory)}")
        print(f"Files Indexed      : {processed_files_count}")
        print(f"Files Skipped      : {len(inventory) - processed_files_count}")
        print(f"Files Deleted Sync : {len(files_deleted)}")
        print(f"Total Chunks       : {len(all_chunks)}")
        print(f"Vectors in Qdrant  : {self.qdrant.count()}")
        print(f"Duration           : {duration_ms} ms")
        print("=" * 60)

        return {
            "files_scanned": len(inventory),
            "python_files": processed_files_count,
            "chunks": len(all_chunks),
            "vectors": self.qdrant.count(),
            "duration_ms": duration_ms
        }