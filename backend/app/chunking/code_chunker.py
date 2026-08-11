import ast
from pathlib import Path


class CodeChunker:

    LANGUAGE_MAP = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript (React)",
        ".ts": "TypeScript",
        ".tsx": "TypeScript (React)",
        ".md": "Markdown",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".sql": "SQL",
        ".sh": "Shell",
        ".css": "CSS",
        ".html": "HTML",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".dockerfile": "Dockerfile",
    }

    def __init__(self, file_path, language: str = None):
        self.file_path = Path(file_path)
        self.language = language or self.LANGUAGE_MAP.get(self.file_path.suffix.lower(), "Text")
        
        try:
            self.source_code = self.file_path.read_text(encoding="utf-8")
        except Exception:
            self.source_code = ""

        self.lines = self.source_code.splitlines()

        # Safely parse AST for Python files
        self.tree = None
        if self.file_path.suffix.lower() == ".py":
            try:
                self.tree = ast.parse(self.source_code)
            except Exception:
                self.tree = None

    def get_function_chunks(self):
        """
        Extract AST-based function and async function chunks (Python specific).
        """
        if not self.tree:
            return []

        chunks = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                chunk_text = "\n".join(self.lines[start - 1 : end])

                chunks.append({
                    "id": f"{self.file_path.as_posix()}::{node.name}",
                    "type": "function",
                    "name": node.name,
                    "path": self.file_path.as_posix(),
                    "language": "Python",
                    "start_line": start,
                    "end_line": end,
                    "content": chunk_text
                })

        return chunks

    def get_class_chunks(self):
        """
        Extract AST-based class chunks (Python specific).
        """
        if not self.tree:
            return []

        chunks = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                chunk_text = "\n".join(self.lines[start - 1 : end])

                chunks.append({
                    "id": f"{self.file_path.as_posix()}::{node.name}",
                    "type": "class",
                    "name": node.name,
                    "path": self.file_path.as_posix(),
                    "language": "Python",
                    "start_line": start,
                    "end_line": end,
                    "content": chunk_text
                })

        return chunks

    def get_sliding_chunks(self, chunk_size: int = 35, overlap: int = 5):
        """
        Language-aware sliding line window chunking for non-Python or fallback files.
        """
        if not self.lines:
            return []

        chunks = []
        total_lines = len(self.lines)
        step = max(1, chunk_size - overlap)
        chunk_idx = 1

        for start_idx in range(0, total_lines, step):
            end_idx = min(total_lines, start_idx + chunk_size)
            chunk_lines = self.lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines).strip()

            if not chunk_text:
                continue

            chunks.append({
                "id": f"{self.file_path.as_posix()}::block_{chunk_idx}",
                "type": "block",
                "name": f"{self.file_path.name}:L{start_idx + 1}-L{end_idx}",
                "path": self.file_path.as_posix(),
                "language": self.language,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "content": chunk_text
            })
            chunk_idx += 1

            if end_idx >= total_lines:
                break

        return chunks

    def get_chunks(self):
        """
        Hybrid multi-language chunking strategy:
        1. If Python: try AST functions & classes.
        2. If non-Python or AST returned no chunks: use sliding line window chunking.
        """
        if self.file_path.suffix.lower() == ".py" and self.tree is not None:
            ast_chunks = self.get_function_chunks()
            if ast_chunks:
                return ast_chunks

        # Fallback for non-Python, non-function Python modules, or syntax errors
        return self.get_sliding_chunks()