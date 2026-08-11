from pathlib import Path


class RepositoryScanner:

    IGNORED_DIRS = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "data",
        "results",
        "figures",
        ".gitignore",
        ".env",
        ".DS_Store",
        "__init__.py",
    }

    IGNORED_EXTENSIONS = {
        ".csv", ".tsv", ".parquet", ".feather", ".pkl", ".h5", ".hdf5",
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf",
        ".zip", ".tar", ".gz", ".7z", ".rar",
        ".exe", ".dll", ".so", ".dylib", ".pbix", ".pyc", ".pyo", ".db", ".sqlite"
    }

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
        ".txt": "Text",
        ".sql": "SQL",
        ".sh": "Shell",
        ".css": "CSS",
        ".html": "HTML",
        ".go": "Go",
        ".rs": "Rust",
        ".java": "Java",
        ".dockerfile": "Dockerfile",
    }

    MAX_FILE_SIZE = 1_000_000  # 1MB limit for text source files

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)

    def detect_language(self, file_path: Path):
        return self.LANGUAGE_MAP.get(file_path.suffix.lower(), "Unknown")

    def scan(self):
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise FileNotFoundError(
                f"Repository path '{self.repo_path}' is not a valid directory."
            )

        inventory = []

        for file in self.repo_path.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() in self.IGNORED_EXTENSIONS:
                continue

            if any(part in self.IGNORED_DIRS for part in file.parts):
                continue

            try:
                size = file.stat().st_size
                if size > self.MAX_FILE_SIZE:
                    continue
            except Exception:
                continue

            lang = self.detect_language(file)
            if lang == "Unknown" and file.suffix.lower() not in self.LANGUAGE_MAP:
                continue

            inventory.append({
                "path": file.relative_to(self.repo_path).as_posix(),
                "language": lang,
                "size": size,
            })

        return inventory