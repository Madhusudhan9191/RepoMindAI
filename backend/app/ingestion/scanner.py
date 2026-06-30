from pathlib import Path

from pathlib import Path


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
        ".gitignore",
        ".env",
        ".DS_Store",
        "__init__.py",
    }

    LANGUAGE_MAP = {
    ".py": "Python",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".txt": "Text",
    ".env": "Environment",
    
}

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)

    def detect_language(self, file_path: Path):
        return self.LANGUAGE_MAP.get(file_path.suffix.lower(), "Unknown")

    def scan(self):
        inventory = []

        for file in self.repo_path.rglob("*"):
            if not file.is_file():
                continue
            
            if any(part in self.IGNORED_DIRS for part in file.parts):
                continue
            
            inventory.append({
               "path": file.relative_to(self.repo_path).as_posix(),
               "language": self.detect_language(file),
               "size": file.stat().st_size,
            })
        return inventory