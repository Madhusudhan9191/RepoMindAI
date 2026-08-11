import ast
from pathlib import Path


class PythonParser:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.source_code = self.read_file()
        try:
            self.tree = ast.parse(self.source_code)
        except Exception:
            self.tree = None

    def read_file(self):
        try:
            return self.file_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def get_functions(self):
        if not self.tree:
            return []
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
        return functions

    def get_classes(self):
        if not self.tree:
            return []
        classes = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes

    def get_imports(self):
        if not self.tree:
            return []
        imports = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def get_functions_metadata(self):
        if not self.tree:
            return []
        functions = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = [arg.arg for arg in node.args.args]
                decorators = [
                    ast.unparse(decorator)
                    for decorator in node.decorator_list
                ]
                functions.append({
                    "name": node.name,
                    "arguments": arguments,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node),
                    "decorators": decorators,
                    "is_async": isinstance(node, ast.AsyncFunctionDef)
                })
        return functions

    def extract_metadata(self):
        return {
            "imports": self.get_imports(),
            "functions": self.get_functions_metadata(),
            "classes": self.get_classes(),
        }