import ast
from pathlib import Path


class PythonParser:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.source_code = self.read_file()
        self.tree = ast.parse(self.source_code)

    def read_file(self):
        return self.file_path.read_text(encoding="utf-8")

    def get_functions(self):
        functions = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        return functions

    def get_classes(self):
        classes = []

        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return classes

    def get_imports(self):
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
        functions = []

        for node in ast.walk(self.tree):

            if isinstance(node, ast.FunctionDef):

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
                    "decorators": decorators
                })

        return functions

    def extract_metadata(self):
        return {
            "imports": self.get_imports(),
            "functions": self.get_functions_metadata(),
            "classes": self.get_classes(),
        }