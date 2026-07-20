import ast
from pathlib import Path

class CodeChunker:

    def __init__(self, file_path):
        self.file_path = Path(file_path)

        self.source_code = self.file_path.read_text(
            encoding="utf-8"
        )

        self.lines = self.source_code.splitlines()

        self.tree = ast.parse(self.source_code)


    def get_function_chunks(self):
        
        chunks = []
        
        for node in ast.walk(self.tree):
            
            if isinstance(node, ast.FunctionDef):
                start = node.lineno
                end = node.end_lineno
                
                chunk = "\n".join(
                    self.lines[start - 1:end])
                
                chunks.append({
                    
                    "id": f"{self.file_path.as_posix()}::{node.name}",

                    "type": "function",

                    "name": node.name,

                    "path": self.file_path.as_posix(),

                    "language": "Python",

                    "start_line": start,

                    "end_line": end,

                    "content": chunk
                })
                
        return chunks