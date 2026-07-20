import logging
import re
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

class ContextCompressor:
    """
    Strips noise from retrieved code blocks by keeping only search-relevant focus lines,
    their surrounding window, and their parent class/function structural signatures.
    """

    KEYWORDS = {
        "def", "class", "import", "return", "self", "from", "as",
        "and", "or", "in", "is", "not", "pass", "try", "except", "elif",
        "where", "how", "what", "who", "why", "when", "the", "a", "an", "this", "that"
    }

    COMMENT_MARKERS = {
        "py": "#",
        "js": "//",
        "ts": "//",
        "java": "//",
        "c": "//",
        "cpp": "//",
        "cs": "//",
        "go": "//",
        "rust": "//",
        "rb": "#",
        "sh": "#",
        "sql": "--",
        "html": "<!-- {} -->",
        "css": "/* {} */",
    }

    def compress(self, query: str, code_content: str, filepath: str = "") -> str:
        """
        Compresses code content by extracting focus lines and structural headers.
        Gracefully falls back to the original content if compression yields
        marginal savings (less than 15% line reduction) or if no matching focus lines exist.
        """
        if not code_content.strip() or not query.strip():
            return code_content

        lines = code_content.splitlines()
        total_lines = len(lines)

        # Safety guard: do not compress short chunks (less than 15 lines)
        if total_lines < 15:
            return code_content

        # 1. Parse and expand query terms (splitting camelCase and snake_case)
        query_words = self._extract_query_words(query)
        if not query_words:
            return code_content

        # 2. Compute focus scores for each line
        scores = self._score_lines(lines, query_words)
        
        # 3. Locate enclosing structural def/class scope indices for all lines
        parent_class, parent_def = self._map_parent_scopes(lines)

        # 4. Compile kept indices (focus line + local window + parent def + parent class)
        kept_indices = set()
        has_focus = False

        for i in range(total_lines):
            # Line is a focus point if its score is positive
            if scores[i] > 0:
                has_focus = True
                # Focus window: +/- 2 lines
                start_win = max(0, i - 2)
                end_win = min(total_lines - 1, i + 2)
                for w_idx in range(start_win, end_win + 1):
                    kept_indices.add(w_idx)
                
                # Enclosing scope preservation
                if i in parent_class:
                    kept_indices.add(parent_class[i])
                if i in parent_def:
                    kept_indices.add(parent_def[i])

        # If no focus line matched, return the original chunk intact
        if not has_focus or not kept_indices:
            return code_content

        # 5. Assemble compressed blocks with language-aware omission markers
        compressed_text = self._build_compressed_text(lines, kept_indices, filepath)
        compressed_lines = len(compressed_text.splitlines())

        # Safety guard: if we didn't compress by at least 15%, return original content
        reduction_ratio = (total_lines - compressed_lines) / total_lines
        if reduction_ratio < 0.15:
            return code_content

        logger.info(
            f"Compressed {filepath or 'chunk'}: {total_lines} -> {compressed_lines} lines "
            f"({reduction_ratio * 100:.1f}% reduction)"
        )

        return compressed_text

    def _split_token(self, token: str) -> List[str]:
        """
        Splits camelCase and snake_case strings into individual lowercased words.
        e.g., "RepositoryScanner" -> ["repository", "scanner"]
        e.g., "initialize_scanner" -> ["initialize", "scanner"]
        """
        words = [token]
        words.extend(token.split("_"))
        
        # Regex split for CamelCase
        camel_splits = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
        if camel_splits:
            words.extend(camel_splits)
            
        return [w.lower() for w in words if w]

    def _extract_query_words(self, query: str) -> Set[str]:
        """
        Tokenizes query, extracts camel/snake case splits, and filters out common keywords.
        """
        raw_tokens = re.findall(r"\w+", query)
        words = set()
        for tok in raw_tokens:
            tok_lower = tok.lower()
            if tok_lower not in self.KEYWORDS:
                words.add(tok_lower)
                words.update(self._split_token(tok))
        return words

    def _score_line(self, line: str, query_words: Set[str]) -> int:
        """
        Calculates focus score of a single line:
        - Exact query word: +2
        - Split word overlap: +1
        - Match is inside def/class header name: +3 extra
        """
        line_lower = line.lower()
        line_tokens = set(re.findall(r"\w+", line))
        score = 0

        is_header = "class " in line_lower or "def " in line_lower

        for tok in line_tokens:
            tok_lower = tok.lower()
            if tok_lower in query_words:
                score += 2
                if is_header:
                    score += 3
            else:
                tok_splits = self._split_token(tok)
                for split in tok_splits:
                    if split in query_words:
                        score += 1
                        if is_header:
                            score += 2
        return score

    def _score_lines(self, lines: List[str], query_words: Set[str]) -> List[int]:
        return [self._score_line(line, query_words) for line in lines]

    def _map_parent_scopes(self, lines: List[str]) -> Tuple[dict, dict]:
        """
        Locates the parent enclosing class and function/def definition line indices
        for each line in the file using indentation rules.
        """
        parent_class = {}
        parent_def = {}

        class_stack = []  # Stack of (indent, line_index)
        def_stack = []

        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if not stripped:
                continue

            # Calculate indentation (count leading whitespace)
            indent = len(line) - len(stripped)

            # Pop stack elements that have larger or equal indentation levels
            while class_stack and class_stack[-1][0] >= indent:
                class_stack.pop()
            while def_stack and def_stack[-1][0] >= indent:
                def_stack.pop()

            # Record def/class lines
            if stripped.startswith("class "):
                class_stack.append((indent, idx))
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                def_stack.append((indent, idx))

            # Store references to immediate enclosing scopes
            if class_stack:
                parent_class[idx] = class_stack[-1][1]
            if def_stack:
                parent_def[idx] = def_stack[-1][1]

        return parent_class, parent_def

    def _build_compressed_text(self, lines: List[str], kept_indices: Set[int], filepath: str) -> str:
        """
        Reconstructs the code file using kept_indices, placing language-aware
        omission comments between discontinuous line blocks.
        """
        total_lines = len(lines)
        sorted_indices = sorted(list(kept_indices))
        
        marker = self._get_omission_marker(filepath)
        compressed_blocks = []
        
        last_idx = -1
        for idx in sorted_indices:
            # Check for range omissions
            if last_idx != -1 and idx > last_idx + 1:
                omitted_count = idx - last_idx - 1
                omitted_range_str = f"... lines {last_idx + 2}-{idx} omitted for brevity ({omitted_count} lines) ..."
                
                # Format comment marker
                if "{}" in marker:
                    omission_text = marker.format(omitted_range_str)
                else:
                    omission_text = f"{marker} {omitted_range_str}"
                compressed_blocks.append(omission_text)
                
            compressed_blocks.append(lines[idx])
            last_idx = idx

        # Handle trailing omission if last line wasn't included
        if last_idx != -1 and last_idx < total_lines - 1:
            omitted_count = total_lines - 1 - last_idx
            omitted_range_str = f"... lines {last_idx + 2}-{total_lines} omitted for brevity ({omitted_count} lines) ..."
            if "{}" in marker:
                omission_text = marker.format(omitted_range_str)
            else:
                omission_text = f"{marker} {omitted_range_str}"
            compressed_blocks.append(omission_text)

        return "\n".join(compressed_blocks)

    def _get_omission_marker(self, filepath: str) -> str:
        """
        Returns file extension specific comment syntax.
        """
        if not filepath:
            return "#"
        ext = filepath.split(".")[-1].lower()
        return self.COMMENT_MARKERS.get(ext, "#")
