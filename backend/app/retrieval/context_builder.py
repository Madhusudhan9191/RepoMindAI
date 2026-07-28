class ContextBuilder:
    """
    Responsible for compiling raw vector store search results into a clean,
    consistently formatted context block for prompts, and generating citations.
    """

    def build_context_and_citations(self, results: list) -> tuple[str, list[dict]]:
        """
        Formats retrieved Qdrant ScoredPoint structures into prompt context blocks
        and a standardized citations dictionary list.

        Args:
            results: List of ScoredPoint results from Qdrant client.

        Returns:
            A tuple of (formatted_context_string, list_of_citations_dicts)
        """
        if not results:
            return "", []

        context_blocks = []
        citations = []

        for result in results:
            payload = result.payload

            # Format the context block for system prompt builder
            block = (
                f"File: {payload['path']}\n"
                f"Function: {payload['name']}\n"
                f"Lines: {payload['start_line']} - {payload['end_line']}\n"
                f"Code:\n{payload['content']}"
            )
            context_blocks.append(block)

            # Format the citation metadata payload
            citation = {
                "file": payload["path"],
                "function": payload["name"],
                "type": payload["type"],
                "start_line": int(payload["start_line"]),
                "end_line": int(payload["end_line"]),
                "score": round(float(result.score), 4),
                "content": payload.get("content", "")
            }
            citations.append(citation)

        # Merge all chunks separated by double newlines
        formatted_context = "\n\n".join(context_blocks)
        
        return formatted_context, citations
