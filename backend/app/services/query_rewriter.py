import logging
import re

logger = logging.getLogger(__name__)

class QueryRewriter:
    """
    Reformulates follow-up questions using conversation history to make them
    self-contained for vector retrieval.
    """

    def __init__(self, llm_service):
        self.llm_service = llm_service

    def rewrite(self, question: str, history: list[dict]) -> str:
        """
        Rewrites the query using prior conversation history if available.
        Falls back to the original question if history is empty, the question is
        too long, or if the LLM call fails/returns empty results.

        Args:
            question: The raw follow-up user question.
            history:  List of prior conversation messages as [{"role": ..., "content": ...}].

        Returns:
            The reformulated standalone search query, or the original question as fallback.
        """
        # Skip rewriting if there is no history or the query is exceptionally long
        if not history or len(question) > 500:
            return question

        try:
            messages = self._build_rewrite_prompt(question, history)
            
            # Request the LLM to generate the standalone query
            rewritten = self.llm_service.generate(messages)
            if not rewritten:
                return question

            cleaned = rewritten.strip().strip('"').strip("'").strip()
            
            # Sanity check: if model output is empty or unusually long, fallback
            if not cleaned or len(cleaned) > 500:
                return question

            return cleaned

        except Exception as e:
            logger.warning(
                f"QueryRewriter failed (falling back to original query): {str(e)}"
            )
            return question

    def _build_rewrite_prompt(self, question: str, history: list[dict]) -> list[dict]:
        """
        Assembles a tightly scoped chat message context for the query rewriter.
        """
        system_instruction = (
            "You are an expert query reformulation engine for a codebase search engine.\n"
            "Your sole task is to rewrite the user's follow-up question into a single, "
            "self-contained search query that integrates key subjects and nouns from the "
            "conversation history.\n\n"
            "Rules:\n"
            "1. Output ONLY the rewritten search query. Do NOT add any preamble, explanation, "
            "markdown block quotes, or conversational phrases.\n"
            "2. Resolve pronouns (such as 'it', 'this', 'its', 'that function', 'the previous file') "
            "to the specific subjects or symbols mentioned in the history.\n"
            "3. Keep technical symbols, filenames, function names, and parameters exact.\n"
            "4. If the follow-up question is already fully self-contained or does not rely on "
            "conversation history, return it exactly as it is without modifications."
        )

        history_str = self._format_history_for_prompt(history)

        user_content = (
            f"Conversation History:\n"
            f"--------------------------------------------------\n"
            f"{history_str}\n"
            f"--------------------------------------------------\n\n"
            f"Follow-up Question: {question}\n\n"
            f"Rewritten Search Query:"
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": user_content}
        ]

    def _format_history_for_prompt(self, history: list[dict]) -> str:
        """
        Serializes user/assistant turns into a readable text log for prompt context.
        """
        formatted = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if not role or not content:
                continue
            
            speaker = "User" if role == "user" else "Assistant"
            
            # Strip excessive newlines for a compact prompt representation
            compact_content = re.sub(r"\s+", " ", content)
            if len(compact_content) > 200:
                compact_content = compact_content[:200] + "..."
                
            formatted.append(f"{speaker}: {compact_content}")
            
        return "\n".join(formatted)
