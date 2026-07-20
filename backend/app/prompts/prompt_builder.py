class PromptBuilder:
    """
    Constructs structured LLM prompt templates combining instructions,
    retrieved repository code context, conversation history, and the user's query.

    Two prompt strategies are available:

        build_rag_prompt()               — stateless single-turn (no history)
        build_rag_prompt_with_history()  — multi-turn with bounded conversation history

    Both methods produce the same message list format accepted by all LLM providers.
    """

    SYSTEM_PROMPT = (
        "You are RepoMindAI, a state-of-the-art AI assistant specializing in analyzing, "
        "explaining, and answering questions about software codebases.\n\n"
        "Instructions:\n"
        "1. Answer the user's question using ONLY the provided repository context below.\n"
        "2. If the answer cannot be found in the provided context, state clearly: "
        "\"I cannot find the answer to this question in the indexed codebase.\"\n"
        "3. Cite the exact file names and line numbers when referencing specific code blocks.\n"
        "4. Be technical, precise, and concise. Explain code structure and usage where helpful."
    )

    SYSTEM_PROMPT_WITH_HISTORY = (
        "You are RepoMindAI, a state-of-the-art AI assistant specializing in analyzing, "
        "explaining, and answering questions about software codebases.\n\n"
        "Instructions:\n"
        "1. Answer the user's question using ONLY the provided repository context below.\n"
        "2. If the answer cannot be found in the provided context, state clearly: "
        "\"I cannot find the answer to this question in the indexed codebase.\"\n"
        "3. Cite the exact file names and line numbers when referencing specific code blocks.\n"
        "4. Be technical, precise, and concise. Explain code structure and usage where helpful.\n"
        "5. Use the conversation history to resolve references such as \"it\", \"that function\", "
        "\"the previous file\", or omitted subjects. "
        "If the history and retrieved repository context conflict, "
        "prefer the retrieved repository context."
    )

    def build_rag_prompt(self, query: str, context: str) -> list[dict]:
        """
        Builds a single-turn structured prompt (no conversation history).

        Used when there is no prior conversation context, or as a fallback
        when history is empty.

        Args:
            query:   The user's natural language question.
            context: Formatted code context string retrieved from Qdrant.

        Returns:
            List of role/content message dicts for the LLM API.
        """
        user_message_content = (
            f"Repository Context:\n"
            f"--------------------------------------------------\n"
            f"{context}\n"
            f"--------------------------------------------------\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_message_content},
        ]

    def build_rag_prompt_with_history(
        self,
        query: str,
        context: str,
        history: list[dict],
        max_history_messages: int = 10,
    ) -> list[dict]:
        """
        Builds a multi-turn prompt with conversation history.

        Message structure:

            [system]    RepoMindAI instructions (with history-aware rule #5)
            [user]      Exchange 1 — user turn
            [assistant] Exchange 1 — assistant turn
            ...         (up to max_history_messages / 2 complete exchanges)
            [user]      CURRENT — repository context + question

        History is trimmed to complete user/assistant exchange pairs so the
        conversation is always structurally balanced. Only completed exchanges
        (both user AND assistant content present) are included.

        Args:
            query:                The current user question.
            context:              Formatted code context string from Qdrant.
            history:              List of past {"role": ..., "content": ...} dicts.
                                  Must contain only "user" and "assistant" roles.
            max_history_messages: Hard cap on total history messages included
                                  (default 10 = 5 complete exchanges). Applied
                                  before exchange-pairing so the final count is
                                  always even.

        Returns:
            List of role/content message dicts for the LLM API.
        """
        # --- 1. Build bounded, balanced exchange history ---
        trimmed = self._build_exchange_history(history, max_history_messages)

        # --- 2. Build the current-turn user message (context + question) ---
        current_user_message = (
            f"Repository Context:\n"
            f"--------------------------------------------------\n"
            f"{context}\n"
            f"--------------------------------------------------\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

        # --- 3. Assemble: system → history turns → current question ---
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT_WITH_HISTORY}]
        messages.extend(trimmed)
        messages.append({"role": "user", "content": current_user_message})

        return messages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_exchange_history(
        self,
        history: list[dict],
        max_messages: int,
    ) -> list[dict]:
        """
        Converts a flat message list into balanced user/assistant exchange pairs.

        Rules:
        - Only "user" and "assistant" roles are accepted.
        - Pairs are built from the end of the list (recency bias).
        - The result always starts with a "user" turn (never "assistant" first).
        - An incomplete trailing assistant turn (no user turn before it) is dropped.

        Args:
            history:      Raw message list from the frontend.
            max_messages: Maximum total messages to consider.

        Returns:
            Balanced list of alternating user/assistant dicts.
        """
        if not history:
            return []

        # Take the most recent max_messages entries
        recent = [
            m for m in history
            if m.get("role") in ("user", "assistant") and m.get("content", "").strip()
        ][-max_messages:]

        # Build complete pairs working backwards so we always include the most
        # recent exchange and never end up with a dangling assistant message.
        pairs: list[tuple[dict, dict]] = []
        i = len(recent) - 1
        while i >= 1:
            if recent[i]["role"] == "assistant" and recent[i - 1]["role"] == "user":
                pairs.append((recent[i - 1], recent[i]))
                i -= 2
            else:
                i -= 1

        # Reverse so chronological order is preserved (oldest first)
        pairs.reverse()

        # Flatten pairs back to a flat list
        result = []
        for user_msg, asst_msg in pairs:
            result.append({"role": "user",      "content": user_msg["content"]})
            result.append({"role": "assistant", "content": asst_msg["content"]})

        return result
