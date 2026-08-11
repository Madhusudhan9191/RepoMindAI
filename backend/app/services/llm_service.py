from abc import ABC, abstractmethod
from typing import Generator
import json
import os
import re
import httpx

class BaseLLMProvider(ABC):
    """
    Abstract Base Class for all LLM providers in RepoMindAI.
    """

    @abstractmethod
    def generate(self, messages: list[dict], **kwargs) -> str:
        """
        Generate completion response from a list of chat messages.

        Args:
            messages: List of chat messages in format [{"role": "user/system", "content": "..."}]
            
        Returns:
            The generated string response from the model.
        """
        pass

    @abstractmethod
    def stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """
        Stream completion response token-by-token.

        Args:
            messages: List of chat messages in format [{"role": "user/system", "content": "..."}]
            
        Yields:
            Token or text chunk strings.
        """
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider for local offline testing.
    Generates structured responses citing files and functions found in the context.
    """

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name

    def generate(self, messages: list[dict], **kwargs) -> str:
        # Retrieve system and user message content
        system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        user_content = next((msg["content"] for msg in messages if msg["role"] == "user"), "")

        # Detect query rewrite/reformulation requests
        if "reformulation" in system_content or "rewrite" in system_content:
            # Extract follow-up question
            match = re.search(r"Follow-up Question:\s*([^\n\r]+)", user_content)
            follow_up = match.group(1).strip() if match else "What about its dependencies?"
            # Mock rewrite substitution for offline testing
            rewritten = follow_up.replace("its", "Scanner's").replace("it", "Scanner")
            # Clear double spacing/cleanup
            return rewritten.strip()

        # Regex search to extract files and functions present in context in order
        blocks = re.findall(r"File:\s*([^\n\r]+)\s*\nFunction:\s*([^\n\r]+)", user_content)
        citations = []
        seen = set()
        for f, func in blocks:
            key = (f.strip(), func.strip())
            if key not in seen:
                seen.add(key)
                citations.append(f"`{f.strip()}` (function: `{func.strip()}`)")
            
        citations_str = ", ".join(citations) if citations else "retrieved repository files"

        return (
            f"[MOCK RESPONSE - MODEL: {self.model_name}]\n\n"
            f"Based on the repository context, I analyzed the following code components: {citations_str}.\n\n"
            f"Here is a mock analysis of your request. Everything seems properly set up! "
            f"If you ask a specific query, you can hook this up to Ollama, OpenAI, or Gemini to get real AI answers."
        )

    def stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        response_text = self.generate(messages, **kwargs)
        import time
        words = response_text.split(" ")
        chunk_size = 4  # Yield small word groups for rendering efficiency
        for i in range(0, len(words), chunk_size):
            time.sleep(0.04)
            # Make sure we don't trailing space the final chunk excessively
            chunk = " ".join(words[i:i+chunk_size])
            if i + chunk_size < len(words):
                yield chunk + " "
            else:
                yield chunk


class OllamaProvider(BaseLLMProvider):
    """
    Ollama Provider for zero-cost, fully local model execution.
    """

    def __init__(self, model_name: str = "llama3.3", api_base: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')

    def generate(self, messages: list[dict], **kwargs) -> str:
        url = f"{self.api_base}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": kwargs.get("options", {})
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["message"]["content"]
        except httpx.HTTPError as e:
            return f"Ollama execution error: Failed to connect or receive response from {url}. Details: {str(e)}"
        except Exception as e:
            return f"Unexpected Ollama provider error: {str(e)}"

    def stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        url = f"{self.api_base}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": kwargs.get("options", {})
        }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
        except httpx.HTTPError as e:
            yield f"Ollama streaming connection error: Failed to connect to {url}. Details: {str(e)}"
        except Exception as e:
            yield f"Ollama streaming execution error: {str(e)}"


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API Provider for GPT-based code reasoning.
    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.client = None
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            pass

    def generate(self, messages: list[dict], **kwargs) -> str:
        if not self.api_key:
            return "OpenAI execution error: API key is not configured. Please set it in Settings."
        if not self.client:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                return "OpenAI execution error: The 'openai' library is required. Please run: pip install openai"
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"OpenAI execution error: {str(e)}"

    def stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        if not self.api_key:
            yield "OpenAI streaming error: API key is not configured. Please set it in Settings."
            return
        if not self.client:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                yield "OpenAI streaming error: The 'openai' library is required. Please run: pip install openai"
                return
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                **kwargs
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            yield f"OpenAI streaming error: {str(e)}"


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini Provider using official API endpoints.
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        self.types = None
        self.legacy_client = None
        self.use_legacy = False
        
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            # Dynamic import to bypass IDE static analysis unresolved import warnings
            import importlib
            genai_module = importlib.import_module("google.genai")
            types_module = importlib.import_module("google.genai.types")
            
            self.client = genai_module.Client(api_key=self.api_key)
            self.types = types_module
            self.use_legacy = False
        except ImportError:
            try:
                # Fallback to legacy google-generativeai SDK if installed
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.legacy_client = legacy_genai
                self.use_legacy = True
            except ImportError:
                pass

    def generate(self, messages: list[dict], **kwargs) -> str:
        if not self.api_key:
            return "Gemini execution error: API key is not configured. Please set it in Settings."
        if not self.client and not self.legacy_client:
            self._init_client()
            if not self.client and not self.legacy_client:
                return "Gemini execution error: Google GenAI or google-generativeai SDK is required. Run 'pip install google-genai'."

        try:
            if not self.use_legacy:
                # Map system message and conversation history to standard format
                system_instruction = None
                contents = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] in ("user", "assistant"):
                        role = "model" if msg["role"] == "assistant" else "user"
                        contents.append(self.types.Content(
                            role=role,
                            parts=[self.types.Part.from_text(text=msg["content"])]
                        ))
                
                config = self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    **kwargs
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                return response.text
            else:
                # Legacy SDK usage fallback (with multi-turn history mapping)
                system_instruction = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                contents = []
                for msg in messages:
                    if msg["role"] in ("user", "assistant"):
                        role = "model" if msg["role"] == "assistant" else "user"
                        contents.append({
                            "role": role,
                            "parts": [msg["content"]]
                        })
                
                model = self.legacy_client.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(contents)
                return response.text
        except Exception as e:
            return f"Gemini execution error: {str(e)}"

    def stream(self, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        if not self.api_key:
            yield "Gemini streaming error: API key is not configured. Please set it in Settings."
            return
        if not self.client and not self.legacy_client:
            self._init_client()
            if not self.client and not self.legacy_client:
                yield "Gemini streaming error: Google GenAI or google-generativeai SDK is required. Run 'pip install google-genai'."
                return

        try:
            if not self.use_legacy:
                system_instruction = None
                contents = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] in ("user", "assistant"):
                        role = "model" if msg["role"] == "assistant" else "user"
                        contents.append(self.types.Content(
                            role=role,
                            parts=[self.types.Part.from_text(text=msg["content"])]
                        ))
                
                config = self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    **kwargs
                )
                
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            else:
                system_instruction = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                contents = []
                for msg in messages:
                    if msg["role"] in ("user", "assistant"):
                        role = "model" if msg["role"] == "assistant" else "user"
                        contents.append({
                            "role": role,
                            "parts": [msg["content"]]
                        })
                
                model = self.legacy_client.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction
                )
                response = model.generate_content(contents, stream=True)
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
        except Exception as e:
            yield f"Gemini streaming error: {str(e)}"


def get_llm_provider(
    provider_name: str,
    model_name: str,
    api_key: str = None,
    api_base: str = None
) -> BaseLLMProvider:
    """
    Factory function to fetch configured LLM Provider.
    """
    provider_name = provider_name.strip().lower()
    
    if provider_name == "mock":
        return MockLLMProvider(model_name=model_name)
    elif provider_name == "ollama":
        if api_base:
            return OllamaProvider(model_name=model_name, api_base=api_base)
        return OllamaProvider(model_name=model_name)
    elif provider_name == "openai":
        return OpenAIProvider(api_key=api_key or os.getenv("OPENAI_API_KEY"), model_name=model_name)
    elif provider_name == "gemini":
        return GeminiProvider(api_key=api_key or os.getenv("GEMINI_API_KEY"), model_name=model_name)
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider_name}'. "
            "Supported providers are: 'mock', 'ollama', 'openai', 'gemini'."
        )
