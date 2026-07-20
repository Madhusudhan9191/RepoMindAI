class RepoMindAIException(Exception):
    """Base exception for all RepoMindAI errors."""
    pass


class RepositoryNotFoundException(RepoMindAIException):
    """Raised when the target repository path does not exist or is not a directory."""
    pass


class EmptyQuestionException(RepoMindAIException):
    """Raised when the submitted question is blank or empty."""
    pass


class NoChunksRetrievedException(RepoMindAIException):
    """Raised when no matching code chunks could be retrieved from the vector store."""
    pass


class LLMTimeoutException(RepoMindAIException):
    """Raised when the LLM service fails to generate a response in time or fails to execute."""
    pass
