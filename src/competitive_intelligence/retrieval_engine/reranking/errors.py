"""Exceptions for the Reranking layer."""

class RerankerError(Exception):
    """Base exception for all reranker-related errors."""
    pass

class BGERerankerError(RerankerError):
    """Exception raised when the underlying BGE cross-encoder fails."""
    pass

class RerankedRetrievalError(RerankerError):
    """Exception raised when the reranked retrieval composed pipeline fails."""
    pass
