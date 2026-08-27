class RagError(Exception):
    """Base exception for RAG application errors."""


class DependencyError(RagError):
    """Raised when a required dependency is unavailable."""


class EmbeddingInitializationError(RagError):
    """Raised when embedding initialization fails."""


class EmbeddingRequestError(RagError):
    """Raised when an embedding provider fails to generate a vector."""


class PineconeInitializationError(RagError):
    """Raised when Pinecone cannot be initialized."""


class PineconeIndexError(RagError):
    """Raised when the Pinecone index cannot be created or accessed."""


class DocumentLoadError(RagError):
    """Raised when document loading fails."""


class UploadError(RagError):
    """Raised when a file upload cannot be processed."""


class ItemNotFoundError(RagError):
    """Exception raised when an item is not found."""

    def __init__(self, item_id: int):
        self.item_id = item_id
        self.message = f"Item with ID {item_id} not found."
        super().__init__(self.message)