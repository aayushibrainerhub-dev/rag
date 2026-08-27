import glob
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from exceptions.rag import (
    EmbeddingRequestError,
    EmbeddingInitializationError,
    PineconeInitializationError,
)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

latest_upload_id: Optional[str] = None


def set_latest_upload_id(upload_id: str) -> None:
    global latest_upload_id
    latest_upload_id = upload_id


class PineconeVectorStore:
    def __init__(self, index_name: str = "rag-documents"):
        self.base_index_name = index_name
        self.index_name = index_name
        self.pc = None
        self.index = None
        self.embeddings = None
        self.embedding_dim = None
        self.documents = []
        self.embeddings_type = "local"
        self.embedding_model_name = os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.embeddings = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = (
            self.embeddings.get_embedding_dimension()
            if hasattr(self.embeddings, "get_embedding_dimension")
            else self.embeddings.get_sentence_embedding_dimension()
        )
        print(f"Using local HuggingFace embeddings model: {self.embedding_model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")

        # This index holds dense vectors only. Lexical retrieval and rank fusion
        # are performed by LangChain's BM25Retriever and EnsembleRetriever.
        self.index_name = f"{self.base_index_name}-{self.embedding_dim}-dense"

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise PineconeInitializationError("PINECONE_API_KEY is required to initialize Pinecone.")

        self.pc = Pinecone(api_key=api_key)
        if not self.pc.has_index(self.index_name):
            print(f"Creating serverless index '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        self.index = self.pc.Index(self.index_name)
        print(f"Successfully initialized Pinecone index: {self.index_name}")

    def _embed_text(self, text: str) -> list[float]:
        if self.embeddings is None:
            raise EmbeddingInitializationError("Embeddings client is unavailable")

        vector = self.embeddings.encode(
            [text],
            convert_to_numpy=True,
        )[0]
        if not hasattr(vector, "tolist"):
            raise EmbeddingRequestError("Embedding encoder returned an unexpected type")
        return vector.tolist()

    def _build_pinecone_filter(self, metadata_filter: dict[str, Any]) -> dict[str, Any]:
        return {key: {"$eq": value} for key, value in metadata_filter.items()}

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to Pinecone vector store, or fall back to local storage."""
        if not self.index or not self.embeddings:
            self.documents.extend(documents)
            return

        vectors_to_upsert = []
        for doc in documents:
            raw_text = doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)
            text = raw_text.strip()
            if not text:
                continue

            embedding = self._embed_text(text)

            metadata = {"text": text}
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)

            vectors_to_upsert.append(
                {
                    "id": str(uuid.uuid4()),
                    "values": embedding,
                    "metadata": metadata,
                }
            )

        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)

        # Keep the process-local corpus for LangChain BM25 retrieval. Pinecone
        # remains the durable dense-vector backend.
        self.documents.extend(documents)

    def similarity_search(self, query: str, k: int = 4, metadata_filter: Optional[dict[str, Any]] = None) -> list[Document]:
        """Search using Pinecone when embeddings are available, otherwise use local keyword matching.

        Supports an optional metadata_filter dict which will be passed to Pinecone's
        filter parameter when using the remote index. When falling back to the local
        store, documents will be pre-filtered using simple equality checks on metadata.
        """
        if not self.embeddings or not self.index:
            return self._local_similarity_search(query, k, metadata_filter)

        query_text = query if isinstance(query, str) else str(query)
        query_text = query_text.strip()
        if not query_text:
            return self._local_similarity_search(query, k, metadata_filter)

        query_embedding = self._embed_text(query_text)

        # Use Pinecone filtering if provided
        if metadata_filter:
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=self._build_pinecone_filter(metadata_filter),
            )
        else:
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
            )
        
        docs = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            docs.append(Document(page_content=text, metadata=metadata))

        return docs
    
    def _local_similarity_search(self, query: str, k: int = 4, metadata_filter: Optional[dict[str, Any]] = None) -> list[Document]:
        """Fallback local similarity search using term matching and optional metadata filtering."""
        if not self.documents:
            return []
        
        # Apply simple metadata filtering first
        candidates = self.documents
        if metadata_filter:
            def matches_metadata(doc: Document) -> bool:
                for key, val in metadata_filter.items():
                    # convert both sides to string for simple equality matching
                    if str(doc.metadata.get(key, "")) != str(val):
                        return False
                return True

            candidates = [doc for doc in self.documents if matches_metadata(doc)]

        if not candidates:
            return []
        
        query_terms = [term.lower() for term in query.split() if term]
        if not query_terms:
            return candidates[:k]
        
        scored_documents: list[tuple[int, Document]] = []
        for document in candidates:
            content = document.page_content.lower()
            score = sum(1 for term in query_terms if term in content)
            scored_documents.append((score, document))
        
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored_documents[:k] if document]


def _normalize_document_stem(pdf_path: str) -> str:
    return Path(pdf_path).stem.strip().lower().replace(" ", "_")


def _build_document_id(pdf_path: str, upload_id: Optional[str]) -> str:
    stem = _normalize_document_stem(pdf_path)
    if upload_id:
        return f"{stem}_{upload_id[:8]}"
    return stem


def load_documents_from_pdf_paths(
    pdf_paths: list[str],
    upload_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_name: Optional[str] = None,
) -> list[Document]:
    if not pdf_paths:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    result: list[Document] = []

    for pdf_path in pdf_paths:
        if not os.path.isfile(pdf_path):
            continue

        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        chunks = splitter.split_documents(raw_docs)
        total_chunks = len(chunks)
        document_id = _build_document_id(pdf_path, upload_id)
        document_type = _normalize_document_stem(pdf_path)

        for chunk_no, chunk in enumerate(chunks, start=1):
            raw_page = getattr(chunk, "metadata", {}).get("page")
            if isinstance(raw_page, int):
                page_number = raw_page + 1
            else:
                page_number = chunk_no

            metadata = {
                "document_id": document_id,
                "source": os.path.basename(pdf_path),
                "chunk_no": chunk_no,
                "total_chunks": total_chunks,
                "page": page_number,
                "document_type": document_type,
            }
            if upload_id is not None:
                metadata["upload_id"] = upload_id
            if user_id is not None:
                metadata["user_id"] = user_id
            if user_name is not None:
                metadata["user_name"] = user_name

            result.append(Document(page_content=chunk.page_content, metadata=metadata))

    return result


def load_documents_from_pdf_folder() -> list[Document]:
    documents_path = os.path.join(os.path.dirname(__file__), "documents")
    if not os.path.isdir(documents_path):
        return []

    pdf_paths = sorted(glob.glob(os.path.join(documents_path, "*.pdf")))
    return load_documents_from_pdf_paths(pdf_paths)


# Initialize Pinecone vector store
vector_store = PineconeVectorStore(index_name="rag-documents")
# Load and add documents
documents = load_documents_from_pdf_folder()
if documents:
    vector_store.add_documents(documents)
