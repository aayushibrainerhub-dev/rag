import os
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict
from pinecone import Pinecone, ServerlessSpec

try:  # LangChain < 1.0
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers.ensemble import EnsembleRetriever
except ImportError:  # LangChain 1.0+ moved legacy retrievers here.
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_classic.retrievers.ensemble import EnsembleRetriever

from database.vectorstore import vector_store

if load_dotenv is not None:
    load_dotenv()


class FilteredVectorStoreRetriever(BaseRetriever):
    """Expose the project's vector store through LangChain's retriever API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_store: Any
    metadata_filter: dict[str, Any]
    k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Any]:
        return self.vector_store.similarity_search(
            query,
            k=self.k,
            metadata_filter=self.metadata_filter,
        )


class EmptyRetriever(BaseRetriever):
    """A retriever that contributes no ranks when the BM25 corpus is empty."""

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Any]:
        return []


def build_bm25_retriever(
    documents: list[Any], metadata_filter: dict[str, Any], k: int
) -> BaseRetriever:
    """Create a BM25 retriever over only the caller's permitted chunks."""
    filtered_documents = [
        document
        for document in documents
        if all(str(document.metadata.get(key, "")) == str(value)
               for key, value in metadata_filter.items())
    ]
    if not filtered_documents:
        return EmptyRetriever()

    retriever = BM25Retriever.from_documents(filtered_documents)
    retriever.k = k
    return retriever


class PrintingMultiQueryRetriever(MultiQueryRetriever):
    """Print generated query variants so retrieval can be inspected locally."""

    def generate_queries(self, question: str, run_manager: Any) -> list[str]:
        queries = super().generate_queries(question, run_manager)
        total_queries = len(queries) + int(self.include_original)
        print(
            f"MultiQueryRetriever generated {len(queries)} alternate queries "
            f"({total_queries} searches including the original query)."
        )
        for number, query in enumerate(queries, start=1):
            print(f"  Query {number}: {query}")
        if self.include_original:
            print(f"  Original query: {question}")
        return queries


class RagService:
    """RAG service with process-local session state backed by LangChain's store."""

    # MultiQueryRetriever can return up to four groups of results. A cross
    # encoder reranks a broader candidate set before this final limit is used.
    max_retrieved_documents = int(os.getenv("RAG_MAX_RETRIEVED_DOCUMENTS", "6"))
    reranker_candidate_documents = int(
        os.getenv("RAG_RERANK_CANDIDATE_DOCUMENTS", "10")
    )
    max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "8000"))
    max_history_chars = int(os.getenv("RAG_MAX_HISTORY_CHARS", "2000"))
    dense_retriever_weight = float(os.getenv("RAG_DENSE_RETRIEVER_WEIGHT", "0.5"))
    reranker_model_name = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    _cross_encoder: Any = None

    # A class-level store lets FastAPI's per-request service instances access the
    # same chat session while keeping session persistence behind LangChain's API.
    session_store: InMemoryStore = InMemoryStore()

    def __init__(self):
        self.vector_store = vector_store
        self.prompt = self._build_prompt()
        self.llm = self._build_llm()

    @classmethod
    def _get_cross_encoder(cls) -> Any:
        """Load the cross encoder only when a request needs reranking."""
        if cls._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            cls._cross_encoder = CrossEncoder(cls.reranker_model_name)
        return cls._cross_encoder

    def _rerank_documents(self, question: str, documents: list[Any]) -> list[Any]:
        """Score each (question, chunk) pair and return descending relevance."""
        candidates = [
            document
            for document in documents[:self.reranker_candidate_documents]
            if getattr(document, "page_content", "").strip()
        ]
        if not candidates:
            return []

        scores = self._get_cross_encoder().predict(
            [(question, document.page_content) for document in candidates]
        )
        ranked = sorted(
            zip(scores, candidates), key=lambda item: float(item[0]), reverse=True
        )
        for score, document in ranked:
            document.metadata["reranker_score"] = float(score)
            print(
                f"Score: {float(score):.4f} | "
                f"{document.metadata.get('source')} | "
                f"{document.page_content[:80]}"
            )
        return [document for _, document in ranked]

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"chat-session:{session_id}"

    def _ensure_session(self, session_id: str) -> dict[str, Any]:
        key = self._session_key(session_id)
        session = self.session_store.mget([key])[0]
        if session is None:
            session = {
                "upload_id": None,
                "user_id": None,
                "user_name": None,
                "uploaded_files": [],
                "available_user_ids": [],
                "user_profiles": {},
                # Each uploaded document gets an independent conversation. This
                # prevents messages about a previous PDF affecting answers for
                # the current PDF in the same browser session.
                "history_by_upload": {},
            }
            self.session_store.mset([(key, session)])
        return session

    def _save_session(self, session_id: str, session: dict[str, Any]) -> None:
        self.session_store.mset([(self._session_key(session_id), session)])

    @staticmethod
    def _history_namespace(session: dict[str, Any]) -> str:
        return session.get("user_id") or session.get("upload_id") or "__no_upload__"

    def _register_user_profile(
        self,
        session: dict[str, Any],
        user_id: str,
        user_name: Optional[str] = None,
        filenames: Optional[list[str]] = None,
        upload_id: Optional[str] = None,
    ) -> None:
        profiles = session.setdefault("user_profiles", {})
        profile = profiles.get(user_id)
        if profile is None:
            profile = {
                "user_name": user_name or "",
                "uploaded_files": [],
                "upload_ids": [],
            }
            profiles[user_id] = profile

        if user_name is not None:
            profile["user_name"] = user_name

        if upload_id and upload_id not in profile["upload_ids"]:
            profile["upload_ids"].append(upload_id)

        if filenames:
            for filename in filenames:
                if filename not in profile["uploaded_files"]:
                    profile["uploaded_files"].append(filename)

        available_user_ids = session.setdefault("available_user_ids", [])
        if user_id not in available_user_ids:
            available_user_ids.append(user_id)

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        session = self._ensure_session(session_id)
        histories = session.setdefault("history_by_upload", {})
        namespace = self._history_namespace(session)
        history = histories.get(namespace)
        if history is None:
            history = InMemoryChatMessageHistory()
            histories[namespace] = history
            self._save_session(session_id, session)
        return history

    def record_upload(
        self,
        session_id: str,
        upload_id: str,
        filenames: list[str] | str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> None:
        session = self._ensure_session(session_id)
        session["upload_id"] = upload_id
        if user_id is not None:
            session["user_id"] = user_id
        if user_name is not None:
            session["user_name"] = user_name

        if isinstance(filenames, str):
            filenames = [filenames]

        if user_id is not None:
            self._register_user_profile(
                session,
                user_id=user_id,
                user_name=user_name,
                filenames=filenames,
                upload_id=upload_id,
            )

        for filename in filenames:
            if filename not in session["uploaded_files"]:
                session["uploaded_files"].append(filename)
        self._save_session(session_id, session)

    def set_session_user_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> None:
        session = self._ensure_session(session_id)
        if user_id is not None:
            session["user_id"] = user_id
            profile = session.get("user_profiles", {}).get(user_id, {})
            resolved_user_name = user_name if user_name is not None else profile.get("user_name", "")
            session["user_name"] = resolved_user_name
            self._register_user_profile(session, user_id=user_id, user_name=resolved_user_name)
        elif user_name is not None:
            session["user_name"] = user_name
        self._save_session(session_id, session)

    def get_session_upload_id(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        session = self.session_store.mget([self._session_key(session_id)])[0]
        return session.get("upload_id") if session else None

    def get_session_files(self, session_id: Optional[str]) -> list[str]:
        if not session_id:
            return []
        session = self.session_store.mget([self._session_key(session_id)])[0]
        return session.get("uploaded_files", []) if session else []

    def get_session_user_options(self, session_id: Optional[str]) -> list[dict[str, str]]:
        if not session_id:
            return []
        session = self.session_store.mget([self._session_key(session_id)])[0]
        if not session:
            return []

        available_user_ids = session.get("available_user_ids", [])
        profiles = session.get("user_profiles", {})
        user_options: list[dict[str, str]] = []
        for user_id in available_user_ids:
            profile = profiles.get(user_id, {})
            user_options.append(
                {
                    "user_id": user_id,
                    "user_name": profile.get("user_name", ""),
                }
            )
        return user_options

    def get_session_user_id(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        session = self.session_store.mget([self._session_key(session_id)])[0]
        return session.get("user_id") if session else None

    def get_session_user_name(self, session_id: Optional[str]) -> Optional[str]:
        if not session_id:
            return None
        session = self.session_store.mget([self._session_key(session_id)])[0]
        return session.get("user_name") if session else None

    def _build_prompt(self):
        if ChatPromptTemplate is None:
            return None
        return ChatPromptTemplate.from_template(
            """
            Answer only from the given context.

            Conversation history:
            {history}

            Context:
            {context}

            Question:
            {question}
            """
        )

    def _build_llm(self):
        if ChatOpenAI is None:
            return None

        api_key = os.getenv("GROK_API_CLOUD_KEY")
        if not api_key:
            return None

        return ChatOpenAI(
            api_key=api_key,
            model=os.getenv("GROK_MODEL", "llama-3.1-8b-instant"),
            base_url=os.getenv("GROK_BASE_URL", "https://api.groq.com/openai/v1"),
        )

    def retrieve_documents(
        self,
        question: str,
        metadata: Optional[dict[str, Any]] = None,
        upload_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        metadata_filter = dict(metadata or {})
        active_user_id = user_id
        if active_user_id is None and "user_id" in metadata_filter:
            active_user_id = metadata_filter["user_id"]
        if active_user_id is None and session_id is not None:
            active_user_id = self.get_session_user_id(session_id)

        # Never fall back to the global index. Answers must stay scoped to the
        # selected user id so older uploads do not leak into the answer.
        if active_user_id is None:
            return []

        metadata_filter["user_id"] = active_user_id

        # EnsembleRetriever performs weighted reciprocal-rank fusion (RRF) over
        # independent dense and BM25 result lists. Both are scoped to the same
        # user before they are combined.
        dense_retriever = FilteredVectorStoreRetriever(
            vector_store=self.vector_store,
            metadata_filter=metadata_filter,
            k=4,
        )
        print("dense_retriever----------", dense_retriever)
        bm25_retriever = build_bm25_retriever(
            self.vector_store.documents, metadata_filter, k=4
        )
        print("bm25_retriever----------", bm25_retriever)
        if not 0.0 <= self.dense_retriever_weight <= 1.0:
            raise ValueError("RAG_DENSE_RETRIEVER_WEIGHT must be between 0.0 and 1.0")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[self.dense_retriever_weight, 1 - self.dense_retriever_weight],
        )

        # MultiQueryRetriever generates alternate phrasings of the question and
        # returns the de-duplicated union of the ensemble results.
        if self.llm is not None:
            multi_query_retriever = PrintingMultiQueryRetriever.from_llm(
                retriever=ensemble_retriever,
                llm=self.llm,
                include_original=True,
            )
            print("called multi query retriever")
            docs = multi_query_retriever.invoke(question)
        else:
            docs = ensemble_retriever.invoke(question)

        docs = self._rerank_documents(question, docs)
            # return multi_query_retriever.invoke(question)[:self.max_retrieved_documents]

        return docs[:self.max_retrieved_documents]

    def build_context(self, docs: list[Any]) -> str:
        context_parts: list[str] = []
        context_length = 0

        for doc in docs:
            content = getattr(doc, "page_content", "")
            if not content:
                continue

            separator_length = 2 if context_parts else 0
            remaining = self.max_context_chars - context_length - separator_length
            if remaining <= 0:
                break

            context_parts.append(content[:remaining])
            context_length += separator_length + len(context_parts[-1])
            if len(content) > remaining:
                break

        return "\n\n".join(context_parts)

    def build_history(self, session_id: Optional[str]) -> str:
        if not session_id:
            return ""

        messages = self._get_history(session_id).messages
        if not messages:
            return ""

        history = "\n".join(
            f"{'User' if message.type == 'human' else 'Assistant'}: {message.content}"
            for message in messages
        )
        return history[-self.max_history_chars:]

    def append_history(self, session_id: str, user_message: str, assistant_message: str) -> None:
        session = self._ensure_session(session_id)
        history = self._get_history(session_id)
        history.add_user_message(user_message)
        history.add_ai_message(assistant_message)
        self._save_session(session_id, session)

    def generate_answer(self, question: str, docs: list[Any] | None = None, history: str = "") -> str:
        docs = docs if docs is not None else self.retrieve_documents(question)
        context = self.build_context(docs)

        if self.prompt is None or self.llm is None:
            return self._fallback_answer(question, context)

        chain = self.prompt | self.llm
        response = chain.invoke({"history": history, "context": context, "question": question})
        return getattr(response, "content", str(response))

    def set_session_upload_id(self, session_id: str, upload_id: str) -> None:
        session = self._ensure_session(session_id)
        session["upload_id"] = upload_id
        self._save_session(session_id, session)

    def ask(
        self,
        question: str,
        metadata: Optional[dict[str, Any]] = None,
        upload_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if session_id is None:
            session_id = str(uuid.uuid4())

        self._ensure_session(session_id)
        
        if user_id is None and metadata is not None and "user_id" in metadata:
            user_id = metadata["user_id"]
            
        if user_id is not None or user_name is not None:
            self.set_session_user_context(session_id, user_id=user_id, user_name=user_name)
        if upload_id is not None:
            self.set_session_upload_id(session_id, upload_id)

        history = self.build_history(session_id)
        docs = self.retrieve_documents(
            question,
            metadata=metadata,
            upload_id=upload_id,
            session_id=session_id,
            user_id=user_id,
        )
        answer = self.generate_answer(question, docs, history)
        self.append_history(session_id, question, answer)

        return {
            "answer": answer,
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
            "session_id": session_id,
            "upload_id": self.get_session_upload_id(session_id),
            "user_id": self.get_session_user_id(session_id),
            "user_name": self.get_session_user_name(session_id),
            "uploaded_files": self.get_session_files(session_id),
            "available_users": self.get_session_user_options(session_id),
        }

    def _fallback_answer(self, question: str, context: str) -> str:
        if context:
            return (
                "I could not reach the Grok API, so I am answering from the local context only.\n\n"
                f"Context summary: {context[:500]}"
            )
        return f"I could not find relevant context for your question: {question}"
