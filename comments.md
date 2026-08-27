# libraries to be added

# uv add langchain-community pypdf langchain-text-splitters
## we had used for hybrid serahc BM25 reteiver and ensemble retreiver
# in ragservice.py
<!-- ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[self.dense_retriever_weight, 1 - self.dense_retriever_weight],
        ) -->

# for reranker we are using cross encoder instead of this we can also use api of reranker from pinecone too and other available too
