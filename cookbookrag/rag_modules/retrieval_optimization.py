import logging
from collections.abc import Iterable

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    def __init__(self, vectorstore: FAISS, chunks: list[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retriever()

    def setup_retriever(self):
        logger.info("Setting up BM25 retriever with provided chunks.")
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        self.bm25_retriever = BM25Retriever.from_documents(self.chunks, k=5)

        logger.info("Retrievers set up successfully.")

    def hybrid_search(self, query: str, top_k: int = 3) -> list[Document]:
        logger.info(f"Running hybrid search for query: {query}")

        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)

        return reranked_docs[:top_k]

    def metadata_filtered_search(
        self, query: str, filters: dict, top_k: int = 3
    ) -> list[Document]:
        logger.info(
            f"Running metadata-filtered search for query: {query} with filters: {filters}"
        )

        docs = self.hybrid_search(query, top_k=max(top_k * 3, 10))
        filtered_docs = []

        for doc in docs:
            if self._matches_filters(doc, filters):
                filtered_docs.append(doc)
                if len(filtered_docs) >= top_k:
                    break

        return filtered_docs

    def _matches_filters(self, doc: Document, filters: dict) -> bool:
        for key, value in filters.items():
            if key not in doc.metadata:
                return False

            metadata_value = doc.metadata[key]
            if isinstance(value, Iterable) and not isinstance(value, str):
                if metadata_value not in value:
                    return False
            elif metadata_value != value:
                return False

        return True

    def _rrf_rerank(
        self, vector_docs: list[Document], bm25_docs: list[Document], k: int = 60
    ) -> list[Document]:
        doc_scores = {}
        doc_objects = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1.0)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(
                f"Vector doc: {doc.page_content[:50]}... | Rank: {rank} | RRF Score: {rrf_score:.4f}"
            )

        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1.0)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(
                f"BM25 doc: {doc.page_content[:50]}... | Rank: {rank} | RRF Score: {rrf_score:.4f}"
            )

        sorted_cocs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        rerank_docs = []
        for doc_id, final_score in sorted_cocs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata["rrf_score"] = final_score
                rerank_docs.append(doc)
                logger.debug(
                    f"Doc: {doc.page_content[:50]}... | Final RRF Score: {final_score:.4f}"
                )

        logger.info(
            f"RRF reranking completed. Total unique docs scored: {len(doc_scores)}"
        )
        return rerank_docs
