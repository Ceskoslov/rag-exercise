import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-V1.5",
        index_save_path: str = "./vector_index",
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embeddings = None
        self.setup_embeddings()

    def setup_embeddings(self):
        logger.info(f"Setting up embeddings with model: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Embeddings setup complete.")

    def build_vector_index(self, chunks: list[Document]) -> FAISS:
        logger.info("Building vector index from document chunks.")
        self.vectorstore = FAISS.from_documents(
            documents=chunks, embedding=self.embeddings
        )

        logger.info("Vector index construction complete.")
        return self.vectorstore

    def add_document(self, new_chunks: list[Document]):
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Please build the vector index first."
            )

        logger.info(
            f"Adding {len(new_chunks)} new document chunks to the vector index."
        )
        self.vectorstore.add_documents(new_chunks)
        logger.info("New document chunks added to the vector index.")

    def save_index(self):
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Please build the vector index first."
            )

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"Vector index saved to {self.index_save_path}.")

    def load_index(self):
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(
                f"No existing index found at {self.index_save_path}. Starting with an empty index."
            )
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"Vector index loaded from {self.index_save_path}.")
            return self.vectorstore
        except Exception as e:
            logger.error(
                f"Failed to load vector index from {self.index_save_path}: {e}"
            )
            return None

    def similarity_search(self, query: str, top_k: int = 5) -> list[Document]:
        if not self.vectorstore:
            raise ValueError(
                "Vector store not initialized. Please build or load the vector index first."
            )

        return self.vectorstore.similarity_search(query, k=top_k)
