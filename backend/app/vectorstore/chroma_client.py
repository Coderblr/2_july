import logging
from functools import lru_cache

import chromadb

from app.core.config import get_settings
from app.vectorstore.embedding_function import OfflineHashEmbeddingFunction

logger = logging.getLogger(__name__)

FEATURE_FILE_COLLECTION = "feature_files"
KNOWLEDGE_COLLECTION = "app_knowledge"


@lru_cache
def get_chroma_client():
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_path)


@lru_cache
def get_feature_file_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=FEATURE_FILE_COLLECTION,
        embedding_function=OfflineHashEmbeddingFunction(),
    )


def add_feature_file_chunks(doc_id: str, chunks: list[str], metadatas: list[dict]) -> None:
    if not chunks:
        return
    try:
        collection = get_feature_file_collection()
        ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    except Exception:
        logger.warning("Failed to embed feature file chunks for %s; continuing without vector index", doc_id)


def query_feature_file_chunks(query_text: str, n_results: int = 5) -> list[dict]:
    try:
        collection = get_feature_file_collection()
        if collection.count() == 0:
            return []
        result = collection.query(query_texts=[query_text], n_results=min(n_results, collection.count()))
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [{"text": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]
    except Exception:
        logger.warning("Feature file vector query failed; returning no matches")
        return []


@lru_cache
def get_knowledge_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        embedding_function=OfflineHashEmbeddingFunction(),
    )


def add_knowledge_items(ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
    if not documents:
        return
    try:
        get_knowledge_collection().upsert(ids=ids, documents=documents, metadatas=metadatas)
    except Exception:
        logger.warning("Failed to index knowledge items; continuing without vector index")


def query_knowledge(query_text: str, n_results: int = 5) -> list[dict]:
    try:
        collection = get_knowledge_collection()
        if collection.count() == 0:
            return []
        result = collection.query(query_texts=[query_text], n_results=min(n_results, collection.count()))
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [{"text": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)]
    except Exception:
        logger.warning("Knowledge vector query failed; returning no matches")
        return []
