import math

import faiss
import numpy as np

from langchain_openai import OpenAIEmbeddings


class MemoryStorage:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_session_id: int = 3,
    ) -> None:
        self.memory_list: list[str] = []
        self.embeddings = OpenAIEmbeddings(model=model, chunk_size=batch_size)
        self.max_session_id = max_session_id
        self.index = None
        self._is_initialized = False

    def _initialize_index(self, dimension: int) -> None:
        if self._is_initialized:
            return
        self.index = faiss.IndexFlatIP(dimension)
        self._is_initialized = True

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def add_memory(self, memories: list[str], session_id: int) -> None:
        embeddings_list = self.embeddings.embed_documents(memories)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        self._initialize_index(embeddings_array.shape[1])

        if self.index is None:
            raise ValueError("Index has not been initialized.")

        normalized_embeddings = self._normalize_vectors(embeddings_array)

        importance = math.exp(-0.2 * (1 - (session_id / self.max_session_id)))

        weighted_embeddings = normalized_embeddings * importance

        self.index.add(weighted_embeddings)

        for content in memories:
            self.memory_list.append(content)

    def find_similar(self, query: str, top_k: int = 5) -> list[str]:
        if self.index is None or len(self.memory_list) == 0:
            return []

        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding], dtype=np.float32)

        normalized_query = self._normalize_vectors(query_vector)

        indices = self.index.search(
            normalized_query, min(top_k, len(self.memory_list))
        )[1]

        results = []
        for idx in indices[0]:
            results.append(self.memory_list[idx])

        return results

    def get_memory_count(self) -> int:
        return len(self.memory_list)
