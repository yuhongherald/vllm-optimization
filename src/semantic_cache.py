from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
from threading import Lock
import logging
from collections import deque
from typing import Deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticCache:
    """
    A thread-safe semantic cache that stores query embeddings and their corresponding results.
    It allows efficient retrieval of cached results based on cosine similarity of embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        max_cache_size: int = 1000,
        device: str = "cpu"
    ):
        """
        Initialize the semantic cache.

        Args:
            model_name (str): The name of the SentenceTransformer model to use.
            similarity_threshold (float): The cosine similarity threshold for cache hits.
            max_cache_size (int): The maximum number of entries the cache can hold.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.keys: Deque[str] = deque()
        self.values: Deque[np.ndarray] = deque()
        self.lock = Lock()
        logger.info(
            "Initialized ProductionSemanticCache with model '%s'", model_name)

    def add_to_cache(self, query: str, result: str) -> None:
        """
        Add a query and its result to the cache.

        Args:
            query (str): The query string to cache.
            result (str): The result corresponding to the query.
        """
        with self.lock:
            if len(self.cache) >= self.max_cache_size:
                self._evict_oldest()

            try:
                embedding = self.model.encode(
                    query, convert_to_numpy=True).astype('float32')
                normalized_embedding = embedding / \
                    np.linalg.norm(embedding) if np.linalg.norm(
                        embedding) != 0 else embedding
            except Exception as e:
                logger.error("Failed to encode query '%s': %s", query, e)
                return

            self.cache[query] = {
                'embedding': normalized_embedding,
                'result': result,
                'timestamp': time.time(),
                'hits': 0
            }
            self.keys.append(query)
            self.values.append(normalized_embedding)
            logger.debug("Added query '%s' to cache", query)

    def get_from_cache(self, query: str) -> Optional[Any]:
        """
        Retrieve a cached result based on the query's embedding similarity.

        Args:
            query (str): The query string to look up.

        Returns:
            Optional[Any]: The cached result if a similar query is found; otherwise, None.
        """

        try:
            query_embedding = self.model.encode(
                query, convert_to_numpy=True).astype('float32')
            norm = np.linalg.norm(query_embedding)
            if norm != 0:
                query_embedding /= norm
        except Exception as e:
            logger.error("Failed to encode query '%s': %s", query, e)
            return None

        with self.lock:
            if not self.cache:
                logger.debug("Cache is empty. No match for query '%s'", query)
                return None

            # Extract all valid embeddings and corresponding queries
            embeddings = self.values
            embeddings_matrix = np.vstack(embeddings)  # Shape: (N, D)
            # Cosine similarity since embeddings are normalized
            similarities = np.dot(embeddings_matrix, query_embedding)

            # Find the best match
            best_idx = np.argmax(similarities)
            highest_similarity = similarities[best_idx]

            if highest_similarity < self.threshold:
                logger.info(
                    "No cache hit for query '%s'. Highest similarity was %.4f", query, highest_similarity)
                return None

            valid_queries = self.keys
            best_match = valid_queries[best_idx]

            logger.debug("Best similarity for query '%s' is %.4f with cached query '%s'",
                         query, highest_similarity, best_match)

            self.cache[best_match]['hits'] += 1
            logger.info("Cache hit for query '%s' with cached query '%s' (Similarity: %.4f)",
                        query, best_match, highest_similarity)
            return self.cache[best_match]['result']

    def _evict_oldest(self) -> None:
        """
        Evict the oldest entry in the cache (FIFO eviction).
        """
        if not self.cache:
            logger.warning("Attempted to evict from an empty cache")
            return

        # Get the first inserted (oldest) key
        oldest_key = next(iter(self.cache))

        del self.cache[oldest_key]
        self.keys.popleft()
        self.values.popleft()
        logger.info("Evicted oldest cache entry: '%s'", oldest_key)

    def clear_cache(self) -> None:
        """
        Clear all entries from the cache.
        """
        with self.lock:
            self.cache.clear()
            self.keys.clear()
            self.values.clear()
            logger.info("Cleared all entries from the cache")

    def cache_size(self) -> int:
        """
        Get the current number of entries in the cache.

        Returns:
            int: Number of cache entries.
        """
        size = len(self.cache)
        logger.debug("Current cache size: %d", size)
        return size

    def get_cache_entries(self) -> List[Tuple[str, Any]]:
        """
        Retrieve all cache entries as a list of tuples.

        Returns:
            List[Tuple[str, Any]]: List containing (query, result) tuples.
        """
        with self.lock:
            entries = list(self.cache.items())
            logger.debug("Retrieved %d valid cache entries", len(entries))
            return entries
