from dataclasses import dataclass
from typing import List
import numpy as np
from google.api_core.exceptions import ServerError, ClientError
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log, retry_if_exception
import logging
from google import genai
from google.genai import types
from config import GOOGLE_API_KEY
logger = logging.getLogger(__name__)

@dataclass
class GeminiEmbeddingConfig:
    """Configuration for Gemini dense embedding generation."""
    model: str = "text-embedding-004"
    batch_size: int = 16
    max_retries: int = 3
    retry_delay: float = 1.0


class GeminiEmbedder:
    """Handles dense embedding generation using Gemini's text-embedding-004 model."""
    
    def __init__(self, config: GeminiEmbeddingConfig):
        self.config = config
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("Gemini embedder initialized")
    
    def should_retry_gemini_error(self, exception):
        """Determine if a Gemini API error should be retried."""
        if isinstance(exception, ServerError):
            return exception.code >= 500
        elif isinstance(exception, ClientError):
            return exception.code == 429  # Rate limiting
        elif "connection" in str(exception).lower() or "timeout" in str(exception).lower():
            return True
        return False
    
    def _create_retry_decorator(self):
        """Create retry decorator for Gemini API calls."""
        return retry(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(multiplier=1, min=self.config.retry_delay, max=30),
            retry=retry_if_exception(self.should_retry_gemini_error),
            before_sleep=before_sleep_log(logger, logging.WARNING)
        )
    
    def encode_queries(self, texts: List[str]) -> List[np.ndarray]:
        return self._encode_texts(texts, "RETRIEVAL_QUERY")

    def encode_documents(self, texts: List[str]) -> List[np.ndarray]:
        return self._encode_texts(texts, "RETRIEVAL_DOCUMENT")

    def _encode_texts(self, texts: List[str], task_type: str) -> List[np.ndarray]:
        """
        Encode texts using Gemini's text-embedding-004 model.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of dense embedding vectors as numpy arrays
        """
        embeddings = []
        retry_decorator = self._create_retry_decorator()
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self._encode_batch(batch, task_type, retry_decorator)
            embeddings.extend(batch_embeddings)
            
        logger.info(f"Generated {len(embeddings)} dense embeddings")
        return embeddings
    
    def _encode_batch(self, texts: List[str], task_type: str, retry_decorator) -> List[np.ndarray]:
        """Encode a batch of texts with retry logic."""
        
        @retry_decorator
        def make_embedding_request():
            return self.client.models.embed_content(
                model=self.config.model,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type)
            )
        
        try:
            response = make_embedding_request()
            embeddings = []
            
            for embedding_data in response.embeddings:
                # Convert to numpy array
                embedding_vector = np.array(embedding_data.values, dtype=np.float32)
                embeddings.append(embedding_vector)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {str(e)}")
            # Return zero vectors as fallback
            return [np.zeros(768, dtype=np.float32) for _ in texts]  # 768 is the dimension for text-embedding-004
