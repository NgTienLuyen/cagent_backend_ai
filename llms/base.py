from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import uuid
import logging

logger = logging.getLogger(__name__)


class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_type = None

    @abstractmethod
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        """
        Create message method for using agentic chunker
        """
        pass

    @abstractmethod
    def generate_content(self, prompt: str, **kwargs):
        """Generate content with given prompt"""
        pass

    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        pass

    def generate_content_with_timing(self, prompt: str, **kwargs):
        """Generate content and track timing"""
        start_time = time.time()
        request_id = uuid.uuid4()

        logger.info(f"Request {request_id}: Starting LLM generation with model {self.model_name}")

        try:
            result = self.generate_content(prompt, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"Request {request_id}: LLM generation completed in {duration:.2f}s")

            return {
                "content": result,
                "request_id": request_id,
                "model": self.model_name,
                "duration": duration
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            logger.error(f"Request {request_id}: LLM generation failed after {duration:.2f}s: {str(e)}")
            raise
