from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class Intent(BaseModel):
    """
    Base class for intent models.
    
    :param name: Name of the intent
    :param confidence: Confidence score of the recognition
    :param metadata: Additional data associated with the intent
    """
    name: str
    confidence: float
    metadata: dict = {}

class BaseIntentRecognizer(ABC):
    """
    Abstract base class for intent recognizers.
    
    This class defines the interface that all intent recognizers must implement.
    Different implementations can use different approaches (LLM, SpaCy, etc.).
    """
    
    @abstractmethod
    async def recognize_intent(self, text: str) -> List[Intent]:
        """
        Recognize intents from the given text.
        
        :param text: Input text to analyze
        :return: List of recognized intents
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the recognizer with necessary resources.
        """
        pass 