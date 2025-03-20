import logging
from typing import List, Optional, Dict
from pydantic_ai import AIModel, AIModelConfig
from .base_recognizer import BaseIntentRecognizer, Intent

logger = logging.getLogger(__name__)

class MedicalIntent(AIModel):
    """
    Medical intent model for LLM-based recognition.
    
    :param intent_name: Name of the recognized intent
    :param description: Detailed description of the intent
    :param required_action: Type of action required
    :param urgency_level: Level of urgency (1-5)
    :param relevant_parameters: Additional parameters extracted from the text
    """
    intent_name: str
    description: str
    required_action: str
    urgency_level: int
    relevant_parameters: Dict[str, str] = {
        "body_part": "",  # Relevant body part (e.g., "knee", "heart")
        "condition": "",  # Medical condition (e.g., "arthritis", "inflammation")
        "test_type": "",  # Type of medical test if applicable
        "visualization_type": "",  # Type of visual aid needed (e.g., "anatomy", "procedure")
        "additional_context": ""  # Any other relevant context
    }

    class Config:
        """Configuration for the AI model."""
        system_prompt = """You are a medical intent recognition system.
        Analyze the conversation between physician and patient to identify key intents.
        Focus on medical procedures, tests, prescriptions, patient concerns, and requests for visual aids.
        
        When visual aids are mentioned:
        - Identify the specific body part or anatomical region
        - Note any medical conditions being discussed
        - Determine if they need an anatomical diagram, procedure illustration, or other type of visual
        
        Provide structured output with:
        - Intent name (e.g., "print_map", "show_diagram", "request_test")
        - Description of what's needed
        - Required action (e.g., "display_anatomy", "schedule_test")
        - Urgency level (1-5)
        - Relevant parameters (body part, condition, etc.)"""

class LLMIntentRecognizer(BaseIntentRecognizer):
    """
    LLM-based implementation of intent recognition.
    
    Uses PydanticAI to structure the LLM output into actionable intents.
    """
    
    def __init__(self, model_endpoint: str, api_key: Optional[str] = None):
        """
        Initialize the LLM recognizer.
        
        :param model_endpoint: URL of the self-hosted LLM API
        :param api_key: Optional API key for authentication
        """
        self.model_endpoint = model_endpoint
        self.api_key = api_key
        self.config = AIModelConfig(
            api_base=model_endpoint,
            api_key=api_key,
            temperature=0.3  # Lower temperature for more focused responses
        )
    
    async def initialize(self) -> None:
        """Initialize the LLM connection and verify the endpoint."""
        try:
            # Test connection with a simple prompt
            test_intent = await MedicalIntent.agenerate(
                "test connection",
                config=self.config
            )
            logger.info("LLM Intent Recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Intent Recognizer: {e}")
            raise
    
    async def recognize_intent(self, text: str) -> List[Intent]:
        """
        Recognize medical intents from the conversation text.
        
        :param text: Transcribed conversation text
        :return: List of recognized intents
        """
        try:
            medical_intent = await MedicalIntent.agenerate(
                text,
                config=self.config
            )
            
            # Convert MedicalIntent to base Intent
            return [Intent(
                name=medical_intent.intent_name,
                confidence=0.9,  # LLM doesn't provide confidence, using high default
                metadata={
                    "description": medical_intent.description,
                    "required_action": medical_intent.required_action,
                    "urgency_level": medical_intent.urgency_level,
                    "parameters": medical_intent.relevant_parameters
                }
            )]
        except Exception as e:
            logger.error(f"Error recognizing intent: {e}")
            return [] 