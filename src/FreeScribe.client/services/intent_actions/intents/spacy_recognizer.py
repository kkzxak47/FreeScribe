import logging
from typing import List, Dict, Optional, Pattern
import re
import spacy
from spacy.matcher import Matcher
from pydantic import BaseModel

from .base import BaseIntentRecognizer, Intent

logger = logging.getLogger(__name__)

class SpacyIntentPattern(BaseModel):
    """
    Pattern definition for SpaCy-based intent matching.
    
    :param intent_name: Name of the intent to match
    :param patterns: List of token patterns for the spaCy Matcher
    :param required_entities: Required entity types for this intent
    :param confidence_weights: Weights for confidence calculation
    """
    intent_name: str
    patterns: List[List[Dict[str, str]]]
    required_entities: List[str] = []
    confidence_weights: Dict[str, float] = {
        "pattern_match": 0.6,
        "entity_match": 0.4
    }

class SpacyIntentRecognizer(BaseIntentRecognizer):
    """
    SpaCy-based implementation of intent recognition.
    
    Uses pattern matching and entity recognition for medical intents.
    """
    
    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initialize the SpaCy recognizer.
        
        :param model_name: Name of the SpaCy model to use
        """
        self.model_name = model_name
        self.nlp = None
        self.matcher = None
        self.patterns = [
            SpacyIntentPattern(
                intent_name="show_directions",
                patterns=[
                    [{"LOWER": "go"}, {"LOWER": "to"}, {"ENT_TYPE": "ORG"}],
                    [{"LOWER": "need"}, {"LOWER": "to"}, {"LOWER": "get"}, {"LOWER": "to"}],
                    [{"LOWER": "directions"}, {"LOWER": "to"}],
                ],
                required_entities=["ORG", "TIME"]
            ),
            SpacyIntentPattern(
                intent_name="schedule_appointment",
                patterns=[
                    [{"LEMMA": "schedule"}, {"LOWER": "appointment"}],
                    [{"LEMMA": "book"}, {"LOWER": "appointment"}],
                    [{"LOWER": "need"}, {"LOWER": "to"}, {"LOWER": "see"}],
                ],
                required_entities=["TIME", "ORG"]
            )
        ]
    
    def initialize(self) -> None:
        """Initialize SpaCy model and configure the matcher."""
        try:
            self.nlp = spacy.load(self.model_name)
            self.matcher = Matcher(self.nlp.vocab)
            
            # Add patterns to matcher
            for pattern in self.patterns:
                self.matcher.add(pattern.intent_name, pattern.patterns)
            
            logger.info("SpaCy Intent Recognizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpaCy Intent Recognizer: {e}")
            raise
    
    def _calculate_confidence(self, pattern: SpacyIntentPattern, doc, matches: List) -> float:
        """
        Calculate confidence score based on pattern matches and entity presence.
        
        :param pattern: Intent pattern definition
        :param doc: SpaCy Doc object
        :param matches: List of pattern matches
        :return: Confidence score between 0 and 1
        """
        if not matches:
            return 0.0
        
        # Calculate pattern match score
        pattern_score = min(len(matches) / len(pattern.patterns), 1.0)
        
        # Calculate entity match score
        found_entities = set(ent.label_ for ent in doc.ents)
        required_entities = set(pattern.required_entities)
        entity_score = len(found_entities.intersection(required_entities)) / len(required_entities) if required_entities else 1.0
        
        # Weighted average
        weights = pattern.confidence_weights
        confidence = (
            weights["pattern_match"] * pattern_score +
            weights["entity_match"] * entity_score
        )
        
        return min(confidence, 1.0)
    
    def _extract_parameters(self, doc) -> Dict[str, str]:
        """
        Extract relevant parameters from the recognized entities.
        
        :param doc: SpaCy Doc object
        :return: Dictionary of extracted parameters
        """
        params = {
            "destination": "",
            "transport_mode": "driving",
            "patient_mobility": "",
            "appointment_time": "",
            "additional_context": ""
        }
        
        for ent in doc.ents:
            if ent.label_ == "ORG":
                params["destination"] = ent.text
            elif ent.label_ == "TIME":
                params["appointment_time"] = ent.text
            elif ent.label_ == "PRODUCT" and any(vehicle in ent.text.lower() for vehicle in ["ambulance", "wheelchair", "taxi"]):
                params["transport_mode"] = ent.text.lower()
        
        return params
    
    def recognize_intent(self, text: str) -> List[Intent]:
        """
        Recognize medical intents from the conversation text using SpaCy.
        
        :param text: Transcribed conversation text
        :return: List of recognized intents
        """
        try:
            doc = self.nlp(text)
            recognized_intents = []
            
            for pattern in self.patterns:
                matches = self.matcher(doc, as_spans=True)
                confidence = self._calculate_confidence(pattern, doc, matches)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    intent = Intent(
                        name=pattern.intent_name,
                        confidence=confidence,
                        metadata={
                            "description": f"Recognized {pattern.intent_name} intent",
                            "required_action": pattern.intent_name,
                            "urgency_level": 2,  # Default urgency
                            "parameters": self._extract_parameters(doc)
                        }
                    )
                    recognized_intents.append(intent)
            
            return recognized_intents
            
        except Exception as e:
            logger.error(f"Error recognizing intent with SpaCy: {e}")
            return [] 