import logging
from typing import List, Dict, Optional, Pattern
import re
import spacy
from spacy.matcher import Matcher
from pydantic import BaseModel, Field, field_validator

from .base import BaseIntentRecognizer, Intent

logger = logging.getLogger(__name__)

class SpacyIntentPattern(BaseModel):
    """
    Pattern definition for SpaCy-based intent matching.
    
    :param intent_name: Name of the intent to match
    :type intent_name: str
    :param patterns: List of token patterns for the spaCy Matcher
    :type patterns: List[List[Dict[str, str]]]
    :param required_entities: Required entity types for this intent
    :type required_entities: List[str]
    :param confidence_weights: Weights for confidence calculation
    :type confidence_weights: Dict[str, float]
    :raises ValueError: If validation fails for any field
    """
    intent_name: str = Field(..., min_length=1, description="Name of the intent to match")
    patterns: List[List[Dict[str, str]]] = Field(..., min_length=1, description="List of token patterns for the spaCy Matcher")
    required_entities: List[str] = Field(default_factory=list, description="Required entity types for this intent")
    confidence_weights: Dict[str, float] = Field(
        default_factory=lambda: {"pattern_match": 0.6, "entity_match": 0.4},
        description="Weights for confidence calculation"
    )

    @field_validator("confidence_weights")
    @classmethod
    def validate_confidence_weights(cls, v):
        """
        Validate that confidence weights sum to 1.0.
        
        :param v: Dictionary of confidence weights
        :type v: Dict[str, float]
        :return: Validated confidence weights
        :rtype: Dict[str, float]
        :raises ValueError: If weights are invalid
        """
        if not v:
            raise ValueError("Confidence weights cannot be empty")
        
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # Allow for small floating point errors
            raise ValueError("Confidence weights must sum to 1.0")
        
        if not all(0 <= w <= 1 for w in v.values()):
            raise ValueError("All confidence weights must be between 0 and 1")
        
        return v

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v):
        """
        Validate that patterns are properly formatted.
        
        :param v: List of patterns
        :type v: List[List[Dict[str, str]]]
        :return: Validated patterns
        :rtype: List[List[Dict[str, str]]]
        :raises ValueError: If patterns are invalid
        """
        if not v:
            raise ValueError("Patterns cannot be empty")
        
        for pattern in v:
            if not isinstance(pattern, list):
                raise ValueError("Each pattern must be a list of dictionaries")
            if not all(isinstance(token, dict) for token in pattern):
                raise ValueError("Each token in a pattern must be a dictionary")
            if not all(isinstance(key, str) and isinstance(value, str) 
                      for token in pattern for key, value in token.items()):
                raise ValueError("All pattern keys and values must be strings")
        
        return v

class SpacyIntentRecognizer(BaseIntentRecognizer):
    """
    SpaCy-based implementation of intent recognition.
    
    Uses pattern matching and entity recognition for medical intents.
    """
    
    def __init__(self, model_name: str = "en_core_web_md"):
        """
        Initialize the SpaCy recognizer.
        
        :param model_name: Name of the SpaCy model to use
        :type model_name: str
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
    
    def add_pattern(self, pattern: SpacyIntentPattern) -> None:
        """
        Add a new pattern to the recognizer.
        
        :param pattern: Pattern to add
        :type pattern: SpacyIntentPattern
        """
        self.patterns.append(pattern)
        if self.matcher is not None:
            self.matcher.add(pattern.intent_name, pattern.patterns)
    
    def initialize(self) -> None:
        """
        Initialize SpaCy model and configure the matcher.
        
        :raises Exception: If initialization fails
        """
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
        :type pattern: SpacyIntentPattern
        :param doc: SpaCy Doc object
        :type doc: spacy.tokens.Doc
        :param matches: List of pattern matches
        :type matches: List[Tuple[int, int, int]]
        :return: Confidence score between 0 and 1
        :rtype: float
        """
        # Calculate pattern match score
        pattern_score = min(len(matches) / len(pattern.patterns), 1.0) if matches else 0.0
        
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
        :type doc: spacy.tokens.Doc
        :return: Dictionary of extracted parameters
        :rtype: Dict[str, str]
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
            elif ent.label_ == "location":  # Handle location entity type
                params["destination"] = ent.text
        
        return params
    
    def recognize_intent(self, text: str) -> List[Intent]:
        """
        Recognize medical intents from the conversation text using SpaCy.
        
        :param text: Transcribed conversation text
        :type text: str
        :return: List of recognized intents
        :rtype: List[Intent]
        """
        try:
            doc = self.nlp(text)
            recognized_intents = []
            
            for pattern in self.patterns:
                # Get matches for this specific pattern
                matches = self.matcher(doc)
                matches = [m for m in matches if self.matcher.vocab.strings[m[0]] == pattern.intent_name]
                
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