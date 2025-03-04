"""
Module for verifying factual consistency between original text and generated summaries.
This module provides a pipeline of different verification methods to ensure accuracy.
"""

import os
import spacy
from typing import Tuple, List, Dict, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VerificationMethod(Enum):
    """Enum for different verification methods"""
    NER = "named_entity_recognition"
    # Add more methods as we implement them
    # KEYWORDS = "keyword_matching"
    # SEMANTIC = "semantic_similarity"
    # NUMERICAL = "numerical_consistency"


@dataclass
class VerificationResult:
    """Data class to hold verification results"""
    is_consistent: bool
    inconsistent_items: List[str]
    confidence: float  # 0.0 to 1.0
    method: VerificationMethod


class ConsistencyVerifier(ABC):
    """Abstract base class for consistency verifiers"""

    @abstractmethod
    def verify(self, original_text: str, generated_summary: str) -> VerificationResult:
        """Verify consistency between original text and generated summary"""
        pass


class NERVerifier(ConsistencyVerifier):
    """Named Entity Recognition based verifier"""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")

    def verify(self, original_text: str, generated_summary: str) -> VerificationResult:
        """
        Verify factual consistency using named entity recognition.
        
        Args:
            original_text: The original transcribed text
            generated_summary: The generated medical note/summary

        Returns:
            VerificationResult containing:
                - is_consistent: True if all entities in summary appear in original text
                - inconsistent_entities: List of entities that appear in summary but not in original text
                - confidence: Confidence score based on entity overlap
                - method: The verification method used
        """
        # Process both texts with spaCy
        original_doc = self.nlp(original_text.lower())
        summary_doc = self.nlp(generated_summary.lower())

        # Extract named entities from both texts
        original_entities = set(ent.text for ent in original_doc.ents)
        summary_entities = set(ent.text for ent in summary_doc.ents)

        # Find entities that appear in summary but not in original text
        inconsistent_entities = list(summary_entities - original_entities)

        # Calculate confidence based on entity overlap
        if not summary_entities:
            confidence = 1.0  # If no entities in summary, consider it consistent
        else:
            overlap = len(original_entities.intersection(summary_entities))
            confidence = overlap / len(summary_entities)

        # Check if there are any inconsistent entities
        is_consistent = len(inconsistent_entities) == 0

        return VerificationResult(
            is_consistent=is_consistent,
            inconsistent_items=inconsistent_entities,
            confidence=confidence,
            method=VerificationMethod.NER
        )


class ConsistencyPipeline:
    """Pipeline for running multiple consistency verifications"""

    def __init__(self):
        self.verifiers: Dict[VerificationMethod, ConsistencyVerifier] = {
            VerificationMethod.NER: NERVerifier(),
            # Add more verifiers as we implement them
        }

    def verify(self, original_text: str, generated_summary: str) -> Dict[VerificationMethod, VerificationResult]:
        """
        Run all verification methods and return their results.
        
        Args:
            original_text: The original transcribed text
            generated_summary: The generated medical note/summary
            
        Returns:
            Dictionary mapping verification methods to their results
        """
        results = {}
        for method, verifier in self.verifiers.items():
            try:
                results[method] = verifier.verify(original_text, generated_summary)
            except Exception as e:
                logger.error(f"Error in {method.value} verification: {str(e)}")
                # Create a failed result
                results[method] = VerificationResult(
                    is_consistent=False,
                    inconsistent_items=[f"Verification failed: {str(e)}"],
                    confidence=0.0,
                    method=method
                )
        return results

    def get_overall_consistency(self, results: Dict[VerificationMethod, VerificationResult]) -> Tuple[bool, List[str], float]:
        """
        Analyze results from all verification methods to determine overall consistency.
        
        Args:
            results: Dictionary of verification results from each method
            
        Returns:
            Tuple containing:
                - is_consistent: True if all methods agree on consistency
                - issues: List of all issues found across methods
                - overall_confidence: Average confidence across all methods
        """
        if not results:
            return False, ["No verification methods available"], 0.0

        # Collect all issues
        all_issues = []
        for result in results.values():
            all_issues.extend(result.inconsistent_items)

        # Calculate overall confidence
        confidences = [result.confidence for result in results.values()]
        overall_confidence = sum(confidences) / len(confidences)

        # Consider it consistent if all methods agree
        is_consistent = all(result.is_consistent for result in results.values())

        return is_consistent, all_issues, overall_confidence


# Create a global pipeline instance
pipeline = ConsistencyPipeline()


def verify_factual_consistency(original_text: str, generated_summary: str) -> Tuple[bool, List[str], float]:
    """
    Verify factual consistency between original text and generated summary using multiple methods.
    
    Args:
        original_text: The original transcribed text
        generated_summary: The generated medical note/summary
        
    Returns:
        Tuple containing:
            - is_consistent: True if all verification methods agree on consistency
            - issues: List of all issues found across methods
            - confidence: Overall confidence score (0.0 to 1.0)
    """
    results = pipeline.verify(original_text, generated_summary)
    return pipeline.get_overall_consistency(results)
