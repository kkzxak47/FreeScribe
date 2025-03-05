import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple

import spacy

logger = logging.getLogger(__name__)


class VerificationMethod(Enum):
    """Enumeration of different verification methods.

    .. py:attribute:: NER
       :annotation: = "named_entity_recognition"

       Named Entity Recognition based verification.

    .. note::
        Additional verification methods may be added in future versions.
    """
    NER = "named_entity_recognition"
    # Add more methods as we implement them
    # KEYWORDS = "keyword_matching"
    # SEMANTIC = "semantic_similarity"
    # NUMERICAL = "numerical_consistency"


@dataclass
class VerificationResult:
    """Data class to hold verification results.

    :param method: The verification method used
    :type method: VerificationMethod
    :param inconsistent_items: List of inconsistent items found, defaults to empty list
    :type inconsistent_items: List[str]

    .. note::
        If inconsistent_items is empty, no inconsistencies were found.
    """
    method: VerificationMethod
    inconsistent_items: List[str] = field(default_factory=list)


class ConsistencyVerifier(ABC):
    """Abstract base class for consistency verifiers.

    This class defines the interface for all concrete verification implementations.
    Subclasses must implement the verify() method.

    .. note::
        All verifiers should be thread-safe as they may be used concurrently.
    """

    @abstractmethod
    def verify(self, original_text: str, generated_summary: str) -> VerificationResult:
        """Verify consistency between original text and generated summary.

        :param original_text: The original transcribed text
        :type original_text: str
        :param generated_summary: The generated medical note/summary
        :type generated_summary: str
        :return: Verification result containing consistency status
        :rtype: VerificationResult

        .. note::
            Both input texts are normalized (lowercased) before verification.

        .. warning::
            Empty input texts may produce unexpected results.
        """
        pass


class NERVerifier(ConsistencyVerifier):
    """Named Entity Recognition based verifier.

    This verifier uses spaCy's NER capabilities to compare medical entities
    between original text and generated summaries.

    :ivar nlp: The spaCy language model for medical text processing
    :vartype nlp: spacy.Language

    .. note::
        The model is automatically downloaded if not present.

    .. warning::
        The 'en_core_web_trf' model requires significant memory resources.
    """
    NLP_MODEL = "en_core_sci_md"

    def __init__(self):
        try:
            self.nlp = spacy.load(self.NLP_MODEL)
        except OSError:
            logger.info("Downloading spaCy model...")
            ret = os.system(f"python -m spacy download {self.NLP_MODEL}")
            if ret != 0:
                raise EnvironmentError(f"{self.NLP_MODEL} not found and could not be downloaded")
            self.nlp = spacy.load(self.NLP_MODEL)

    def verify(self, original_text: str, generated_summary: str) -> VerificationResult:
        """
        Verify factual consistency using named entity recognition.

        Args:
            original_text: The original transcribed text
            generated_summary: The generated medical note/summary

        Returns:
            VerificationResult containing:
                - inconsistent_items: List of entities that appear in summary but not in original text
                - method: The verification method used
        """
        if not generated_summary or not original_text:
            return VerificationResult(method=VerificationMethod.NER)

        # Process both texts with spaCy
        original_doc = self.nlp(original_text.lower())
        summary_doc = self.nlp(generated_summary.lower())

        # Extract named entities from both texts
        original_entities = set(ent.text for ent in original_doc.ents)
        summary_entities = set(ent.text for ent in summary_doc.ents)
        logger.debug(f"summary entities: {[(x.text, x.label_) for x in summary_doc.ents]}")

        # Find entities that appear in summary but not in original text
        inconsistent_entities = list(summary_entities - original_entities)

        return VerificationResult(
            inconsistent_items=inconsistent_entities,
            method=VerificationMethod.NER
        )


class ConsistencyPipeline:
    """Pipeline for running multiple consistency verifications.

    This class orchestrates the execution of multiple verification methods
    and aggregates their results.

    :ivar verifiers: Dictionary mapping verification methods to their implementations
    :vartype verifiers: Dict[VerificationMethod, ConsistencyVerifier]

    .. note::
        New verification methods can be added by extending the verifiers dictionary.

    .. warning::
        The pipeline assumes all verifiers are independent and can be run in any order.
    """

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
                    inconsistent_items=[f"Verification failed: {str(e)}"],
                    method=method
                )
        return results

    def get_inconsistent_entities(self, results: Dict[VerificationMethod, VerificationResult]) -> List[str]:
        """
        Get all inconsistent entities found across verification methods.

        Args:
            results: Dictionary of verification results from each method

        Returns:
            List of all inconsistent entities found
        """
        if not results:
            return ["No verification methods available"]

        # Collect all issues
        all_issues = []
        for result in results.values():
            all_issues.extend(result.inconsistent_items)

        return all_issues
