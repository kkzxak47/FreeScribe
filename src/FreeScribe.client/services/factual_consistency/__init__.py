"""
.. module:: factual_consistency
   :synopsis: Module for verifying factual consistency between original text and generated summaries.

This module provides a pipeline of different verification methods to ensure accuracy 
in medical note generation. It includes named entity recognition and other verification
techniques to validate that generated summaries accurately reflect the original content.

.. note::
    The module is designed specifically for medical/clinical text verification.

.. warning::
    This is not a general-purpose fact verification system and should only be used
    for medical note generation scenarios.
"""

import logging
from typing import Tuple, List

from services.factual_consistency.verifiers import ConsistencyPipeline

logger = logging.getLogger(__name__)


FACTUAL_CONFIDENCE_THRESHOLD = 0.6

# Create a global pipeline instance
pipeline = ConsistencyPipeline()


def verify_factual_consistency(original_text: str, generated_summary: str) -> Tuple[bool, List[str], float]:
    """Verify factual consistency between original text and generated summary using multiple methods.

    :param original_text: The original transcribed text
    :type original_text: str
    :param generated_summary: The generated medical note/summary
    :type generated_summary: str
    :return: Tuple containing verification results
    :rtype: Tuple[bool, List[str], float]
    
    The return tuple contains:
        - is_consistent (bool): True if all verification methods agree on consistency
        - issues (List[str]): List of all issues found across methods
        - confidence (float): Overall confidence score (0.0 to 1.0)

    .. note::
        A confidence score below 0.6 indicates potential factual inconsistencies.

    .. warning::
        This function may raise exceptions if verification methods fail.
        Callers should handle potential errors appropriately.
    """
    results = pipeline.verify(original_text, generated_summary)
    return pipeline.get_overall_consistency(results)
