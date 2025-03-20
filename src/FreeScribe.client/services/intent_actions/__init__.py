"""
Intent recognition and action system for FreeScribe.

This module provides a unified interface for intent recognition and action execution
in the FreeScribe medical transcription system.
"""

from .intents import (
    Intent,
    ActionResult,
    BaseIntentRecognizer,
    BaseAction,
    LLMIntentRecognizer,
    MedicalIntentResult
)
from .actions import PrintMapAction

__all__ = [
    'Intent',
    'ActionResult',
    'BaseIntentRecognizer',
    'BaseAction',
    'LLMIntentRecognizer',
    'MedicalIntentResult',
    'PrintMapAction'
] 