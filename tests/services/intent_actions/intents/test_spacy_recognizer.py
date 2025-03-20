"""
Tests for SpaCy-based intent recognition.
"""

import pytest
from pydantic import ValidationError
import spacy
from unittest.mock import patch, MagicMock

from services.intent_actions.intents.spacy_recognizer import SpacyIntentPattern, SpacyIntentRecognizer

@pytest.fixture
def mock_nlp():
    """Create a mock SpaCy NLP model."""
    mock = MagicMock()
    mock.vocab = spacy.blank("en").vocab
    return mock

@pytest.fixture
def test_pattern():
    """Create a test intent pattern."""
    return SpacyIntentPattern(
        intent_name="test_intent",
        patterns=[[{"LOWER": "test"}, {"LOWER": "pattern"}]],
        required_entities=["location"],
        confidence_weights={"pattern_match": 0.6, "entity_match": 0.4}
    )

@pytest.fixture
def recognizer(mock_nlp):
    """Create a SpaCy recognizer with mocked model."""
    with patch("spacy.load", return_value=mock_nlp):
        recognizer = SpacyIntentRecognizer()
        recognizer.initialize()
        return recognizer

def test_pattern_creation():
    """Test creating a valid pattern."""
    pattern = SpacyIntentPattern(
        intent_name="test_intent",
        patterns=[[{"LOWER": "test"}]],
        required_entities=["ORG"],
        confidence_weights={"pattern_match": 0.6, "entity_match": 0.4}
    )
    assert pattern.intent_name == "test_intent"
    assert len(pattern.patterns) == 1
    assert pattern.required_entities == ["ORG"]
    assert pattern.confidence_weights == {"pattern_match": 0.6, "entity_match": 0.4}

def test_pattern_validation():
    """Test pattern validation rules."""
    # Test empty intent name
    with pytest.raises(ValidationError) as exc_info:
        SpacyIntentPattern(
            intent_name="",
            patterns=[[{"LOWER": "test"}]]
        )
    assert "intent_name" in str(exc_info.value)

    # Test empty patterns
    with pytest.raises(ValidationError) as exc_info:
        SpacyIntentPattern(
            intent_name="test_intent",
            patterns=[]
        )
    assert "patterns" in str(exc_info.value)

    # Test invalid pattern format
    with pytest.raises(ValidationError) as exc_info:
        SpacyIntentPattern(
            intent_name="test_intent",
            patterns=[{"invalid": "format"}]
        )
    assert "patterns" in str(exc_info.value)

    # Test invalid confidence weights
    with pytest.raises(ValidationError) as exc_info:
        SpacyIntentPattern(
            intent_name="test_intent",
            patterns=[[{"LOWER": "test"}]],
            confidence_weights={"pattern_match": 0.7, "entity_match": 0.4}
        )
    assert "confidence_weights" in str(exc_info.value)

def test_recognizer_initialization(recognizer):
    """Test recognizer initialization."""
    assert recognizer.nlp is not None
    assert recognizer.matcher is not None
    assert len(recognizer.patterns) > 0

def test_add_pattern(recognizer):
    """Test adding a new pattern to the recognizer."""
    pattern = SpacyIntentPattern(
        intent_name="new_intent",
        patterns=[[{"LOWER": "new"}]]
    )
    recognizer.patterns.append(pattern)
    recognizer.matcher.add(pattern.intent_name, pattern.patterns)
    assert len(recognizer.patterns) > 1
    assert pattern.intent_name in recognizer.matcher

def test_calculate_confidence(recognizer):
    """Test confidence calculation."""
    pattern = recognizer.patterns[0]
    doc = MagicMock()
    doc.ents = [MagicMock(label_="ORG"), MagicMock(label_="TIME")]
    matches = [MagicMock()]
    
    confidence = recognizer._calculate_confidence(pattern, doc, matches)
    assert 0 <= confidence <= 1

def test_extract_parameters(recognizer):
    """Test parameter extraction from entities."""
    doc = MagicMock()
    doc.ents = [
        MagicMock(label_="ORG", text="Test Hospital"),
        MagicMock(label_="TIME", text="tomorrow"),
        MagicMock(label_="PRODUCT", text="ambulance")
    ]
    
    params = recognizer._extract_parameters(doc)
    assert params["destination"] == "Test Hospital"
    assert params["appointment_time"] == "tomorrow"
    assert params["transport_mode"] == "ambulance"

def test_recognize_intent(recognizer):
    """Test intent recognition."""
    text = "I need to go to Test Hospital tomorrow"
    intents = recognizer.recognize_intent(text)
    assert isinstance(intents, list)
    if intents:
        intent = intents[0]
        assert intent.name in [p.intent_name for p in recognizer.patterns]
        assert 0 <= intent.confidence <= 1
        assert "parameters" in intent.metadata

def test_no_matches(recognizer):
    """Test handling of text with no matching patterns."""
    text = "This is a completely unrelated text"
    intents = recognizer.recognize_intent(text)
    assert isinstance(intents, list)
    assert len(intents) == 0

def test_recognizer_add_pattern(recognizer, test_pattern):
    """Test adding patterns to recognizer."""
    recognizer.add_pattern(test_pattern)
    assert len(recognizer.patterns) > 1  # Original patterns + new pattern
    assert recognizer.patterns[-1] == test_pattern

def test_recognizer_confidence_calculation(recognizer, test_pattern):
    """Test confidence calculation."""
    # Test pattern match only
    confidence = recognizer._calculate_confidence(
        test_pattern,
        MagicMock(ents=[]),
        [(0, 1, 0.8)]  # Mock match
    )
    assert confidence == 0.6  # Only pattern match weight
    
    # Test entity match only
    mock_doc = MagicMock()
    mock_doc.ents = [MagicMock(label_="location")]
    confidence = recognizer._calculate_confidence(
        test_pattern,
        mock_doc,
        []
    )
    assert confidence == 0.4  # Only entity match weight
    
    # Test both matches
    confidence = recognizer._calculate_confidence(
        test_pattern,
        mock_doc,
        [(0, 1, 0.8)]
    )
    assert confidence == 1.0  # Sum of both weights

def test_recognizer_parameter_extraction(recognizer, test_pattern):
    """Test parameter extraction from text."""
    mock_doc = MagicMock()
    mock_ent = MagicMock()
    mock_ent.label_ = "location"
    mock_ent.text = "test location"
    mock_doc.ents = [mock_ent]
    
    params = recognizer._extract_parameters(mock_doc)
    assert params["destination"] == "test location"

def test_recognizer_intent_recognition(recognizer, test_pattern, mock_nlp):
    """Test intent recognition process."""
    # Setup mock doc
    mock_doc = MagicMock()
    mock_doc.text = "test pattern at test location"
    mock_doc.ents = [MagicMock(label_="location", text="test location")]
    mock_nlp.return_value = mock_doc
    
    # Mock matcher to return matches
    mock_match = (0, 1, 0.8)  # (match_id, start, end)
    recognizer.matcher = MagicMock()
    recognizer.matcher.return_value = [mock_match]
    recognizer.matcher.vocab.strings = {0: "test_intent"}
    
    # Add pattern and recognize
    recognizer.add_pattern(test_pattern)
    intents = recognizer.recognize_intent("test pattern at test location")
    
    assert len(intents) == 1
    assert intents[0].name == "test_intent"
    assert intents[0].confidence == 1.0  # Both pattern and entity match
    assert intents[0].metadata["parameters"]["destination"] == "test location"

def test_recognizer_no_match(recognizer, test_pattern, mock_nlp):
    """Test when no pattern matches."""
    mock_doc = MagicMock()
    mock_doc.text = "unrelated text"
    mock_doc.ents = []
    mock_nlp.return_value = mock_doc
    
    recognizer.add_pattern(test_pattern)
    intents = recognizer.recognize_intent("unrelated text")
    
    assert len(intents) == 0 