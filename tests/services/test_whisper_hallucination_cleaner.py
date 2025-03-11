import pytest
from services.whisper_hallucination_cleaner import WhisperHallucinationCleaner, COMMON_HALUCINATIONS, SIMILARITY_THRESHOLD, download_spacy_model
import spacy


@pytest.fixture
def cleaner():
    """Create a WhisperHallucinationCleaner instance for testing.

    :returns: A configured WhisperHallucinationCleaner instance
    :rtype: WhisperHallucinationCleaner
    """
    return WhisperHallucinationCleaner(similarity_threshold=SIMILARITY_THRESHOLD)

def test_initialization(cleaner):
    """Test the initialization of WhisperHallucinationCleaner.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    assert cleaner.similarity_threshold == SIMILARITY_THRESHOLD
    assert isinstance(cleaner.hallucinations, set)
    # Both should already be sets of lowercase strings
    assert cleaner.hallucinations == set(h.lower() for h in COMMON_HALUCINATIONS)

def test_nlp_loading(cleaner):
    """Test that the spacy model loads correctly.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    nlp = cleaner.nlp
    assert nlp is not None
    # Test that it can process text
    doc = nlp("Test sentence.")
    assert len(list(doc.sents)) == 1

def test_is_similar_to_hallucination(cleaner):
    """Test similarity checking against hallucinations.
    
    Tests exact matches, similar matches, and completely different text.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Test with exact match
    assert cleaner._is_similar_to_hallucination("thank you") is True
    
    # Test with similar but not exact match
    assert cleaner._is_similar_to_hallucination("thanks a lot") is True
    
    # Test with completely different text
    assert not cleaner._is_similar_to_hallucination("the quick brown fox jumps over the lazy dog")

def test_split_into_sentences(cleaner):
    """Test sentence splitting functionality.
    
    Tests multiple sentences, empty text, and various punctuation.
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Test with multiple sentences
    text = "First sentence. Second sentence. Third sentence."
    sentences = cleaner._split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].strip() == "First sentence."
    assert sentences[1].strip() == "Second sentence."
    assert sentences[2].strip() == "Third sentence."
    
    # Test with empty text
    assert cleaner._split_into_sentences("") == []
    
    # Test with text containing various punctuation
    text = "Sentence one! Sentence two? Sentence three."
    sentences = cleaner._split_into_sentences(text)
    assert len(sentences) == 3

def test_clean_text(cleaner):
    """Test the main text cleaning functionality."""
    # Test with text containing hallucinations
    text = "This is a normal sentence. thank you. This is another normal sentence. thanks for watching!"
    cleaned = cleaner.clean_text(text)
    assert "thank you" not in cleaned
    assert "thanks for watching" not in cleaned
    assert "This is a normal sentence" in cleaned
    assert "This is another normal sentence" in cleaned
    
    # Test with empty text
    assert cleaner.clean_text("") == ""
    
    # Test with text containing no hallucinations
    text = "This is a normal sentence. This is another normal sentence."
    cleaned = cleaner.clean_text(text)
    assert cleaned == text  # No double periods should occur
    
    # Test with multiple sentence endings
    text = "First sentence! Second sentence? Third sentence."
    cleaned = cleaner.clean_text(text)
    assert cleaned == text  # Punctuation should be preserved correctly

def test_clean_text_with_different_threshold():
    """Test text cleaning with different similarity thresholds."""
    # Create cleaner with higher threshold (more strict)
    strict_cleaner = WhisperHallucinationCleaner(similarity_threshold=0.95)
    
    # Create cleaner with lower threshold (more lenient)
    lenient_cleaner = WhisperHallucinationCleaner(similarity_threshold=0.5)
    
    text = "This is a normal sentence. thank you. This is another normal sentence. thanks for watching!"
    
    # Strict cleaner should remove fewer matches
    strict_cleaned = strict_cleaner.clean_text(text)
    assert "thank you" not in strict_cleaned
    
    # Lenient cleaner should remove more similar phrases
    lenient_cleaned = lenient_cleaner.clean_text(text)
    assert "thank you" not in lenient_cleaned
    assert "thanks for watching" not in lenient_cleaned

def test_clean_text_with_edge_cases(cleaner):
    """Test text cleaning with various edge cases.
    
    Tests handling of:
        * Multiple spaces
        * Mixed case text
        * Various punctuation
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Test with text containing multiple spaces
    text = "This  has  multiple  spaces.  thank  you."
    cleaned = cleaner.clean_text(text)
    assert "thank  you" not in cleaned
    assert "This  has  multiple  spaces" in cleaned
    
    # Test with text containing mixed case
    text = "This is a sentence. THANK YOU. This is another sentence."
    cleaned = cleaner.clean_text(text)
    assert "THANK YOU" not in cleaned
    
    # Test with text containing punctuation
    text = "This is a sentence! thank you? This is another sentence."
    cleaned = cleaner.clean_text(text)
    assert "thank you" not in cleaned
    assert "This is a sentence" in cleaned
    assert "This is another sentence" in cleaned 

def test_normalize_text(cleaner):
    """Test text normalization functionality.
    
    Tests handling of:
        * Extra whitespace
        * Mixed case text
        * Empty strings
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Test with extra whitespace
    assert cleaner._normalize_text("  Hello   World  ") == "hello world"
    
    # Test with mixed case
    assert cleaner._normalize_text("HeLLo WoRLD") == "hello world"
    
    # Test with empty string
    assert cleaner._normalize_text("") == ""

def test_hallucination_docs_property(cleaner):
    """Test the hallucination_docs property.
    
    Tests:
        * Correct creation of spaCy docs
        * Proper handling of case-insensitive duplicates
        * Caching behavior
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Test that docs are created correctly
    docs = cleaner.hallucination_docs
    # Get unique lowercase hallucinations since some might be duplicates when lowercased
    unique_hallucinations = len(set(h.lower() for h in COMMON_HALUCINATIONS))
    assert len(docs) == unique_hallucinations
    
    # Test caching - should return same object
    assert cleaner.hallucination_docs is docs

def test_is_similar_to_hallucination_with_long_sentences(cleaner):
    """Test similarity checking with sentences of varying lengths.
    
    Tests:
        * Very long sentences (should not be detected)
        * Single hallucination
        * Hallucination within a reasonable length sentence
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Create a sentence longer than MAX_SENTENCE_LENGTH
    long_sentence = "This is a very long sentence that " * 10
    assert not cleaner._is_similar_to_hallucination(long_sentence)
    
    # Test with a hallucination by itself
    hallucination = COMMON_HALUCINATIONS[0]
    assert cleaner._is_similar_to_hallucination(hallucination)
    
    # Test with a hallucination in a reasonable length sentence
    sentence_with_hallucination = f"I wanted to say {COMMON_HALUCINATIONS[0]} to everyone"
    assert cleaner._is_similar_to_hallucination(sentence_with_hallucination)

def test_vector_similarity(cleaner):
    """Test vector similarity comparisons.
    
    Tests:
        * Semantically similar phrases that should be detected
        * Dissimilar phrases that should not be detected
    
    :param cleaner: The WhisperHallucinationCleaner fixture
    :type cleaner: WhisperHallucinationCleaner
    """
    # Test phrases that are semantically similar but not exact matches
    similar_phrases = [
        "thanks for your time",
        "appreciate your watching",
        "see you in our next video",
        "goodbye everyone"
    ]
    
    for phrase in similar_phrases:
        # These should be caught by vector similarity
        assert cleaner._is_similar_to_hallucination(phrase)
    
    # Test phrases that shouldn't be similar
    dissimilar_phrases = [
        "the weather is nice today",
        "please pass the salt",
        "what time is the meeting"
    ]
    
    for phrase in dissimilar_phrases:
        assert not cleaner._is_similar_to_hallucination(phrase)

@pytest.mark.parametrize("attempt_scenario", [
    (True, None),  # Success on first try
    (False, RuntimeError("Download failed")),  # Failure
])
def test_download_spacy_model(monkeypatch, attempt_scenario):
    """Test spacy model download functionality.
    
    Tests:
        * Successful download on first attempt
        * Failed download with retry behavior
    
    :param monkeypatch: pytest's monkeypatch fixture
    :type monkeypatch: pytest.MonkeyPatch
    :param attempt_scenario: Tuple of (success_flag, error_to_raise)
    :type attempt_scenario: tuple[bool, Exception|None]
    """
    success, error = attempt_scenario
    calls = []
    
    def mock_is_package(name):
        return False
        
    def mock_run(*args, **kwargs):
        calls.append(args[0])
        # Create a mock CompletedProcess object
        class MockCompletedProcess:
            def __init__(self, returncode, stderr=""):
                self.returncode = returncode
                self.stderr = stderr
                
        if success:
            return MockCompletedProcess(0)
        else:
            return MockCompletedProcess(1, "Download failed")
    
    monkeypatch.setattr(spacy.util, "is_package", mock_is_package)
    monkeypatch.setattr("subprocess.run", mock_run)
    
    if success:
        assert download_spacy_model()
        assert len(calls) == 1  # Should be called once on success
    else:
        assert not download_spacy_model()
        assert len(calls) == 3  # Should try 3 times on failure

def test_global_hallucination_cleaner():
    """Test the global hallucination cleaner instance.
    
    Tests:
        * Instance type
        * Default configuration
    """
    from services.whisper_hallucination_cleaner import hallucination_cleaner
    
    assert isinstance(hallucination_cleaner, WhisperHallucinationCleaner)
    assert hallucination_cleaner.similarity_threshold == SIMILARITY_THRESHOLD 