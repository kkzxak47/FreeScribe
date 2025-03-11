"""Whisper Hallucination Cleaner.

This module provides functionality to clean common hallucinations from Whisper transcriptions.
It uses spaCy's vector similarity to detect and remove common phrases that Whisper tends to
hallucinate at the end of transcriptions.

Example:
    >>> from services.whisper_hallucination_cleaner import hallucination_cleaner
    >>> text = "This is a real transcription. Thanks for watching!"
    >>> cleaned = hallucination_cleaner.clean_text(text)
    >>> print(cleaned)
    'This is a real transcription.'
"""

from typing import List
import string
import spacy
import spacy.cli
import time
import logging
import subprocess
import sys
from pathlib import Path


logger = logging.getLogger(__name__)


COMMON_HALUCINATIONS = [
    'sorry don t ask me if i asked about this question right and i mean the lightening',
    'don t forget to like comment and subscribe to the channel',
    'thank you very much thank you very much',
    'thank you very much for watching',
    'i ll see you in the next video',
    'thank you for your watching',
    'see you in the next video',
    'bye ladies and gentlemen',
    'thank you all very much',
    'see you in the next one',
    'thank you for watching',
    'i ll see you next time',
    'thank you all so much',
    'he was gonna catch it',
    'thanks for watching',
    'thank you very much',
    'thank you everyone',
    'see you next video',
    'it s no good to me',
    'thank you so much',
    'thank you bye bye',
    'see you next time',
    'thanks very much',
    'thanks everyone',
    'thank you thank',
    'thank you again',
    'thanks so much',
    'thank you too',
    'thank you sir',
    'thank you for',
    'thank you bye',
    'thank you all',
    'thanks a lot',
    'thanks mate',
    'thank y all',
    'thank you',
    'thanks for',
    'your watching',
]

# Calculate max length based on tokens instead of characters for more accurate comparison
MAX_SENTENCE_LENGTH = max(len(hallucination.split()) for hallucination in COMMON_HALUCINATIONS)
SIMILARITY_THRESHOLD = 0.95
SPACY_MODEL_NAME = "en_core_web_md"


def download_spacy_model():
    """Download the spacy model with retries.
    
    Attempts to download the spaCy model if not already installed.
    Will retry up to 3 times with a 2-second delay between attempts.
    Uses subprocess.run to execute the download command for better error handling
    and output capture.
    
    :returns: True if model was downloaded successfully, False otherwise
    :rtype: bool
    
    :raises: No exceptions are raised, failures are logged and False is returned
    """
    max_retries = 3
    retry_delay = 2  # seconds
    python_executable = sys.executable
    
    logger.info(f"Checking/downloading spacy model {SPACY_MODEL_NAME}...")
    for attempt in range(max_retries):
        try:
            # Check if model is already installed
            if spacy.util.is_package(SPACY_MODEL_NAME):
                logger.info("Spacy model already installed")
                return True
            
            logger.info(f"Downloading spacy model (attempt {attempt + 1}/{max_retries})...")
            
            # Use subprocess.run to execute the download command
            result = subprocess.run(
                [python_executable, "-m", "spacy", "download", SPACY_MODEL_NAME],
                capture_output=True,
                text=True,
                check=False  # Don't raise exception on non-zero return code
            )
            
            # Check if the command was successful
            if result.returncode == 0:
                logger.info("Spacy model downloaded successfully")
                return True
            else:
                logger.error(f"Error downloading spacy model: {result.stderr}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    
        except Exception as e:
            logger.error(f"Unexpected error downloading spacy model: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    logger.error("Failed to download spacy model after all retries")
    return False


class WhisperHallucinationCleaner:
    """A class to clean common hallucinations from Whisper transcriptions.
    
    This class uses spaCy's vector similarity to detect and remove common phrases
    that Whisper tends to hallucinate, particularly at the end of transcriptions.
    
    :param similarity_threshold: The minimum similarity ratio (0-1) between a sentence
                               and a hallucination to consider it a match
    :type similarity_threshold: float
    """
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        """Initialize the cleaner with a similarity threshold.
        
        :param similarity_threshold: The minimum similarity ratio (0-1) between a sentence
                                   and a hallucination to consider it a match
        :type similarity_threshold: float
        """
        self.similarity_threshold = similarity_threshold
        # Create a translation table (all punctuation -> spaces)
        self._trans_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        # Store normalized hallucinations for exact matching
        self.hallucinations = {self._normalize_text(h) for h in COMMON_HALUCINATIONS}
        self._nlp = None
        self._hallucination_docs = None
        
    @property
    def nlp(self):
        """Lazy load the spacy model.
        
        :returns: The loaded spaCy model
        :rtype: spacy.language.Language
        :raises RuntimeError: If the spaCy model fails to download
        """
        if self._nlp is None:
            if not download_spacy_model():
                raise RuntimeError("Failed to download spacy model")
            self._nlp = spacy.load(SPACY_MODEL_NAME)
        return self._nlp
    
    @property
    def hallucination_docs(self):
        """Lazy load the hallucination docs.
        
        :returns: List of processed spaCy docs for each hallucination
        :rtype: list[spacy.tokens.Doc]
        """
        if self._hallucination_docs is None:
            # Process all hallucinations and store their docs
            self._hallucination_docs = [
                self.nlp(h) for h in sorted(COMMON_HALUCINATIONS)  # Use original text for semantic similarity
            ]
        return self._hallucination_docs
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing punctuation and extra whitespace.
        
        :param text: Text to normalize
        :type text: str
        :returns: Normalized text
        :rtype: str
        """
        # Remove punctuation and normalize whitespace
        text = text.strip().lower()
        # remove all punctuation
        text = text.translate(self._trans_table)
        text = ' '.join(t for t in text.split() if t.isalnum())
        logger.debug(f"Normalized text: {text}")
        return text
    
    def _is_similar_to_hallucination(self, sentence: str) -> bool:
        """Check if a sentence is similar to any known hallucination using vector similarity.
        
        :param sentence: The sentence to check
        :type sentence: str
        :returns: True if the sentence is similar to a hallucination, False otherwise
        :rtype: bool
        """
        if not sentence:
            return False

        # Normalize for exact matching
        normalized = self._normalize_text(sentence)
        # First check for exact matches (case insensitive)
        if any(h in normalized for h in self.hallucinations):
            logger.debug(f"Sentence contains a hallucination: {normalized}")
            return True
            
        # Process the original sentence for semantic similarity
        doc = self.nlp(sentence)
        
        # Longer sentences are less likely to be hallucinations
        if len(doc) > MAX_SENTENCE_LENGTH * 1.5:
            return False
            
        # Use pre-processed hallucination docs for similarity check
        result = any(doc.similarity(h_doc) >= self.similarity_threshold 
                  for h_doc in self.hallucination_docs)
        if result:
            logger.debug(f"Sentence is similar to hallucination: {sentence}")
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy.
        
        :param text: The text to split
        :type text: str
        :returns: List of sentences
        :rtype: list[str]
        """
        if not text:
            return []
            
        # Process the text with spacy
        doc = self.nlp(text)
        # Get sentences and preserve their original text
        return [sent.text.strip() for sent in doc.sents]
    
    def clean_text(self, text: str) -> str:
        """Clean the text by removing sentences similar to known hallucinations.
        
        :param text: The text to clean
        :type text: str
        :returns: The cleaned text with hallucinations removed
        :rtype: str
        
        Example:
            >>> cleaner = WhisperHallucinationCleaner()
            >>> text = "This is a real transcription. Thanks for watching!"
            >>> cleaned = cleaner.clean_text(text)
            >>> print(cleaned)
            'This is a real transcription.'
        """
        if not text:
            return text
            
        sentences = self._split_into_sentences(text)
        cleaned_sentences = [s for s in sentences if not self._is_similar_to_hallucination(s)]
        
        # Join sentences back together with a single space, since each sentence already has its punctuation
        result = ' '.join(s.strip() for s in cleaned_sentences)
        logger.debug(f"Cleaned text: {result}")
        return result

# Initialize the hallucination cleaner
hallucination_cleaner = WhisperHallucinationCleaner()
