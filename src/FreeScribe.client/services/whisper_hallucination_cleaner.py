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

from typing import List, Optional
import string
import spacy
import logging
from spacy.language import Language
from spacy.tokens import Doc
import os
# Create a punctuation string without apostrophe
punct_without_apostrophe = string.punctuation.replace("'", "")

# Default logger
default_logger = logging.getLogger(__name__)

class HallucinationCleanerException(Exception):
    """Exception raised for errors in the hallucination cleaner."""
    pass


COMMON_HALUCINATIONS = [
    "thank you",
    "thank y'all",
    "thanks mate",
    "thanks a lot",
    "thank you all",
    "thank you bye",
    "thank you for",
    "thank you sir",
    "thank you too",
    "thanks so much",
    "thank you again",
    "thank you thank",
    "thanks everyone",
    "thanks very much",
    "see you next time",
    "thank you bye bye",
    "thank you so much",
    "it's no good to me",
    "see you next video",
    "thank you everyone",
    "thank you very much",
    "thanks for watching",
    "he was gonna catch it",
    "thank you all so much",
    "i'll see you next time",
    "thank you for watching",
    "see you in the next one",
    "thank you all very much",
    "bye ladies and gentlemen",
    "thanks for your watching",
    "see you in the next video",
    "subtitles by steamteamextra",
    "thank you for your watching",
    "i'll see you in the next video",
    "i'm not sure what i'm doing here",
    "thank you very much for watching",
    "hello everyone welcome to my channel",
    "subtitles by the amara org community",
    "thank you very much thank you very much",
    "like comment and subscribe to the channel",
    "don't forget to like comment and subscribe to the channel",
    "sorry don't ask me if i asked about this question right and i mean the lightening",
]

# Calculate max length based on tokens instead of characters for more accurate comparison
MAX_SENTENCE_LENGTH = max(len(hallucination.split()) for hallucination in COMMON_HALUCINATIONS)
SIMILARITY_THRESHOLD = 0.95
SPACY_MODEL_NAME = "en_core_web_md"
SPACY_MODEL_PATH = os.path.join(spacy.util.get_package_path(SPACY_MODEL_NAME), f"{SPACY_MODEL_NAME}-3.7.1")


class WhisperHallucinationCleaner:
    """A class to clean common hallucinations from Whisper transcriptions.
    
    This class uses spaCy's vector similarity to detect and remove common phrases
    that Whisper tends to hallucinate, particularly at the end of transcriptions.
    
    :param similarity_threshold: The minimum similarity ratio (0-1) between a sentence
                               and a hallucination to consider it a match
    :param spacy_model_path: Path of the spaCy model to use
    :param hallucinations: List of hallucination phrases to check against
    :param nlp: Optional pre-configured spaCy model
    :param logger: Logger instance for debugging
    :type similarity_threshold: float
    :type spacy_model_path: str
    :type hallucinations: List[str]
    :type nlp: Optional[Language]
    :type logger: logging.Logger
    """
    
    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        spacy_model_path: str = SPACY_MODEL_PATH,
        hallucinations: List[str] = COMMON_HALUCINATIONS,
        nlp: Optional[Language] = None,
        logger: logging.Logger = default_logger
    ):
        """Initialize the cleaner with configurable dependencies.
        
        :param similarity_threshold: The minimum similarity ratio (0-1)
        :param spacy_model_path: Path of the spaCy model to use
        :param hallucinations: List of hallucination phrases to check against
        :param nlp: Optional pre-configured spaCy model
        :param logger: Logger instance for debugging
        """
        self.logger = logger
        self.similarity_threshold = similarity_threshold
        self.spacy_model_path = spacy_model_path
        self._trans_table = str.maketrans(punct_without_apostrophe, ' ' * len(punct_without_apostrophe))
        self.hallucinations = {self._normalize_text(h) for h in hallucinations}
        self._nlp = nlp
        self._hallucination_docs = None
        
    def initialize_model(self) -> Optional[str]:
        """Initialize the spaCy model proactively.
        
        This method should be called when the hallucination cleaning feature is enabled
        in settings. It downloads and loads the model if necessary.
        
        :return: Error message if initialization fails, None if successful
        :rtype: Optional[str]
        """
        try:
            if self._nlp is not None:
                return None
            try:
                self._nlp = spacy.load(self.spacy_model_path)
            except IOError as e:
                return f"Failed to load spaCy model. {e}"
            # Pre-process hallucination docs
            self._hallucination_docs = [
                self._nlp(h) for h in sorted(COMMON_HALUCINATIONS)
            ]
            return None
        except Exception as e:
            error_msg = f"Failed to initialize spaCy model: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

    def unload_model(self):
        """Unload the spaCy model and free resources."""
        self._nlp = None
        self._hallucination_docs = None
        self.logger.info("Unloaded spaCy model")
        
    @property
    def nlp(self) -> Language:
        """Lazy load the spacy model.
        
        :return: The loaded spaCy model
        :rtype: spacy.language.Language
        :raises RuntimeError: If the spaCy model fails to download
        """
        if self._nlp is None:
            if error := self.initialize_model():
                raise RuntimeError(error)
        return self._nlp
    
    @property
    def hallucination_docs(self) -> List[Doc]:
        """Lazy load the hallucination docs.
        
        :return: List of processed spaCy docs for each hallucination
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
        
        :param text: The text to normalize
        :type text: str
        :return: Normalized text with punctuation removed and whitespace normalized
        :rtype: str
        """
        # Remove punctuation and normalize whitespace
        text = text.strip().lower()
        # remove all punctuation
        text = text.translate(self._trans_table)
        text = ' '.join(t for t in text.split() if t.isalnum())
        self.logger.debug(f"Normalized text: {text}")
        return text
    
    def _is_similar_to_hallucination(self, sentence: str) -> bool:
        """Check if a sentence is similar to any known hallucination using vector similarity.
        
        :param sentence: The sentence to check for hallucination similarity
        :type sentence: str
        :return: True if the sentence is similar to a known hallucination, False otherwise
        :rtype: bool
        """
        if not sentence:
            return False

        # Normalize for exact matching
        normalized = self._normalize_text(sentence)
        # First check for exact matches (case insensitive)
        if any(h in normalized for h in self.hallucinations):
            self.logger.debug(f"Sentence contains a hallucination: {normalized}")
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
            self.logger.debug(f"Sentence is similar to hallucination: {sentence}")
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spacy.
        
        :param text: The text to split into sentences
        :type text: str
        :return: List of sentences
        :rtype: List[str]
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
        :return: Cleaned text with hallucination sentences removed
        :rtype: str
        """
        if not text:
            return text
            
        sentences = self._split_into_sentences(text)
        cleaned_sentences = [s for s in sentences if not self._is_similar_to_hallucination(s)]
        
        # Join sentences back together with a single space, since each sentence already has its punctuation
        result = ' '.join(s.strip() for s in cleaned_sentences)
        self.logger.debug(f"Cleaned text: {result}")
        return result

# Initialize the hallucination cleaner with default settings
hallucination_cleaner = WhisperHallucinationCleaner()
