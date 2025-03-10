"""
This module provides functionality to clean common hallucinations from Whisper transcriptions.
"""

from typing import List
import spacy
import spacy.cli
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


COMMON_HALUCINATIONS = [
    "thank y'all",
    "thank you again",
    "thank you all so much",
    "thank you all very much",
    "thank you all",
    "thank you everyone",
    "thank you for",
    "thank you so much",
    "thank you too",
    "thank you very much",
    "thank you",
    "thank you, bye",
    "thank you, sir",
    "thank you. bye",
    "thank you. thank",
    "thanks a lot",
    "thanks so much",
    "thanks very much",
    "thanks",
    "thanks, everyone",
    "sorry, don't ask me if i asked about this question right, and i mean the lightening",
    "don't forget to like, commentary, and subscribe to the channel",
    "thank you very much. thank you very much",
    "thank you very much for watching!",
    "thank you very much for watching",
    "i'll see you in the next video",
    "thank you for your watching",
    "bye, ladies and gentlemen",
    "see you in the next video",
    "thank you for watching",
    "i'll see you next time",
    "he was gonna catch it",
    "thank you. bye bye",
    "Thanks for watching!",
    "thanks for watching",
    "thanks for watching!",
    "thank you very much",
    "it's no good to me",
    "see you next video",
    "see you next time",
    "see you in the next one",
    "thanks mate",
]

SIMILARITY_THRESHOLD = 0.9


def download_spacy_model():
    """
    Download the spacy model with retries.
    
    Returns:
        bool: True if model was downloaded successfully, False otherwise
    """
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if model is already installed
            if spacy.util.is_package("en_core_web_md"):
                print("Spacy model already installed")
                return True
                
            print(f"Downloading spacy model (attempt {attempt + 1}/{max_retries})...")
            spacy.cli.download("en_core_web_sm")
            print("Spacy model downloaded successfully")
            return True
            
        except Exception as e:
            print(f"Error downloading spacy model: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Failed to download spacy model after all retries")
                return False


class WhisperHallucinationCleaner:
    """A class to clean common hallucinations from Whisper transcriptions."""
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        """
        Initialize the cleaner with a similarity threshold.
        
        Args:
            similarity_threshold (float): The minimum similarity ratio (0-1) 
                                        between a sentence and a hallucination to consider it a match.
        """
        self.similarity_threshold = similarity_threshold
        self.hallucinations = set(COMMON_HALUCINATIONS)
        self._nlp = None
        self._hallucination_vectors = None
        
    @property
    def nlp(self):
        """Lazy load the spacy model."""
        if self._nlp is None:
            if not download_spacy_model():
                raise RuntimeError("Failed to download spacy model")
            self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    @property
    def hallucination_vectors(self):
        """Lazy load the hallucination vectors."""
        if self._hallucination_vectors is None:
            # Process all hallucinations and store their vectors
            self._hallucination_vectors = [
                self.nlp(hallucination).vector 
                for hallucination in self.hallucinations
            ]
        return self._hallucination_vectors
        
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1 (np.ndarray): First vector
            vec2 (np.ndarray): Second vector
            
        Returns:
            float: Cosine similarity between 0 and 1
        """
        result = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        logger.debug(f"vec1: {vec1}")
        logger.debug(f"vec2: {vec2}")
        logger.debug(f"Cosine similarity: {result}")
        return result
    
    def _is_similar_to_hallucination(self, sentence: str) -> bool:
        """
        Check if a sentence is similar to any known hallucination using vector similarity.
        
        Args:
            sentence (str): The sentence to check
            
        Returns:
            bool: True if the sentence is similar to a hallucination, False otherwise
        """
        if not sentence:
            return True
        if sentence in self.hallucinations:
            return True
        sentence = sentence.strip().lower()
        sentence_vector = self.nlp(sentence).vector
        
        # Calculate similarity with all hallucination vectors
        similarities = [
            self._cosine_similarity(sentence_vector, vec)
            for vec in self.hallucination_vectors
        ]
        
        # Return True if any similarity is above threshold
        return max(similarities) >= self.similarity_threshold
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spacy.
        
        Args:
            text (str): The text to split
            
        Returns:
            List[str]: List of sentences
        """
        if not text:
            return []
            
        # Process the text with spacy
        doc = self.nlp(text)
        # Get sentences and clean them
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    def clean_text(self, text: str) -> str:
        """
        Clean the text by removing sentences similar to known hallucinations.
        
        Args:
            text (str): The text to clean
            
        Returns:
            str: The cleaned text
        """
        if not text:
            return text
            
        sentences = self._split_into_sentences(text)
        logger.debug(f"Sentences: {sentences}")
        cleaned_sentences = [s for s in sentences if not self._is_similar_to_hallucination(s)]
        
        logger.debug(f"Cleaned sentences: {cleaned_sentences}")
        # Join sentences back together with periods
        return '. '.join(cleaned_sentences) 

# Initialize the hallucination cleaner
hallucination_cleaner = WhisperHallucinationCleaner()
