"""
Example usage of the SpaCy Intent Recognizer.

This module demonstrates how to use the SpaCy Intent Recognizer
to identify medical intents from conversation text.
"""

import logging
from services.intent_actions.intents.spacy_recognizer import SpacyIntentRecognizer, SpacyIntentPattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Demonstrate basic usage of the SpaCy Intent Recognizer.
    """
    # Initialize the recognizer
    recognizer = SpacyIntentRecognizer()
    
    try:
        # Initialize the model
        recognizer.initialize()
        
        # Example conversation texts
        texts = [
            "I need to get to Memorial Hospital for my appointment at 2 PM",
            "Can you help me schedule an appointment with Dr. Smith?",
            "I need directions to City Medical Center"
        ]
        
        # Process each text
        for text in texts:
            logger.info(f"\nProcessing text: {text}")
            intents = recognizer.recognize_intent(text)
            
            # Print results
            for intent in intents:
                logger.info(f"Recognized intent: {intent.name}")
                logger.info(f"Confidence: {intent.confidence:.2f}")
                logger.info(f"Parameters: {intent.metadata['parameters']}")
                logger.info("-" * 50)
    
    except Exception as e:
        logger.error(f"Error in example: {e}")

if __name__ == "__main__":
    main() 