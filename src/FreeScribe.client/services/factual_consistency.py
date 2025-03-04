"""
Module for verifying factual consistency between original text and generated summaries.
This module provides a pipeline of different verification methods to ensure accuracy.
"""

import os
import spacy
from typing import Tuple, List, Dict, Any
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


FACTUAL_CONFIDENCE_THRESHOLD = 0.6


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
    method: VerificationMethod
    is_consistent: bool = True
    inconsistent_items: List[str] = field(default_factory=list)
    confidence: float = 1.0 # 0.0 to 1.0



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
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            logger.info("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_trf")
            self.nlp = spacy.load("en_core_web_trf")

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
        if not generated_summary:
            return VerificationResult(
                method=VerificationMethod.NER
            )
        if not original_text:
            return VerificationResult(
                is_consistent=False,
                confidence=0.0,
                method=VerificationMethod.NER
            )
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

def main():
    original = """ Hi, Mr. Jones.  How are you?  I'm good, Dr. Smith.  Thanks to see you.  Thanks to see you again.  What brings you back?  Well, my back's been hurting again.  I see.  I've seen you a number of times for this, haven't I?  Yeah, well, ever since I got hurt on the job three years ago.  It's something that just keeps coming back.  You know, it'll be fine for a while.  And then I'll end down or I'll move in a weird way.  And then boom, it'll just go out again.  Unfortunately, that can happen.  And I do have quite a few patients who get reoccurring episodes of back pain.  Have you been keeping up with the therapy that we had you on before?  Which the pills?  Actually, I was talking about the physical therapy that we had you doing.  The pills are only meant for short term because they don't actually prevent the back pain from coming back.  See, yeah, I've once my back started feeling better.  I was happy not to go to the therapist anymore.  Why was that?  Well, it started to become kind of a hassle with my work schedule and the cost was an issue.  But I was able to get back to work.  And I could use the money.  Do you think the physical therapy was helping?  Yeah, what was slow going at first?  I see.  Physical therapy is a bit slower than medications.  But the point is to build up the core muscles in your back and your abdomen.  Physical therapy is also less invasive than medications.  So that's why we had you doing the therapy.  But you mentioned that cost was getting to be a real issue for you.  Can you tell me more about that?  Well, the insurance I had only covered a certain number of sessions.  And then they moved my therapy office because they were trying to work out my schedule at work.  But that was really far away.  And then I had to deal with parking and it just started to get really expensive.  Got it.  I understand.  So for now, I'd like you to try using a heating pad for your back pain.  So that should help in the short term.  Our goal is to get your back pain under better control without creating additional problems for you like cost.  Let's talk about some different options and the pros and cons of each.  So the physical therapy is actually really good for your back pain.  But there are other things we can be doing to help.  Yes, I definitely don't need to lose any more time at work and just lie around the house all day.  Okay, whether some alternative therapies like yoga or tai chi classes or meditation therapies that might be able to help.  And they might also be closer to you and be less expensive.  Would that be something you'd be interested in?  Sure, that'd be great.  Good.  Let's talk about some of the other costs of your care.  In the past we had you on some tram at all because the physical therapy alone wasn't working.  Yeah, that medicine was working really well.  But again, the cost to take out really expensive.  Yeah, yeah.  So that is something in the future we could order something like a generic medication.  And then there are also resources for people to look up the cheapest cost of their medications.  But for now, I'd like to stick with the non prescription medications.  And if we can have you go to yoga or tai chi classes like I mentioned, that could alleviate the need for ordering prescriptions.  Okay, yeah, that sounds good.  Okay, great, great.  Are there any other costs that are a problem for you and your care?  Well, my insurance isn't going down, but that seems to be the case for everybody that I talk to, but I should be able to make it work.  Yeah.  And fortunately, that is an issue for a lot of people.  But I would encourage you during open season to look at your different insurance options,  to see which plan is more cost-effective for you.  Okay, yeah, that sounds great.  Great.  Well, I appreciate you talking to me today.  Yeah, I'm glad you're able to come in.  What I'll do is I'll have my office team research, the different things that you and I talked about today.  And then let's set a time early next week, say Tuesday, where we can talk over the phone,  about what we were able to come up with for you and see if those would work for you.  Okay, great.  Great. """
    summary = """## SOAP Note

**Subjective:** 
Mr. Jones reports his back pain has been recurring since an injury at work three years ago. He describes it as coming and going, with periods of improvement followed by episodes where he experiences discomfort and limited mobility.  He notes that the pain is exacerbated by movement and can be debilitating. He previously had physical therapy but stopped due to cost concerns related to his insurance coverage and the distance of his therapy office. He also mentions a preference for non-prescription medications, which were effective in the past, but are expensive. 

**Objective:**
None.

**Assessment:**  
Mr. Jones presents with chronic back pain likely stemming from an occupational injury. His history suggests a pattern of recurring episodes, potentially related to muscle weakness and instability. He reports difficulty managing the cost of his care, including physical therapy and medications. 

**Plan:**
1. **Short-term Management:** Recommend using a heating pad for pain relief.
2. **Alternative Therapies:** Explore options like yoga, tai chi, or meditation classes as potential alternatives to prescription medication and potentially more affordable.
3. **Insurance Review:** Encourage Mr. Jones to review his insurance plan during open enrollment to identify cost-effective options. 
4. **Follow-up:** Schedule a phone call with Mr. Jones next week to discuss the proposed management strategies, including alternative therapies and potential cost-saving measures.  


"""
    is_consistent, issues, confidence = verify_factual_consistency(original, summary)
    print(f"{is_consistent=} {issues=} {confidence=}")


if __name__ == '__main__':
    main()
