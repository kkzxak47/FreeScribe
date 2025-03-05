import spacy
import services.factual_consistency.verifiers

# Load the general English model
model_name = services.factual_consistency.verifiers.NERVerifier.NLP_MODEL
nlp = spacy.load(model_name)

# Example conversation text
conversation = """
Dr. Smith met with John Doe on March 5, 2025. John mentioned back pain and hypertension.
Dr. Smith prescribed lisinopril and scheduled a blood test for next week.
"""

# Process the text
doc = nlp(conversation)

# Extract named entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Named Entities:", entities)

# Example summary text (with a potential hallucination)
summary = """
Dr. Smith met John Doe on March 5, 2025, to discuss back pain and cancer.
He prescribed lisinopril and metformin.
"""

# Process the summary
summary_doc = nlp(summary)
summary_entities = [(ent.text, ent.label_) for ent in summary_doc.ents]
print("Summary Entities:", summary_entities)

# Compare entities to detect hallucinations
original_set = set([ent[0] for ent in entities])
summary_set = set([ent[0] for ent in summary_entities])
hallucinations = summary_set - original_set
print("Potential Hallucinations:", hallucinations)
