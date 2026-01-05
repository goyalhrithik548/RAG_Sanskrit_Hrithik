# generator.py
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Lightweight CPU-safe model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def generate_answer(context: str, question: str) -> str:
    """
    Extractive QA:
    Returns the most relevant sentence from context based on semantic similarity.
    """

    if not context.strip():
        return "No context available."

    # Split context into sentences (very important)
    sentences = [
        s.strip()
        for s in context.replace("\n", " ").split("।")
        if len(s.strip()) > 10
    ]

    if not sentences:
        return "No meaningful sentences found in context."

    # Encode
    question_embedding = model.encode(question, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Similarity
    similarities = util.cos_sim(question_embedding, sentence_embeddings)[0]
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    # Confidence threshold
    if best_score < 0.25:
        return "Answer not clearly found in the text."

    return sentences[best_idx]
