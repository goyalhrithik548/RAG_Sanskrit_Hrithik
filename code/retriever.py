import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model ONCE
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Load FAISS index and chunks ONCE
index = faiss.read_index("faiss_index.bin")
chunks = np.load("chunk_texts.npy", allow_pickle=True)


def retrieve_chunks(query, top_k=3):
    """
    Retrieve top_k most relevant text chunks for a query.
    Works for Sanskrit / transliterated input at embedding level.
    """

    # Encode query
    query_embedding = model.encode([query]).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results
