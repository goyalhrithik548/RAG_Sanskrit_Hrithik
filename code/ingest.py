import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "../data"
OUTPUT_FILE = "processed_chunks.txt"


def load_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)
    return documents


def clean_text(text):
    """
    Minimal text cleaning:
    - Remove excessive newlines
    - Normalize spaces
    Sanskrit text is preserved as-is.
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = []
    for doc in documents:
        cleaned = clean_text(doc)
        doc_chunks = splitter.split_text(cleaned)
        chunks.extend(doc_chunks)
    return chunks


def save_chunks(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n---\n\n")


if __name__ == "__main__":
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    save_chunks(chunks, OUTPUT_FILE)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")
