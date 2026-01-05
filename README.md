# 📘 RAG-based Sanskrit Question Answering System

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions from **Sanskrit texts** using semantic search and context-grounded answering.

The system allows users to:

* Ask questions in **Sanskrit or English**
* Retrieve relevant Sanskrit passages using **vector similarity**
* Generate answers **strictly from retrieved context**
* Avoid hallucinations and unsupported answers

---

## 📂 Project Structure

```
RAG_Sanskrit_<InternName>/
│
├── code/
│   ├── rag_pipeline.py      # Main pipeline (entry point)
│   ├── retriever.py         # FAISS-based semantic retriever
│   ├── generator.py         # Context-grounded answer generator
│   ├── ingest.py            # Data ingestion & indexing script
│   ├── chunk_texts.npy      # Stored Sanskrit text chunks
│   ├── faiss_index.bin      # FAISS vector index
│   └── rag_env/             # Python virtual environment
│
├── data/
│   └── processed_chunks.txt # Sample Sanskrit documents
│
├── report/
│   └── RAG_Sanskrit_Report.pdf
│
└── README.md
```

---

## ⚙️ Requirements

* Python **3.9+**
* CPU-only system (no GPU required)

### Python Libraries

```bash
faiss-cpu
sentence-transformers
numpy
torch
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd RAG_Sanskrit_<InternName>
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv rag_env
```

### 3️⃣ Activate Virtual Environment

**Windows**

```bash
rag_env\Scripts\activate
```

**Linux / macOS**

```bash
source rag_env/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install faiss-cpu sentence-transformers numpy torch
```

---

## 📥 Data Preparation (One-Time)

If FAISS index files are **not already present**, run:

```bash
python code/ingest.py
```

This will:

* Read Sanskrit text from `processed_chunks.txt`
* Generate embeddings using
  `sentence-transformers/all-MiniLM-L6-v2`
* Create:

  * `chunk_texts.npy`
  * `faiss_index.bin`

> ⚠️ Skip this step if these files already exist.

---

## ▶️ Running the System

Start the interactive RAG pipeline:

```bash
python code/rag_pipeline.py
```

You should see:

```
>>> SCRIPT STARTED <<<
>>> ENTERED MAIN LOOP <<<
RAG pipeline ready. Ask a question (or type exit):
```

---

## ❓ Usage Examples

### Sanskrit Questions

```
भक्तः कः?
सः किमर्थं देवम् उपासते?
```

### English Questions

```
Who is called a devotee?
What does the text say about true devotion?
```

### Exit

```
exit
```

---

## 🧠 How It Works (Brief)

1. **Query Encoding**

   * Uses `sentence-transformers/all-MiniLM-L6-v2`

2. **Semantic Retrieval**

   * FAISS `IndexFlatL2` retrieves top-k relevant chunks

3. **Context Construction**

   * Chunks concatenated with a fixed character limit

4. **Answer Generation**

   * Answers strictly from retrieved context
   * No hallucination or external knowledge

---

## 🌐 Language Handling

| Question Language | Answer Language |
| ----------------- | --------------- |
| Sanskrit          | Sanskrit        |
| English           | English         |

(No translation layer is used.)

---

## ⚠️ Limitations

* Answers limited to provided corpus
* No abstractive summarization
* No Sanskrit-specific fine-tuned LLM
* Designed for correctness over creativity

---

## 🚀 Future Improvements

* Sanskrit-specific embedding models
* Hybrid keyword + semantic retrieval
* Larger corpus support
* Optional translation layer
* Abstractive answer generation

---

## 📄 License

This project is created for **academic / evaluation purposes** as part of an AI/ML assignment.

---
