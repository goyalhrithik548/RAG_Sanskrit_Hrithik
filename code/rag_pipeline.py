print(">>> SCRIPT STARTED <<<")

from retriever import retrieve_chunks
from generator import generate_answer

def main():
    print(">>> ENTERED MAIN LOOP <<<")
    print("RAG pipeline ready. Ask a question (or type exit):")

    while True:
        query = input("\nAsk a question (or type exit): ").strip()
        if query.lower() == "exit":
            print("Exiting...")
            break

        chunks = retrieve_chunks(query)

        MAX_CONTEXT_CHARS = 1200
        context = ""

        for chunk in chunks:
            if len(context) + len(chunk) > MAX_CONTEXT_CHARS:
                break
            context += chunk + "\n"

        print("\n--- CONTEXT SENT TO LLM ---")
        print(context)
        print("--- END CONTEXT ---\n")

        answer = generate_answer(context, query)
        print("Answer:")
        print(answer)

if __name__ == "__main__":
    main()
