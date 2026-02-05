from rerank import rerank_chunks
from transformers import pipeline
from eval_rag import evaluate_chat

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512
)

llm = HuggingFacePipeline(pipeline=generator)




chat_history = []
eval_log = []
chat_logs = []



def generate_answer(query, best_chunks, history):
    context = "\n\n".join(c.page_content for c in best_chunks)
    memory = "\n".join(history[-6:])

    prompt = f"""
Conversation so far:
{memory}

Use only this context:

{context}

Question:
{query}

Answer:
"""
    return llm.invoke(prompt)



print("\n✅ Hybrid Conversational RAG Ready")
print("👉 Ctrl+N = new chat (saved)")
print("👉 Ctrl+L = view logs")
print("👉 Ctrl+C = exit + evaluate\n")


while True:
    try:
        raw = input("Ask: ").strip()

        # Ctrl+N → new chat
        if raw == "\x0e":
            if eval_log:
                chat_logs.append(list(eval_log))

            chat_history.clear()
            eval_log.clear()

            print("\n🔄 New chat started (previous saved)\n")
            continue

        # Ctrl+L → show logs
        if raw == "\x0c":
            print("\n📜 CHAT LOGS\n")
            if not chat_logs:
                print("No previous chats.\n")
            else:
                for i, session in enumerate(chat_logs, 1):
                    print(f"--- Chat {i} ---")
                    for turn in session:
                        print("Q:", turn["question"])
                        print("A:", turn["answer"])
                        print()
            continue

        query = raw
        chat_history.append(f"User: {query}")

        
        retrieved_chunks = vectorstore.similarity_search(query, k=10)
        best_chunks = rerank_chunks(query, retrieved_chunks, top_k=3)

        if not best_chunks:
            best_chunks = retrieved_chunks[:3]

        
        answer = generate_answer(query, best_chunks, chat_history)

        chat_history.append(f"Assistant: {answer}")

        print("\nAnswer:\n", answer)

        eval_log.append({
            "question": query,
            "answer": answer,
            "chunks": best_chunks
        })

        print("\nSources:")
        for c in best_chunks:
            print(c.metadata)

    except KeyboardInterrupt:
        print("\n\n🧠 Evaluating last chat...\n")

        if eval_log:
            report = evaluate_chat(eval_log)
            print("📊 Evaluation Report:")
            print(report)

            chat_logs.append(list(eval_log))

        print("\nAll chats kept in memory. Exiting.\n")
        break
        