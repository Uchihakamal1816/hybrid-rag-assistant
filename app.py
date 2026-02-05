import streamlit as st
from rerank import rerank_chunks
from transformers import pipeline
from eval_rag import evaluate_chat

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline



@st.cache_resource
def load_system():
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

    return vectorstore, llm


vectorstore, llm = load_system()




if "chat" not in st.session_state:
    st.session_state.chat = []

if "eval_log" not in st.session_state:
    st.session_state.eval_log = []

if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []




def generate_answer(query, chunks, history):
    context = "\n\n".join(c.page_content for c in chunks)
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



st.title("📄 Hybrid Conversational RAG Assistant")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🆕 New Chat"):
        if st.session_state.eval_log:
            st.session_state.chat_logs.append(
                list(st.session_state.eval_log)
            )
        st.session_state.chat = []
        st.session_state.eval_log = []

with col2:
    if st.button("📜 View Logs"):
        st.subheader("Previous Chats")
        for i, chat in enumerate(st.session_state.chat_logs, 1):
            st.markdown(f"### Chat {i}")
            for turn in chat:
                st.write("**Q:**", turn["question"])
                st.write("**A:**", turn["answer"])

with col3:
    if st.button("📊 Evaluate Chat"):
        if st.session_state.eval_log:
            report = evaluate_chat(st.session_state.eval_log)
            st.subheader("Evaluation Report")
            st.json(report)




query = st.text_input("Ask about AI governance:")

if query:
    st.session_state.chat.append(f"User: {query}")

    retrieved_chunks = vectorstore.similarity_search(query, k=10)
    best_chunks = rerank_chunks(query, retrieved_chunks, top_k=3)

    if not best_chunks:
        best_chunks = retrieved_chunks[:3]

    answer = generate_answer(query, best_chunks, st.session_state.chat)

    st.session_state.chat.append(f"Assistant: {answer}")

    st.session_state.eval_log.append({
        "question": query,
        "answer": answer,
        "chunks": best_chunks
    })

    st.markdown("### 🤖 Answer")
    st.write(answer)

    st.markdown("### 📚 Sources")
    for c in best_chunks:
        st.write(c.metadata)


st.markdown("## 💬 Conversation")

for msg in st.session_state.chat:
    if msg.startswith("User"):
        st.write("🧑", msg.replace("User: ", ""))
    else:
        st.write("🤖", msg.replace("Assistant: ", ""))
