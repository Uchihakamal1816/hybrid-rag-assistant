# 📄 Hybrid Conversational RAG System for Regulatory Intelligence

An enterprise-style Retrieval Augmented Generation (RAG) system for
intelligent querying of long-form regulatory and governance documents.

The system combines semantic vector search, LLM-based re-ranking,
conversational memory, automated evaluation, and an interactive
Streamlit UI.

------------------------------------------------------------------------

## 🚀 Features

-   🔍 FAISS vector semantic search over full document corpus\
-   🧠 HuggingFace embeddings for high-quality retrieval\
-   ⚡ LLM-based semantic re-ranking (hybrid RAG)\
-   💬 Conversational memory across turns\
-   📊 Automated evaluation (grounding + faithfulness scoring)\
-   🖥 Streamlit chat interface\
-   📚 Source-aware grounded responses

------------------------------------------------------------------------

## 🏗 System Architecture

User Query → Embeddings → FAISS → LLM Rerank → Grounded Answer →
Evaluation

------------------------------------------------------------------------

## 📁 Project Structure
```
├── app.py
├── rag.py
├── eval_rag.py
├── rerank.py
├── build_index.py
├── requirements.txt
├── .env.example
├── agora/fulltext/
└── faiss_index
```
------------------------------------------------------------------------

## ⚙️ Installation
```
pip install -r requirements.txt

cp .env.example .env

Add API key in .env
```
------------------------------------------------------------------------

## 📦 Build Vector Index
```
python build_index.py
```
------------------------------------------------------------------------

## ▶️ Run App
```
streamlit run app.py
```
------------------------------------------------------------------------

## 🎬 Demo

### 💬 Interactive Chat Interface

-   Multi-turn memory\
-   Hybrid retrieval\
-   Source grounded answers\
-   Automated evaluation

### 📸 Add screenshots:

screenshots/chat_interface.png\
screenshots/sources.png\
screenshots/evaluation.png

Embed:


------------------------------------------------------------------------

## 📊 Evaluation

-   Grounding Score\
-   Faithfulness Score\
-   Session Averages

------------------------------------------------------------------------

## 🧠 Tech Stack

HuggingFace • FAISS • LangChain • Streamlit • Python • Groq

------------------------------------------------------------------------

## 🎯 Use Cases

-   Regulatory intelligence\
-   Compliance automation\
-   Policy analysis\
-   Enterprise knowledge assistant

------------------------------------------------------------------------

## 👨‍💻 Author

Nannuri Sai Kamal\
GitHub: https://github.com/Uchihakamal1816
