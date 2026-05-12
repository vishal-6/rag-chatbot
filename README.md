# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, FAISS, Groq LLM, and Flask.

## Features
- Document-based Q&A using RAG pipeline
- FAISS vector store for semantic search
- Groq LLM (llama-3.1-8b-instant) for fast responses
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Flask REST API endpoint

## Tech Stack
- Python, Flask
- LangChain, FAISS
- Groq API
- HuggingFace Transformers

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GROQ_API_KEY` in a `.env` file
4. Run `python ingest.py` to build the vector store
5. Run `python app.py` to start the server

## Usage
```
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```
