# Mini RAG + Agentic AI System

A lightweight Retrieval-Augmented Generation (RAG) system with agentic fallback handling, built to demonstrate end-to-end GenAI engineering — from document ingestion to API deployment.

## Features
- PDF document ingestion and semantic chunking
- Vector embeddings using Sentence Transformers
- FAISS vector store for similarity search
- LangGraph-based agentic workflow with fallback/retry logic
- FastAPI backend exposing a REST API
- Interaction logging for observability

## Tech Stack
Python, LangChain, LangGraph, FastAPI, FAISS, Sentence Transformers, Ollama (local LLM)

## Architecture
## Setup

1. Clone the repo
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Add PDF files to `data/` folder
6. Run ingestion: `python ingest.py`
7. Install Ollama and pull model: `ollama pull llama3.2:1b`
8. Start API: `uvicorn main:app --reload`
9. Test at `http://127.0.0.1:8000/docs`

## Example
**Request:**
```json
POST /ask
{ "question": "Can Medicare cover respite care?" }
```

**Response:**
```json
{ "question": "Can Medicare cover respite care?", "answer": "Medicare covers respite care for up to 5 days each time.", "retries": 0 }
```