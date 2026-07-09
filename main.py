from fastapi import FastAPI
from pydantic import BaseModel
from agent_graph import build_graph
import csv, datetime, os

app = FastAPI(title="Mini RAG Agent API")
graph = build_graph()

class Query(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "RAG Agent API is running. Go to /docs to test."}

@app.post("/ask")
def ask(query: Query):
    result = graph.invoke({"query": query.question, "answer": "", "retries": 0})
    log_interaction(query.question, result["answer"])
    return {
        "question": query.question,
        "answer": result["answer"],
        "retries": result["retries"]
    }

def log_interaction(question, answer):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.isfile("logs/interactions.csv")
    with open("logs/interactions.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer"])
        writer.writerow([datetime.datetime.now(), question, answer])