from langgraph.graph import StateGraph, END
from typing import TypedDict
from rag_pipeline import get_answer

class AgentState(TypedDict):
    query: str
    answer: str
    retries: int

def retriever_agent(state: AgentState):
    answer, docs = get_answer(state["query"])
    state["answer"] = answer
    return state

def validator_agent(state: AgentState):
    # Fallback loop: answer weak ah irundha retry pannurathu
    if "don't have enough information" in state["answer"].lower() and state["retries"] < 2:
        state["retries"] += 1
        return "retry"
    return "done"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retriever", retriever_agent)
    graph.set_entry_point("retriever")
    graph.add_conditional_edges("retriever", validator_agent, {
        "retry": "retriever",
        "done": END
    })
    return graph.compile()

if __name__ == "__main__":
    app = build_graph()
    query = input("Ask a question: ")
    result = app.invoke({"query": query, "answer": "", "retries": 0})
    print("\n--- Final Answer ---")
    print(result["answer"])
    print(f"\n(Retries used: {result['retries']})")