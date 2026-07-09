from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

def get_answer(query: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    relevant_docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""Answer the question using ONLY the context below. Do not ask for clarification.
If the answer is not in the context, respond exactly: "I don't have enough information."

Context:
{context}

Question: {query}
Answer:"""

    llm = Ollama(model="llama3.2:1b", temperature=0)
    response = llm.invoke(prompt)
    return response, relevant_docs

if __name__ == "__main__":
    query = input("Ask a question: ")
    answer, docs = get_answer(query)
    print("\n--- Answer ---")
    print(answer)