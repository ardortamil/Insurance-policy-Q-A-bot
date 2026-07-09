from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def load_and_chunk(data_dir="data"):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, filename))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store saved.")
    return vectorstore

if __name__ == "__main__":
    chunks = load_and_chunk()
    build_vectorstore(chunks)