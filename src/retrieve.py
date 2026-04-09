from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import argparse

CHROMA_PATH = "./chroma_db"

def test_retrieval(query_text: str):
    print(f"Query: {query_text}\n")
    
    # TODO: Initialize the exact same HuggingFaceEmbeddings model used in ingest.py
    # Hint: model_name="all-MiniLM-L6-v2"
    model_name="all-MiniLM-L6-v2"
    hf_embedding = HuggingFaceEmbeddings(model_name = model_name)
    
    # TODO: Load the existing Chroma vector database
    # Hint: Use Chroma(persist_directory=CHROMA_PATH, embedding_function=...)
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=hf_embedding
    )
    
    # TODO: Perform the similarity search
    # Hint: Look for a method on your database object like .similarity_search()
    # You should pass the query_text and k=3 (to return the top 3 chunks)
    results = db.similarity_search(query_text, k=3)
    
    # Print the results
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'Unknown')}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG retrieval from ChromaDB.")
    parser.add_argument("--query", type=str, default="What is the role of the attention mechanism?", help="The question to search for.")
    args = parser.parse_args()

    test_retrieval(args.query)