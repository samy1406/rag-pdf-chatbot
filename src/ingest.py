import argparse
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# The directory where local vector database will be saved
CHROMA_PATH = "./chroma_db"

def load_documents(pdf_path: str):
    """Loads a PDF and returns a list of LangChain Document objects."""
    print(f"Loading {pdf_path}...")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def split_documents(documents):
    """Splits documents into smaller overlapping chunks."""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters in each chunk
        chunk_overlap=200,    # Overlapping characters between chunks
        add_start_index=True  # Optional: keeps track of where each chunk starts
    )
   
    chunks = text_splitter.split_documents(documents)
    return chunks

def build_vector_database(chunks):
    """Embeds text chunks and saves them to a local Chroma database."""
    print("Generating embeddings and building vector database...")
    
    # TODO: Initialize HuggingFaceEmbeddings
    # Hint: Use the argument model_name="all-MiniLM-L6-v2"
    model_name="all-MiniLM-L6-v2"
    hf_embedding = HuggingFaceEmbeddings(model_name = model_name)
    
    # TODO: Create the Chroma vector store
    # Hint: Use the class method Chroma.from_documents()
    # You will need to pass the 'chunks', the embedding model, and persist_directory=CHROMA_PATH
    Chroma.from_documents(
        documents = chunks,
        embedding = hf_embedding,
        persist_directory = CHROMA_PATH
    )

def main():
    # Handle command line arguments so you can run: python ingest.py --file data/sample.pdf
    parser = argparse.ArgumentParser(description="Ingest a PDF into a local Chroma vector database.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF file to ingest")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        return

    # The execution pipeline
    documents = load_documents(args.file)
    chunks = split_documents(documents)
    build_vector_database(chunks)
    
    print(f"Ingestion complete. Database saved to {CHROMA_PATH}")

if __name__ == "__main__":
    main()