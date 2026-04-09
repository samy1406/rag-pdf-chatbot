# RAG Chatbot over PDFs

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)

## Problem Statement
Large Language Models have fixed knowledge cutoffs and cannot natively read proprietary documents. [cite_start]This project implements a Retrieval-Augmented Generation (RAG) pipeline to allow users to query custom PDF documents. It extracts, embeds, and indexes document text, enabling an LLM to answer questions using strictly the provided context.

## Architecture & Workflow
This project is designed with a hybrid compute architecture:
1. **CPU Pipeline (GitHub Codespaces):** Handles document ingestion. Uses LangChain to parse PDFs, recursively chunk text, and generate dense vectors using lightweight sentence-transformers. [cite_start]The vectors are stored in a local ChromaDB instance.
2. **GPU Pipeline (Google Colab):** Handles the LLM inference. Loads the finalized ChromaDB index and connects it to an open-source LLM for context-aware question answering.

## Setup Instructions (Codespaces)

1. **Clone and Install:**
   ```bash
   git clone [https://github.com/samy1406/rag-pdf-chatbot.git](https://github.com/samy1406/rag-pdf-chatbot.git)
   cd rag-pdf-chatbot
   pip install -r requirements.txt

2. **Run Data Ingestion:**
    ```bash
    Place your target PDF in the data/ folder and run the pipeline to build the vector database.
    python src/ingest.py --file data/sample.pdf