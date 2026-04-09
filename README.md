# RAG Chatbot over PDFs

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Zephyr_7B-yellow.svg)

## Problem Statement
Large Language Models have fixed knowledge cutoffs and cannot natively read proprietary documents. This project implements a privacy-preserving Retrieval-Augmented Generation (RAG) pipeline to allow users to query custom PDF documents. It extracts, embeds, and indexes document text, enabling a locally-run open-source LLM to answer questions using strictly the provided context without relying on paid APIs.

## Architecture & Workflow
This project utilizes a hybrid compute architecture to balance cost and performance:
1. **CPU Pipeline (GitHub Codespaces):** Handles document ingestion. Uses LangChain to parse PDFs, recursively chunk text, and generate dense vectors using the lightweight `all-MiniLM-L6-v2` sentence-transformer. The vectors are stored in a local ChromaDB instance.
2. **GPU Pipeline (Google Colab):** Handles LLM inference. Loads the ChromaDB index and connects it to a 4-bit quantized `Zephyr-7B-beta` model via HuggingFace and `bitsandbytes` for context-aware question answering on a free T4 GPU.

![Architecture Diagram](./demo/architecture.png) *(Note: Add diagram later)*

## Setup Instructions

### Part 1: Data Ingestion (Local/Codespaces)
1. Clone the repository and install dependencies:
   ```bash
   git clone [https://github.com/samy1406/rag-pdf-chatbot.git](https://github.com/samy1406/rag-pdf-chatbot.git)
   cd rag-pdf-chatbot
   pip install -r requirements.txt

2. Place your target PDF in the data/ folder and run the ingestion script to build the vector database:
    ```bash
    python src/ingest.py --file data/sample.pdf

3. Test retrieval independently to ensure vector math is accurate:
    ```bash
    python src/retrieve.py --query "Your test question here"

### Part 2: Generation (Google Colab)
1. Open notebooks/rag_generation.ipynb in Google Colab.

2. Ensure the runtime is set to T4 GPU.

3. Run the cells to clone the repo, rebuild the Chroma database, load the 4-bit quantized Zephyr model, and execute the LangChain Expression Language (LCEL) chain.

### What I Learned
- Semantic Chunking: Implemented recursive character splitting, learning how chunk size and overlap directly impact retrieval accuracy.

- Vector Embeddings: Utilized lightweight HuggingFace models to generate embeddings efficiently on CPU environments.

- LLM Quantization: Successfully loaded a 7-billion parameter model into a constrained GPU environment using bitsandbytes NormalFloat4 (nf4) quantization.

- LCEL: Built a modern LangChain execution pipeline to format retrieved documents and enforce strict prompt templating.
