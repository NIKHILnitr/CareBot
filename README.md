# CareBot – AI Medical Chatbot

CareBot is an AI-powered medical chatbot built using Retrieval-Augmented Generation (RAG), LangChain, FAISS, Hugging Face Transformers, and Streamlit. The chatbot answers healthcare-related queries by retrieving relevant information from medical encyclopedia PDFs and generating contextual responses using Large Language Models (LLMs).

---

# Features

* AI-powered medical question answering
* Retrieval-Augmented Generation (RAG) pipeline
* Semantic search using FAISS vector database
* PDF-based medical knowledge retrieval
* Source attribution with PDF filename and page numbers
* Streamlit chat interface
* Hugging Face Transformer integration
* Context-aware response generation
* Optimized chunking and retrieval pipeline

---

# Tech Stack

## Frontend

* Streamlit

## Backend

* Python
* LangChain
* Hugging Face Transformers
* FAISS Vector Database
* Sentence Transformers

## Models

* google/flan-t5-large
* sentence-transformers/all-MiniLM-L6-v2

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader (PyPDFLoader)
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Store
      ↓
User Query
      ↓
Similarity Search
      ↓
Retrieved Context
      ↓
Flan-T5 Large
      ↓
Generated Medical Response
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/NIKHILnitr/CareBot.git
cd CareBot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Create Vector Database

Run the following command to process PDFs and create the FAISS vector database:

```bash
python create_memory_for_llm.py
```

This will:

* Load medical PDF files
* Split text into chunks
* Generate embeddings
* Store vectors inside FAISS

---

# Run the Application

```bash
streamlit run medibot.py
```

---

# Example Questions

* What are common symptoms of diabetes?
* How is migraine treated?
* What causes high blood pressure?
* How can asthma be managed?
* What are treatment options for cancer?

---

# Project Workflow

## Step 1 – Load Medical PDFs

Medical encyclopedia PDFs are loaded using PyPDFLoader.

## Step 2 – Create Text Chunks

Documents are divided into smaller chunks using RecursiveCharacterTextSplitter.

## Step 3 – Generate Embeddings

Sentence Transformer embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Step 4 – Store Embeddings in FAISS

Embeddings are stored inside a FAISS vector database for fast semantic retrieval.

## Step 5 – User Query Processing

The user query is converted into embeddings and matched against stored vectors.

## Step 6 – Response Generation

Relevant chunks are passed to Flan-T5 Large through LangChain RetrievalQA to generate contextual answers.

---

# Challenges Faced

* Noisy OCR text from PDFs
* Incomplete sentence extraction
* Irrelevant chunk retrieval
* Hallucinated responses from LLMs
* Large model deployment constraints

---

# Optimizations Implemented

* Improved chunk size and overlap strategy
* Added OCR text cleaning
* Reduced retrieval noise using optimized k-value
* Prompt engineering for concise answers
* Added source attribution with page references

---

# Future Improvements

* Conversational memory support
* Hybrid retrieval (BM25 + Vector Search)
* Medical image understanding
* Voice-based interaction
* Cloud deployment with GPU inference
* Authentication and user profiles
* Better reranking models


---

# GitHub Repository

[https://github.com/NIKHILnitr/CareBot](https://github.com/NIKHILnitr/CareBot)

---

# Author

Nikhil Bhoi
