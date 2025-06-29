# Document Search RAG Prototype

## What is this application?

This is a **Retrieval-Augmented Generation (RAG)** prototype that demonstrates how to build an intelligent document search system.
The application combines document processing, semantic search, and AI-powered question answering to create a comprehensive
document management solution.

### Key Features:

**📄 Document Processing:**

- Upload PDF documents and extract text content
- Automatic text extraction with page-by-page analysis
- Store documents in a SQLite database for persistent storage

**🔍 Semantic Search:**

- Convert document text into vector embeddings using AI models
- Perform semantic similarity search to find relevant document excerpts
- Use FAISS (Facebook AI Similarity Search) for fast vector retrieval

**🤖 AI-Powered Answers:**

- Generate intelligent responses based on document content
- Combine retrieved document excerpts with Large Language Model (LLM) processing
- Provide context-aware answers that reference specific document sections

### How it works:

1. **Upload & Process:** Upload a PDF document, extract text, and save to database
2. **Vector Embeddings:** Convert document text into numerical vectors for semantic understanding
3. **Search & Retrieve:** When you ask a question, find the most relevant document excerpts
4. **Generate Answer:** Use AI to create a comprehensive answer based on retrieved context

### Technology Stack:

- **Frontend:** Streamlit for user interface
- **Document Processing:** PyPDF for PDF text extraction
- **Vector Database:** FAISS for similarity search
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **AI Model:** Groq API with compound-beta model
- **Database:** SQLite for document storage

## Setup Instructions

### Prerequisites

- Python 3.8+
- Groq API key (free tier available)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free Groq API key from https://console.groq.com/
4. Run the application: `streamlit run app.py`

### Usage

1. Enter your Groq API key in the sidebar
2. Upload a PDF document
3. Save it to the database
4. Ask questions about your documents using the RAG search feature

## Project Structure

- `app.py` - Main Streamlit application
- `pdf_processor.py` - PDF text extraction and database operations
- `rag_processor.py` - RAG functionality with embeddings and similarity search
- `requirements.txt` - Python dependencies
- `documents.db` - SQLite database (created automatically)
