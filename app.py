import streamlit as st
import requests
import json
import os
from pdf_processor import PDFProcessor
from rag_processor import RAGProcessor

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "compound-beta"

def test_llm_api(prompt, api_key):
    """Test LLM API with Groq compound-beta model"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "Invalid API key. Please check your Groq API key."
        elif response.status_code == 429:
            return "You've used your daily free limit."
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

def rag_llm_api(prompt, context, api_key):
    """RAG-enhanced LLM API call"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    rag_prompt = f"""Based on the following document excerpts, please answer the question.

Document excerpts:
{context}

Question: {prompt}

Please provide a comprehensive answer based on the document excerpts above. If the information is not available in the excerpts, please say so."""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": rag_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 800
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "Invalid API key. Please check your Groq API key."
        elif response.status_code == 429:
            return "You've used your daily free limit."
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

def main():
    st.title("🔍 Document Search RAG Prototype")
    
    # Simple user-friendly description
    st.markdown("""
    
    This application helps you **search and ask questions about your PDF documents** using AI. 
    
    ### How to use this app:
    
    1. **📄 Upload a PDF** - Upload any PDF document you want to search through
    2. **💾 Save to Database** - Store your document so the AI can access it
    3. **🤖 Ask Questions** - Type questions about your documents and get AI-powered answers
    4. **🔍 Search Documents** - Find specific parts of your documents that match your question

    """)
    
    pdf_processor = PDFProcessor()
    rag_processor = RAGProcessor()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Setup instructions in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Setup Instructions")
    st.sidebar.markdown("""
    **To get responses, get a free Groq API key:**
    
    1. **Sign up for free:**
       - Go to https://console.groq.com/
       - Create a free account
       - No credit card required!
    
    2. **Get your API key:**
       - Go to API Keys section
       - Create a new API key
       - Copy it to the field above
    
    3. **Have fun testing the model!**
    
    """)
    
    st.markdown("---")
    
    st.subheader("📄 PDF Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to extract text and store in database"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        with st.spinner("Extracting text from PDF..."):
            result = pdf_processor.extract_text_from_pdf(uploaded_file)
        
        if result["success"]:
            st.success(f"✅ Text extracted successfully! ({result['page_count']} pages)")
            
            if "total_chars" in result:
                st.info(f"📊 Total characters extracted: {result['total_chars']}")
            
            with st.expander("📖 View extracted text (first 500 characters)"):
                preview = result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
                st.text(preview)
                
                if len(result["content"]) < 100:
                    st.warning("⚠️ Very little text extracted. This PDF might contain only images or be a scanned document.")
            
            if st.button("💾 Save to Database"):
                if pdf_processor.save_document_to_db(result["filename"], result["content"], result["page_count"]):
                    st.success("✅ Document saved to database!")
                    
                    with st.spinner("Processing document for RAG (this may take a few minutes)..."):
                        documents = pdf_processor.get_all_documents()
                        latest_doc = documents[0]  # Most recent
                        
                        if rag_processor.process_document_for_rag(latest_doc["id"], result["content"], result["page_count"]):
                            st.success("✅ Document processed for RAG!")
                        else:
                            st.error("❌ Failed to process document for RAG.")
                else:
                    st.error("❌ Failed to save document to database.")
        else:
            st.error(f"❌ Error extracting text: {result['error']}")
    
    st.markdown("---")
    
    st.subheader("📚 Document Management")
    
    documents = pdf_processor.get_all_documents()
    
    if documents:
        st.write(f"**Total documents in database: {len(documents)}**")
        
        for doc in documents:
            with st.expander(f"📄 {doc['filename']} ({doc['page_count']} pages)"):
                st.write(f"**Uploaded:** {doc['upload_date']}")
                st.write(f"**Content preview:**")
                preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                st.text(preview)
    else:
        st.info("No documents uploaded yet. Upload a PDF to get started!")
    
    st.markdown("---")
    
    st.subheader("🔍 RAG Document Search")
    
    search_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What does the report say about AI policy?",
        help="Ask questions about your uploaded documents"
    )
    
    if st.button("🤖 Ask RAG"):
        if search_query and api_key:
            with st.spinner("Generating RAG response..."):
                context = rag_processor.get_rag_context(search_query, top_k=3)
                
                if context:
                    response = rag_llm_api(search_query, context, api_key)
                    st.success("✅ RAG response generated!")
                    st.write("**Answer:**")
                    st.write(response)
                    
                    with st.expander("📖 Context used"):
                        st.write(context)
                else:
                    st.warning("No relevant context found. Try a different question.")
        elif not api_key:
            st.error("❌ Please enter Groq API key in the sidebar")
        else:
            st.error("Please enter a question.")
    
    if st.button("🔍 Search Documents"):
        if search_query:
            with st.spinner("Searching documents..."):
                results = rag_processor.search_documents(search_query, top_k=3)
            
            if results:
                st.success(f"✅ Found {len(results)} relevant excerpts")
                
                for result in results:
                    with st.expander(f"📄 {result['filename']} (Score: {result['similarity_score']:.3f})"):
                        st.write(result['chunk_text'])
            else:
                st.warning("No relevant documents found. Try a different query.")
        else:
            st.error("Please enter a search query.")
    
    st.markdown("---")
    
    st.subheader("🔄 Compare: RAG vs Regular AI")
    
    st.markdown("""
    **Test the difference:** Ask the same question to both systems and see how they respond differently.
    
    - **🤖 RAG (above):** Uses your uploaded documents to provide accurate, fact-based answers
    - **🧠 Regular AI (below):** Relies only on its general knowledge (may not know about your specific documents)
    """)
    
    comparison_prompt = st.text_area(
        "Enter the same question you asked above:",
        value=search_query if search_query else "",
        height=100,
        help="Try asking the same question to see the difference between RAG and regular AI"
    )
    
    if st.button("🧠 Ask Regular AI (No Document Context)"):
        if api_key and comparison_prompt:
            with st.spinner("Calling AI without document context..."):
                response = test_llm_api(comparison_prompt, api_key)
                st.success("✅ Regular AI response generated!")
                st.write("**Regular AI Answer (no document context):**")
                st.write(response)
                
                st.info("💡 **Notice the difference:** The regular AI can only use its general knowledge, while RAG above used your specific documents!")
        elif not api_key:
            st.error("❌ Please enter Groq API key in the sidebar")
        else:
            st.error("Please enter a question to compare.")

if __name__ == "__main__":
    main() 