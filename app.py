import streamlit as st
import requests
import json
import os
from pdf_processor import PDFProcessor

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

def main():
    st.title("🔍 Document Search RAG Prototype")
    st.markdown("This is a mini prototype of a RAG (Retrieval-Augmented Generation) system for document search.")
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Main content
    st.markdown("---")
    
    # PDF Upload Section
    st.subheader("📄 PDF Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file to extract text and store in database"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            result = pdf_processor.extract_text_from_pdf(uploaded_file)
        
        if result["success"]:
            st.success(f"✅ Text extracted successfully! ({result['page_count']} pages)")
            
            # Show more detailed information
            if "total_chars" in result:
                st.info(f"📊 Total characters extracted: {result['total_chars']}")
            
            # Show extracted text preview
            with st.expander("📖 View extracted text (first 500 characters)"):
                preview = result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
                st.text(preview)
                
                # Show warning if content is very short
                if len(result["content"]) < 100:
                    st.warning("⚠️ Very little text extracted. This PDF might contain only images or be a scanned document.")
            
            # Save to database
            if st.button("💾 Save to Database"):
                if pdf_processor.save_document_to_db(result["filename"], result["content"], result["page_count"]):
                    st.success("✅ Document saved to database!")
                else:
                    st.error("❌ Failed to save document to database.")
        else:
            st.error(f"❌ Error extracting text: {result['error']}")
    
    st.markdown("---")
    
    # Document Management Section
    st.subheader("📚 Document Management")
    
    # Show all documents
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
    
    # LLM Testing Section
    st.markdown("### Testing LLM API Integration")
    
    test_prompt = st.text_area(
        "Enter test prompt:",
        value="",
        height=100
    )
    
    if st.button("Test LLM API"):
        if api_key:
            with st.spinner("Calling Groq LLM..."):
                response = test_llm_api(test_prompt, api_key)
                st.success("✅ API call successful!")
                st.write("**Response:**")
                st.write(response)
        else:
            st.error("❌ Please enter Groq API key in the sidebar")
            st.info("💡 Groq is completely free with ?? requests/day limit")
    
    # Setup instructions
    st.markdown("---")
    st.subheader("🔧 Setup Instructions")
    st.markdown("""
    **To get responses, get a free Groq API key:**
    
    1. **Sign up for free:**
       - Go to https://console.groq.com/
       - Create a free account
       - No credit card required!
    
    2. **Get your API key:**
       - Go to API Keys section
       - Create a new API key
       - Copy it to the sidebar
    
    3. **Test the model!**
    """)

if __name__ == "__main__":
    main() 