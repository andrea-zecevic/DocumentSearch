import streamlit as st
import requests
import json
import os

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
    st.markdown("### Testing LLM API Integration")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    
    # Main content
    st.markdown("---")

    
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