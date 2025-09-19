import streamlit as st
import os
from dotenv import load_dotenv
import PyPDF2
import numpy as np
from groq import Groq
import tempfile
import pickle
from typing import List, Dict, Any
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize the TF-IDF vectorizer for embeddings
@st.cache_resource
def load_embedding_model():
    """Load and cache the TF-IDF vectorizer"""
    return TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))

# Initialize Groq client
@st.cache_resource
def initialize_groq_client():
    """Initialize and cache the Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Try to get from Streamlit secrets as fallback
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            st.error("GROQ_API_KEY not found in environment variables or Streamlit secrets!")
            st.stop()
    return Groq(api_key=api_key)

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract text from PDF
        text = ""
        with open(tmp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk.strip())
        
        if end == text_length:
            break
            
        start += chunk_size - overlap
    
    return [chunk for chunk in chunks if chunk.strip()]

def create_vector_store(chunks: List[str], vectorizer) -> tuple:
    """Create TF-IDF vector store from text chunks"""
    if not chunks:
        return None, [], None
    
    # Generate TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    return tfidf_matrix, chunks, vectorizer

def search_similar_chunks(query: str, tfidf_matrix, chunks: List[str], vectorizer, k: int = 3) -> List[str]:
    """Search for similar chunks using TF-IDF and cosine similarity"""
    if tfidf_matrix is None or not chunks:
        return []
    
    # Transform query using the same vectorizer
    query_vector = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top k most similar chunks
    top_indices = similarities.argsort()[-k:][::-1]
    
    # Return the most similar chunks
    similar_chunks = []
    for idx in top_indices:
        if idx < len(chunks) and similarities[idx] > 0:
            similar_chunks.append(chunks[idx])
    
    return similar_chunks

def generate_response(query: str, context_chunks: List[str], groq_client) -> str:
    """Generate response using Groq with context from PDF"""
    if not context_chunks:
        context = "No relevant context found in the uploaded document."
    else:
        context = "\n\n".join(context_chunks)
    
    prompt = f"""You are a helpful assistant that answers questions based on the provided context from a PDF document.

Context from the document:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, please say so and provide what information you can based on the available context."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖RAG Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Initialize models
    vectorizer = load_embedding_model()
    groq_client = initialize_groq_client()
    
    # Initialize session state
    if 'tfidf_matrix' not in st.session_state:
        st.session_state.tfidf_matrix = None
    if 'vectorizer_fitted' not in st.session_state:
        st.session_state.vectorizer_fitted = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to ask questions about its content"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text from PDF
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text.strip():
                        # Chunk the text
                        chunks = chunk_text(text)
                        
                        if chunks:
                            # Create vector store
                            tfidf_matrix, processed_chunks, fitted_vectorizer = create_vector_store(chunks, vectorizer)
                            
                            # Store in session state
                            st.session_state.tfidf_matrix = tfidf_matrix
                            st.session_state.vectorizer_fitted = fitted_vectorizer
                            st.session_state.chunks = processed_chunks
                            st.session_state.document_processed = True
                            
                            st.success(f"✅ Document processed successfully!")
                            st.info(f"📊 Created {len(chunks)} text chunks for search")
                        else:
                            st.error("No text chunks could be created from the PDF")
                    else:
                        st.error("No text could be extracted from the PDF")
        
        # Document status
        if st.session_state.document_processed:
            st.success("📄 Document ready for questions!")
            if st.button("Clear Document"):
                st.session_state.tfidf_matrix = None
                st.session_state.vectorizer_fitted = None
                st.session_state.chunks = []
                st.session_state.document_processed = False
                st.session_state.chat_history = []
                st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
        
        # Chat input
        if st.session_state.document_processed:
            user_question = st.chat_input("Ask a question about your document...")
            
            if user_question:
                # Add user question to chat
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Search for relevant chunks
                        similar_chunks = search_similar_chunks(
                            user_question, 
                            st.session_state.tfidf_matrix, 
                            st.session_state.chunks, 
                            st.session_state.vectorizer_fitted
                        )
                        
                        # Generate response
                        response = generate_response(user_question, similar_chunks, groq_client)
                        st.write(response)
                
                # Add to chat history
                st.session_state.chat_history.append((user_question, response))
        else:
            st.info("👆 Please upload and process a PDF document first to start chatting!")
    
    with col2:
        st.header("📊 Document Info")
        
        if st.session_state.document_processed:
            st.metric("Text Chunks", len(st.session_state.chunks))
            st.metric("Chat Messages", len(st.session_state.chat_history))
            
            # Show sample chunks
            if st.session_state.chunks:
                st.subheader("📝 Sample Content")
                with st.expander("View first chunk"):
                    st.text_area(
                        "First chunk preview:",
                        st.session_state.chunks[0][:500] + "..." if len(st.session_state.chunks[0]) > 500 else st.session_state.chunks[0],
                        height=200,
                        disabled=True
                    )
        else:
            st.info("Upload a document to see information here")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>
  🚀 Powered by Streamlit, TF-IDF, Groq, and LLaMA | Built by 
  <a href="https://www.linkedin.com/in/isaac-obeng-obeng/" target="_blank" rel="noopener noreferrer">
    Isaac Obeng
  </a>
</p>

        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
