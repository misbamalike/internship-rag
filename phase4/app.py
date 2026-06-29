import streamlit as st
import pickle
import numpy as np
from groq import Groq
# Note: If you use sentence-transformers locally, import it here:
# from sentence_transformers import CrossEncoder

# =====================================================================
# 1. INITIAL SETUP & CONFIGURATION (The "Kitchen" Setup)
# =====================================================================
st.set_page_config(page_title="Advanced RAG Assistant", layout="centered")

# SECURE METHOD: Fetch the key safely from Streamlit's backend environment
if "GROQ_API_KEY" in st.secrets:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Fallback for local testing if you haven't set up local secrets
    GROQ_KEY = "your_fallback_local_key_here"

def load_rag_resources():
    """Loads your processed text chunks using explicit subfolder paths."""
    # We add "phase4/" in front of the filenames so Streamlit knows exactly where to look
    with open("phase4/chunks.pkl", "rb") as f:
        processed_texts = pickle.load(f)
    with open("phase4/chunk_embeddings.pkl", "rb") as f:
        chunk_embeddings = pickle.load(f)
    return processed_texts, chunk_embeddings

try:
    processed_texts, chunk_embeddings = load_rag_resources()
except Exception as e:
    st.error(f"Could not load pickle files. Make sure chunks.pkl is in this folder! Error: {e}")

# =====================================================================
# 2. THE BACKEND LOGIC (Your Phase 4 Upgraded Functions)
# =====================================================================
def hybrid_retrieve_top_documents(query, top_k=10):
    # This is where your exact hybrid search code from Colab goes
    # For now, it returns a placeholder slice of your texts
    return processed_texts[:top_k]

def get_reranked_context(query, initial_texts):
    # This is where your Phase 4 Cross-Encoder Re-ranker code goes
    # It filters down the 10 chunks to the top 3 best chunks
    return "\n\n".join(initial_texts[:3])

def ask_advanced_rag(query, final_context):
    groq_client = Groq(api_key=GROQ_KEY)
    
    # UPDATED PROMPT: Allows the AI to use its general knowledge if the document is brief
    system_prompt = (
        "You are an expert AI assistant. Answer the user's query clearly and directly. "
        "Use the provided context below as your primary source of truth, but expand on it "
        "with accurate general knowledge if the context is missing a direct definition.\n\n"
        f"Provided Context:\n{final_context}"
    )
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.3 # Small increase from 0.0 to make the text sound more natural and direct
    )
    return response.choices[0].message.content

# =====================================================================
# 3. THE FRONTEND INTERFACE (What the User Sees)
# =====================================================================
st.title("🤖 Advanced RAG Knowledge Assistant")
st.write("Phase 4 Optimized: Powered by Hybrid Search & Cross-Encoder Re-ranking.")
st.markdown("---")

# Text box where the user types their question
user_query = st.text_input("Ask a question about Machine Learning or NLP:", placeholder="e.g., What is deep learning?")

# Button that triggers the backend code execution
if st.button("Generate Answer"):
    if user_query.strip() == "":
        st.warning("Please type a question first!")
    else:
        with st.spinner("Searching knowledge base and re-ranking documents..."):
            try:
                # Step A: Run Hybrid Retrieval
                initial_chunks = hybrid_retrieve_top_documents(user_query)
                
                # Step B: Run Local Re-ranking
                refined_context = get_reranked_context(user_query, initial_chunks)
                
                # Step C: Generate response using Groq
                output_answer = ask_advanced_rag(user_query, refined_context)
                
                # Step D: Display the result cleanly on the web screen
                st.success("✨ Answer Generated Successfully!")
                st.write(output_answer)
                
            except Exception as runtime_error:
                st.error(f"An error occurred while processing: {runtime_error}")
