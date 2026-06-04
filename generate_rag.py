import os
import streamlit as st # <-- 1. Added the Streamlit import
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Initialize the models outside the function so they stay cached in memory
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    # 2. Use a placeholder name here! NEVER put the "gsk_" key in this file.
    api_key=st.secrets["GROQ_API_KEY"] 
)

# Pass the specific user's database directly into the query function
def query_rag(query_text: str, user_db):
    print(f"--- Searching Session Database for: '{query_text}' ---")
    
    # Use the database instance passed from the frontend session
    docs = user_db.similarity_search(query_text, k=2)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""
    You are a helpful study assistant. Answer the question in a clear, concise way based ONLY on the provided context. 
    If the answer is not in the context, say "I don't know based on the document." Do not make things up.

    Context from PDF:
    {context}

    Question: {query_text}
    """
    
    response = llm.invoke(prompt)
    return response.content