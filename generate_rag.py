import os
import streamlit as st
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Initialize the models outside the function so they stay cached in memory
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
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
    
    response = get_llm().invoke(prompt)
    return response.content

# Make sure this is pushed all the way to the left!
def generate_flashcards(user_db, topic="key concepts and definitions"):
    print(f"--- Extracting Flashcards for: '{topic}' ---")
    
    # 1. Pull the largest chunk of relevant context from the syllabus/notes
    docs = user_db.similarity_search(topic, k=4) 
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. The Strict JSON Prompt
    prompt = f"""
    You are an autonomous study agent. Read the provided course materials and extract the 5 most critical concepts into flashcards.
    You must output ONLY valid, parsable JSON. Do not include any conversational text, introductions, or explanations.

    Output format:
    {{
      "flashcards": [
        {{
          "concept": "Name of the concept",
          "definition": "Clear, concise definition"
        }}
      ]
    }}

    Course Materials:
    {context}
    """
    
    # 3. Force the Llama 3 model into JSON mode
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
        model_kwargs={"response_format": {"type": "json_object"}} 
    )
    
    response = llm.invoke(prompt)
    
    # 4. Parse the text back into a usable Python dictionary
    try:
        # Clean up the string just in case the LLM wrapped it in markdown
        raw_text = response.content.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3]
            
        return json.loads(raw_text)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return {"flashcards": []} # Return empty list if it fails, so the app doesn't crash