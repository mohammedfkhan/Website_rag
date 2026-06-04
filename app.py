import streamlit as st
import time
import st_yled
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from generate_rag import query_rag, embeddings_model

st.set_page_config(page_title="Dynamic AI Assistant", layout="centered")
st_yled.init()

st_yled.title("🤖 Shared RAG Assistant", color="#FF4655", font_weight="800")
st_yled.caption("Upload any PDF and start chatting instantly.", color="#A0A0A0")

# 1. Add a File Uploader Component to the sidebar or main page
uploaded_file = st.file_uploader("Drop your study guide or PDF here", type=["pdf"])

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# 2. If a file is uploaded and we haven't processed it yet, build the database in memory
if uploaded_file and st.session_state.vector_db is None:
    with st.spinner("Processing document on-the-fly..."):
        # Read text straight from the uploaded bytes
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            
        # Split the text into clean chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        # Build an EPHEMERAL (in-memory) Chroma store just for this session
        st.session_state.vector_db = Chroma.from_texts(
            texts=chunks, 
            embedding=embeddings_model
        )
        st.success("Document ready! Ask your questions below.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat loop execution
if user_input := st.chat_input("Ask me anything about your document..."):
    if st.session_state.vector_db is None:
        st.warning("Please upload a PDF file first before asking a question!")
    else:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                # Pass the session's custom vector store into your backend logic
                response = query_rag(user_input, st.session_state.vector_db)
            st.markdown(response)
            
        st.session_state.messages.append({"role": "assistant", "content": response})