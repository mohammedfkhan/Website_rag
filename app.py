import streamlit as st
import time
import st_yled
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# 1. Added generate_flashcards to the import list
from generate_rag import query_rag, embeddings_model, generate_flashcards

st.set_page_config(page_title="Dynamic AI Assistant", layout="centered")
st_yled.init()

st.title("🤖 Shared RAG Assistant")
st.caption("Upload a PDF to chat or generate an interactive study deck.")

# Add a File Uploader Component
uploaded_file = st.file_uploader("Drop your study guide or PDF here", type=["pdf"])

# Initialize all session states (Memory for the app)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "study_deck" not in st.session_state:
    st.session_state.study_deck = []
if "card_index" not in st.session_state:
    st.session_state.card_index = 0
if "flipped" not in st.session_state:
    st.session_state.flipped = False

# Document Processing Engine
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

    if not chunks:
        st.error(
            "Couldn't extract any text from this PDF. It may be a scanned or "
            "image-only document. Please upload a PDF that contains selectable text."
        )
        st.stop()

    with st.spinner("Building vector database..."):
        # Build an EPHEMERAL (in-memory) vector store just for this session
        st.session_state.vector_db = InMemoryVectorStore.from_texts(
            texts=chunks, 
            embedding=embeddings_model
        )
        st.success("Document ready!")

# Only show the UI if a document is loaded
if st.session_state.vector_db is not None:
    
    # 2. Create Tabs for a clean UI separation
    tab1, tab2 = st.tabs(["💬 Chat with PDF", "🧠 Interactive Flashcards"])
    
    # ==========================================
    # TAB 1: THE STANDARD CHAT INTERFACE
    # ==========================================
    with tab1:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat loop execution
        if user_input := st.chat_input("Ask me anything about your document..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    # Pass the session's custom vector store into your backend logic
                    response = query_rag(user_input, st.session_state.vector_db)
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})

    # ==========================================
    # TAB 2: THE AUTONOMOUS STUDY AGENT
    # ==========================================
    with tab2:
        st.subheader("Automated Study Deck")
        
        # If no deck exists yet, show the generation button
        if not st.session_state.study_deck:
            if st.button("Generate Flashcards from Document", type="primary", use_container_width=True):
                with st.spinner("Agent is extracting core concepts..."):
                    
                    # Call the JSON LLM function
                    study_data = generate_flashcards(st.session_state.vector_db)
                    
                    # Save the results to session state
                    st.session_state.study_deck = study_data.get("flashcards", [])
                    st.session_state.card_index = 0
                    st.session_state.flipped = False
                    st.rerun() # Refresh the page to show the cards
                    
        # If the deck exists, display the interactive UI
        if st.session_state.study_deck:
            deck = st.session_state.study_deck
            current_card = deck[st.session_state.card_index]
            
            # Card Counter
            st.caption(f"Card {st.session_state.card_index + 1} of {len(deck)}")
            
            # The physical card display
            if st.session_state.flipped:
                st.info(f"**Definition:** \n\n {current_card['definition']}")
            else:
                st.warning(f"**Concept:** \n\n {current_card['concept']}")
                
            # The Control Buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Flip Card", use_container_width=True):
                    st.session_state.flipped = not st.session_state.flipped
                    st.rerun()
                    
            with col2:
                if st.button("Next Card", use_container_width=True):
                    # Loop back to 0 if we hit the end of the deck
                    st.session_state.card_index = (st.session_state.card_index + 1) % len(deck)
                    st.session_state.flipped = False
                    st.rerun()
                    
            with col3:
                # Let the user wipe the deck to generate a new one
                if st.button("Discard Deck", use_container_width=True):
                    st.session_state.study_deck = []
                    st.rerun()