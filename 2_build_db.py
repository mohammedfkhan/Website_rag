import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Get absolute path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "sample.pdf")
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

print("--- 1. Loading and Chunking PDF ---")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"✅ Document split into {len(chunks)} chunks.")

print("\n--- 2. Building Vector Database (Downloading free local model...) ---")
# 2. Use a FREE, local HuggingFace model instead of OpenAI!
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Convert chunks to math and save them
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings_model, 
    persist_directory=DB_DIR
)

print("✅ Success! Your chunks were converted to vectors and saved locally.")