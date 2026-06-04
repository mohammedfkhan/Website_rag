import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Get the path to your saved database
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

# 2. Load the exact same math model we used to build the database
print("--- Loading Local Embeddings Engine ---")
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Connect to the existing database on your hard drive
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings_model)

# 4. Ask your database a question!
# 🚨 CHANGE THIS QUESTION to match something actually inside your sample.pdf! 🚨
query = "how does attendance work?" 

print(f"\n🔍 Searching database for: '{query}'...")

# 5. Retrieve the top 2 closest matching chunks
docs = db.similarity_search(query, k=2)

print("\n=========================================")
print("🤖 RETRIEVED KNOWLEDGE CHUNKS")
print("=========================================")
for i, doc in enumerate(docs):
    print(f"\n[Match #{i+1}] (Page {doc.metadata.get('page', 'Unknown') + 1}):")
    print(doc.page_content)
    print("-" * 40)