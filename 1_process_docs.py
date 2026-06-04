import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Python version guard ──────────────────────────────────────────────────────
if sys.version_info < (3, 8):
    sys.exit("❌ Python 3.8+ is required. Run with: python3 1_process_docs.py")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH   = os.path.join(SCRIPT_DIR, "sample.pdf")

print("=========================================")
print("🔍 AUTOMATIC PATH RESOLUTION")
print("=========================================")
print(f"Python version : {sys.version}")
print(f"Script location: {SCRIPT_DIR}")
print(f"Looking for PDF: {PDF_PATH}")
print(f"File exists?   : {os.path.exists(PDF_PATH)}")
print("=========================================\n")

# ── Step 1: Load ──────────────────────────────────────────────────────────────
print("--- Step 1: Loading Document ---")

if not os.path.exists(PDF_PATH):
    sys.exit(
        f"❌ ERROR: Could not find 'sample.pdf' at:\n  {PDF_PATH}\n"
        "Make sure sample.pdf is in the same folder as this script."
    )

try:
    loader    = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    if not documents:
        sys.exit("❌ ERROR: PDF loaded but contains no pages.")

    print(f"✅ Loaded {len(documents)} page(s) from the PDF.")

    # ── Step 2: Chunk ──────────────────────────────────────────────────────────
    print("\n--- Step 2: Chunking Text ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        sys.exit("❌ ERROR: Document was loaded but produced no chunks.")

    print(f"✅ Split into {len(chunks)} searchable chunk(s).")

    # ── Step 3: Preview ────────────────────────────────────────────────────────
    print("\n--- Preview of Chunk 1 ---")
    print(chunks[0].page_content)

    print("\n✅ Done — document is ready for embedding.")

except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {e}")
    raise   # shows full traceback so you can debug