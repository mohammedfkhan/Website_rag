# Retrieval-Augmented Generation (RAG) Document Assistant

A production-grade Retrieval-Augmented Generation (RAG) application built to parse, embed, and query custom documentation using a local vector store and large language models. This system enables intelligent semantic search and contextual question-answering over uploaded files.

## 🚀 Live Demo

Explore the live application deployed on Streamlit Community Cloud:  
👉 **[mfk-ragfull.streamlit.app](https://mfk-ragfull.streamlit.app/)**

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit
* **Vector Database:** ChromaDB
* **Data Processing:** Python
* **LLM Orchestration:** Groq API / Google AI Studio (Gemini)

## 📋 Features

* **Document Ingestion:** Automatically processes and segments text from uploaded documents.
* **Semantic Search & Retrieval:** Embeds text chunks into a vector store to retrieve the most contextually relevant information.
* **Context-Aware Generation:** Feeds retrieved context into a high-performance LLM to prevent hallucinations and provide accurate answers.
* **Cloud Architecture:** Fully optimized configuration deployment ready for zero-latency execution.

## ⚙️ Local Setup & Installation

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/mohammedfkhan/Website_rag.git
cd Website_rag
```

### 2. Configure the Virtual Environment

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```
GROQ_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```

### 5. Launch the Application

```bash
streamlit run app.py
```

After running this command, open your browser and navigate to `http://localhost:8501`.
