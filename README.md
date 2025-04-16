# RAG App - Local Setup Instructions

Follow the steps below to set up and run the Retrieval-Augmented Generation (RAG) app locally on your system.

---

## 📥 1. Download Ollama

Download and install Ollama from the official website:
👉 [https://ollama.com/](https://ollama.com/)

---

## 🧠 2. Install Local LLMs and Embedding Models

Once Ollama is installed, open your terminal and run the following commands to download the LLM and embedding models:

```bash
# Run a local LLM model (choose one based on your system specs)
ollama run mistral  # Example model

# Run the embedding model
ollama run nomic-embed-text
```

You can explore and choose other models here:
👉 [https://ollama.com/search](https://ollama.com/search)

---

## 📦 3. Install Required Python Dependencies

In the terminal, install the required Python libraries:

```bash
pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community
```

---

## 🚀 4. Run the App

Once the dependencies are installed, start the app using the command:

```bash
streamlit run rag_app.py
```

Note: Ollama should be running in the backgroud before running this python script.
---

## 🎉 You're Ready!

Happy Learning! 🙌

