Download ollama on your system. - link: https://ollama.com/
Install local LLMs model and embedding model using ollama function in terminal, use the below code.
  ollama run <select your own model as per your machine specifics from the> link https://ollama.com/search
  Example - ollama run mistral,
  For embedding model , ollama run nomic-embed-text

Once this is done, use the below the code to install the dependencies via Terminal.
pip install ollama chromadb sentence-transformers streamlit pymupdf langchain-community

Once these depencies are install, simply run command "streamlit run rag_app.py"

Happy learning!!
