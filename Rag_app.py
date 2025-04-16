import os               # For file system operation like adding, deleting, intereact with syatem
import tempfile         # To create and safely store temperoy files of the processed pdf files
 
import chromadb         # open soruce vector database to store and query vector embedding (numerical representation of text). store embedding are compared using cosine similarity
import re               # To clean and normalize file names or text using regex Replacing non-alphanumeric characters with _
import ollama           # To interact with LLMs running locally via Ollama
import streamlit as st  # library for interactive UI application, user friendly web interface
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction, # utility from chromadb that lets you generate embeddings using an Ollama-hosted model, 
)                            # Embeddings are numerical representations of text used for semantic search and stored in vector databases like ChromaDB.
from langchain_community.document_loaders import PyMuPDFLoader # document loader provided by Langchain Community, used specifically to read PDF files using the PyMuPDF backend, It extracts the text content, returning a list of Document objects (one per page). These documents are then split into chunks and used for embedding and vector storage.
from langchain_core.documents import Document             # Representing chunks of data (text + metadata) from loaded documents
from langchain_text_splitters import RecursiveCharacterTextSplitter  # text chunking utility designed to break large text blocks into smaller, overlapping chunks—while trying to keep them semantically meaningful.
from sentence_transformers import CrossEncoder
""" A CrossEncoder is a model that takes a pair of inputs (e.g. a query and a passage) and directly predicts a relevance score.
Unlike embedding-based models (which encode query and document separately), CrossEncoders consider both together for more precise ranking."""
from streamlit.runtime.uploaded_file_manager import UploadedFile   # helper class used internally by Streamlit to represent files uploaded 

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False) # Create temporary file on disk in write-binary (wb) mode.

    temp_file.write(uploaded_file.read())  # Reads file into memory and writes into the temp file

    loader = PyMuPDFLoader(temp_file.name) # Initializes a PyMuPDFLoader from LangChain with the path of the temporary PDF.
    docs = loader.load()                   # Loads the content of the PDF file
    os.unlink(temp_file.name)  # Delete temp file after it has been processed

    text_splitter = RecursiveCharacterTextSplitter(   # Initializes a Recursive Character Splitter from LangChain to chunk the texxt in 400 with overlapping
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs) # returns a list of LangChain Document objects.


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(             # convert text into vectors before storing
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",        # specify the model used for embedding
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")   # persistent ChromaDB client — vector data is stored on disk at the specified path, so that it is retained across restarts.
    return chroma_client.get_or_create_collection(
        name="rag_app",                # returns a collection named "rag_app"
        embedding_function=ollama_ef,  # Specify Ollama’s embedding model, that is used to generate the vectors added documents.
        metadata={"hnsw:space": "cosine"},   # Tells ChromaDB to use cosine similarity for nearest neighbor search, most common similarity measure for semantic search
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    collection = get_vector_collection()      # gives access to ChromaDB collection (named "rag_app")
    documents, metadatas, ids = [], [], []    # Initializes three empty lists

    for idx, split in enumerate(all_splits):  # Loops through all the document chunks, using enumerate() for both index and the chunk.
        documents.append(split.page_content)  # actual text of the chunk, added to documents
        metadatas.append(split.metadata)      # metadata from the document (e.g. page number) → added to metadatas
        ids.append(f"{file_name}_{idx}")      # creates a unique ID by combining the file name and the index of the chunk → added to ids.

    collection.upsert(                        # Adds the documents into the ChromaDB vector store using the upsert() method
        documents=documents,                  
        metadatas=metadatas,                  
        ids=ids,                              
    )
    st.success("Data added to the vector store!")   # Uses Streamlit to show a success message, when PDF is successfully processed.

def delete_from_vector_collection(file_name_prefix: str):  # function that takes a string input for file names
    """Deletes all vectors whose IDs start with the given file_name_prefix."""
    collection = get_vector_collection()        # Calls function to get access to the ChromaDB collection
    all_ids = collection.get(ids=None)["ids"]   # retrieve all stored vector IDs in the collection
    ids_to_delete = [doc_id for doc_id in all_ids if doc_id.startswith(file_name_prefix)]  # list comprehension that checks each doc_id from all_ids
    
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)  # if match found, call function to remove them from the vector database
        st.success(f"🗑️ Deleted {len(ids_to_delete)} chunks for '{file_name_prefix}' from the vector store.")  # Display success message.
    else:
        st.warning(f"No vector data found with prefix: {file_name_prefix}")  # if no match found, show a warning message


def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    response = ollama.chat(      # Sends the context + question to the model
        model="mistral:latest",  # Select the LLM model
        stream=True,             # Enable real-time streaming
        messages=[
            {
                "role": "system",       # Set the LLM's tone and behavior
                "content": system_prompt,
            },
            {
                "role": "user",         # Send actual context + question
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:              # Stream response in real-time
        if chunk["done"] is False:
            yield chunk["message"]["content"]      # Return chunks as they're generated
        else:
            break                       # Stop when generation ends


def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # pretrained cross-encoder model from Hugging Face.
    ranks = encoder_model.rank(prompt, documents, top_k=3)                 # rank the documents against the prompt and return top 3 documents
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":        # run only when the file is executed directly, not imported
    # Document Upload Area
    with st.sidebar:              # sidebar layout using streamlit UI
        st.set_page_config(page_title="RAG Question Answer")  # Set browser tab title
        uploaded_files = st.file_uploader(     #  file uploader widget to the sidebar. accept multip[le PDf files]
            "**📑 Upload PDF files for QnA**", type=["pdf"], accept_multiple_files=True
        )

        process = st.button(     # button labled as Process to process the uploaded the documents
            "⚡️ Process",
        )
        if uploaded_files and process:   # procees onlt when Files were uploaded and the user clicked the "Process" button
            for uploaded_file in uploaded_files:     # iterate through each uploaded file
                normalize_uploaded_file_name = re.sub(r"[^\w]", "_", uploaded_file.name)   # clean file, Replaces character that's not a word character (letters, digits, underscores) with _
                with st.spinner(f"Processing {uploaded_file.name}..."):    #  loading spinner in the UI with the file name
                    all_splits = process_document(uploaded_file)           # Calls the process_document() function
                    add_to_vector_collection(all_splits, normalize_uploaded_file_name)    # Sends the split document chunks to ChromaDB

        st.markdown("---")       # Add horizontal rule line
        st.subheader("🗑️ Delete Stored Vector Data")    # subheading for the delete section
        file_to_delete = st.text_input("Enter filename (without extension, e.g. 'Report')")   # text input for file name
        delete_button = st.button("❌ Delete Vector Data")   # delete button to trigger delete logic

        if delete_button and file_to_delete:   # only proceed when both filename is provided and delete button is clicked.
            normalized_name = file_to_delete.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})  # Cleans the input filename to match the original file name when provided for proceessing
            )
            delete_from_vector_collection(normalized_name) # function to delete the vector entries

    # Question and Answer Area
    st.header("🗣️ RAG Question Answer")  # Displays header at the top (Gives clear indication of where user can interact with the app)
    prompt = st.text_area("**Ask a question related to your document:**")   # a text area input field in the UI to captures the user's question, which will be used to perform semantic search
    ask = st.button(  
        "🔥 Ask",   # A labeled button acting as trigger point to generate the answer 
    )

    if ask and prompt:     # Proceed only when the user submits a question
        results = query_collection(prompt)    # Retrieve matching chunks using semantic search from chromaDB
        context = results.get("documents")[0] # extract the list of retrieved documents, and selects the first group to get the actual list of text chunks to re-rank them for better accuracy
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context)  # Re-rank the retrieved documents, Return the top 3 most relevant text chunks and their corresponding indices, Vector search is fast but not always precise,
        response = call_llm(context=relevant_text, prompt=prompt) # call the llm via ollama to stream the response llm's response chunk by chunk.
        st.write_stream(response)             # Displays the LLM’s streamed output in the app in real-time.

        with st.expander("See retrieved documents"):
            st.write(results)     # A collapsible section showing all results from the initial vector search.

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)             # The IDs of the top-ranked chunks and the actual content of those chunks.
            st.write(relevant_text)

